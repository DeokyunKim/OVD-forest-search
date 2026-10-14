#!/usr/bin/env python
"""Phase 2 LoRA trainer for GroundingDINO Swin-B, under the frozen spec.

Spec: the frozen-hyperparameter specification §1 (LoRA), §2 (placement),
§3 (optimizer), §4.0/§4.1 (data policy and loop), §5 (resolution).
Pre-registered collapse thresholds: the training risk register §5.

What this is not
----------------
It is not ``train_gdino_hinge.py`` (research tree) (now in ``backup/scripts_20260808/``).
That script's *integration pattern* — bridge the external ZiRaGroundingDINO
repo through ``sys.path``, resolve the BERT hub cache before any transformers
import, decompose fused attention, inject the repo's own ``lora.Linear`` — is
reused here because it is the only pattern known to work against this
codebase. Its **recipe** is retired: alpha 8, 216 modules spanning the frozen
Swin backbone and BERT, LR 1e-3, grad clip 0.1, task-loss hinge. None of that
appears below.

It also does not use detectron2's ``do_train``. Spec §3 needs three learning
rates on three parameter roles, §3 needs a cosine schedule with 3% warmup
computed against the frozen 5,667-step budget, §4.1 needs bf16 with
``use_reentrant=False`` checkpointing and grad accumulation to an effective
batch of 16, and the risk register needs an **in-process** collapse guard that
can abort. The retired system's out-of-process collapse watcher is the single
cheapest thing that design got wrong: a run could burn hours after collapsing.

The gate before step 1
----------------------
Nothing trains until, on the **live** model:

1. every §2 group matched exactly the module count measured on Swin-B
   (per group — 72 is also reachable as 36 + 36 + 0);
2. zero adapters landed in ``backbone`` or ``bert``;
3. zero adapters landed outside the feature enhancer and the decoder;
4. ``bbox_embed`` / ``class_embed`` exist and are trainable;
5. every adapter's live ``.scaling`` reads alpha/r = 2.0 (alpha was silently
   halved once on this project, and a config inspection would have passed);
6. gradient actually reaches the checkpointed encoder adapters after the first
   backward. The external repo calls ``torch.utils.checkpoint.checkpoint``
   without ``use_reentrant=``; under the reentrant default, a checkpointed
   block whose input tensors do not require grad returns **no** gradient at
   all, and every encoder adapter would train on zeros in silence.

Usage (gate run, seed 0)::

    CUDA_VISIBLE_DEVICES=0 TMPDIR="$TMPDIR" \\
      ~/venvs/uc-gdino/bin/python train.py --model gdino \\
        --seed 0 --gate-only 1000 --device cuda:0 \\
        --output-dir outputs/v6/train/gdino_seed0_gate

Always pin the GPU with ``CUDA_VISIBLE_DEVICES`` and keep ``--device cuda:0``:
the external repo's detectron2-style model holds ``self.device`` as a plain
attribute that ``nn.Module.to()`` never updates, and non-zero device indices
have produced ``illegal memory access`` and silently unstable results.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Before any first-party import: this file is run as a script, so the repo
# root is not on sys.path yet. It has to come first, not beside the other path
# work further down, because the imports below need it already done.
_sys.path.insert(0, str(next(q for q in _Path(__file__).resolve().parents
                             if (q / 'pyproject.toml').is_file())))


import os
from pathlib import Path

from utils.runner.paths import paths

import argparse
import functools
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# ── Environment BEFORE any transformers import (CLAUDE.md gotchas) ───────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# The tokenizer runs before dataloader workers fork; its own warning recommends
# setting this explicitly rather than letting it self-disable per worker.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.pop("HF_HOME", None)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / 'pyproject.toml').is_file())
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "groundingdino_swinb_v6_lora_a32.py"
#: The same file `configs/models.yaml` names for evaluation, at the same flat
#: path. It used to be spelled as a HuggingFace cache tree
#: (`models--…/snapshots/<sha>/…`), so training and evaluation expected two
#: different layouts for one weight file and only training failed -- the same
#: shape of defect as the two entry points disagreeing about `output_dir`.
DEFAULT_CHECKPOINT = (paths().checkpoint_root / "gdino"
                      / "groundingdino_swinb_cogcoor.pth")


def _resolve_bert_cache() -> str:
    """First hub cache that really holds ``bert-base-uncased``.

    The home cache is not dependable: the root filesystem sits near capacity
    and its HF cache may be evicted between runs. When that happened here,
    eviction emptied the snapshot between a passing smoke test and the full
    launch, and the run died 40s in. Probing for a snapshot that actually
    contains ``config.json`` turns a delayed failure into a deterministic
    choice made before transformers is imported.
    """
    candidates = [
        os.environ.get("HF_HUB_CACHE"),
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    for root in candidates:
        if not root:
            continue
        snapshots = os.path.join(root, "models--bert-base-uncased", "snapshots")
        if not os.path.isdir(snapshots):
            continue
        for snap in os.listdir(snapshots):
            if os.path.exists(os.path.join(snapshots, snap, "config.json")):
                return root
    raise RuntimeError(
        "bert-base-uncased not found in any known hub cache "
        f"({[c for c in candidates if c]}); GroundingDINO's text encoder cannot "
        "be built offline")


os.environ["HF_HUB_CACHE"] = _resolve_bert_cache()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The GroundingDINO package, with its compiled CUDA extension, lives in a
# checkout outside this repository. Evaluation finds it through `UC_GDINO_REPO`
# and `ensure_external_repo`; training used to carry a second, hardcoded answer,
# so the two could resolve to different checkouts on the same machine -- and on
# any other machine training put a path that does not exist on `sys.path` and
# failed later, at `import groundingdino`, saying nothing about why.
from utils.runner.repo_bridge import ensure_external_repo  # noqa: E402

_external_repo = os.environ.get("UC_GDINO_REPO")
if not _external_repo:
    raise RuntimeError(
        "UC_GDINO_REPO is not set. Point it at the Grounding DINO checkout "
        "holding the `groundingdino` package, the same one `configs/models.yaml` "
        "reads for evaluation.")
ensure_external_repo(_external_repo)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.utils.checkpoint as torch_checkpoint  # noqa: E402

from utils.runner.logging_utils import get_logger  # noqa: E402
from utils.train.checkpoint import build_manifest, save_v6_checkpoint  # noqa: E402
from utils.train.loop import (  # noqa: E402
    LoopConfig,
    StepResult,
    run_training_loop,
)
from utils.train.collapse_guard import (  # noqa: E402
    ACTION_ABORT,
    ACTION_OK,
    CollapseGuard,
    CollapseThresholds,
    ProbeMetrics,
    measure_discrimination,
)
from utils.data.dataset_register import load_v6_dataset_dicts  # noqa: E402
from models.gdino.lora import (  # noqa: E402
    DEFAULT_SCOPE,
    SCOPES,
    build_param_groups,
    build_placement,
    collect_targets,
    describe_param_groups,
    describe_schedule,
    expected_group_counts,
    group_counts,
    make_lr_lambda,
    scope_groups,
)
from models.shared.lora_spec_v6line import (  # noqa: E402
    SpecViolationError,
    assert_compliant,
    assert_scaling_compliant,
    format_report,
    verify_injection,
    verify_lora_scaling,
)
from utils.protocol.train_caption import training_caption, training_class_names  # noqa: E402
from utils.train.training_schedule import (  # noqa: E402
    EFFECTIVE_BATCH_SIZE,
    LORA,
    MAX_GRAD_NORM,
    TOTAL_STEPS,
    USE_REENTRANT,
    plan_batch,
)

logger = get_logger("v6.train")

MODEL_KEY = "gdino"
CHECKPOINT_EVERY_STEPS = 2_000
#: Spec §5: short side 800, long side <= 1333 (evaluation and probing).
EVAL_SHORT_EDGE = 800
EVAL_MAX_SIZE = 1333
#: Standard DETR/GDINO multi-scale training augmentation, unchanged from every
#: previous forest arm in this project so the comparison stays like-for-like.
TRAIN_SHORT_EDGES = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
#: Score floor of the collapse probe. NOT spec §6 (which fixes 0.0 for AP).
#: The pre-registered healthy/collapsed numbers in the risk register come from
#: the retired ``measure_collapse_trajectory.py``, whose ``--score-thresh``
#: default is 0.05; the guard's thresholds are only meaningful against a probe
#: measured the same way.
PROBE_SCORE_THRESHOLD = 0.05


# --- gradient checkpointing --------------------------------------------------

def force_non_reentrant_checkpointing() -> None:
    """Make every ``checkpoint.checkpoint`` call non-reentrant (§4.1).

    ``transformer_for_adapter.py`` and ``swin_transformer.py`` call
    ``checkpoint.checkpoint(fn, *args)`` with no ``use_reentrant=``. Torch 2.4
    then uses the reentrant implementation, which computes **no** gradient when
    none of ``*args`` requires grad. At encoder layer 0 the image features come
    from a frozen backbone through a frozen ``input_proj``, so nothing requires
    grad, the block runs under ``no_grad``, its output also does not require
    grad — and the same holds for every following layer. All encoder adapters
    would receive exactly zero gradient, with no error and a plausible loss
    curve.

    Both call sites reach the function through the *module attribute*, so
    rebinding it here covers them without touching the external repo on disk.
    """
    original = torch_checkpoint.checkpoint
    if getattr(original, "_v6_non_reentrant", False):
        return

    def checkpoint(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", USE_REENTRANT)
        return original(function, *args, **kwargs)

    checkpoint._v6_non_reentrant = True  # type: ignore[attr-defined]
    torch_checkpoint.checkpoint = checkpoint  # type: ignore[assignment]
    logger.info("gradient checkpointing forced to use_reentrant=%s (spec §4.1)",
                USE_REENTRANT)


def enable_bf16_deformable_attention() -> None:
    """Run multi-scale deformable attention in fp32 under bf16 autocast.

    The compiled CUDA op raises ``"ms_deform_attn_forward_cuda" not implemented
    for 'BFloat16'``. ``ms_deform_attn.py`` already special-cases fp16 by
    up-casting ``value``/``sampling_locations``/``attention_weights`` to fp32
    and casting the output back; bf16 simply predates the file. This applies
    the *same* rule to bf16 rather than inventing one, by rebinding the module
    global the layer resolves at call time — the external repo is not modified
    on disk.

    Numerically this means the deformable sampling kernel runs at fp32 while
    the rest of the network runs at bf16 (spec §4.1). That is the repo's own
    established behaviour for reduced precision, and it is recorded in the run
    config so it is not an undocumented deviation.
    """
    from groundingdino.models.GroundingDINO import ms_deform_attn

    real = ms_deform_attn.MultiScaleDeformableAttnFunction
    if getattr(real, "_v6_bf16_safe", False):
        return

    class _Bf16SafeDeformableAttention:
        """Drop-in for the autograd ``Function``, exposing only ``apply``."""

        _v6_bf16_safe = True

        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations,
                  attention_weights, im2col_step):
            if value.dtype is not torch.bfloat16:
                return real.apply(value, spatial_shapes, level_start_index,
                                  sampling_locations, attention_weights, im2col_step)
            output = real.apply(value.float(), spatial_shapes, level_start_index,
                                sampling_locations.float(), attention_weights.float(),
                                im2col_step)
            return output.to(torch.bfloat16)

    ms_deform_attn.MultiScaleDeformableAttnFunction = _Bf16SafeDeformableAttention
    logger.info("deformable attention will run in fp32 under bf16 autocast "
                "(the compiled CUDA op has no BFloat16 kernel)")


# --- determinism -------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch, and make cuDNN deterministic-ish.

    Full bitwise determinism is not achievable here (deformable attention's
    CUDA kernel uses atomics), so this fixes data order and initialisation and
    the limitation is stated rather than papered over.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    logger.info("seeded python/numpy/torch with %d (deformable attention uses "
                "atomics, so bitwise reproducibility is not claimed)", seed)


# --- model -------------------------------------------------------------------

@dataclass
class BuiltModel:
    """A gated, LoRA-injected model plus the evidence that it is compliant."""

    model: nn.Module
    report: object
    scaling_report: object
    per_group: dict
    scope: str


def build_lora_model(model_config: Path, checkpoint_path: Path, device: str,
                     scope: str, output_dir: Path) -> BuiltModel:
    """Build Swin-B, inject §2 LoRA, and run the §2 gate. Never trains on failure."""
    from groundingdino.models import build_model
    # Carried in this repository rather than imported from the checkout: the
    # fork never committed its LoRA layer. See models/gdino/lora.
    from models.gdino.lora import (Linear as LoRALinear, apply_custom_attention,
                                   mark_only_lora_as_trainable)
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    cfg = SLConfig.fromfile(str(model_config))
    cfg.device = device
    if int(cfg.lora_r) != LORA.r or int(cfg.lora_alpha) != LORA.lora_alpha \
            or float(cfg.lora_dropout) != LORA.lora_dropout:
        raise SpecViolationError(
            f"{model_config} declares r={cfg.lora_r} alpha={cfg.lora_alpha} "
            f"dropout={cfg.lora_dropout}, but spec §1 fixes r={LORA.r} "
            f"alpha={LORA.lora_alpha} dropout={LORA.lora_dropout}")

    model = build_model(cfg)
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(clean_state_dict(payload["model"]), strict=False)
    logger.info("loaded %s: missing=%d unexpected=%d",
                checkpoint_path, len(missing), len(unexpected))

    # Fused nn.MultiheadAttention exposes ZERO nn.Linear children; the §2
    # attention groups only exist after this decomposition. Collecting targets
    # first would silently drop 72 modules.
    for layer in model.transformer.decoder.layers:
        apply_custom_attention(layer, ["self_attn", "ca_text"])
    for layer in model.transformer.encoder.text_layers:
        apply_custom_attention(layer, ["self_attn"])

    targets = collect_targets(model, scope=scope)
    for target in targets:
        old = target.linear
        adapted = LoRALinear(
            in_features=old.in_features,
            out_features=old.out_features,
            r=LORA.r,
            lora_alpha=LORA.lora_alpha,
            lora_dropout=LORA.lora_dropout,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            adapted.weight.copy_(old.weight)
            if old.bias is not None:
                adapted.bias.copy_(old.bias)
        setattr(target.parent, target.attr, adapted.to(old.weight.device))

    mark_only_lora_as_trainable(model, bias=LORA.bias)
    n_head_tensors = 0
    for name, param in model.named_parameters():
        if any(head in name for head in ("bbox_embed", "class_embed")):
            param.requires_grad_(True)
            n_head_tensors += 1
    if n_head_tensors == 0:
        raise SpecViolationError(
            "no bbox_embed/class_embed parameter was made trainable; §2 "
            "requires the detection head fully trained")
    logger.info("detection head: %d trainable tensors (class_embed is "
                "ContrastiveEmbed and is parameter-free in this architecture, "
                "so the trained head is bbox_embed + enc_out_bbox_embed)",
                n_head_tensors)

    model.to(device)
    if hasattr(model, "device"):
        # Plain attribute, not updated by nn.Module.to(); preprocess_image reads it.
        model.device = device

    placement = build_placement(scope)
    report = verify_injection(model, placement)
    logger.info("%s", format_report(report))
    assert_compliant(report)

    per_group = group_counts(targets)
    expected = expected_group_counts(scope)
    if per_group != expected:
        raise SpecViolationError(
            f"per-group injection counts {per_group} != expected {expected}")
    if report.n_lora_modules != sum(expected.values()):
        raise SpecViolationError(
            f"{report.n_lora_modules} live LoRA modules but {sum(expected.values())} "
            f"targets were injected")
    logger.info("per-group LoRA counts (scope=%s, total %d): %s",
                scope, sum(per_group.values()), per_group)

    scaling_report = verify_lora_scaling(model, expected_r=LORA.r,
                                         expected_alpha=LORA.lora_alpha)
    assert_scaling_compliant(scaling_report)
    logger.info("LoRA scaling verified on %d live modules: %s (alpha/r = %.4f)",
                scaling_report.n_modules_checked,
                list(scaling_report.observed_scalings), scaling_report.expected_scaling)

    _write_injection_map(output_dir, scope, per_group, report, scaling_report)
    return BuiltModel(model=model, report=report, scaling_report=scaling_report,
                      per_group=per_group, scope=scope)


def _write_injection_map(output_dir: Path, scope: str, per_group: dict,
                         report, scaling_report) -> None:
    """Record what was adapted, at build time (risk register §6).

    A value reconstructed from config at load time and never stored beside the
    weights is indistinguishable from a wrong one.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": scope,
        "spec_version": "frozen-2026-08-08",
        "lora_r": LORA.r,
        "lora_alpha": LORA.lora_alpha,
        "lora_dropout": LORA.lora_dropout,
        "scaling": LORA.lora_alpha / LORA.r,
        "observed_scalings": list(scaling_report.observed_scalings),
        "per_group_modules": per_group,
        "total_modules": sum(per_group.values()),
        "per_region_modules": dict(report.per_region_counts),
        "trainable_params": report.trainable_params,
        "total_params": report.total_params,
        "trainable_pct": report.trainable_pct,
        "training_caption": training_caption(),
        "input_resolution": {"train_short_edges": list(TRAIN_SHORT_EDGES),
                             "eval_short_edge": EVAL_SHORT_EDGE,
                             "max_size": EVAL_MAX_SIZE},
    }
    (output_dir / "injection_map.json").write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", output_dir / "injection_map.json")


# --- data --------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _detr_dataset_mapper():
    """`DetrDatasetMapper`, around a package that cannot be imported cleanly.

    `groundingdino/datasets/__init__.py` ends by importing `builtin`, and the
    fork's committed `builtin` registers five lab-specific datasets at module
    scope -- and then calls `DatasetCatalog.get` on them, which is not lazy: it
    runs the loader, which opens `datasets/mechmind_easy/mechmind_easy.json`.
    Nobody outside that lab has the file, so on a clean clone of the fork the
    import dies with `FileNotFoundError` before any of our code runs. The
    checkout this work was done on has those lines deleted, uncommitted, which
    is why it was never seen here.

    The mapper itself is fine and is imported four lines earlier in that same
    `__init__`, so by the time `builtin` fails the submodule is complete and
    registered. Taking it from `sys.modules` uses exactly the class the normal
    import would have bound; what is skipped is only `builtin`'s tail, which
    registers datasets this repository never asks for.

    Only `FileNotFoundError` is caught, and only after checking the submodule
    really is there -- a fork that fixes `builtin` takes the direct path, and
    any other failure is still a failure.

    Cached, and that is load-bearing rather than an optimisation. `_build_mapper`
    runs twice per training run, once for the loader and once for the probe.
    When a package import raises, Python removes the half-initialised package
    from `sys.modules`, so a second attempt re-executes `builtin` from the top
    -- and `register_all_coco` then re-registers names the first attempt already
    put in detectron2's process-global `DatasetCatalog`. The second call died on
    `AssertionError: Dataset 'coco_trainval_all' is already registered!`, which
    is not a `FileNotFoundError` and so went straight through the guard above.
    Resolving once per process is what makes the recovery survive being needed
    twice.

    The catalogue is global, so this holds only for callers that come through
    here. Anything else importing `groundingdino.datasets` after the first
    failure meets the same assertion; nothing in this repository does.
    """
    try:
        from groundingdino.datasets import DetrDatasetMapper
        return DetrDatasetMapper
    except FileNotFoundError as exc:
        mapper = sys.modules.get("groundingdino.datasets.detr_dataset_mapper")
        if mapper is None or not hasattr(mapper, "DetrDatasetMapper"):
            raise
        logger.warning(
            "groundingdino.datasets could not be imported (%s: %s); using the "
            "dataset mapper directly. The checkout's `builtin` registers "
            "datasets it does not ship.", type(exc).__name__, exc)
        return mapper.DetrDatasetMapper


def _build_mapper(is_train: bool):
    """DetrDatasetMapper that emits the **V6 caption verbatim**.

    The external mapper builds ``".".join(names) + "."`` — ``"person.clothes.…"``
    — while ``utils/protocol/train_caption.py`` (and therefore evaluation condition
    ``A4_full_v6``) uses ``"person . clothes . … ."``. Those are different
    strings. Training on one and scoring on the other would make the FT arm's
    in-vocabulary result incomparable to its own zero-shot baseline, and
    nothing would error, so the caption is overwritten with the single source
    of truth here.
    """
    import detectron2.data.transforms as T

    DetrDatasetMapper = _detr_dataset_mapper()

    caption = training_caption()

    class V6DatasetMapper(DetrDatasetMapper):
        def __call__(self, dataset_dict):
            out = super().__call__(dataset_dict)
            out["captions"] = caption
            return out

    if is_train:
        augmentation = [
            T.RandomFlip(),
            T.ResizeShortestEdge(short_edge_length=TRAIN_SHORT_EDGES,
                                 max_size=EVAL_MAX_SIZE, sample_style="choice"),
        ]
        augmentation_with_crop = [
            T.RandomFlip(),
            T.ResizeShortestEdge(short_edge_length=(400, 500, 600), sample_style="choice"),
            T.RandomCrop(crop_type="absolute_range", crop_size=(384, 600)),
            T.ResizeShortestEdge(short_edge_length=TRAIN_SHORT_EDGES,
                                 max_size=EVAL_MAX_SIZE, sample_style="choice"),
        ]
    else:
        augmentation = [T.ResizeShortestEdge(short_edge_length=EVAL_SHORT_EDGE,
                                             max_size=EVAL_MAX_SIZE)]
        augmentation_with_crop = None

    return V6DatasetMapper(
        augmentation=augmentation,
        augmentation_with_crop=augmentation_with_crop,
        is_train=is_train,
        mask_on=False,
        img_format="RGB",
        categories_names=training_class_names(),
    )


def build_train_loader(per_device_batch: int, num_workers: int, seed: int):
    """Deterministic training loader over the full V6 train split (§4.0).

    ``filter_empty`` is False inside :func:`load_v6_dataset_dicts`; the sampler
    seed is explicit because detectron2's default draws a shared random seed
    from the process, which would make the run order differ between seeds for
    reasons unrelated to ``--seed``.
    """
    from detectron2.data import build_detection_train_loader
    from detectron2.data.samplers import TrainingSampler

    dicts = load_v6_dataset_dicts("train")
    sampler = TrainingSampler(len(dicts), shuffle=True, seed=seed)
    loader = build_detection_train_loader(
        dataset=dicts,
        mapper=_build_mapper(is_train=True),
        sampler=sampler,
        total_batch_size=per_device_batch,
        num_workers=num_workers,
    )
    logger.info("train loader: %d images, per-device batch %d, %d workers, "
                "filter_empty=False (spec §4.0 forbids dropping negatives)",
                len(dicts), per_device_batch, num_workers)
    return loader


def build_probe_inputs(n_images: int) -> list[dict]:
    """A FIXED held-out subset, evenly spaced across the val split.

    Evenly spaced rather than the first N: the split is ForestPersons followed
    by ForestBelongings, so a prefix would be all-person and could not detect a
    collapse across the six V6 words at all.
    """
    dicts = load_v6_dataset_dicts("val")
    if len(dicts) < n_images:
        raise RuntimeError(f"val split has {len(dicts)} images, need {n_images}")
    stride = len(dicts) // n_images
    chosen = [dicts[i * stride] for i in range(n_images)]
    logger.info("collapse probe subset: %d held-out val images, stride %d "
                "(fixed across seeds)", len(chosen), stride)
    return chosen


# --- collapse probe ----------------------------------------------------------

@torch.no_grad()
def run_probe(model: nn.Module, probe_dicts: Sequence[dict], mapper, device: str,
              batch_size: int,
              score_floor: float = PROBE_SCORE_THRESHOLD) -> tuple[float, float, float, int]:
    """Measure phrase discrimination on the probe subset with the V6 caption.

    Returns ``(dup_score_rate, labels_per_box, mean_score_spread, n_boxes)``.

    The score floor is **not** spec §6's evaluation threshold. §6 fixes 0.0 for
    AP, which is rank-based and needs the full ranking; this is a different
    quantity, and its thresholds are pre-registered against numbers the retired
    ``measure_collapse_trajectory.py`` produced at ``--score-thresh 0.05``.
    Measuring at a different floor would compare against a threshold table that
    was never calibrated for it, so the retired convention is kept.
    """
    was_training = model.training
    model.eval()
    class_names = training_class_names()
    detections: dict[int, list[tuple[str, float, tuple]]] = {}
    try:
        for start in range(0, len(probe_dicts), batch_size):
            batch = [mapper(d) for d in probe_dicts[start:start + batch_size]]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                outputs = model(batch)
            for offset, output in enumerate(outputs):
                instances = output["instances"].to("cpu")
                boxes = instances.pred_boxes.tensor.float().tolist()
                scores = instances.scores.float().tolist()
                labels = instances.pred_classes.tolist()
                rows: list[tuple[str, float, tuple]] = []
                for box, score, label in zip(boxes, scores, labels):
                    if score < score_floor or label >= len(class_names):
                        continue
                    rows.append((class_names[label], float(score), tuple(box)))
                if rows:
                    detections[start + offset] = rows
    finally:
        model.train(was_training)
    if not detections:
        raise RuntimeError(
            "the collapse probe produced no detections above the score floor; "
            "refusing to report a health metric from an empty measurement")
    return measure_discrimination(detections)


# --- training ----------------------------------------------------------------

def _first_backward_gradient_gate(model: nn.Module) -> None:
    """Assert gradient reached the checkpointed encoder adapters.

    ``lora_B`` is zero-initialised, so ``d loss / d lora_A`` is exactly zero at
    step 1 by construction while ``d loss / d lora_B`` is not. Checking
    ``lora_B`` therefore distinguishes "LoRA has not moved yet" (expected) from
    "the checkpointed block returned no gradient at all" (the reentrant-
    checkpointing failure, which is silent).
    """
    checked = 0
    alive = 0
    for name, param in model.named_parameters():
        if "lora_B" not in name or not name.startswith("transformer.encoder"):
            continue
        checked += 1
        if param.grad is not None and torch.count_nonzero(param.grad) > 0:
            alive += 1
    if checked == 0:
        raise SpecViolationError("no encoder lora_B parameter found to check")
    if alive == 0:
        raise SpecViolationError(
            f"none of {checked} encoder `lora_B` tensors received a non-zero "
            f"gradient after the first backward. The external repo calls "
            f"torch.utils.checkpoint.checkpoint without use_reentrant=, and "
            f"the reentrant implementation returns no gradient when the "
            f"checkpointed inputs do not require grad — every encoder adapter "
            f"would then train on zeros with a plausible loss curve")
    logger.info("gradient gate: %d/%d encoder lora_B tensors received gradient "
                "through the checkpointed blocks", alive, checked)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class _GdinoTraining:
    """Grounding DINO behind the shared loop's interface.

    The running totals stay float32 tensors and are converted once, which is
    what this trainer has always done. Converting each term instead would sum
    in float64 and change the logged loss in its last digits.
    """

    def __init__(self, built, device, plan, probe_dicts, probe_mapper, seed):
        self.built, self.model = built, built.model
        self.device, self.plan, self.seed = device, plan, seed
        self.probe_dicts, self.probe_mapper = probe_dicts, probe_mapper
        self.trainable = [p for p in built.model.parameters() if p.requires_grad]

    def training_step(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.device.startswith("cuda")):
            loss_dict = self.model(batch)
            losses = sum(loss_dict.values())
        # `.sum()` reduces defensively: some criterion terms come back with
        # shape [1] rather than [], and a broadcast error here would only
        # surface after the model is built and the first batch is loaded.
        return StepResult(loss=losses, loss_scalar=losses.detach().sum(),
                          loss_class=loss_dict["loss_class"].detach().sum())

    def clip_parameters(self):
        return self.trainable

    def probe(self, step, loss_class_ema):
        m = run_probe(self.model, self.probe_dicts, self.probe_mapper,
                      self.device, self.plan.per_device_batch)
        return ProbeMetrics(step=step, n_boxes=m[3], dup_score_rate=m[0],
                            labels_per_box=m[1], mean_score_spread=m[2],
                            loss_class_ema=loss_class_ema)

    def save_checkpoint(self, output_dir: Path, step: int, tag: str) -> Path:
        return _save_checkpoint(self.built, output_dir, step, TOTAL_STEPS,
                                self.seed, tag=tag)


def _save_checkpoint(built: BuiltModel, output_dir: Path, step: int, total_steps: int,
                     seed: int, tag: str) -> None:
    manifest = build_manifest(
        model_name=MODEL_KEY,
        report=built.report,
        scaling_report=built.scaling_report,
        step=step,
        total_steps=total_steps,
        seed=seed,
        effective_batch=EFFECTIVE_BATCH_SIZE,
        training_caption=training_caption(),
        lora_r=LORA.r,
        lora_alpha=LORA.lora_alpha,
        lora_dropout=LORA.lora_dropout,
        notes=f"scope={built.scope}; per_group={built.per_group}; tag={tag}",
    )
    save_v6_checkpoint(built.model, manifest, output_dir / f"ckpt_{tag}")


def train(args: argparse.Namespace) -> int:
    """Run the loop. Returns a process exit code (0 ok, 2 collapse abort)."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger("detectron2").setLevel(logging.WARNING)

    force_non_reentrant_checkpointing()
    enable_bf16_deformable_attention()
    seed_everything(args.seed)

    plan = plan_batch(MODEL_KEY, world_size=1)
    logger.info("%s", plan.describe())
    logger.info("%s", describe_schedule(TOTAL_STEPS))

    n_steps = args.gate_only if args.gate_only is not None else args.steps
    if n_steps > TOTAL_STEPS:
        raise ValueError(f"--steps {n_steps} exceeds the frozen budget {TOTAL_STEPS}")
    if n_steps < TOTAL_STEPS:
        logger.warning("running %d of %d steps: this is a GATE/SMOKE run and its "
                       "checkpoint must not be reported as a result (spec §4.1 "
                       "fixes the final step)", n_steps, TOTAL_STEPS)

    built = build_lora_model(Path(args.model_config), Path(args.checkpoint_path),
                             args.device, args.scope, output_dir)
    model = built.model

    param_groups, specs = build_param_groups(model)
    logger.info("%s", describe_param_groups(specs))
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(TOTAL_STEPS))

    thresholds = CollapseThresholds(probe_every_steps=args.probe_every,
                                    probe_images=args.probe_images)
    guard = CollapseGuard(thresholds=thresholds, armed=True,
                          trajectory_path=output_dir / "collapse_trajectory.jsonl")
    logger.info("%s", guard.describe())
    if args.probe_every != CollapseThresholds().probe_every_steps:
        logger.warning("probe cadence overridden to %d (pre-registered: %d); "
                       "allowed only for gate/smoke runs", args.probe_every,
                       CollapseThresholds().probe_every_steps)

    loader = build_train_loader(plan.per_device_batch, args.num_workers, args.seed)
    probe_dicts = build_probe_inputs(args.probe_images)
    probe_mapper = _build_mapper(is_train=False)

    train_log_path = output_dir / "train_log.jsonl"
    (output_dir / "run_config.json").write_text(json.dumps({
        "seed": args.seed, "steps": n_steps, "total_steps": TOTAL_STEPS,
        "scope": args.scope, "device": args.device,
        "per_device_batch": plan.per_device_batch,
        "grad_accumulation": plan.grad_accumulation,
        "effective_batch": plan.effective_batch,
        "precision": "bf16", "max_grad_norm": MAX_GRAD_NORM,
        "gradient_checkpointing_use_reentrant": USE_REENTRANT,
        "deformable_attention_precision": "fp32 (no bf16 CUDA kernel)",
        "probe_score_threshold": PROBE_SCORE_THRESHOLD,
        "training_caption": training_caption(),
        "thresholds": thresholds.as_record(),
        "param_groups": [s.as_record() for s in specs],
    }, indent=2))

    # Zero-shot baseline probe: records where discrimination starts, so the
    # onset step of any collapse is bracketed from below.
    baseline = run_probe(model, probe_dicts, probe_mapper, args.device,
                         plan.per_device_batch)
    logger.info("zero-shot probe: dup_score_rate=%.4f labels_per_box=%.3f "
                "score_spread=%.4f boxes=%d", *baseline)
    guard.evaluate(ProbeMetrics(step=0, n_boxes=baseline[3], dup_score_rate=baseline[0],
                                labels_per_box=baseline[1], mean_score_spread=baseline[2]))

    model.train()
    trainer = _GdinoTraining(built, args.device, plan, probe_dicts,
                             probe_mapper, args.seed)
    trainable = trainer.trainable
    exit_code, step_times = run_training_loop(
        trainer, loader, optimizer, scheduler, guard,
        LoopConfig(n_steps=n_steps, accumulation=plan.grad_accumulation,
                   max_grad_norm=MAX_GRAD_NORM, log_every=args.log_every,
                   checkpoint_every=CHECKPOINT_EVERY_STEPS, seed=args.seed),
        output_dir, torch=torch,
        on_first_backward=lambda: _first_backward_gradient_gate(model))


    if exit_code == 0:
        _save_checkpoint(built, output_dir, n_steps, TOTAL_STEPS, args.seed,
                         tag=f"final_step{n_steps}")

    if guard.alarms:
        guard.write_alarm_file(output_dir / "ALARM.md", context={
            "final_step": len(step_times), "seed": args.seed, "scope": args.scope})

    if step_times:
        warm = step_times[1:] or step_times
        logger.info("timing: %.3f s/step (mean over %d steps, first excluded) -> "
                    "%.2f h for %d steps",
                    sum(warm) / len(warm), len(warm),
                    sum(warm) / len(warm) * TOTAL_STEPS / 3600.0, TOTAL_STEPS)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        logger.info("peak GPU memory: %.2f GiB allocated, %.2f GiB reserved",
                    torch.cuda.max_memory_allocated() / 2 ** 30,
                    torch.cuda.max_memory_reserved() / 2 ** 30)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2),
                        help="spec §4.1 fixes the three seeds")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help=f"optimizer steps to run (frozen budget {TOTAL_STEPS})")
    parser.add_argument("--gate-only", type=int, default=None,
                        help="stop after N steps and mark the run as a gate/smoke "
                             "run. The learning-rate schedule stays on the "
                             "full budget and the data order is unchanged, so "
                             "these steps are the real first N of a full run "
                             "and can be compared against one.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="keep this cuda:0 and pin the GPU with CUDA_VISIBLE_DEVICES")
    parser.add_argument("--scope", type=str, default=DEFAULT_SCOPE, choices=sorted(SCOPES),
                        help="LoRA placement scope (spec §2)")
    parser.add_argument("--model-config", type=str, default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--probe-every", type=int,
                        default=CollapseThresholds().probe_every_steps)
    parser.add_argument("--probe-images", type=int,
                        default=CollapseThresholds().probe_images)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device.startswith("cuda") and args.device not in ("cuda", "cuda:0"):
        logger.warning("--device %s: adapted GDINO variants have failed with "
                       "illegal memory access on non-zero device indices; pin "
                       "the GPU with CUDA_VISIBLE_DEVICES and use cuda:0",
                       args.device)
    logger.info("scope=%s groups=%s", args.scope,
                [g.name for g in scope_groups(args.scope)])
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
