#!/usr/bin/env python
"""Phase 2 LoRA trainer for OmDet-Turbo (swin-tiny), under the frozen spec.

Spec: the frozen-hyperparameter specification §1 (LoRA), §2 (placement + the
measured 60-module table), §3 (optimizer), §4.0/§4.1 (data policy and loop),
§5 (640x640), §6 (evaluation — untouched here).
Placement: ``models/shared/lora_spec.py``. Gate evidence:
``outputs/v6/train_gates/omdet.json``.

Reference implementation is ``train.py --model gdino``, which is **not
modified**; every gate it runs before step 1 is run here against OmDet's own
measured counts.

Three things this model needs that Grounding DINO does not
----------------------------------------------------------
1. ``from_pretrained`` leaves timm's ``relative_position_index`` buffers
   uninitialised. Indexing ``relative_position_bias_table`` with that garbage
   trips a **device-side assert inside a softmax**, which surfaces as an
   unrelated CUDA error hundreds of frames away. The inference detector already
   repairs it; the repair is applied on the training path too.
2. The processor needs **one class list per image**. A single flat list against
   a batch of N images mis-aligns ``classes_structure`` and produces the same
   assert.
3. ``OmDetTurboForObjectDetection.forward`` raises ``NotImplementedError`` when
   given ``labels`` — HuggingFace ships inference only. The set-prediction loss
   is therefore reconstructed outside the model by
   ``models/shared/detection_loss.py`` (Hungarian matching + sigmoid focal +
   L1 + GIoU on ``decoder_class_logits`` / ``decoder_coord_logits``), which is
   the same module the retired ``train_omdet_vis4.py`` used.

Recorded spec deviation (§4.1)
------------------------------
transformers 4.57.3 declares ``OmDetTurboForObjectDetection.supports_gradient_
checkpointing = False``, so §4.1's "gradient checkpointing on" cannot be
satisfied. It is **not** worked around: the flag is recorded as unavailable in
``run_config.json``. The model is 116M parameters so memory is not the binding
constraint, and the ``use_reentrant`` hazard the flag exists to control cannot
arise where there is no checkpointing.

Usage (smoke, seed 0)::

    CUDA_VISIBLE_DEVICES=7 TMPDIR="$TMPDIR" \\
      ~/venvs/uc-omdet/bin/python train_v6_omdet.py (research tree) \\
        --seed 0 --gate-only 30 --device cuda:0 \\
        --output-dir outputs/v6/train/omdet_seed0_smoke

Always pin the GPU with ``CUDA_VISIBLE_DEVICES`` and keep ``--device cuda:0``:
merging a PEFT adapter on a non-zero device index has produced **silently
different results between runs** on this model (0 detections vs 4 on the same
image and query).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Before any first-party import: this file is run as a script, so the repo
# root is not on sys.path yet. It has to come first, not beside the other path
# work further down, because the imports below need it already done.
_sys.path.insert(0, str(next(q for q in _Path(__file__).resolve().parents
                             if (q / 'pyproject.toml').is_file())))


import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / 'pyproject.toml').is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from models.shared.detection_loss import LossWeights, set_prediction_loss  # noqa: E402
from utils.runner.logging_utils import get_logger  # noqa: E402
from utils.train.checkpoint import (  # noqa: E402
    build_manifest,
    load_adapter_weights,
    read_manifest,
    save_v6_checkpoint,
    verify_loaded_checkpoint,
)
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
from models.gdino.lora import describe_schedule, make_lr_lambda  # noqa: E402
from models.shared.lora_spec_v6line import (  # noqa: E402
    SpecViolationError,
    assert_compliant,
    assert_scaling_compliant,
    format_report,
    select_targets,
    verify_injection,
    verify_lora_scaling,
)
from models.shared.lora_spec import (  # noqa: E402
    EXPECTED_GROUP_COUNTS,
    EXPECTED_LORA_MODULES,
    get_placement,
    group_of,
)
from utils.protocol.train_caption import training_caption, training_class_names  # noqa: E402
from utils.train.training_schedule import (  # noqa: E402
    EFFECTIVE_BATCH_SIZE,
    LORA,
    MAX_GRAD_NORM,
    TOTAL_STEPS,
)
from utils.data.xarch_data import V6Record, load_v6_records, probe_records  # noqa: E402
from models.shared.train import (  # noqa: E402
    assert_group_counts,
    build_param_groups,
    describe_param_groups,
    mark_trainable,
    omdet_default_task_string,
    plan_batch_xarch,
)

logger = get_logger("v6.train.omdet")

MODEL_KEY = "omdet"
CHECKPOINT_ID = "omlab/omdet-turbo-swin-tiny-hf"
#: HuggingFace cache. `HF_HOME` is the standard variable and is honoured first;
#: the fallback keeps a run working when it is unset.
#: `cache_dir` wants the HUB directory, not HF_HOME. They differ by one path
#: component and the difference is invisible until an offline load fails with
#: "can't load image processor", which names the model rather than the path.
HF_CACHE = (os.environ.get("HF_HUB_CACHE")
            or (os.path.join(os.environ["HF_HOME"], "hub")
                if os.environ.get("HF_HOME") else None)
            or str(Path.home() / ".cache" / "huggingface" / "hub"))
CHECKPOINT_EVERY_STEPS = 2_000
#: Score floor of the collapse probe. NOT spec §6 (which fixes 0.0 for AP).
#: Same 0.05 the Grounding DINO trainer uses, so the pre-registered thresholds
#: are compared against a probe measured the same way.
PROBE_SCORE_THRESHOLD = 0.05
#: §5: the processor forces 640x640 with do_pad=False. Recorded, not chosen.
INPUT_RESOLUTION = (640, 640)


# --- data --------------------------------------------------------------------

class OmDetV6Dataset(Dataset):
    """V6 images with normalised ``cxcywh`` boxes and a horizontal-flip aug.

    Normalising against the *original* size is exact for this model: its
    ``DetrImageProcessor`` rescales to a fixed 640x640 with ``do_pad=False``,
    i.e. a pure (anisotropic) rescale, and post-processing maps normalised
    predictions straight back to the original size.

    Horizontal flip is the only augmentation. Grounding DINO's multi-scale
    resize has no counterpart here — the processor's output size is fixed by
    the model (§5) — so adding a scale jitter would mean re-implementing the
    processor, which §6's "model default" rule argues against.
    """

    def __init__(self, records: Sequence[V6Record], train: bool) -> None:
        self._records = list(records)
        self._train = train

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict:
        record = self._records[index]
        with Image.open(record.image_path) as raw:
            image = raw.convert("RGB")
        boxes = np.asarray(record.boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        flipped = self._train and random.random() < 0.5
        if flipped:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                x1 = record.width - boxes[:, 2]
                x2 = record.width - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x1, x2
        if len(boxes):
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / record.width
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / record.height
            bw = (boxes[:, 2] - boxes[:, 0]) / record.width
            bh = (boxes[:, 3] - boxes[:, 1]) / record.height
            norm = np.stack([cx, cy, bw, bh], axis=1).clip(0.0, 1.0)
        else:
            norm = np.zeros((0, 4), dtype=np.float32)
        return {
            "image": image,
            "boxes": torch.from_numpy(norm.astype(np.float32)),
            "labels": torch.as_tensor(record.labels, dtype=torch.long),
            "image_id": record.image_id,
            "orig_size": (record.height, record.width),
        }


class OmDetV6Collator:
    """Batch into processor inputs plus set-prediction targets.

    The class list is passed **per image** — a single flat list against a batch
    mis-aligns ``classes_structure`` and trips a device-side assert inside the
    decoder softmax. ``task`` is never passed, so the processor builds its own
    default (``"Detect person, clothes, ...".``) from the class list; that is
    the same string the evaluation path produces, which is what makes the FT
    arm comparable to its own zero-shot baseline.
    """

    def __init__(self, processor, class_names: Sequence[str]) -> None:
        self._processor = processor
        self.class_names = list(class_names)

    def __call__(self, batch: list[dict]) -> tuple[dict, list[dict], list[tuple]]:
        images = [item["image"] for item in batch]
        encoding = self._processor(images=images,
                                   text=[self.class_names] * len(images),
                                   return_tensors="pt")
        targets = [{"boxes": item["boxes"], "labels": item["labels"]} for item in batch]
        sizes = [item["orig_size"] for item in batch]
        return dict(encoding), targets, sizes


#: The task prompt the processor builds when ``task`` is not supplied. Defined
#: in :mod:`models.shared.train` so a test can assert the string without
#: instantiating the processor.
default_task_string = omdet_default_task_string


# --- model -------------------------------------------------------------------

@dataclass
class BuiltModel:
    """A gated, LoRA-injected model plus the evidence that it is compliant."""

    model: nn.Module
    report: object
    scaling_report: object
    per_group: dict
    placement: object


def _load_base_model():
    from transformers import AutoProcessor, OmDetTurboForObjectDetection

    from models.omdet.detector import (_materialise_meta_buffers,
                                         _recompute_timm_relative_position_index)

    processor = AutoProcessor.from_pretrained(CHECKPOINT_ID, cache_dir=HF_CACHE)
    model = OmDetTurboForObjectDetection.from_pretrained(
        CHECKPOINT_ID, low_cpu_mem_usage=False, cache_dir=HF_CACHE)
    _materialise_meta_buffers(model)
    _recompute_timm_relative_position_index(model)
    return processor, model


def _inject_lora(model) -> object:
    """Inject §1 LoRA at the §2 targets, in place. Returns the placement."""
    from peft import LoraConfig, inject_adapter_in_model

    placement = get_placement(MODEL_KEY)
    targets = select_targets([n for n, _ in model.named_modules()], placement)
    if len(targets) != EXPECTED_LORA_MODULES[MODEL_KEY]:
        raise SpecViolationError(
            f"§2 regex selected {len(targets)} modules but the measured table "
            f"says {EXPECTED_LORA_MODULES[MODEL_KEY]}")
    # `inject_adapter_in_model`, never `get_peft_model`: the wrapper form adds
    # nodes that confuse the shared verifier and clones `modules_to_save`,
    # inflating the parameter denominator (116.2M -> 118.1M).
    inject_adapter_in_model(
        LoraConfig(r=LORA.r, lora_alpha=LORA.lora_alpha,
                   lora_dropout=LORA.lora_dropout, bias=LORA.bias,
                   target_modules=targets),
        model)
    return placement


def build_lora_model(device: str, output_dir: Path) -> tuple[object, BuiltModel]:
    """Build OmDet-Turbo, inject §2 LoRA, run the gate. Never trains on failure."""
    processor, model = _load_base_model()
    placement = _inject_lora(model)
    n_lora, n_head = mark_trainable(model, placement)
    logger.info("trainable tensors: %d LoRA, %d detection-head (§2 heads: %s)",
                n_lora, n_head, list(placement.modules_to_save))

    if not bool(getattr(model, "supports_gradient_checkpointing", False)):
        logger.warning("spec §4.1 DEVIATION: transformers marks "
                       "OmDetTurboForObjectDetection as not supporting gradient "
                       "checkpointing; recorded in run_config.json, not worked around")

    model.to(device).train()

    report = verify_injection(model, placement)
    logger.info("%s", format_report(report))
    assert_compliant(report)
    per_group = assert_group_counts(model, placement,
                                    EXPECTED_GROUP_COUNTS[MODEL_KEY], group_of)
    if report.n_lora_modules != EXPECTED_LORA_MODULES[MODEL_KEY]:
        raise SpecViolationError(
            f"{report.n_lora_modules} live LoRA modules, expected "
            f"{EXPECTED_LORA_MODULES[MODEL_KEY]}")
    logger.info("per-group LoRA counts (total %d): %s", sum(per_group.values()), per_group)

    scaling_report = verify_lora_scaling(model, expected_r=LORA.r,
                                         expected_alpha=LORA.lora_alpha)
    assert_scaling_compliant(scaling_report)
    logger.info("LoRA scaling verified on %d live modules: %s (alpha/r = %.4f)",
                scaling_report.n_modules_checked,
                list(scaling_report.observed_scalings), scaling_report.expected_scaling)

    built = BuiltModel(model=model, report=report, scaling_report=scaling_report,
                       per_group=per_group, placement=placement)
    _write_injection_map(output_dir, built)
    return processor, built


def _write_injection_map(output_dir: Path, built: BuiltModel) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_KEY,
        "spec_version": "frozen-2026-08-08",
        "lora_r": LORA.r, "lora_alpha": LORA.lora_alpha,
        "lora_dropout": LORA.lora_dropout,
        "scaling": LORA.lora_alpha / LORA.r,
        "observed_scalings": list(built.scaling_report.observed_scalings),
        "per_group_modules": built.per_group,
        "total_modules": sum(built.per_group.values()),
        "per_region_modules": dict(built.report.per_region_counts),
        "trainable_params": built.report.trainable_params,
        "total_params": built.report.total_params,
        "trainable_pct": built.report.trainable_pct,
        "training_caption": training_caption(),
        "training_class_names": training_class_names(),
        "task_prompt": default_task_string(training_class_names()),
        "input_resolution": {"height": INPUT_RESOLUTION[0], "width": INPUT_RESOLUTION[1],
                             "source": "DetrImageProcessor default, do_pad=False"},
    }
    (output_dir / "injection_map.json").write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", output_dir / "injection_map.json")


# --- collapse probe ----------------------------------------------------------

@torch.no_grad()
def run_probe(model, processor, records: Sequence[V6Record], device: str,
              batch_size: int, nms_threshold: float,
              score_floor: float = PROBE_SCORE_THRESHOLD) -> tuple[float, float, float, int]:
    """Phrase discrimination on the fixed probe subset, with the V6 class list."""
    was_training = model.training
    model.eval()
    terms = training_class_names()
    detections: dict[int, list[tuple[str, float, tuple]]] = {}
    try:
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            images = []
            sizes = []
            for record in chunk:
                with Image.open(record.image_path) as raw:
                    images.append(raw.convert("RGB"))
                sizes.append((record.height, record.width))
            inputs = processor(images=images, text=[terms] * len(images),
                               return_tensors="pt").to(device)
            # fp32, deliberately: the pre-registered healthy anchor in
            # outputs/v6/train_gates/omdet.json was measured in fp32, and a
            # threshold is only meaningful against a probe measured the same way.
            outputs = model(**inputs)
            # One class list PER IMAGE. A flat list of strings is wrapped as a
            # single group by the processor and then fails the batch-size check
            # -- the inference detector never hit this because it runs batch 1.
            results = processor.post_process_grounded_object_detection(
                outputs, text_labels=[terms] * len(images), threshold=score_floor,
                nms_threshold=nms_threshold, target_sizes=sizes)
            for offset, result in enumerate(results):
                rows = [(str(label), float(score),
                         tuple(float(v) for v in box))
                        for box, score, label in zip(
                            result["boxes"].detach().float().cpu().numpy(),
                            result["scores"].detach().float().cpu().numpy(),
                            result["text_labels"])]
                if rows:
                    detections[start + offset] = rows
    finally:
        model.train(was_training)
    if not detections:
        raise RuntimeError(
            "the collapse probe produced no detections above the score floor; "
            "refusing to report a health metric from an empty measurement")
    return measure_discrimination(detections)


# --- checkpoint round trip ---------------------------------------------------

def verify_checkpoint_roundtrip(directory: Path, device: str) -> dict:
    """Rebuild from the hub, load the adapter, and re-run the §2 gate.

    A checkpoint that cannot be reloaded into a compliant model is not a
    result. This is deliberately done on a **fresh** model rather than the live
    one, because the live one would pass by construction.
    """
    manifest = read_manifest(directory)
    _, model = _load_base_model()
    placement = _inject_lora(model)
    mark_trainable(model, placement)
    model.to(device)
    n_loaded = load_adapter_weights(model, directory, strict=True)
    verify_loaded_checkpoint(model, manifest, placement,
                             expected_caption=training_caption())
    per_group = assert_group_counts(model, placement,
                                    EXPECTED_GROUP_COUNTS[MODEL_KEY], group_of)
    del model
    torch.cuda.empty_cache()
    record = {"directory": str(directory), "tensors_loaded": n_loaded,
              "per_group": per_group, "step": manifest.step,
              "scaling": manifest.scaling, "caption": manifest.training_caption,
              "passes": True}
    logger.info("checkpoint round trip OK: %d tensors, per-group %s", n_loaded, per_group)
    return record


# --- training ----------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    # Seeding alone does not make a run reproducible: several backward kernels
    # accumulate with atomics, whose order depends on how the GPU schedules
    # blocks. UC_DETERMINISTIC=1 asks torch to refuse those kernels where a
    # deterministic one exists and to name the op where none does, which is how
    # the cause of a run-to-run difference gets identified rather than guessed.
    if os.environ.get("UC_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("deterministic algorithms requested (warn_only); ops with no "
                    "deterministic implementation will be named in warnings")
    logger.info("seeded python/numpy/torch with %d", seed)


def _first_backward_gradient_gate(model: nn.Module) -> None:
    """Assert every ``lora_B`` received a non-zero gradient after backward 1.

    ``lora_B`` is zero-initialised, so ``d loss / d lora_A`` is exactly zero at
    step 1 by construction while ``d loss / d lora_B`` is not; checking
    ``lora_B`` therefore separates "LoRA has not moved yet" from "this block
    returned no gradient at all".
    """
    total = alive = 0
    dead: list[str] = []
    for name, param in model.named_parameters():
        if "lora_B" not in name or not param.requires_grad:
            continue
        total += 1
        if param.grad is not None and torch.count_nonzero(param.grad) > 0:
            alive += 1
        else:
            dead.append(name)
    if total == 0:
        raise SpecViolationError("no lora_B parameter found to check")
    if dead:
        raise SpecViolationError(
            f"{len(dead)} of {total} lora_B tensors received no gradient after the "
            f"first backward, e.g. {dead[:5]}. Those adapters would train on zeros "
            f"behind a plausible loss curve")
    logger.info("gradient gate: %d/%d lora_B tensors received gradient", alive, total)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class _OmDetTraining:
    """OmDet-Turbo behind the shared loop's interface.

    Four things are architecture-specific and live here; the loop around them
    is in ``utils/train/loop.py``. The autocast window covers the forward only,
    as it did when this was written inline: the Hungarian set-prediction loss
    runs in float32 on logits cast up from bfloat16, and moving the cast would
    change the numbers.
    """

    def __init__(self, built, processor, device: str, plan, weights,
                 probe_subset, nms_threshold: float, seed: int):
        self.built = built
        self.model = built.model
        self.processor = processor
        self.device = device
        self.plan = plan
        self.weights = weights
        self.probe_subset = probe_subset
        self.nms_threshold = nms_threshold
        self.seed = seed
        self.trainable = [p for p in built.model.parameters() if p.requires_grad]

    def training_step(self, batch):
        inputs, targets, _ = batch
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.device.startswith("cuda")):
            outputs = self.model(**inputs)
        losses = set_prediction_loss(outputs.decoder_class_logits.float(),
                                     outputs.decoder_coord_logits.float(),
                                     targets, weights=self.weights)
        return StepResult(loss=losses["loss_detection"],
                          loss_scalar=float(losses["loss_detection"].detach()),
                          loss_class=float(losses["loss_class"].detach()))

    def clip_parameters(self):
        return self.trainable

    def probe(self, step: int, loss_class_ema):
        metrics = run_probe(self.model, self.processor, self.probe_subset,
                            self.device, self.plan.per_device_batch,
                            self.nms_threshold)
        return ProbeMetrics(step=step, n_boxes=metrics[3],
                            dup_score_rate=metrics[0], labels_per_box=metrics[1],
                            mean_score_spread=metrics[2],
                            loss_class_ema=loss_class_ema)

    def save_checkpoint(self, output_dir: Path, step: int, tag: str) -> Path:
        return _save_checkpoint(self.built, output_dir, step, self.seed, tag=tag)


def _save_checkpoint(built: BuiltModel, output_dir: Path, step: int, seed: int,
                     tag: str) -> Path:
    manifest = build_manifest(
        model_name=MODEL_KEY, report=built.report,
        scaling_report=built.scaling_report, step=step, total_steps=TOTAL_STEPS,
        seed=seed, effective_batch=EFFECTIVE_BATCH_SIZE,
        training_caption=training_caption(), lora_r=LORA.r,
        lora_alpha=LORA.lora_alpha, lora_dropout=LORA.lora_dropout,
        notes=f"per_group={built.per_group}; tag={tag}; "
              f"task_prompt={default_task_string(training_class_names())!r}")
    return save_v6_checkpoint(built.model, manifest, output_dir / f"ckpt_{tag}")


def train(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    plan = (plan_batch_xarch(MODEL_KEY, world_size=1)
            if args.per_device_batch is None
            else _plan_override(args.per_device_batch))
    logger.info("%s", plan.describe())
    logger.info("%s", describe_schedule(TOTAL_STEPS))

    n_steps = args.gate_only if args.gate_only is not None else args.steps
    if n_steps > TOTAL_STEPS:
        raise ValueError(f"--steps {n_steps} exceeds the frozen budget {TOTAL_STEPS}")
    if n_steps < TOTAL_STEPS:
        logger.warning("running %d of %d steps: this is a GATE/SMOKE run and its "
                       "checkpoint must not be reported as a result (§4.1 fixes "
                       "the final step)", n_steps, TOTAL_STEPS)

    processor, built = build_lora_model(args.device, output_dir)
    model = built.model

    param_groups, specs = build_param_groups(model, built.placement)
    logger.info("%s", describe_param_groups(specs))
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(TOTAL_STEPS))

    thresholds = CollapseThresholds(probe_every_steps=args.probe_every,
                                    probe_images=args.probe_images)
    guard = CollapseGuard(thresholds=thresholds, armed=True,
                          trajectory_path=output_dir / "collapse_trajectory.jsonl")
    logger.info("%s", guard.describe())

    records = load_v6_records("train", max_images=args.max_images)
    dataset = OmDetV6Dataset(records, train=True)
    collator = OmDetV6Collator(processor, training_class_names())
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=plan.per_device_batch, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collator,
                        drop_last=True, pin_memory=args.device.startswith("cuda"),
                        generator=generator, persistent_workers=args.num_workers > 0)
    logger.info("train loader: %d images, per-device batch %d, %d workers "
                "(no empty-image filtering; §4.0)", len(dataset),
                plan.per_device_batch, args.num_workers)

    probe_subset = probe_records("val", args.probe_images)
    from models.omdet.detector import processor_nms_default
    nms_threshold = processor_nms_default()

    (output_dir / "run_config.json").write_text(json.dumps({
        "model": MODEL_KEY, "seed": args.seed, "steps": n_steps,
        "total_steps": TOTAL_STEPS, "device": args.device,
        "per_device_batch": plan.per_device_batch,
        "grad_accumulation": plan.grad_accumulation,
        "effective_batch": plan.effective_batch,
        "precision": "bf16", "max_grad_norm": MAX_GRAD_NORM,
        "gradient_checkpointing": False,
        "gradient_checkpointing_spec_deviation": (
            "§4.1 requires gradient checkpointing; transformers 4.57.3 marks "
            "OmDetTurboForObjectDetection unsupported "
            "(supports_gradient_checkpointing=False). Not worked around. The model "
            "is 116M params so memory is not binding, and the use_reentrant hazard "
            "cannot arise without checkpointing."),
        "input_resolution": {"height": INPUT_RESOLUTION[0], "width": INPUT_RESOLUTION[1]},
        "augmentation": "horizontal flip p=0.5 only (processor fixes 640x640)",
        "probe_score_threshold": PROBE_SCORE_THRESHOLD,
        "probe_nms_threshold": nms_threshold,
        "training_caption": training_caption(),
        "training_class_names": training_class_names(),
        "task_prompt": default_task_string(training_class_names()),
        "loss": "reconstructed set-prediction (models/shared/detection_loss.py); "
                "HF forward raises NotImplementedError on labels",
        "loss_weights": {"classification": LossWeights().classification,
                         "bbox_l1": LossWeights().bbox_l1,
                         "giou": LossWeights().giou},
        "max_images": args.max_images,
        "thresholds": thresholds.as_record(),
        "param_groups": [s.as_record() for s in specs],
    }, indent=2))

    baseline = run_probe(model, processor, probe_subset, args.device,
                         plan.per_device_batch, nms_threshold)
    logger.info("zero-shot probe: dup_score_rate=%.6f labels_per_box=%.4f "
                "score_spread=%.4f boxes=%d", *baseline)
    guard.evaluate(ProbeMetrics(step=0, n_boxes=baseline[3], dup_score_rate=baseline[0],
                                labels_per_box=baseline[1], mean_score_spread=baseline[2]))

    model.train()
    trainer = _OmDetTraining(built, processor, args.device, plan, LossWeights(),
                             probe_subset, nms_threshold, args.seed)
    trainable = trainer.trainable
    exit_code, step_times = run_training_loop(
        trainer, loader, optimizer, scheduler, guard,
        LoopConfig(n_steps=n_steps, accumulation=plan.grad_accumulation,
                   max_grad_norm=MAX_GRAD_NORM, log_every=args.log_every,
                   checkpoint_every=CHECKPOINT_EVERY_STEPS, seed=args.seed),
        output_dir, torch=torch,
        on_first_backward=lambda: _first_backward_gradient_gate(model))

    final_dir = None
    if exit_code == 0:
        final_dir = _save_checkpoint(built, output_dir, n_steps, args.seed,
                                     tag=f"final_step{n_steps}")
    if guard.alarms:
        guard.write_alarm_file(output_dir / "ALARM.md",
                               context={"final_step": len(step_times), "seed": args.seed})

    timing = {}
    if step_times:
        warm = step_times[1:] or step_times
        mean = sum(warm) / len(warm)
        timing = {"sec_per_step_mean": mean, "n_steps_timed": len(warm),
                  "projected_hours_full_run": mean * TOTAL_STEPS / 3600.0}
        logger.info("timing: %.3f s/step (mean over %d, first excluded) -> %.2f h "
                    "for %d steps", mean, len(warm), timing["projected_hours_full_run"],
                    TOTAL_STEPS)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        timing["peak_gib_allocated"] = torch.cuda.max_memory_allocated() / 2 ** 30
        timing["peak_gib_reserved"] = torch.cuda.max_memory_reserved() / 2 ** 30
        logger.info("peak GPU memory: %.2f GiB allocated, %.2f GiB reserved",
                    timing["peak_gib_allocated"], timing["peak_gib_reserved"])

    roundtrip = None
    if final_dir is not None and args.verify_roundtrip:
        # Free the live model first: the round trip builds a second copy.
        del model, built.model, optimizer, trainable
        torch.cuda.empty_cache()
        roundtrip = verify_checkpoint_roundtrip(final_dir, args.device)
    (output_dir / "run_summary.json").write_text(json.dumps(
        {"model": MODEL_KEY, "seed": args.seed, "steps_run": len(step_times),
         "exit_code": exit_code, "timing": timing,
         "checkpoint_roundtrip": roundtrip,
         "alarms": guard.alarms}, indent=2))
    return exit_code


def _cycle(loader):
    while True:
        yield from loader


def _plan_override(per_device: int):
    """A batch plan for an explicitly requested per-device batch (probing only)."""
    from utils.train.training_schedule import BatchPlan

    if EFFECTIVE_BATCH_SIZE % per_device:
        raise ValueError(f"per-device batch {per_device} does not divide the "
                         f"effective batch {EFFECTIVE_BATCH_SIZE}")
    logger.warning("--per-device-batch %d overrides utils/train/training_schedule.py; "
                   "allowed only for memory probing, never for a reported run",
                   per_device)
    return BatchPlan(model=MODEL_KEY, per_device_batch=per_device,
                     grad_accumulation=EFFECTIVE_BATCH_SIZE // per_device, world_size=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--gate-only", type=int, default=None,
                        help="stop after N steps and mark the run as a gate/smoke "
                             "run. The learning-rate schedule stays on the "
                             "full budget and the data order is unchanged, so "
                             "these steps are the real first N of a full run "
                             "and can be compared against one.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None,
                        help="SMOKE ONLY; §4.0 forbids reporting a subsampled run")
    parser.add_argument("--per-device-batch", type=int, default=None,
                        help="memory probing only; normally read from "
                             "utils/train/training_schedule.py")
    parser.add_argument("--probe-every", type=int,
                        default=CollapseThresholds().probe_every_steps)
    parser.add_argument("--probe-images", type=int,
                        default=CollapseThresholds().probe_images)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verify-roundtrip", action="store_true",
                        help="after the final checkpoint, rebuild from the hub, "
                             "reload the adapter and re-run the §2 gate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device.startswith("cuda") and args.device not in ("cuda", "cuda:0"):
        logger.warning("--device %s: merging a PEFT adapter on a non-zero device "
                       "index has silently changed OmDet results between runs; pin "
                       "the GPU with CUDA_VISIBLE_DEVICES and use cuda:0", args.device)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
