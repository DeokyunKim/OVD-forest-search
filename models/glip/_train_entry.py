#!/usr/bin/env python
"""Phase 2 LoRA trainer for GLIP-T(C), under the frozen spec.

Spec: the frozen-hyperparameter specification §1 (LoRA), §2 (placement + the
measured 72-module table), §3 (optimizer), §4.0/§4.1 (data policy and loop),
§5 (800/1333), §6 (evaluation — untouched here).
Placement: ``models/shared/lora_spec.py``. Gate evidence:
``outputs/v6/train_gates/glip.json``.

Reference implementation is ``train.py --model gdino``, which is **not**
modified; every gate it runs before step 1 is run here against GLIP's own
measured counts.

Two things this model needs that Grounding DINO does not
--------------------------------------------------------
1. **The fusion tower's 18 modulated deformable convolutions have no bf16 CUDA
   kernel.** ``"modulated_deformable_im2col_gpu" not implemented for
   'BFloat16'``. They sit inside the tower §2 adapts, so they cannot be routed
   around. :func:`models.shared.train.enable_bf16_modulated_deform_conv` runs
   that kernel at fp32 and casts back, which is the identical remedy (and the
   identical precedent) as the Grounding DINO trainer's
   ``enable_bf16_deformable_attention`` for multi-scale deformable attention.
   Recorded in ``run_config.json`` and in spec §8; it is not undocumented.
2. **One caption per image.** ``VLFuse`` expands the language attention mask to
   the image batch, so a batch-1 language tensor against batch-N images dies
   inside ``single_attention_call``.

Why the loss is not ``GLIP.loss``
---------------------------------
mmdet's ``GLIP.loss`` rebuilds the caption itself with ``to_plain_text_prompts``,
producing ``"person. clothes. hat. shoes. gloves. bag"``. The frozen V6 caption
is ``"person . clothes . hat . shoes . gloves . bag ."`` — a **different
string**. Training on one and scoring on the other would make the FT arm's
in-vocabulary result incomparable to its own zero-shot baseline, and nothing
would error. :func:`glip_loss` is mmdet's own ~20 lines with the caption rebuild
replaced by the frozen caption plus the explicit character spans that
``models/glip/detector.py`` already uses at evaluation time, so the two paths
are byte-identical by construction. the regression tests asserts
the bytes rather than the intent.

Usage (smoke, seed 1)::

    CUDA_VISIBLE_DEVICES=6 TMPDIR="$TMPDIR" \\
      ~/venvs/uc-glip/bin/python train_v6_glip.py (research tree) \\
        --seed 1 --gate-only 30 --device cuda:0 \\
        --output-dir outputs/v6/train/glip_seed1_smoke
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
from models.glip.detector import GLIP_VARIANTS, build_tokens_positive  # noqa: E402
from utils.protocol.eval_protocol import EVAL_PROTOCOL  # noqa: E402
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
    USE_REENTRANT,
)
from utils.data.xarch_data import (  # noqa: E402
    EVAL_MAX_SIZE,
    V6Record,
    load_v6_records,
    probe_records,
)
from models.shared.train import (  # noqa: E402
    assert_group_counts,
    build_param_groups,
    describe_param_groups,
    enable_bf16_modulated_deform_conv,
    force_non_reentrant_checkpointing,
    mark_trainable,
    plan_batch_xarch,
)

logger = get_logger("v6.train.glip")

MODEL_KEY = "glip"
VARIANT = "tiny_c"
CHECKPOINT_EVERY_STEPS = 2_000
PROBE_SCORE_THRESHOLD = 0.05
#: Confirmed 2026-08-10 (spec §8): GLIP's mmdet default ``max_per_img`` is
#: exactly 100, which coincides with §6's cap, so under the per-category cap
#: policy GLIP could never fill more than one category's budget. 500 is the same
#: remedy, for the same reason, that OWLv2 already uses. **No threshold changes**
#: — only how many detections the model emits before §6 scores them. Every
#: completed GLIP zero-shot condition ran at 500, so ZS and FT match.
MAX_OUTPUT_DETECTIONS = 500


# --- data --------------------------------------------------------------------

#: Short edges of mmdetection's own GLIP fine-tuning pipeline
#: (``configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py``).
#: Adopted verbatim rather than reusing the Grounding DINO trainer's 11-value
#: tuple: GLIP's box convention differs (see ``GTBoxSubOne_GLIP``) and this is
#: the recipe the architecture's own maintainers train it with, which is the
#: same "keep the model default" argument §6 makes for NMS.
GLIP_TRAIN_SHORT_EDGES: tuple[int, ...] = (480, 560, 640, 720, 800)


def _train_pipeline_cfg() -> list[dict]:
    """mmdetection's own GLIP fine-tuning train pipeline.

    Every stage is from mmdet's GLIP fine-tune config, with one change:
    ``FilterAnnotations`` runs with ``keep_empty=False``. Its default returns
    ``None`` for an image with no surviving box, which would silently drop the
    V6 negatives that spec §4.0 forbids removing — and would crash the loader
    rather than fail quietly, but the fix is required either way.

    ``GTBoxSubOne_GLIP`` and ``RandomFlip_GLIP`` are not interchangeable with
    their generic counterparts: GLIP's boxes are inclusive at the far edge (the
    head's own ``anchors[:, 2:] -= 1``), so using plain ``RandomFlip`` would
    shift every flipped box by one pixel.
    """
    return [
        dict(type="LoadImageFromFile", imdecode_backend="pillow"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="GTBoxSubOne_GLIP"),
        dict(type="RandomChoiceResize",
             scales=[(EVAL_MAX_SIZE, int(s)) for s in GLIP_TRAIN_SHORT_EDGES],
             keep_ratio=True, resize_type="FixScaleResize", backend="pillow"),
        dict(type="RandomFlip_GLIP", prob=0.5),
        dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type="PackDetInputs",
             meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                        "scale_factor", "flip", "flip_direction")),
    ]


class GlipV6Dataset(Dataset):
    """V6 records fed through an mmdetection training pipeline."""

    def __init__(self, records: Sequence[V6Record]) -> None:
        from mmengine.dataset import Compose

        self._records = list(records)
        self._pipeline = Compose(_train_pipeline_cfg())

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict:
        record = self._records[index]
        instances = [
            {"bbox": list(box), "bbox_label": int(label), "ignore_flag": 0}
            for box, label in zip(record.boxes_xyxy, record.labels)
        ]
        return self._pipeline({
            "img_path": record.image_path,
            "img_id": record.image_id,
            "height": record.height,
            "width": record.width,
            "instances": instances,
        })


def collate_mm(batch: list[dict]) -> dict:
    """mmengine's ``pseudo_collate`` shape: keep per-sample objects in lists."""
    return {"inputs": [item["inputs"] for item in batch],
            "data_samples": [item["data_samples"] for item in batch]}


# --- model -------------------------------------------------------------------

@dataclass
class BuiltModel:
    model: nn.Module
    report: object
    scaling_report: object
    per_group: dict
    placement: object
    prompt: object


def _build_cfg():
    from mmengine.config import Config

    config_path, checkpoint_path = GLIP_VARIANTS[VARIANT]
    cfg = Config.fromfile(config_path)
    if cfg.model.bbox_head.get("early_fuse") is not True:
        raise SpecViolationError(
            f"{config_path} has early_fuse={cfg.model.bbox_head.get('early_fuse')!r}. "
            f"Spec §2's GLIP placement needs the VLFuse cross-attention and the "
            f"in-tower BERT layers, and under early_fuse=False neither group "
            f"exists — there would be nothing to adapt")
    # GLIP's own gradient-checkpointing switch inside VLFuse (§4.1).
    cfg.model.bbox_head.use_checkpoint = True
    # The zero-shot config carries no loss configuration, so `ATSSHead` would
    # fall back to `AnchorHead`'s generic SmoothL1/CrossEntropy defaults, which
    # do not even have compatible shapes here (the box loss is called with a
    # per-box centerness weight). These four values are copied verbatim from
    # mmdetection's own GLIP fine-tuning config, i.e. the recipe the
    # architecture is actually trained with. The neck is deliberately NOT
    # switched to that config's `FPN_DropBlock`: §2 freezes the neck, and adding
    # a train-time regulariser inside a frozen module is a change to the model
    # rather than to the optimisation.
    cfg.model.bbox_head.loss_cls = dict(type="FocalLoss", use_sigmoid=True,
                                        gamma=2.0, alpha=0.25, loss_weight=1.0)
    cfg.model.bbox_head.loss_bbox = dict(type="GIoULoss", loss_weight=2.0)
    cfg.model.bbox_head.loss_centerness = dict(type="CrossEntropyLoss",
                                               use_sigmoid=True, loss_weight=1.0)
    cfg.model.train_cfg.assigner.iou_calculator = dict(type="BboxOverlaps2D_GLIP")
    # §6 values, pushed into the model so the in-loop probe reads out exactly as
    # evaluation does. Only `score_thr` and `max_per_img` are touched; `nms_pre`
    # and the NMS IoU stay at GLIP's own defaults per §6.
    cfg.model.test_cfg.score_thr = EVAL_PROTOCOL.score_threshold
    cfg.model.test_cfg.max_per_img = MAX_OUTPUT_DETECTIONS
    return cfg, checkpoint_path


def _load_base_model(device: str):
    from mmdet.apis import init_detector

    cfg, checkpoint_path = _build_cfg()
    model = init_detector(cfg, checkpoint_path, device=device, palette="none")
    return cfg, model


def _inject_lora(model) -> object:
    from peft import LoraConfig, inject_adapter_in_model

    placement = get_placement(MODEL_KEY)
    targets = select_targets([n for n, _ in model.named_modules()], placement)
    if len(targets) != EXPECTED_LORA_MODULES[MODEL_KEY]:
        raise SpecViolationError(
            f"§2 regex selected {len(targets)} modules but the measured table "
            f"says {EXPECTED_LORA_MODULES[MODEL_KEY]}")
    inject_adapter_in_model(
        LoraConfig(r=LORA.r, lora_alpha=LORA.lora_alpha,
                   lora_dropout=LORA.lora_dropout, bias=LORA.bias,
                   target_modules=targets),
        model)
    return placement


def build_lora_model(device: str, output_dir: Path) -> BuiltModel:
    """Build GLIP-T(C), inject §2 LoRA, run the gate. Never trains on failure."""
    _, model = _load_base_model(device)
    placement = _inject_lora(model)
    n_lora, n_head = mark_trainable(model, placement)
    logger.info("trainable tensors: %d LoRA, %d detection-head (§2 heads: %s). "
                "GLIP has no learnable classification head — class scores are a "
                "dot product, so dot_product_projection_text is the only "
                "classification-side tensor", n_lora, n_head,
                list(placement.modules_to_save))

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

    prompt = build_tokens_positive(training_caption(), tuple(training_class_names()))
    if prompt.caption != training_caption():
        raise SpecViolationError("the prompt builder altered the frozen caption")

    built = BuiltModel(model=model, report=report, scaling_report=scaling_report,
                       per_group=per_group, placement=placement, prompt=prompt)
    _write_injection_map(output_dir, built)
    return built


def _write_injection_map(output_dir: Path, built: BuiltModel) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_KEY,
        "variant": VARIANT,
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
        "tokens_positive": [[list(span) for span in term]
                            for term in built.prompt.tokens_positive],
        "input_resolution": {"train_short_edges": list(GLIP_TRAIN_SHORT_EDGES),
                             "max_size": EVAL_MAX_SIZE},
    }
    (output_dir / "injection_map.json").write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", output_dir / "injection_map.json")


# --- loss --------------------------------------------------------------------

def glip_loss(model, batch: dict, prompt, tokenized) -> dict:
    """mmdet's ``GLIP.loss`` with the caption rebuild replaced by the V6 caption.

    Every other step is mmdet's: per-box positive maps over the tokenised
    caption, one caption per image through the language model, the frozen-backbone
    visual features, and ``bbox_head.loss`` (ATSS assignment, sigmoid BCE over the
    256 token columns, GIoU, centerness).
    """
    from mmdet.models.detectors.glip import create_positive_map

    samples = batch["data_samples"]
    device = batch["inputs"].device
    for sample in samples:
        labels = sample.gt_instances.labels
        per_box = [[list(span) for span in prompt.tokens_positive[int(label)]]
                   for label in labels]
        positive_map = create_positive_map(tokenized, per_box)
        # `.bool().float()` is mmdet's own comment-flagged requirement: the raw
        # map is span-length-normalised and the head expects a 0/1 target.
        sample.gt_instances.positive_maps = positive_map.to(device).bool().float()

    language = model.language_model([prompt.caption] * len(samples))
    visual = model.extract_feat(batch["inputs"])
    return model.bbox_head.loss(visual, language, samples)


# --- collapse probe ----------------------------------------------------------

@torch.no_grad()
def run_probe(model, pipeline, records: Sequence[V6Record], prompt, device: str,
              batch_size: int,
              score_floor: float = PROBE_SCORE_THRESHOLD) -> tuple[float, float, float, int]:
    """Phrase discrimination on the fixed probe subset, through mmdet's test path."""
    was_training = model.training
    model.eval()
    terms = prompt.terms
    detections: dict[int, list[tuple[str, float, tuple]]] = {}
    try:
        for start in range(0, len(records), batch_size):
            inputs, samples = [], []
            for record in records[start:start + batch_size]:
                with Image.open(record.image_path) as raw:
                    array = np.ascontiguousarray(
                        np.asarray(raw.convert("RGB"))[:, :, ::-1])
                data = pipeline(dict(img=array, img_id=0, text=prompt.caption,
                                     custom_entities=False))
                sample = data["data_samples"]
                sample.tokens_positive = [list(span) for span in prompt.tokens_positive]
                inputs.append(data["inputs"])
                samples.append(sample)
            results = model.test_step({"inputs": inputs, "data_samples": samples})
            for offset, result in enumerate(results):
                instances = result.pred_instances
                rows: list[tuple[str, float, tuple]] = []
                for box, score, label in zip(
                        instances.bboxes.detach().cpu().numpy(),
                        instances.scores.detach().cpu().numpy(),
                        instances.labels.detach().cpu().numpy()):
                    if float(score) < score_floor:
                        continue
                    index = int(label)
                    if not 0 <= index < len(terms):
                        raise ValueError(
                            f"GLIP returned label index {index} for a {len(terms)}-term "
                            f"query; the positive map and the term list are out of step")
                    rows.append((terms[index], float(score),
                                 tuple(float(v) for v in box)))
                if rows:
                    detections[start + offset] = rows
    finally:
        model.train(was_training)
    if not detections:
        raise RuntimeError(
            "the collapse probe produced no detections above the score floor; "
            "refusing to report a health metric from an empty measurement")
    return measure_discrimination(detections)


def build_probe_pipeline(cfg):
    from mmdet.utils import get_test_pipeline_cfg
    from mmengine.dataset import Compose

    pipeline_cfg = [t for t in get_test_pipeline_cfg(cfg.copy())
                    if t["type"] != "LoadAnnotations"]
    pipeline_cfg[0]["type"] = "mmdet.LoadImageFromNDArray"
    return Compose(pipeline_cfg)


# --- checkpoint round trip ---------------------------------------------------

def verify_checkpoint_roundtrip(directory: Path, device: str) -> dict:
    """Rebuild from the pretrained checkpoint, reload the adapter, re-gate §2."""
    manifest = read_manifest(directory)
    _, model = _load_base_model(device)
    placement = _inject_lora(model)
    mark_trainable(model, placement)
    n_loaded = load_adapter_weights(model, directory, strict=True)
    verify_loaded_checkpoint(model, manifest, placement,
                             expected_caption=training_caption())
    per_group = assert_group_counts(model, placement,
                                    EXPECTED_GROUP_COUNTS[MODEL_KEY], group_of)
    del model
    torch.cuda.empty_cache()
    logger.info("checkpoint round trip OK: %d tensors, per-group %s", n_loaded, per_group)
    return {"directory": str(directory), "tensors_loaded": n_loaded,
            "per_group": per_group, "step": manifest.step,
            "scaling": manifest.scaling, "caption": manifest.training_caption,
            "passes": True}


# --- training ----------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    logger.info("seeded python/numpy/torch with %d", seed)


def _first_backward_gradient_gate(model: nn.Module) -> None:
    """Assert every ``lora_B`` received a non-zero gradient after backward 1."""
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
            f"first backward, e.g. {dead[:5]}. GLIP's VLFuse calls "
            f"torch.utils.checkpoint.checkpoint without use_reentrant=, and the "
            f"reentrant implementation returns no gradient when the checkpointed "
            f"inputs do not require grad")
    logger.info("gradient gate: %d/%d lora_B tensors received gradient through the "
                "checkpointed VLFuse blocks", alive, total)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class _GlipTraining:
    """GLIP behind the shared loop's interface.

    mmdet's data preprocessor runs inside the step rather than in the collator:
    it moves and normalises the batch, and it wants the model's own device
    placement, which only exists once the model is built.
    """

    def __init__(self, built, device, plan, prompt, tokenized, probe_pipeline,
                 probe_subset, seed):
        self.built, self.model = built, built.model
        self.device, self.plan, self.seed = device, plan, seed
        self.prompt, self.tokenized = prompt, tokenized
        self.probe_pipeline, self.probe_subset = probe_pipeline, probe_subset
        self.trainable = [p for p in built.model.parameters() if p.requires_grad]

    def training_step(self, batch):
        batch = self.model.data_preprocessor(batch, True)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.device.startswith("cuda")):
            loss_dict = glip_loss(self.model, batch, self.prompt, self.tokenized)
            losses = sum(v.sum() for v in loss_dict.values())
        return StepResult(loss=losses, loss_scalar=float(losses.detach()),
                          loss_class=float(loss_dict["loss_cls"].detach().sum()))

    def clip_parameters(self):
        return self.trainable

    def probe(self, step, loss_class_ema):
        m = run_probe(self.model, self.probe_pipeline, self.probe_subset,
                      self.prompt, self.device, self.plan.per_device_batch)
        return ProbeMetrics(step=step, n_boxes=m[3], dup_score_rate=m[0],
                            labels_per_box=m[1], mean_score_spread=m[2],
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
        notes=f"variant={VARIANT}; per_group={built.per_group}; tag={tag}; "
              f"max_output_detections={MAX_OUTPUT_DETECTIONS}")
    return save_v6_checkpoint(built.model, manifest, output_dir / f"ckpt_{tag}")


def train(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    force_non_reentrant_checkpointing(USE_REENTRANT)
    enable_bf16_modulated_deform_conv()
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
                       "checkpoint must not be reported as a result", n_steps, TOTAL_STEPS)

    cfg, _ = _build_cfg()
    built = build_lora_model(args.device, output_dir)
    model = built.model
    prompt = built.prompt
    tokenized = model.language_model.tokenizer([prompt.caption], return_tensors="pt")

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
    dataset = GlipV6Dataset(records)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=plan.per_device_batch, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_mm,
                        drop_last=True, generator=generator,
                        persistent_workers=args.num_workers > 0)
    logger.info("train loader: %d images, per-device batch %d, %d workers "
                "(no empty-image filtering; §4.0)", len(dataset),
                plan.per_device_batch, args.num_workers)

    probe_subset = probe_records("val", args.probe_images)
    probe_pipeline = build_probe_pipeline(cfg)

    (output_dir / "run_config.json").write_text(json.dumps({
        "model": MODEL_KEY, "variant": VARIANT, "seed": args.seed,
        "steps": n_steps, "total_steps": TOTAL_STEPS, "device": args.device,
        "per_device_batch": plan.per_device_batch,
        "grad_accumulation": plan.grad_accumulation,
        "effective_batch": plan.effective_batch,
        "precision": "bf16", "max_grad_norm": MAX_GRAD_NORM,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": USE_REENTRANT,
        "gradient_checkpointing_detail": "bbox_head.use_checkpoint=True (VLFuse)",
        "modulated_deform_conv_precision": (
            "fp32 under bf16 autocast — mmcv's modulated_deformable_im2col_gpu has "
            "no BFloat16 kernel and all 18 DCNs sit inside the §2-adapted fusion "
            "tower. Same remedy as the GDINO trainer's deformable-attention patch. "
            "Spec §8, 2026-08-10."),
        "input_resolution": {"train_short_edges": list(GLIP_TRAIN_SHORT_EDGES),
                             "max_size": EVAL_MAX_SIZE,
                             "source": "mmdet GLIP fine-tune config"},
        "augmentation": "mmdet GLIP fine-tune pipeline (GTBoxSubOne_GLIP + "
                        "RandomChoiceResize/FixScaleResize + RandomFlip_GLIP + "
                        "FilterAnnotations keep_empty=False)",
        "task_losses": {"loss_cls": "FocalLoss(gamma=2.0, alpha=0.25, w=1.0)",
                        "loss_bbox": "GIoULoss(w=2.0)",
                        "loss_centerness": "CrossEntropyLoss(sigmoid, w=1.0)",
                        "assigner_iou": "BboxOverlaps2D_GLIP",
                        "source": "mmdet configs/glip/*_funtune_coco.py; the "
                                  "zero-shot config ships no loss config",
                        "neck": "FPN kept (NOT the fine-tune config's "
                                "FPN_DropBlock) — §2 freezes the neck"},
        "probe_score_threshold": PROBE_SCORE_THRESHOLD,
        "max_output_detections": MAX_OUTPUT_DETECTIONS,
        "test_cfg_protocol_overrides": ["score_thr", "max_per_img"],
        "training_caption": training_caption(),
        "training_class_names": training_class_names(),
        "loss": "mmdet ATSSVLFusionHead.loss with the frozen V6 caption and "
                "explicit tokens_positive (see glip_loss)",
        "max_images": args.max_images,
        "thresholds": thresholds.as_record(),
        "param_groups": [s.as_record() for s in specs],
    }, indent=2))

    baseline = run_probe(model, probe_pipeline, probe_subset, prompt, args.device,
                         plan.per_device_batch)
    logger.info("zero-shot probe: dup_score_rate=%.6f labels_per_box=%.4f "
                "score_spread=%.4f boxes=%d", *baseline)
    guard.evaluate(ProbeMetrics(step=0, n_boxes=baseline[3], dup_score_rate=baseline[0],
                                labels_per_box=baseline[1], mean_score_spread=baseline[2]))

    model.train()
    trainer = _GlipTraining(built, args.device, plan, prompt, tokenized,
                            probe_pipeline, probe_subset, args.seed)
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
                    "for %d steps", mean, len(warm),
                    timing["projected_hours_full_run"], TOTAL_STEPS)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        timing["peak_gib_allocated"] = torch.cuda.max_memory_allocated() / 2 ** 30
        timing["peak_gib_reserved"] = torch.cuda.max_memory_reserved() / 2 ** 30
        logger.info("peak GPU memory: %.2f GiB allocated, %.2f GiB reserved",
                    timing["peak_gib_allocated"], timing["peak_gib_reserved"])

    roundtrip = None
    if final_dir is not None and args.verify_roundtrip:
        del model, built.model, optimizer, trainable
        torch.cuda.empty_cache()
        roundtrip = verify_checkpoint_roundtrip(final_dir, args.device)
    (output_dir / "run_summary.json").write_text(json.dumps(
        {"model": MODEL_KEY, "seed": args.seed, "steps_run": len(step_times),
         "exit_code": exit_code, "timing": timing,
         "checkpoint_roundtrip": roundtrip, "alarms": guard.alarms}, indent=2))
    return exit_code


def _cycle(loader):
    while True:
        yield from loader


def _plan_override(per_device: int):
    from utils.train.training_schedule import BatchPlan

    if EFFECTIVE_BATCH_SIZE % per_device:
        raise ValueError(f"per-device batch {per_device} does not divide the "
                         f"effective batch {EFFECTIVE_BATCH_SIZE}")
    logger.warning("--per-device-batch %d overrides the registered table; allowed "
                   "only for memory probing, never for a reported run", per_device)
    return BatchPlan(model=MODEL_KEY, per_device_batch=per_device,
                     grad_accumulation=EFFECTIVE_BATCH_SIZE // per_device, world_size=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--gate-only", type=int, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None,
                        help="SMOKE ONLY; §4.0 forbids reporting a subsampled run")
    parser.add_argument("--per-device-batch", type=int, default=None,
                        help="memory probing only")
    parser.add_argument("--probe-every", type=int,
                        default=CollapseThresholds().probe_every_steps)
    parser.add_argument("--probe-images", type=int,
                        default=CollapseThresholds().probe_images)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verify-roundtrip", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device.startswith("cuda") and args.device not in ("cuda", "cuda:0"):
        logger.warning("--device %s: pin the GPU with CUDA_VISIBLE_DEVICES and use "
                       "cuda:0", args.device)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
