#!/usr/bin/env python
"""Phase 2 LoRA trainer for OWLv2-large, under the frozen spec.

Spec: the frozen-hyperparameter specification §1 (LoRA), §2 (placement — the
per-model OWLv2 table), §3 (optimizer), §4.0/§4.1 (data policy and loop),
§5 (resolution), §6 (evaluation — untouched here).
Machinery: ``models/owlv2/train.py``. Placement: ``models/shared/lora_spec_v6line.py``.
Pre-registered thresholds: the training risk register §9.

The reference implementation is ``train.py --model gdino``, which is **not
modified**. Every gate it runs before step 1 is run here against OWLv2's own
measured counts, and two things are different by necessity:

**The collapse metrics are not the same metrics.** ``dup_score_rate`` and
``labels_per_box`` describe a joint-caption detector. OWLv2 takes an ordered
query list, encodes each term separately, and scores every (patch, term) pair
independently — so those two are structurally incapable of firing here and
would have produced an armed guard that logs reassuring numbers forever. The
four signals that replace them, their measured zero-shot baselines and their
headroom are in ``models/owlv2/guard.py`` and risk register §9.

**The loss is reconstructed.** ``Owlv2ForObjectDetection.forward`` takes no
``labels``; HuggingFace ships inference only. ``models/shared/detection_loss.py``
rebuilds the set-prediction objective from ``logits`` and ``pred_boxes`` — the
same module the OmDet-Turbo arm uses.

Usage (smoke, seed 1)::

    CUDA_VISIBLE_DEVICES=3 TMPDIR="$TMPDIR" \\
      ~/venvs/uc-owlv2/bin/python train_v6_owlv2.py (research tree) \\
        --seed 1 --gate-only 20 --device cuda:0 \\
        --output-dir outputs/v6/train/owlv2_seed1_smoke

Always pin the GPU with ``CUDA_VISIBLE_DEVICES`` and keep ``--device cuda:0``
(CLAUDE.md: adapted variants have failed or silently changed results on
non-zero device indices).
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
from pathlib import Path
from typing import Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / 'pyproject.toml').is_file())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from models.shared.detection_loss import LossWeights, set_prediction_loss  # noqa: E402
from utils.runner.logging_utils import get_logger  # noqa: E402
from utils.train.checkpoint import build_manifest, save_v6_checkpoint  # noqa: E402
from utils.train.loop import (  # noqa: E402
    LoopConfig,
    StepResult,
    run_training_loop,
)
from models.gdino.lora import describe_schedule, make_lr_lambda  # noqa: E402
from models.shared.lora_spec_v6line import SpecViolationError  # noqa: E402
from models.owlv2.guard import (  # noqa: E402
    ACTION_ABORT,
    ACTION_OK,
    ZERO_SHOT_BASELINE,
    Owlv2CollapseGuard,
    Owlv2Thresholds,
    summarise_thresholds,
)
from models.owlv2.train import (  # noqa: E402
    CHECKPOINT_ID,
    EXPECTED_LORA_MODULES,
    HF_CACHE,
    INPUT_RESOLUTION,
    MODEL_KEY,
    N_PATCHES,
    Owlv2Collator,
    Owlv2V6Dataset,
    ProbeConfig,
    assert_query_identity,
    baseline_record,
    build_lora_model,
    first_backward_gradient_gate,
    head_gradient_gate,
    measure_health,
    text_query_embeddings,
    training_query_list,
)
from utils.protocol.train_caption import training_caption  # noqa: E402
from utils.train.training_schedule import (  # noqa: E402
    EFFECTIVE_BATCH_SIZE,
    LORA,
    MAX_GRAD_NORM,
    TOTAL_STEPS,
    USE_REENTRANT,
    plan_batch,
)
from utils.data.xarch_data import load_v6_records, probe_records  # noqa: E402
from models.shared.train import build_param_groups, describe_param_groups  # noqa: E402

logger = get_logger("v6.train.owlv2")

CHECKPOINT_EVERY_STEPS = 2_000


# --- determinism -------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch. Fixes data order and initialisation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    logger.info("seeded python/numpy/torch with %d", seed)


# --- io ----------------------------------------------------------------------

def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _write_injection_map(output_dir: Path, built, query_identity: dict) -> None:
    """Record what was adapted, at build time (risk register §6).

    A value reconstructed from config at load time and never stored beside the
    weights is indistinguishable from a wrong one — that is exactly how alpha
    was silently halved on this project.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": MODEL_KEY,
        "checkpoint": CHECKPOINT_ID,
        "spec_version": "frozen-2026-08-08",
        "lora_r": LORA.r, "lora_alpha": LORA.lora_alpha,
        "lora_dropout": LORA.lora_dropout,
        "scaling": LORA.lora_alpha / LORA.r,
        "observed_scalings": list(built.scaling_report.observed_scalings),
        "total_modules": built.report.n_lora_modules,
        "expected_modules": EXPECTED_LORA_MODULES,
        "per_region_modules": dict(built.report.per_region_counts),
        "lora_modules_in_text_tower": 0,
        "trainable_params": built.report.trainable_params,
        "total_params": built.report.total_params,
        "trainable_pct": built.report.trainable_pct,
        "heads": built.head_summary,
        "training_query_list": query_identity,
        "training_caption_equivalent": training_caption(),
        "input_resolution": {"height": INPUT_RESOLUTION[0], "width": INPUT_RESOLUTION[1],
                             "n_patches": N_PATCHES,
                             "source": "Owlv2ImageProcessor of the checkpoint",
                             "spec_5_deviation": (
                                 "§5 lists 960x960, which is the owlv2-base-patch16 "
                                 "value; owlv2-large-patch14-ensemble is fixed at "
                                 "1008x1008 by its own processor. ZS and FT use the "
                                 "same value, which is what §5 requires")},
    }
    (output_dir / "injection_map.json").write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", output_dir / "injection_map.json")


class _Owlv2Training:
    """OWLv2 behind the shared loop's interface.

    Two gradient gates run after the first backward, not one: the second checks
    that the detection head received gradient, which the first does not cover.
    """

    def __init__(self, built, processor, device, plan, weights, probe_subset,
                 terms, reference_text, probe_config, seed):
        self.built, self.model = built, built.model
        self.processor, self.device, self.plan = processor, device, plan
        self.weights, self.seed = weights, seed
        self.probe_subset, self.terms = probe_subset, terms
        self.reference_text, self.probe_config = reference_text, probe_config
        self.trainable = [p for p in built.model.parameters() if p.requires_grad]

    def training_step(self, batch):
        inputs, targets, _ = batch
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.device.startswith("cuda")):
            outputs = self.model(**inputs)
        losses = set_prediction_loss(outputs.logits.float(),
                                     outputs.pred_boxes.float(),
                                     targets, weights=self.weights)
        return StepResult(loss=losses["loss_detection"],
                          loss_scalar=float(losses["loss_detection"].detach()),
                          loss_class=float(losses["loss_class"].detach()))

    def clip_parameters(self):
        return self.trainable

    def probe(self, step, loss_class_ema):
        return measure_health(self.model, self.processor, self.probe_subset,
                              self.device, self.terms, self.reference_text,
                              self.probe_config, step=step,
                              loss_class_ema=loss_class_ema)

    def save_checkpoint(self, output_dir: Path, step: int, tag: str) -> Path:
        return _save_checkpoint(self.built, output_dir, step, self.seed, tag=tag)


def _save_checkpoint(built, output_dir: Path, step: int, seed: int, tag: str) -> Path:
    manifest = build_manifest(
        model_name=MODEL_KEY, report=built.report,
        scaling_report=built.scaling_report, step=step, total_steps=TOTAL_STEPS,
        seed=seed, effective_batch=EFFECTIVE_BATCH_SIZE,
        training_caption=training_caption(), lora_r=LORA.r,
        lora_alpha=LORA.lora_alpha, lora_dropout=LORA.lora_dropout,
        notes=f"tag={tag}; checkpoint={CHECKPOINT_ID}; "
              f"query_list={training_query_list()}; "
              f"heads_fully_trained=box_head+class_head")
    return save_v6_checkpoint(built.model, manifest, output_dir / f"ckpt_{tag}")


# --- training ----------------------------------------------------------------

def train(args: argparse.Namespace) -> int:
    """Run the loop. Returns a process exit code (0 ok, 2 collapse abort).

    A ``STOP_TRAIN_owlv2_seed<N>`` file under ``outputs/v6/`` disarms this seed
    before anything is built. It exists because a dispatcher that is already
    running cannot be disarmed any other way: its shell script must not be
    edited (bash re-reads a running script by byte offset) and signalling it is
    not permitted. The dispatcher therefore still fires, this guard still runs,
    and the card is handed back within seconds instead of being held for eight
    hours. The check is per SEED, so disarming seed 2 never touches a seed 0 or
    seed 1 run already in flight.
    """
    output_dir = Path(args.output_dir)
    stop_file = (Path(__file__).resolve().parents[2] / "outputs" / "v6"
                 / f"STOP_TRAIN_owlv2_seed{args.seed}")
    if stop_file.exists():
        logger.error("STOP_TRAIN present: %s", stop_file)
        logger.error("refusing to start owlv2 seed %d. Reason:\n%s",
                     args.seed, stop_file.read_text().strip())
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    plan = plan_batch(MODEL_KEY, world_size=1)
    logger.info("%s", plan.describe())
    logger.info("%s", describe_schedule(TOTAL_STEPS))

    n_steps = args.gate_only if args.gate_only is not None else args.steps
    if n_steps > TOTAL_STEPS:
        raise ValueError(f"--steps {n_steps} exceeds the frozen budget {TOTAL_STEPS}")
    if n_steps < TOTAL_STEPS:
        logger.warning("running %d of %d steps: this is a GATE/SMOKE run and its "
                       "checkpoint must not be reported as a result (§4.1 fixes "
                       "the final step)", n_steps, TOTAL_STEPS)

    processor, built = build_lora_model(args.device, cache_dir=args.cache_dir,
                                        gradient_checkpointing=True)
    model = built.model
    terms = training_query_list()
    query_identity = assert_query_identity(processor, terms)
    logger.info("query-list identity OK: training list %s is byte-identical, "
                "token for token, to what evaluation sends (%s)",
                query_identity["terms"], query_identity["input_ids_shape"])
    _write_injection_map(output_dir, built, query_identity)

    param_groups, specs = build_param_groups(model, built.placement)
    logger.info("%s", describe_param_groups(specs))
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(TOTAL_STEPS))

    thresholds = Owlv2Thresholds(probe_every_steps=args.probe_every,
                                 probe_images=args.probe_images)
    guard = Owlv2CollapseGuard(thresholds=thresholds, armed=True,
                               trajectory_path=output_dir / "collapse_trajectory.jsonl")
    logger.info("%s", guard.describe())
    for line in summarise_thresholds(thresholds, ZERO_SHOT_BASELINE):
        logger.info("%s", line)

    records = load_v6_records("train", max_images=args.max_images)
    dataset = Owlv2V6Dataset(records, train=True)
    collator = Owlv2Collator(processor, terms)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=plan.per_device_batch, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collator,
                        drop_last=True, pin_memory=False, generator=generator,
                        persistent_workers=args.num_workers > 0)
    logger.info("train loader: %d images, per-device batch %d, %d workers "
                "(no empty-image filtering; §4.0)", len(dataset),
                plan.per_device_batch, args.num_workers)

    probe_subset = probe_records("val", args.probe_images)
    probe_config = ProbeConfig(batch_size=plan.per_device_batch,
                               permutation_images=thresholds.permutation_images)
    reference_text = text_query_embeddings(model, processor, terms, args.device)

    (output_dir / "run_config.json").write_text(json.dumps({
        "model": MODEL_KEY, "checkpoint": CHECKPOINT_ID, "seed": args.seed,
        "steps": n_steps, "total_steps": TOTAL_STEPS, "device": args.device,
        "per_device_batch": plan.per_device_batch,
        "grad_accumulation": plan.grad_accumulation,
        "effective_batch": plan.effective_batch,
        "effective_batch_line": plan.describe(),
        "precision": "bf16", "max_grad_norm": MAX_GRAD_NORM,
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": USE_REENTRANT,
        "input_resolution": {"height": INPUT_RESOLUTION[0],
                             "width": INPUT_RESOLUTION[1], "n_patches": N_PATCHES},
        "augmentation": "horizontal flip p=0.5 only (processor fixes 1008x1008)",
        "box_normalisation": "padded square max(H,W); Owlv2ImageProcessor pads "
                             "to square before resizing and _scale_boxes "
                             "multiplies by max(H,W)",
        "training_query_list": query_identity,
        "loss": "reconstructed set-prediction (models/shared/detection_loss.py); "
                "Owlv2ForObjectDetection.forward takes no labels",
        "loss_weights": {"classification": LossWeights().classification,
                         "bbox_l1": LossWeights().bbox_l1,
                         "giou": LossWeights().giou},
        "max_images": args.max_images,
        "thresholds": thresholds.as_record(),
        "zero_shot_baseline": dict(ZERO_SHOT_BASELINE),
        "param_groups": [s.as_record() for s in specs],
    }, indent=2))

    baseline = measure_health(model, processor, probe_subset, args.device, terms,
                              reference_text, probe_config, step=0)
    logger.info("zero-shot probe: %s", baseline.as_record())
    guard.assert_healthy_baseline(baseline)
    (output_dir / "baseline_probe.json").write_text(
        json.dumps(baseline_record(baseline, thresholds), indent=2))

    model.train()
    trainer = _Owlv2Training(built, processor, args.device, plan, LossWeights(),
                             probe_subset, terms, reference_text, probe_config,
                             args.seed)
    trainable = trainer.trainable

    def _gates() -> None:
        first_backward_gradient_gate(model)
        head_gradient_gate(model)

    exit_code, step_times = run_training_loop(
        trainer, loader, optimizer, scheduler, guard,
        LoopConfig(n_steps=n_steps, accumulation=plan.grad_accumulation,
                   max_grad_norm=MAX_GRAD_NORM, log_every=args.log_every,
                   checkpoint_every=CHECKPOINT_EVERY_STEPS, seed=args.seed),
        output_dir, torch=torch, on_first_backward=_gates)


    if exit_code == 0:
        _save_checkpoint(built, output_dir, n_steps, args.seed, tag=f"final_step{n_steps}")
    if guard.alarms:
        guard.write_alarm_file(output_dir / "ALARM.md",
                               context={"final_step": len(step_times), "seed": args.seed})

    timing: dict = {}
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

    (output_dir / "run_summary.json").write_text(json.dumps(
        {"model": MODEL_KEY, "seed": args.seed, "steps_run": len(step_times),
         "exit_code": exit_code, "timing": timing,
         "lora_modules": built.report.n_lora_modules,
         "trainable_params": built.report.trainable_params,
         "total_params": built.report.total_params,
         "trainable_pct": built.report.trainable_pct,
         "heads": built.head_summary,
         "probe_history": [m.as_record() for m in guard.history],
         "alarms": guard.alarms}, indent=2))
    return exit_code


def _cycle(loader):
    while True:
        yield from loader


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
    parser.add_argument("--cache-dir", type=str, default=HF_CACHE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None,
                        help="SMOKE ONLY; §4.0 forbids reporting a subsampled run")
    parser.add_argument("--probe-every", type=int,
                        default=Owlv2Thresholds().probe_every_steps)
    parser.add_argument("--probe-images", type=int,
                        default=Owlv2Thresholds().probe_images)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device.startswith("cuda") and args.device not in ("cuda", "cuda:0"):
        logger.warning("--device %s: adapted variants on this machine have failed "
                       "or silently changed results on non-zero device indices; "
                       "pin the GPU with CUDA_VISIBLE_DEVICES and use cuda:0",
                       args.device)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
