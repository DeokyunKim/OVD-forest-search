"""Measure what `train.py --config` produced, and write the numbers as JSON.

    python eval.py --config configs/experiments/gdino_seed1.yaml

One option, because the config describes the whole run: which model, which
adapter, which queries, and which of the two kinds of number to report.

**Detection** (`eval.mode: detection`, the default) runs the frozen protocol
over the benchmark and writes a fragment per condition: per-class AP/AP50/AP75/AR,
the query, the post-processing settings and the environment that produced them.
With `eval.dump` it also writes the raw boxes, which the flip mode reads.

**Flip** (`eval.mode: flip`) computes the query-sensitivity rates -- retention,
negative-flip, positive-flip, stealing -- from two dumps of the same model under
two queries. No inference: a re-reading of boxes already on disk.

Pick the card with `CUDA_VISIBLE_DEVICES=N`. There is no device option: adapted
Grounding DINO variants fail on a non-zero ordinal, and OmDet-Turbo corrupts
silently there -- same image, same query, a different number of detections
between runs, with nothing in the log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from utils.runner.logging_utils import get_logger  # noqa: E402
from utils.runner.paths import paths  # noqa: E402
from utils.runner.model_registry import (  # noqa: E402
    default_variants,
    model_settings,
)

logger = get_logger(__name__)

#: One runner for every model. They differ in how a detector is constructed --
#: mmdet for GLIP, transformers for OWLv2 and OmDet, the external repo for
#: Grounding DINO -- and that difference belongs in the detector factory, not in
#: which script runs. Scoring is the same protocol either way.
RUNNER = "utils/runner/_eval.py"
#: Both come from configs/models.yaml. `max_output_detections` widens the
#: candidate set BEFORE the protocol's per-category cap of 100, so it is part of
#: the published read-out rather than a tuning knob -- and a reader who opens the
#: config should see it, which they could not while it lived in a dict here.
DEFAULT_VARIANT = default_variants()

#: Always cuda:0, with the card chosen by CUDA_VISIBLE_DEVICES. Not an option:
#: adapted Grounding DINO variants fail on a non-zero ordinal, and OmDet-Turbo
#: corrupts silently there -- same image, same query, a different detection
#: count between runs, with nothing in the log. A flag would let a caller opt
#: into that, which is not a choice worth offering.
DEVICE = "cuda:0"


def extra_args(model: str) -> list[str]:
    """Model-specific read-out flags, if the config declares any."""
    cap = model_settings(model).max_output_detections
    return ["--max-output-detections", str(cap)] if cap is not None else []


def resolve_adapter(output_dir: Path, steps: int, config: Path) -> Path:
    """The adapter directory `train.py` actually wrote, under `output_dir`.

    The two entry points read the same config field and meant different things
    by it: training writes `<output_dir>/ckpt_final_step<N>/`, and evaluation
    was handed `<output_dir>` itself. That is the promise of one file driving
    both failing exactly where it matters, and it only shows when a config is
    used for training and then for evaluation -- which is the intended workflow.

    A directory holding `v6_manifest.json` IS the checkpoint, so a config that
    names one directly (as the validation configs do) still works.
    """
    if (output_dir / "v6_manifest.json").is_file():
        return output_dir
    final = output_dir / f"ckpt_final_step{steps}"
    if (final / "v6_manifest.json").is_file():
        return final
    if not output_dir.exists():
        raise SystemExit(
            f"{output_dir} does not exist. Run `train.py --config {config}` "
            f"first; evaluating a fine-tuned arm without its adapter would "
            f"silently measure the zero-shot model.")
    found = sorted(p.name for p in output_dir.iterdir() if p.is_dir())
    raise SystemExit(
        f"no V6 checkpoint under {output_dir}: neither it nor "
        f"ckpt_final_step{steps}/ holds a v6_manifest.json. It contains "
        f"{found or 'no subdirectories'}. A checkpoint without its manifest is "
        f"refused rather than loaded blind.")


def run_experiment(args: argparse.Namespace) -> int:
    """Evaluate what `train.py --config` produced, per corpus.

    Every condition is run once with the detection dump on, then the dump is
    re-scored for ForestPersons and ForestBelongings separately. Scoring twice
    from one dump gives the same numbers a second inference pass would and costs
    no GPU time; more to the point, both splits then come from the identical set
    of boxes, so a difference between them is a property of the corpora and not
    of two runs.
    """
    from utils.runner.experiment import load_experiment
    from utils.metrics.by_dataset import score_by_dataset

    experiment = load_experiment(args.config)
    # The zero-shot arm reads no adapter. That is the whole difference between
    # the arms -- same protocol, same read-out -- so it is one branch here and
    # nothing else changes.
    adapter = (None if experiment.eval.arm == "zeroshot"
               else resolve_adapter(experiment.output_dir, experiment.train.steps,
                                    args.config))

    variant = DEFAULT_VARIANT[experiment.model]
    if experiment.eval.mode == "flip":
        return run_flip(experiment, variant)

    conditions = experiment.eval.conditions
    if not conditions:
        from utils.protocol.conditions import all_conditions
        conditions = tuple(c.condition_id for c in all_conditions()
                           if c.condition_id.startswith("A"))

    logger.info("%s seed %d: %d condition(s), adapter %s",
                experiment.model, experiment.seed, len(conditions), adapter)
    failed = []
    for i, condition in enumerate(conditions, 1):
        cmd = [sys.executable, str(REPO / RUNNER),
               "--condition", condition, "--model", experiment.model,
               "--variant", variant, "--device", DEVICE,
               "--fragment-variant", experiment.fragment_label(variant),
               "--out-dir", str(paths().output("fragments"))]
        if adapter is not None:
            cmd += ["--v6-adapter", str(adapter)]
        if experiment.eval.resume:
            cmd += ["--resume"]
        cmd += extra_args(experiment.model)
        if not experiment.eval.dump:
            cmd += ["--no-detection-dump"]
        logger.info("[%d/%d] %s", i, len(conditions), condition)
        if subprocess.call(cmd, cwd=str(REPO)) != 0:
            failed.append(condition)

    if failed:
        logger.error("%d condition(s) failed: %s", len(failed), ", ".join(failed))
        return 1
    if experiment.eval.per_dataset:
        out = score_by_dataset(experiment.model, experiment.fragment_label(variant),
                               conditions)
        print(f"per-corpus metrics written to {out}")
    return 0


def run_flip(experiment, variant: str) -> int:
    """Query-sensitivity rates from dumps this model already wrote."""
    dumps = paths().output("detection_dumps")
    fragments = paths().output("fragments")
    label = experiment.fragment_label(variant)
    failed = []
    for condition in experiment.eval.conditions:
        out = (paths().output("analysis") /
               f"flip__{experiment.model}__{label}__{condition}.json")
        cmd = [sys.executable, str(REPO / "utils/metrics/_flip_entry.py"),
               "--dump-q0", str(dumps / f"{experiment.model}__{label}__"
                                        f"{experiment.eval.reference}.npz"),
               "--dump-q1", str(dumps / f"{experiment.model}__{label}__{condition}.npz"),
               "--test-json", str(paths().require("benchmark") / "test.json"),
               "--target-class", experiment.eval.target,
               "--out", str(out)]
        # The fragments carry the AP/AR the two queries scored, which turns the
        # flip rates into a delta-accuracy layer and lets the identity be checked
        # against an independently computed AR rather than only against itself.
        # Passed only when present: a missing fragment is a reason to report
        # less, not a reason to fail.
        for flag, cond in (("--fragment-q0", experiment.eval.reference),
                           ("--fragment-q1", condition)):
            fragment = fragments / f"{experiment.model}__{label}__{cond}.json"
            if fragment.is_file():
                cmd += [flag, str(fragment)]
        if subprocess.call(cmd, cwd=str(REPO)) != 0:
            failed.append(condition)
    if failed:
        logger.error("%d condition(s) failed: %s", len(failed), ", ".join(failed))
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="experiment YAML, the same file train.py read. It "
                         "names the adapter, the queries and the mode, so the "
                         "measurement and its description cannot drift apart.")
    args = ap.parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
