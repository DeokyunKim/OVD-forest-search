"""Fine-tune one model with LoRA on the six-class benchmark.

    python train.py --model gdino --seed 0
    python train.py --model owlv2 --seed 1 --out checkpoints/owlv2/seed1

One trainer per architecture, dispatched here. They cannot be one script: each
model needs its own interpreter, so importing all four would fail in every
environment. Run this with the interpreter for the model you are training --
`models/<model>/README.md` says which.

Run with `CUDA_VISIBLE_DEVICES=N` and leave the device at cuda:0.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from utils.runner.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

TRAINER = {model: f"models/{model}/_train_entry.py"
           for model in ("gdino", "owlv2", "glip", "omdet")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="experiment YAML. The only option: everything a run "
                         "depends on is described there, and eval.py reads the "
                         "same file, so the two cannot disagree about which "
                         "adapter this run wrote.")
    args, passthrough = ap.parse_known_args()

    from utils.runner.experiment import load_experiment

    experiment = load_experiment(args.config)
    logger.info("config %s: %s seed %d -> %s", experiment.config_file,
                experiment.model, experiment.seed, experiment.output_dir)
    cmd = [sys.executable, str(REPO / TRAINER[experiment.model]),
           "--seed", str(experiment.seed),
           "--output-dir", str(experiment.output_dir),
           "--steps", str(experiment.train.steps),
           "--num-workers", str(experiment.train.num_workers)]
    cmd += passthrough
    logger.info("running %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO))


if __name__ == "__main__":
    raise SystemExit(main())
