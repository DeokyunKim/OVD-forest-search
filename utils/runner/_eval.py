#!/usr/bin/env python
"""Evaluate one condition on the full test census and write its fragment.

One runner for all four models. `eval.py` invokes this; run it directly only
when you want a single condition without the config machinery::

    CUDA_VISIBLE_DEVICES=0 TMPDIR="$TMPDIR" \\
    ~/venvs/uc-gdino/bin/python utils/runner/_eval.py \\
        --condition A4_full_v6 --model gdino --variant swinb_zeroshot --device cuda:0

The fine-tuned arm is the same command plus an adapter and a label that says so
(the label guard below refuses anything else)::

    ... --v6-adapter checkpoints/gdino/seed1/ckpt_final_step5667 \\
        --fragment-variant swinb_ft_seed1

Probe mode measures throughput and writes nothing::

    ... --condition C5_v6_plus_coco80 --limit 40 --probe-out outputs/speed/

One runner rather than one per model family, so the set of runnable conditions
is a property of the benchmark rather than of which file a model happens to
reach. The benchmark's claim is four architectures measured the same way, and
that has to include which queries they can be asked.

Always pin the GPU with ``CUDA_VISIBLE_DEVICES`` and keep ``--device cuda:0``.
Adapted GDINO variants break on non-zero ordinals, and OmDet-Turbo corrupts
silently there -- same image, same query, a different detection count between
runs, with nothing in the log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# The repository root: the directory that holds `pyproject.toml`. Found by
# walking up rather than by a fixed number of parents, and anchored on a file
# rather than a package name -- the package name changed once, and every anchor
# spelled with it had to be found by hand.
sys.path.insert(0, str(next(q for q in Path(__file__).resolve().parents
                            if (q / 'pyproject.toml').is_file())))

from utils.runner.logging_utils import get_logger  # noqa: E402
from utils.runner.paths import paths  # noqa: E402
from utils.runner.adapter_eval import assert_ft_label  # noqa: E402
from utils.protocol.cap_policy import CAP_POLICIES, DEFAULT_CAP_POLICY  # noqa: E402
from utils.protocol.conditions import get_condition  # noqa: E402
from utils.protocol.conditions_belongings_pairs import (  # noqa: E402
    get_belongings_pair_condition,
)
from utils.protocol.conditions_triples import get_triple_condition  # noqa: E402
from utils.data.dataset_build import EXPECTED_IMAGES  # noqa: E402
from utils.runner.detector import build_detector  # noqa: E402
from utils.runner.environment import FRAGMENT_KEY, environment_record  # noqa: E402
from utils.protocol.eval_protocol import EVAL_PROTOCOL  # noqa: E402
from utils.runner.fragment import write_fragment  # noqa: E402
from utils.runner.prefetch import (  # noqa: E402
    DEFAULT_PREFETCH_DEPTH,
    DEFAULT_PREFETCH_WORKERS,
)
from utils.runner.runner import (  # noqa: E402
    DEFAULT_DUMP_FORMAT,
    DUMP_FORMATS,
    load_test_set,
    run_condition,
)
from utils.runner.runner_version import describe, runner_fingerprint  # noqa: E402
from utils.metrics.text_similarity import cosine_lookup, load_cosines  # noqa: E402

logger = get_logger(__name__)

#: Locations come from configs/paths.yaml, never assembled from a literal: this
#: file lives inside the package, and joining "datasets/..." onto its parent is
#: what put the benchmark under the package directory when it last moved.
TEST_JSON = paths().dataset("benchmark") / "test.json"
FRAGMENT_DIR = paths().output("fragments")
DUMP_DIR = paths().output("detection_dumps")
COSINE_PATH = paths().output("analysis") / "text_cosines.json"

MODELS = ("gdino", "owlv2", "glip", "omdet")

#: Every condition registry, tried in order. All three are exact-key lookups, so
#: a typo raises rather than resolving to a neighbour. Listed here once so the
#: set of runnable conditions is the same for every model.
_REGISTRIES = (get_condition, get_triple_condition, get_belongings_pair_condition)


def resolve_condition(condition_id: str):
    """The condition with this exact id, from whichever registry defines it."""
    for lookup in _REGISTRIES:
        try:
            return lookup(condition_id)
        except KeyError:
            continue
    raise SystemExit(
        f"unknown condition {condition_id!r}. `python eval.py --list-conditions` "
        f"prints every id; they are matched exactly, never by prefix.")


def cosines_for(condition, model: str) -> dict[str, float] | None:
    """Axis-B cosines for this condition, from this model's own encoder.

    Driven by whether the table has an entry, not by which model is running:
    absent is absent, and a model gaining an entry needs no code change.
    """
    if condition.axis != "B" or not COSINE_PATH.is_file():
        return None
    table = cosine_lookup(load_cosines(COSINE_PATH))
    out: dict[str, float] = {}
    for term, gt_class in condition.term_to_gt_class.items():
        if gt_class is not None and term != gt_class:
            key = (model, gt_class, term)
            if key in table:
                out[term] = table[key]
    return out or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, help="exact condition_id")
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-override", default=None)
    parser.add_argument("--config-override", default=None,
                        help="GLIP: an mmdet config to build from instead of the "
                             "variant's own")
    parser.add_argument("--v6-adapter", default=None,
                        help="A LoRA checkpoint directory (e.g. "
                             "checkpoints/gdino/seed1/ckpt_final_step5667). "
                             "Switches the arm from zero-shot to fine-tuned; the "
                             "protocol, the read-out and the post-processing are "
                             "unchanged. Requires a --fragment-variant containing "
                             "'ft'.")
    parser.add_argument("--fragment-variant", default=None,
                        help="Label used in the fragment and dump FILENAMES; the "
                             "model is still built from --variant. Required with "
                             "--v6-adapter so a fine-tuned fragment cannot land on "
                             "top of its zero-shot counterpart.")
    parser.add_argument("--cache-dir", default=None,
                        help="OWLv2 / OmDet: HuggingFace cache directory")
    parser.add_argument("--test-json", type=Path, default=TEST_JSON)
    parser.add_argument("--out-dir", type=Path, default=FRAGMENT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="probe mode: process N images, write no fragment")
    parser.add_argument("--probe-out", type=Path, default=None,
                        help="probe mode: write the throughput record here")
    parser.add_argument("--cap-policy", default=DEFAULT_CAP_POLICY, choices=CAP_POLICIES,
                        help="per_category is the COCO standard and the default; "
                             "global_per_image is the deployment view.")
    parser.add_argument("--max-output-detections", type=int, default=None,
                        help="OWLv2 and GLIP: emit more candidates than the "
                             "protocol cap, so the cap selects from a full pool. "
                             "Part of the published read-out, not a tuning knob.")
    parser.add_argument("--no-prefetch", dest="prefetch", action="store_false",
                        help="escape hatch: run the serial loop (load -> "
                             "preprocess -> forward, one image at a time). The "
                             "fragment records which path was taken.")
    parser.add_argument("--prefetch-workers", type=int, default=DEFAULT_PREFETCH_WORKERS,
                        help="threads running image load + detector preprocessing")
    parser.add_argument("--prefetch-depth", type=int, default=DEFAULT_PREFETCH_DEPTH,
                        help="max staged images held in memory (bounded queue)")
    parser.add_argument("--detection-dump-dir", type=Path, default=DUMP_DIR,
                        help="one file of the SCORED detections per condition")
    parser.add_argument("--no-detection-dump", dest="detection_dump",
                        action="store_false",
                        help="do not write the detection dump (it is on by default)")
    parser.add_argument("--detection-dump-format", default=DEFAULT_DUMP_FORMAT,
                        choices=list(DUMP_FORMATS),
                        help="v6_npz.1 (default) packs the six columns as float32, "
                             "~6x smaller than rev2_jsonl and lossless for float32 "
                             "model outputs; rev2_jsonl is the old text format.")
    parser.add_argument("--detection-dump-compress", "--detection-dump-gzip",
                        dest="detection_dump_compress", action="store_true",
                        default=None,
                        help="compress the dump: gzip for rev2_jsonl, deflate "
                             "inside the container for v6_npz.1. The default is "
                             "the format's own (on for v6_npz.1, off for JSONL).")
    parser.add_argument("--no-detection-dump-compress", dest="detection_dump_compress",
                        action="store_false",
                        help="store the binary dump raw: ~40%% larger, ~2s faster "
                             "to commit per condition")
    parser.add_argument("--resume", action="store_true",
                        help="leave a condition alone if its fragment already "
                             "exists. Off by default: asking for a measurement "
                             "and getting a skip is how a check passes without "
                             "checking. Turn it on to continue a long run that "
                             "died partway, and say so in the config so the file "
                             "records what the run actually did.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Refuse BEFORE anything is loaded: the fragment and the detection dump are
    # named after this label, so a fine-tuned run under a zero-shot label would
    # overwrite the zero-shot result and be indistinguishable from it after.
    fragment_variant = args.fragment_variant or args.variant
    assert_ft_label(fragment_variant, args.v6_adapter)

    condition = resolve_condition(args.condition)
    logger.info("condition %s | %d terms | caption %r",
                condition.condition_id, len(condition.terms), condition.caption)

    out_path = (args.out_dir /
                f"{args.model}__{fragment_variant}__{condition.condition_id}.json")
    if out_path.exists() and args.resume and args.limit is None:
        logger.info("resume: %s already exists, leaving it", out_path)
        return

    _run(args, condition, fragment_variant, out_path)


def _run(args, condition, fragment_variant: str, out_path: Path) -> None:
    """The measured work, once this process owns the condition."""
    coco_gt, records = load_test_set(args.test_json)
    logger.info("test census: %d images", len(records))

    load_start = time.monotonic()
    detector = build_detector(args.model, args.variant, args.device, EVAL_PROTOCOL,
                              checkpoint_override=args.checkpoint_override,
                              cache_dir=args.cache_dir,
                              max_output_detections=args.max_output_detections,
                              v6_adapter_dir=args.v6_adapter,
                              config_override=args.config_override)
    logger.info("model loaded in %.1fs", time.monotonic() - load_start)
    logger.info("%s", describe(runner_fingerprint(args.model)))

    result = run_condition(
        condition, detector, coco_gt, records,
        text_cosines=cosines_for(condition, args.model),
        protocol=EVAL_PROTOCOL, limit=args.limit, cap_policy=args.cap_policy,
        prefetch=args.prefetch, prefetch_workers=args.prefetch_workers,
        prefetch_depth=args.prefetch_depth,
        dump_dir=args.detection_dump_dir if args.detection_dump else None,
        dump_pool=str(args.test_json), dump_compress=args.detection_dump_compress,
        dump_label=fragment_variant,
        dump_format=args.detection_dump_format,
    )

    if result.is_probe:
        record = {
            "model": args.model, "variant": fragment_variant,
            "condition_id": condition.condition_id,
            "n_terms": len(condition.terms), "n_images": result.n_images,
            "elapsed_seconds": result.elapsed_seconds,
            "images_per_second": result.images_per_second,
            "seconds_per_full_census": (
                EXPECTED_IMAGES["test"] / result.images_per_second
                if result.images_per_second > 0 else None),
            "readout": detector.postprocessing_notes(),
        }
        logger.info("PROBE %s", json.dumps(record))
        if args.probe_out:
            args.probe_out.mkdir(parents=True, exist_ok=True)
            target = (args.probe_out /
                      f"{args.model}__{fragment_variant}__{condition.condition_id}.json")
            target.write_text(json.dumps(record, indent=2))
            logger.info("wrote probe record %s", target)
        return

    # Which interpreter produced this fragment. A venv change silently changes
    # preprocessing -- transformers 4.57.3 pads the OWLv2 canvas with 0.5 (grey)
    # via the slow PIL processor while 5.14.1 pads with 0 (black) via
    # torchvision, so the same image fills ~44% of the input differently and
    # nothing in the log says so. Written here rather than inside the runner
    # because runner_version hashes the runner, and recording an environment
    # must not move the fingerprint of the code that produced the numbers.
    result.fragment.setdefault(FRAGMENT_KEY, environment_record())
    write_fragment(result.fragment, out_path, expected_images=EXPECTED_IMAGES["test"])
    logger.info("%s done in %.1f min (%.2f img/s)",
                condition.condition_id, result.elapsed_seconds / 60,
                result.images_per_second)


if __name__ == "__main__":
    main()
