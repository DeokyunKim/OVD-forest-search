#!/usr/bin/env python
"""Measure what one query-vocabulary change does, per object, to one class.

    $PY analyze_v6_query_sensitivity.py \\
        --dump-q0 outputs/v6/detection_dumps/<arm>__A1_singleton_person.jsonl \\
        --dump-q1 outputs/v6/detection_dumps/<arm>__A2_pair_person_clothes.jsonl \\
        --test-json datasets/v6_fp_fb/test.json \\
        --fragment-q0 outputs/v6/fragments/<...>__A1_singleton_person.json \\
        --fragment-q1 outputs/v6/fragments/<...>__A2_pair_person_clothes.json \\
        --permutation-dump outputs/v6/detection_dumps/<arm>__A2_pair_clothes_person.jsonl \\
        --out outputs/v6/analysis/qsens__<arm>__person__A1_vs_A2.json

CPU only; it reads dumps and fragments and never loads a model.

Layers 1 and 2 are always emitted together. If the ``consistency`` block comes
back ``defect``, the layer-1.5 numbers in the same file are NOT results: the
detection set the flip analysis saw and the detection set COCOeval scored have
diverged, and the matching rule or the max-detections setting has to explain it
first.

Without ``--permutation-dump`` the noise floor is unknown and ``s_norm`` is
emitted as ``_unmeasured_``. It is never estimated: an invented floor makes
every normalised number look decisive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The repository root, found by walking up to the directory that holds
# `pyproject.toml`. Written this way rather than as a fixed
# number of parents because this file has moved once already, and a
# hard-coded depth is only correct until it moves again.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(next(p for p in _HERE.parents
                            if (p / 'pyproject.toml').is_file())))

from utils.runner.logging_utils import get_logger  # noqa: E402
from utils.protocol.condition_registry import resolve_condition  # noqa: E402
from utils.protocol.conditions import get_condition  # noqa: E402
from utils.metrics import consistency  # noqa: E402
from utils.metrics.dump_adapter import (  # noqa: E402
    fragment_lossless_guard,
    load_dump,
    load_ground_truth,
)
from utils.metrics.flip import flip_analysis  # noqa: E402
from utils.metrics.layer2 import accuracy_gap, delta_accuracy  # noqa: E402
from utils.metrics.layer3 import (  # noqa: E402
    absorption_stealing,
    false_alarm_sweep,
    froc,
    union_intersection_recall,
)
from utils.metrics.matching import IOU_MATCH, THRESHOLDS  # noqa: E402
from utils.metrics.report import write_record  # noqa: E402
from utils.metrics.stability import (  # noqa: E402
    noise_floor_normalised,
    stability_pair,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-q0", type=Path, required=True,
                        help="reference condition, e.g. `person` alone")
    parser.add_argument("--dump-q1", type=Path, required=True,
                        help="comparison condition, e.g. `person . clothes`")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--target-class", default="person")
    parser.add_argument("--fragment-q0", type=Path)
    parser.add_argument("--fragment-q1", type=Path)
    parser.add_argument("--permutation-dump", type=Path,
                        help="the order-only control that defines the noise floor")
    parser.add_argument("--iou-threshold", type=float, default=IOU_MATCH)
    parser.add_argument("--score-threshold", type=float, default=THRESHOLDS[0],
                        help="operating point for the layer-1 metrics; layer 1.5 "
                             "uses --flip-score-threshold")
    parser.add_argument("--flip-score-threshold", type=float, default=0.0,
                        help="0.0 keeps the frozen protocol's full ranked list")
    parser.add_argument("--max-dets-per-category", type=int, default=100,
                        help="only used to detect a saturated false-alarm rate")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def permuted_partner(control, candidates):
    """The condition ``control`` is an order-only permutation OF.

    The noise floor is a contrast in which **only the word order changes**, so
    the control must be paired with the condition holding the same term
    multiset — for ``A2_pair_clothes_person`` that is ``A2_pair_person_clothes``,
    NOT the ``person`` singleton. Pairing it with the singleton would put a
    vocabulary-size change inside the "noise floor" and make the floor an
    upper bound on itself, which silently deflates every ``S_norm``.

    Term lists come from the condition registries, the single definition site,
    so this cannot drift from the captions that were actually evaluated.
    Resolution goes through :func:`utils.protocol.condition_registry.resolve_condition`
    rather than one registry: the A5 and A6 axes live in their own modules
    (``conditions.py`` is inside the ``runner_version`` fingerprint), and a
    single-registry lookup here raised ``KeyError: 'A6_pair_bag_clothes'`` on
    64 of 320 runs — losing exactly the permutation controls that the other
    A6 results are read against.
    """
    control_terms = frozenset(resolve_condition(control.condition_id).terms)
    matches = [c for c in candidates
               if frozenset(resolve_condition(c.condition_id).terms) == control_terms]
    if len(matches) != 1:
        raise ValueError(
            f"{control.condition_id!r} has terms {sorted(control_terms)}, which match "
            f"{[c.condition_id for c in matches]} among the analysed conditions "
            f"{[c.condition_id for c in candidates]}. A permutation floor needs exactly "
            "one partner with the SAME term multiset; supply the permutation of a "
            "condition being analysed, or omit --permutation-dump and report S_norm "
            "as _unmeasured_.")
    return matches[0]


def _sweep_or_unmeasured(dump, gt, target_class: str, cap: int):
    """M12 for one condition, or a named reason why it is unmeasurable.

    A class with no GT-absent frame in this pool has no false-alarm
    denominator. ``false_alarm_sweep`` refuses to invent one; the driver
    records the refusal instead of dying, so the other layers still land.
    """
    try:
        return false_alarm_sweep(dump, gt, target_class, cap)
    except ValueError as exc:
        logger.warning("false-alarm sweep unmeasurable for %s: %s", target_class, exc)
        return {"unmeasured_reason": str(exc)}


def main() -> int:
    args = parse_args()
    q0 = load_dump(args.dump_q0)
    q1 = load_dump(args.dump_q1)
    gt = load_ground_truth(args.test_json, args.target_class, q0.image_ids)
    logger.info("%s: %s vs %s over %d images, %d ground-truth %s",
                q0.arm, q0.condition_id, q1.condition_id, len(q0.detections),
                gt.n_objects, args.target_class)

    flip = flip_analysis(q0, q1, gt, args.target_class,
                         args.iou_threshold, args.flip_score_threshold)
    stability = stability_pair(q0, q1, args.target_class,
                               args.score_threshold, args.iou_threshold)

    floor = None
    control = None
    floor_partner = None
    if args.permutation_dump:
        control = load_dump(args.permutation_dump)
        floor_partner = permuted_partner(control, (q0, q1))
        floor = stability_pair(floor_partner, control, args.target_class,
                               args.score_threshold, args.iou_threshold)
        logger.info("noise floor from the order-only contrast %s vs %s",
                    floor_partner.condition_id, control.condition_id)

    fragments = {}
    for name, path in (("q0", args.fragment_q0), ("q1", args.fragment_q1)):
        fragments[name] = json.loads(path.read_text()) if path else None

    # The float32 guard is recorded on BOTH sides on purpose: an npz-only copy
    # is one nobody reads while building a table. The dump side was already
    # checked at load; the fragment side is checked here.
    guards = {name: fragment_lossless_guard(frag)
              for name, frag in fragments.items() if frag}
    for name, guard in guards.items():
        guard.assert_ok(f"fragment {name}")

    gate = consistency.check(flip, fragments["q0"], fragments["q1"])
    if gate.is_defect:
        logger.error("CONSISTENCY DEFECT — the layer-1.5 numbers in %s are not "
                     "results until this is explained: %s", args.out, gate.detail)

    record = {
        "schema_version": "v6-qsens.1",
        "arm": q0.arm,
        "pool": q0.pool,
        "target_class": args.target_class,
        "q0": q0.condition_id,
        "q1": q1.condition_id,
        "dump_format": {"q0": args.dump_q0.name, "q1": args.dump_q1.name},
        "lossless_guard": guards or None,
        "iou_threshold": args.iou_threshold,
        "score_threshold_layer1": args.score_threshold,
        "score_threshold_layer15": args.flip_score_threshold,
        "layer1_stability": stability,
        "noise_floor": {
            "permutation_condition": (control.condition_id if floor else None),
            "permutation_partner": (floor_partner.condition_id if floor else None),
            "pi_f1_floor": (floor.pi_f1 if floor else None),
            "s_norm": noise_floor_normalised(stability.pi_f1,
                                             floor.pi_f1 if floor else None),
            "rule": ("S_norm at or below 0 means the contrast is indistinguishable "
                     "from sequence noise and is NOT reported as a composition effect"),
        },
        "layer15_flip": flip,
        "consistency": gate,
        "layer2": {
            "delta_accuracy": (delta_accuracy(fragments["q0"], fragments["q1"],
                                              args.target_class)
                               if fragments["q0"] and fragments["q1"] else None),
            "ar_ap_gap_q0": (accuracy_gap(fragments["q0"], args.target_class)
                             if fragments["q0"] else None),
            "ar_ap_gap_q1": (accuracy_gap(fragments["q1"], args.target_class)
                             if fragments["q1"] else None),
        },
        "layer3": {
            "absorption_stealing": absorption_stealing(
                q0, q1, gt, args.target_class, args.iou_threshold,
                args.flip_score_threshold),
            "false_alarms_q0": _sweep_or_unmeasured(q0, gt, args.target_class,
                                                    args.max_dets_per_category),
            "false_alarms_q1": _sweep_or_unmeasured(q1, gt, args.target_class,
                                                    args.max_dets_per_category),
            "froc_q0": froc(q0, gt, args.target_class, iou_threshold=args.iou_threshold),
            "froc_q1": froc(q1, gt, args.target_class, iou_threshold=args.iou_threshold),
            "union_recall": union_intersection_recall(
                [q0, q1], gt, args.target_class, args.iou_threshold,
                args.flip_score_threshold),
        },
    }
    write_record(record, args.out)
    logger.info("NFR=%s PFR=%s SR=%s | Δrecall=%s | consistency=%s",
                flip.negative_flip_rate, flip.positive_flip_rate, flip.stealing_rate,
                flip.delta_recall, gate.status)
    return 1 if gate.is_defect else 0


if __name__ == "__main__":
    raise SystemExit(main())
