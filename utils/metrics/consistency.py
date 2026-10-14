"""The M16 consistency gate: ``ΔAR ≈ PFR − NFR``.

Layer 1.5 counts individual ground-truth objects; layer 2 reports COCOeval's
AP/AR. The two are statements about the *same* detections, so they are related
by an identity:

    recall(q1) - recall(q0)  =  (|s1 = P| - |s0 = P|) / N  =  PFR - NFR

The identity holds *exactly* inside this package (:data:`INTERNAL_TOLERANCE`);
a residual there is a bug in the matching code, full stop. The version worth
running is the external one, against the ``ΔAR`` a fragment reports from
pycocotools. That comparison crosses two implementations, so it carries a
**pre-registered** tolerance (:data:`FRAGMENT_TOLERANCE`, fixed here before any
dump existed) and, when exceeded, is reported as a **defect, not a result**:
the suspects are the matching logic and the max-detections setting, in that
order, and no flip number should be published until one of them explains it.

Which AR key
------------
The dump-side recall is recall at a single IoU threshold (``iou_threshold``,
0.5 by default) with the per-category cap of 100. The fragment key that means
the same thing is **``AR50``**. ``AR`` and ``AR100`` are COCO's average over
the 0.50:0.95 sweep and are numerically much lower; comparing against them
would produce a large residual with no bug behind it. Picking a key by prefix
has already cost this project one reversed conclusion, so :func:`check` refuses
a mismatched key instead of trusting the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from utils.runner.logging_utils import get_logger
from utils.metrics.analysis import metric as fragment_metric
from utils.metrics.flip import FlipResult

logger = get_logger(__name__)

#: The internal identity is algebraic; anything above float noise is a bug.
INTERNAL_TOLERANCE = 1e-9
#: Pre-registered before the first dump landed. Absolute recall units.
FRAGMENT_TOLERANCE = 0.02

#: Fragment keys that are a recall at ONE IoU threshold, keyed by that value.
SINGLE_IOU_AR_KEYS: Mapping[float, str] = {0.5: "AR50", 0.75: "AR75"}
#: Fragment keys that average recall over the 0.50:0.95 sweep.
SWEPT_AR_KEYS: frozenset[str] = frozenset({"AR", "AR100"})

STATUS_OK = "ok"
STATUS_DEFECT = "defect"
STATUS_UNMEASURED = "unmeasured"


class MetricKeyMismatchError(ValueError):
    """The requested AR key does not describe the IoU the flip analysis used."""


@dataclass(frozen=True)
class ConsistencyCheck:
    """Result of the ``ΔAR ≈ PFR − NFR`` gate for one (class, q0, q1)."""

    target_class: str
    q0: str
    q1: str
    metric_key: str
    iou_threshold: float
    pfr_minus_nfr: float | None
    #: Recall difference recomputed from the dump. Must equal ``pfr_minus_nfr``.
    delta_recall_dump: float | None
    residual_internal: float | None
    #: ``AR(q1) - AR(q0)`` read from the two fragments.
    delta_ar_fragment: float | None
    residual_fragment: float | None
    tolerance: float
    status: str
    detail: str

    @property
    def is_defect(self) -> bool:
        return self.status == STATUS_DEFECT


def expected_ar_key(iou_threshold: float) -> str:
    """The fragment key whose definition matches a single-IoU dump recall."""
    try:
        return SINGLE_IOU_AR_KEYS[float(iou_threshold)]
    except KeyError as exc:
        raise MetricKeyMismatchError(
            f"no fragment AR key corresponds to IoU {iou_threshold}; the fragment "
            f"stores {sorted(SINGLE_IOU_AR_KEYS.values())} plus the swept "
            f"{sorted(SWEPT_AR_KEYS)}. Run the flip analysis at 0.5 or 0.75, or "
            "pass allow_mismatched_iou=True and say so in the report.") from exc


def check(flip: FlipResult,
          fragment_q0: Mapping | None = None,
          fragment_q1: Mapping | None = None,
          metric_key: str | None = None,
          tolerance: float = FRAGMENT_TOLERANCE,
          allow_mismatched_iou: bool = False) -> ConsistencyCheck:
    """Run both halves of the gate.

    Args:
        fragment_q0/fragment_q1: The §7 result fragments for the two
            conditions. Omit them to run the internal identity only — which is
            still worth running, because it is where a matching bug shows up.
        metric_key: Defaults to the key implied by the flip analysis's IoU
            threshold (:func:`expected_ar_key`).
        allow_mismatched_iou: Permit a swept key (``AR``/``AR100``) against a
            single-IoU recall. Only for a deliberate, annotated comparison.
    """
    key = metric_key or expected_ar_key(flip.iou_threshold)
    if key in SWEPT_AR_KEYS and not allow_mismatched_iou:
        raise MetricKeyMismatchError(
            f"{key!r} is COCO's recall averaged over IoU 0.50:0.95, but the flip "
            f"analysis ran at a single IoU {flip.iou_threshold}. The matching key is "
            f"{expected_ar_key(flip.iou_threshold)!r}. Comparing against {key!r} "
            "produces a large residual that has no bug behind it.")

    pfr, nfr = flip.positive_flip_rate, flip.negative_flip_rate
    difference = None if (pfr is None or nfr is None) else pfr - nfr
    residual_internal = (None if (difference is None or flip.delta_recall is None)
                         else flip.delta_recall - difference)

    if residual_internal is not None and abs(residual_internal) > INTERNAL_TOLERANCE:
        detail = (
            f"INTERNAL IDENTITY BROKEN: Δrecall(dump) - (PFR - NFR) = "
            f"{residual_internal:.3e}, which is algebraically impossible. This is a "
            "bug in the state assignment or the transition tally, not a finding.")
        logger.error("%s %s vs %s: %s", flip.target_class, flip.q0, flip.q1, detail)
        return ConsistencyCheck(
            flip.target_class, flip.q0, flip.q1, key, flip.iou_threshold, difference,
            flip.delta_recall, residual_internal, None, None, tolerance,
            STATUS_DEFECT, detail)

    if fragment_q0 is None or fragment_q1 is None:
        return ConsistencyCheck(
            flip.target_class, flip.q0, flip.q1, key, flip.iou_threshold, difference,
            flip.delta_recall, residual_internal, None, None, tolerance,
            STATUS_UNMEASURED,
            "internal identity holds; no fragments supplied, so ΔAR was not compared")

    ar0 = fragment_metric(fragment_q0, flip.target_class, key)
    ar1 = fragment_metric(fragment_q1, flip.target_class, key)
    delta_ar = None if (ar0 is None or ar1 is None) else ar1 - ar0
    if delta_ar is None or difference is None:
        return ConsistencyCheck(
            flip.target_class, flip.q0, flip.q1, key, flip.iou_threshold, difference,
            flip.delta_recall, residual_internal, delta_ar, None, tolerance,
            STATUS_UNMEASURED,
            f"{key} absent for {flip.target_class!r} in one of the fragments "
            "(the class was not queried there); the gate is unmeasured, not passed")

    residual = delta_ar - difference
    if abs(residual) > tolerance:
        detail = (
            f"Δ{key} = {delta_ar:+.4f} but PFR - NFR = {difference:+.4f}; residual "
            f"{residual:+.4f} exceeds the pre-registered tolerance {tolerance}. "
            "Report this as a DEFECT, not as a result. Suspect, in order: (1) the "
            "greedy matching rule or IoU threshold used for the object states, "
            "(2) the max-detections setting — the dump must be post-protocol, "
            "per-category maxDets applied, or the two layers are describing "
            "different detection sets.")
        logger.error("%s %s vs %s: %s", flip.target_class, flip.q0, flip.q1, detail)
        status = STATUS_DEFECT
    else:
        detail = (f"Δ{key} = {delta_ar:+.4f} vs PFR - NFR = {difference:+.4f}; "
                  f"residual {residual:+.4f} within {tolerance}")
        status = STATUS_OK
    return ConsistencyCheck(
        flip.target_class, flip.q0, flip.q1, key, flip.iou_threshold, difference,
        flip.delta_recall, residual_internal, delta_ar, residual, tolerance,
        status, detail)
