"""Layer 3 — false alarms, the absorption/stealing decomposition, and FROC.

* **M12** false alarms on frames with no ground truth of the class. The
  fragment-level version of this number is *saturated and carries no
  information*: under the frozen protocol (score threshold 0.0, per-category
  cap) every image emits exactly the cap, so the rate is a constant — 100.0/
  image for a cap of 100, 500.0 for GLIP/OWLv2's 500. The dump makes the
  threshold sweep possible, and :func:`false_alarm_curve` refuses to let the
  saturated point pass unlabelled: :func:`detect_saturation` marks it
  automatically.
* **M13** absorption vs stealing at the DETECTION level. The layer-2 version
  (``utils.metrics.analysis.axis_c_mechanism``) infers the mechanism from the sign of
  ΔAP and ΔAR; this one names the individual boxes that disappeared and asks
  whether they had been true or false positives. ``A >> S`` means the extra
  class is acting as a false-positive sink — a deployment prescription. A
  significant ``S`` means it is eating true detections — the opposite
  prescription. Reporting one number for both would recommend the wrong thing
  half the time.
* **M14** FROC: recall at fixed false-positives-per-image budgets, which is
  how a search-and-rescue operator actually reads a detector.
* **M11** union and intersection recall across several wordings. A union
  recall far above the best single wording means recall is recoverable by
  querying more than once; no gain with more false positives means the extra
  wordings are hallucinating. The prescriptions are opposite, so the two are
  never collapsed.

Every function takes detections that are already post-protocol, so a rate here
and an AP there describe the same set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from utils.runner.logging_utils import get_logger
from utils.data.empty_frames import DEFAULT_THRESHOLDS, FalseAlarmPoint, false_alarm_curve
from utils.metrics.dump_adapter import ConditionDump, GroundTruth, assert_comparable
from utils.metrics.matching import (
    IOU_MATCH,
    STATE_P,
    assign_gt_states,
    class_detections,
    iou_matrix,
    matched_pairs,
)

logger = get_logger(__name__)

#: False-positives-per-image budgets the FROC curve is read at.
FROC_BUDGETS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)

#: A per-image false-positive count within this fraction of the cap, with no
#: spread across images, is the saturated protocol artefact rather than a
#: measurement.
SATURATION_FRACTION = 0.999


@dataclass(frozen=True)
class Saturation:
    """Whether a false-alarm rate is the protocol artefact rather than a signal."""

    saturated: bool
    cap: int | None
    mean_per_image: float
    min_per_image: int
    max_per_image: int
    reason: str


def detect_saturation(counts_per_image: Sequence[int], cap: int | None) -> Saturation:
    """Flag a false-alarm rate that is pinned to the detection cap.

    Saturation is declared when the per-image count has no spread AND sits at
    the cap. Both conditions are required: a genuinely constant low rate is not
    saturated, and a high rate that varies is a real measurement.
    """
    counts = [int(c) for c in counts_per_image]
    if not counts:
        return Saturation(False, cap, 0.0, 0, 0, "no images in the denominator")
    lo, hi = min(counts), max(counts)
    mean = sum(counts) / len(counts)
    if cap is None:
        return Saturation(False, None, mean, lo, hi, "no cap supplied; not checked")
    if lo == hi and hi >= cap * SATURATION_FRACTION:
        reason = (f"every image returned exactly {hi} detections against a cap of "
                  f"{cap}: the rate is the cap, not a false-alarm measurement. Read "
                  "the threshold sweep instead.")
        logger.warning("saturated false-alarm rate — %s", reason)
        return Saturation(True, cap, mean, lo, hi, reason)
    return Saturation(False, cap, mean, lo, hi, "per-image counts vary below the cap")


@dataclass(frozen=True)
class FalseAlarmSweep:
    """M12 — the false-alarm curve plus its saturation verdict."""

    target_class: str
    condition_id: str
    n_images: int
    curve: tuple[FalseAlarmPoint, ...]
    saturation_at_zero: Saturation


def gt_absent_ids(gt: GroundTruth, image_ids: Sequence[int]) -> list[int]:
    """Images holding no non-crowd ground truth of the class. The denominator."""
    return [i for i in image_ids if not len(gt.get(i))]


def false_alarm_sweep(dump: ConditionDump, gt: GroundTruth, target_class: str,
                      cap: int | None = None,
                      thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> FalseAlarmSweep:
    """M12 for one class and one condition, over that class's own denominator.

    The curve itself is computed by ``utils.data.empty_frames.false_alarm_curve``
    — imported, not reimplemented, so this line's numbers and the false-alarm
    line's numbers are the same quantity.
    """
    index = dump.class_index(target_class)
    ids = gt_absent_ids(gt, dump.image_ids)
    if not ids:
        raise ValueError(f"{target_class}: no GT-absent frames; refusing to fabricate "
                         "a false-alarm rate")
    by_image = {
        i: [(target_class, float(s))
            for s in class_detections(dump.get(i), index, 0.0)[:, 4]]
        for i in ids
    }
    curve = false_alarm_curve(by_image, ids, [target_class], target_class, thresholds)
    counts = [len(by_image[i]) for i in ids]
    return FalseAlarmSweep(target_class=target_class, condition_id=dump.condition_id,
                           n_images=len(ids), curve=tuple(curve),
                           saturation_at_zero=detect_saturation(counts, cap))


def label_detections(detections: np.ndarray, gt_boxes: np.ndarray,
                     ignore_boxes: np.ndarray | None = None,
                     iou_threshold: float = IOU_MATCH) -> np.ndarray:
    """Per detection: ``1`` true positive, ``0`` false positive, ``-1`` ignored.

    Greedy by descending score against the ground truth first — COCO's own
    order — then, for whatever is left, against the ignore regions. A detection
    landing on an ``etc`` belonging is neither right nor wrong and must not
    inflate the false-alarm rate.
    """
    detections = np.asarray(detections, dtype=float).reshape(-1, 6)
    labels = np.zeros(len(detections), dtype=int)
    if not len(detections):
        return labels
    gt6 = np.zeros((len(np.asarray(gt_boxes).reshape(-1, 4)), 6))
    gt6[:, :4] = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    for ai, _ in matched_pairs(detections, gt6, iou_threshold):
        labels[ai] = 1
    if ignore_boxes is not None and len(np.asarray(ignore_boxes).reshape(-1, 4)):
        ig = np.asarray(ignore_boxes, dtype=float).reshape(-1, 4)
        rest = np.flatnonzero(labels == 0)
        if len(rest):
            ious = iou_matrix(detections[rest][:, :4], ig)
            labels[rest[ious.max(axis=1) >= iou_threshold]] = -1
    return labels


@dataclass(frozen=True)
class AbsorptionStealing:
    """M13 at the detection level, for one class between two conditions."""

    target_class: str
    q0: str
    q1: str
    iou_threshold: float
    score_threshold: float
    n_dets_q0: int
    n_dets_q1: int
    n_true_positive_q0: int
    n_false_positive_q0: int
    #: q0 detections with no counterpart in q1.
    absorbed_false_positives: int
    stolen_true_positives: int
    #: q1 detections with no counterpart in q0.
    gained_true_positives: int
    gained_false_positives: int
    verdict: str


def absorption_stealing(q0: ConditionDump, q1: ConditionDump, gt: GroundTruth,
                        target_class: str, iou_threshold: float = IOU_MATCH,
                        score_threshold: float = 0.0,
                        stealing_floor: int = 1) -> AbsorptionStealing:
    """Split the detections q1 removed into absorbed FPs and stolen TPs."""
    image_ids = assert_comparable(q0, q1)
    index = q0.class_index(target_class)
    absorbed = stolen = gained_tp = gained_fp = 0
    n_tp = n_fp = n_a = n_b = 0

    for image_id in image_ids:
        boxes, ignore = gt.get(image_id), gt.get_ignore(image_id)
        a = class_detections(q0.get(image_id), index, score_threshold)
        b = class_detections(q1.get(image_id), index, score_threshold)
        n_a += len(a)
        n_b += len(b)
        la = label_detections(a, boxes, ignore, iou_threshold)
        lb = label_detections(b, boxes, ignore, iou_threshold)
        n_tp += int((la == 1).sum())
        n_fp += int((la == 0).sum())
        pairs = matched_pairs(a, b, iou_threshold)
        kept_a = {ai for ai, _ in pairs}
        kept_b = {bj for _, bj in pairs}
        for ai in range(len(a)):
            if ai in kept_a:
                continue
            if la[ai] == 1:
                stolen += 1
            elif la[ai] == 0:
                absorbed += 1
        for bj in range(len(b)):
            if bj in kept_b:
                continue
            if lb[bj] == 1:
                gained_tp += 1
            elif lb[bj] == 0:
                gained_fp += 1

    if stolen < stealing_floor and absorbed > stolen:
        verdict = "absorption"
    elif stolen >= stealing_floor and absorbed > 10 * max(stolen, 1):
        verdict = "absorption_dominant"
    elif stolen >= stealing_floor:
        verdict = "stealing_present"
    else:
        verdict = "no_effect"
    return AbsorptionStealing(
        target_class=target_class, q0=q0.condition_id, q1=q1.condition_id,
        iou_threshold=iou_threshold, score_threshold=score_threshold,
        n_dets_q0=n_a, n_dets_q1=n_b,
        n_true_positive_q0=n_tp, n_false_positive_q0=n_fp,
        absorbed_false_positives=absorbed, stolen_true_positives=stolen,
        gained_true_positives=gained_tp, gained_false_positives=gained_fp,
        verdict=verdict,
    )


@dataclass(frozen=True)
class FrocCurve:
    """M14 — recall at fixed false-positives-per-image budgets."""

    target_class: str
    condition_id: str
    n_images: int
    n_ground_truth: int
    budgets: tuple[float, ...]
    #: budget -> recall. ``None`` only when there is no ground truth at all.
    recall_at_budget: Mapping[float, float | None]
    #: budget -> the score threshold that realises it, for reproduction.
    score_at_budget: Mapping[float, float | None]
    max_fppi: float


def froc(dump: ConditionDump, gt: GroundTruth, target_class: str,
         budgets: Sequence[float] = FROC_BUDGETS,
         iou_threshold: float = IOU_MATCH) -> FrocCurve:
    """Recall the model reaches while staying under each false-alarm budget.

    The operating points are the detection scores themselves, so no binning
    choice enters. Ignored (``iscrowd``) matches count as neither.
    """
    index = dump.class_index(target_class)
    scores: list[float] = []
    labels: list[int] = []
    n_gt = 0
    for image_id in dump.image_ids:
        boxes = gt.get(image_id)
        n_gt += len(boxes)
        dets = class_detections(dump.get(image_id), index, 0.0)
        if not len(dets):
            continue
        lab = label_detections(dets, boxes, gt.get_ignore(image_id), iou_threshold)
        scores.extend(dets[:, 4].tolist())
        labels.extend(lab.tolist())

    n_images = len(dump.image_ids)
    if n_gt == 0 or not scores:
        return FrocCurve(target_class, dump.condition_id, n_images, n_gt,
                         tuple(budgets), {b: None if n_gt == 0 else 0.0 for b in budgets},
                         {b: None for b in budgets}, 0.0)

    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    lab = np.asarray(labels, dtype=int)[order]
    srt = np.asarray(scores, dtype=float)[order]
    tp = np.cumsum(lab == 1)
    fp = np.cumsum(lab == 0)
    recall = tp / n_gt
    fppi = fp / max(n_images, 1)

    recall_at: dict[float, float | None] = {}
    score_at: dict[float, float | None] = {}
    for budget in budgets:
        allowed = np.flatnonzero(fppi <= budget)
        if not len(allowed):
            # Even the single highest-scoring detection exceeds the budget;
            # the achievable recall is 0 by definition, not unmeasured.
            recall_at[budget] = 0.0
            score_at[budget] = None
            continue
        best = allowed[int(np.argmax(recall[allowed]))]
        recall_at[budget] = float(recall[best])
        score_at[budget] = float(srt[best])
    return FrocCurve(target_class, dump.condition_id, n_images, n_gt, tuple(budgets),
                     recall_at, score_at, float(fppi[-1]))


@dataclass(frozen=True)
class UnionRecall:
    """M11 — recall of the union and the intersection of several wordings."""

    target_class: str
    condition_ids: tuple[str, ...]
    n_ground_truth: int
    per_condition: Mapping[str, float | None]
    best_single: float | None
    union: float | None
    intersection: float | None
    union_gain_over_best: float | None
    note: str


def union_intersection_recall(dumps: Sequence[ConditionDump], gt: GroundTruth,
                              target_class: str, iou_threshold: float = IOU_MATCH,
                              score_threshold: float = 0.0) -> UnionRecall:
    """How much recall several wordings jointly reach, and how much they share."""
    if len(dumps) < 2:
        raise ValueError("union recall needs at least two conditions")
    reference = dumps[0]
    for other in dumps[1:]:
        assert_comparable(reference, other)
    index = reference.class_index(target_class)

    n_gt = 0
    per_condition = {d.condition_id: 0 for d in dumps}
    n_union = n_intersection = 0
    for image_id in reference.image_ids:
        boxes = gt.get(image_id)
        if not len(boxes):
            continue
        n_gt += len(boxes)
        found = []
        for dump in dumps:
            states = assign_gt_states(dump.get(image_id), boxes, index,
                                      iou_threshold, score_threshold)
            hit = np.asarray([s == STATE_P for s in states.states])
            per_condition[dump.condition_id] += int(hit.sum())
            found.append(hit)
        stacked = np.vstack(found)
        n_union += int(stacked.any(axis=0).sum())
        n_intersection += int(stacked.all(axis=0).sum())

    if n_gt == 0:
        return UnionRecall(target_class, tuple(d.condition_id for d in dumps), 0,
                           {k: None for k in per_condition}, None, None, None, None,
                           "no ground truth of this class; unmeasured, not 0.0")
    rates = {k: v / n_gt for k, v in per_condition.items()}
    best = max(rates.values())
    union = n_union / n_gt
    return UnionRecall(
        target_class=target_class,
        condition_ids=tuple(d.condition_id for d in dumps), n_ground_truth=n_gt,
        per_condition=rates, best_single=best, union=union,
        intersection=n_intersection / n_gt, union_gain_over_best=union - best,
        note=("a union far above the best single wording means recall is "
              "recoverable by asking twice; no gain with more false positives "
              "means the extra wordings hallucinate"),
    )
