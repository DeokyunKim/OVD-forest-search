"""Matching wrappers over ``utils.metrics.invariance``.

Nothing here redefines a matching rule. ``iou_matrix``, ``greedy_match`` and
``class_detections`` are imported from the retired-but-verified rev2 module and
re-exported, so a number produced by this line and a number produced by that
one cannot differ for definitional reasons. The additions are strictly
mechanical:

* :func:`matched_pairs` — the same greedy rule at an arbitrary IoU threshold.
  At the default it *delegates* to ``greedy_match`` rather than duplicating it;
  the generalised path is pinned to the delegated one by a randomised
  equivalence test (the regression tests).
* :func:`matched_ious` — ``greedy_match`` discards the IoU it matched on, and
  M3 (location jitter) needs it. This is the IoU-preserving wrapper.
* :func:`assign_gt_states` — the three-valued P / O / M assignment that layer
  1.5 is built on.
* :func:`pairwise_pi_f1` — PI-F1 for exactly two conditions at an explicit
  score threshold. rev2's ``_one_order_contrast`` hard-codes ``THRESHOLDS[0]``;
  a test asserts this function reproduces it bit-for-bit at that threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from utils.metrics.invariance import (
    IOU_MATCH,
    THRESHOLDS,
    class_detections,
    greedy_match,
    iou_matrix,
)

__all__ = [
    "IOU_MATCH", "THRESHOLDS", "class_detections", "greedy_match", "iou_matrix",
    "matched_pairs", "matched_ious", "assign_gt_states", "pairwise_pi_f1",
    "GtStates", "STATE_P", "STATE_O", "STATE_M", "STATES",
]

#: Detected as the target class (IoU >= tau).
STATE_P = "P"
#: Detected, but claimed by some OTHER queried class. Distinct from a miss:
#: the object was found and then re-labelled, which has a different mechanism
#: and a different deployment remedy than not finding it at all.
STATE_O = "O"
#: Not detected by any queried class.
STATE_M = "M"
STATES: tuple[str, ...] = (STATE_P, STATE_O, STATE_M)


def _as_scored(boxes: np.ndarray) -> np.ndarray:
    """Pad an ``(n, 4)`` box array to the ``(n, 6)`` form the rev2 helpers take.

    Ground truth carries no score; the padded column is never read, because
    ``greedy_match`` orders by the score of its FIRST argument only and ground
    truth is always passed second.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = np.zeros((len(boxes), 6), dtype=float)
    out[:, :4] = boxes
    return out


def matched_pairs(a: np.ndarray, b: np.ndarray,
                  iou_threshold: float = IOU_MATCH) -> list[tuple[int, int]]:
    """Greedy 1:1 matching by descending score of ``a``, at ``iou_threshold``.

    At :data:`IOU_MATCH` this is ``greedy_match`` itself — called, not copied.
    Any other threshold runs the identical algorithm with the comparison value
    substituted; the regression tests asserts the two agree on
    randomised inputs at 0.5, which is what licenses the generalisation.
    """
    if iou_threshold == IOU_MATCH:
        return greedy_match(a, b)
    if len(a) == 0 or len(b) == 0:
        return []
    ious = iou_matrix(a[:, :4], b[:, :4])
    taken: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for ai in np.argsort(-a[:, 4]):
        row = ious[ai].copy()
        for bj in taken:
            row[bj] = -1.0
        bj = int(np.argmax(row))
        if row[bj] >= iou_threshold:
            taken.add(bj)
            pairs.append((int(ai), bj))
    return pairs


def matched_ious(a: np.ndarray, b: np.ndarray,
                 iou_threshold: float = IOU_MATCH) -> list[tuple[int, int, float]]:
    """:func:`matched_pairs` with the matched IoU retained.

    M3 (location jitter) is ``1 - mean(IoU)`` over matched pairs and M1 (PI-F1)
    is a count of matched pairs. They must be reported separately: "the same
    object, a little displaced" and "a different object" are different
    failures, and a single number cannot say which occurred.
    """
    pairs = matched_pairs(a, b, iou_threshold)
    if not pairs:
        return []
    ious = iou_matrix(np.asarray(a)[:, :4], np.asarray(b)[:, :4])
    return [(ai, bj, float(ious[ai, bj])) for ai, bj in pairs]


@dataclass(frozen=True)
class GtStates:
    """P / O / M for every ground-truth object of one image, one condition.

    Attributes:
        states: One of :data:`STATES` per ground-truth index.
        det_box: ``(n_gt, 4)`` — the matched target-class box, ``nan`` unless
            the state is ``P``. Layer 1.5's box-preservation term (M18) reads
            this for objects that are ``P`` in both conditions.
        det_score: ``(n_gt,)`` — the matched target-class score, ``nan``
            unless ``P``.
        other_class: The class index that claimed the object, or ``None``
            unless the state is ``O``. This is what makes "stolen" nameable.
    """

    states: tuple[str, ...]
    det_box: np.ndarray
    det_score: np.ndarray
    other_class: tuple[int | None, ...]

    def __len__(self) -> int:
        return len(self.states)


def assign_gt_states(detections: np.ndarray, gt_boxes: np.ndarray, target_index: int,
                     iou_threshold: float = IOU_MATCH,
                     score_threshold: float = 0.0) -> GtStates:
    """Three-valued state of every ground-truth object under one condition.

    The target class is resolved first and keeps whatever it matches, so an
    object found by both ``person`` and ``clothes`` is ``P`` — it *was*
    detected as a person, and calling it stolen would overstate the effect.
    Only the objects the target class did not claim are offered to the other
    queried classes.

    Args:
        detections: ``(n, 6)`` rows for this image, all classes.
        gt_boxes: ``(m, 4)`` xyxy ground truth of the target class.
        score_threshold: Applied through rev2's ``class_detections``. The
            frozen protocol dumps at 0.0; a higher value here is an operating
            point, and it is recorded in every result that uses it.
    """
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    n_gt = len(gt_boxes)
    states = [STATE_M] * n_gt
    det_box = np.full((n_gt, 4), np.nan)
    det_score = np.full(n_gt, np.nan)
    other_class: list[int | None] = [None] * n_gt
    if n_gt == 0:
        return GtStates(tuple(states), det_box, det_score, tuple(other_class))

    detections = np.asarray(detections, dtype=float).reshape(-1, 6)
    gt6 = _as_scored(gt_boxes)

    target = class_detections(detections, target_index, score_threshold)
    for ai, gj in matched_pairs(target, gt6, iou_threshold):
        states[gj] = STATE_P
        det_box[gj] = target[ai, :4]
        det_score[gj] = target[ai, 4]

    remaining = [j for j in range(n_gt) if states[j] == STATE_M]
    if remaining and len(detections):
        others = detections[(detections[:, 5] != target_index)
                            & (detections[:, 4] >= score_threshold)].reshape(-1, 6)
        if len(others):
            sub = gt6[remaining]
            for ai, sj in matched_pairs(others, sub, iou_threshold):
                gj = remaining[sj]
                states[gj] = STATE_O
                other_class[gj] = int(others[ai, 5])
    return GtStates(tuple(states), det_box, det_score, tuple(other_class))


def pairwise_pi_f1(a: np.ndarray, b: np.ndarray, class_index: int,
                   score_threshold: float = THRESHOLDS[0],
                   iou_threshold: float = IOU_MATCH) -> float:
    """PI-F1 of one class between two conditions, for ONE image.

    ``2 |match| / (|A| + |B|)``, with rev2's two conventions kept verbatim:
    both-empty scores 1.0 (the two prompts agreed there is nothing) and
    one-empty scores 0.0. The both-empty convention is what makes a silent
    model look invariant, so callers count those images separately.
    """
    da = class_detections(np.asarray(a, dtype=float).reshape(-1, 6), class_index,
                          score_threshold)
    db = class_detections(np.asarray(b, dtype=float).reshape(-1, 6), class_index,
                          score_threshold)
    if len(da) == 0 and len(db) == 0:
        return 1.0
    if len(da) == 0 or len(db) == 0:
        return 0.0
    return 2 * len(matched_pairs(da, db, iou_threshold)) / (len(da) + len(db))


def both_empty(a: np.ndarray, b: np.ndarray, class_index: int,
               score_threshold: float = THRESHOLDS[0]) -> bool:
    """True when neither condition produced a detection of this class."""
    da = class_detections(np.asarray(a, dtype=float).reshape(-1, 6), class_index,
                          score_threshold)
    db = class_detections(np.asarray(b, dtype=float).reshape(-1, 6), class_index,
                          score_threshold)
    return len(da) == 0 and len(db) == 0


def concat_scores(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate score columns, tolerating an all-empty input."""
    parts = [np.asarray(r, dtype=float).reshape(-1, 6)[:, 4] for r in rows]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.zeros(0)
