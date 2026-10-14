"""Score-priority greedy one-to-one box matching (standard COCO-style assignment).

Distinct from the IoU-priority greedy matcher used for the rank-free recall
metric in ``eval_visibility_grounding_generic.py`` (research tree) (which ignores
confidence entirely): here detections are matched in DESCENDING score order,
each consuming its best-IoU still-unused ground-truth box, because the
occlusion-ordering measurement needs both the matched GT's continuous
visibility ratio AND the matching detection's score (for the score-weighted
mean).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes, shape ``(len(a), len(b))``.

    Args:
        boxes_a: ``(n, 4)`` array of ``(x1, y1, x2, y2)`` boxes.
        boxes_b: ``(m, 4)`` array of ``(x1, y1, x2, y2)`` boxes.

    Returns:
        ``(n, m)`` IoU matrix; zero where the union area is non-positive.
    """
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


@dataclass(frozen=True)
class Match:
    """One matched (detection, ground-truth) pair."""

    det_index: int
    gt_index: int
    iou: float


def greedy_match_by_score(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> list[Match]:
    """Match detections to ground truth, descending-score-first, one GT per detection.

    Standard COCO-style assignment: detections are visited in decreasing
    confidence order; each is matched to its highest-IoU still-unmatched GT
    box if that IoU clears ``iou_thresh``, and that GT is then removed from
    the pool for subsequent (lower-score) detections.

    Args:
        det_boxes: ``(n, 4)`` array of detection ``xyxy`` boxes.
        det_scores: ``(n,)`` array of detection confidences.
        gt_boxes: ``(m, 4)`` array of candidate ground-truth ``xyxy`` boxes
            (callers must pre-filter out ignore/crowd boxes -- this function
            treats every row as a valid match target).
        iou_thresh: Minimum IoU for a valid match.

    Returns:
        A list of ``Match`` objects, at most ``min(n, m)`` long, in the
        score-descending order the detections were visited.
    """
    n_det = det_boxes.shape[0]
    n_gt = gt_boxes.shape[0]
    if n_det == 0 or n_gt == 0:
        return []

    iou = iou_matrix(det_boxes, gt_boxes)
    order = np.argsort(-det_scores, kind="mergesort")
    gt_available = np.ones(n_gt, dtype=bool)

    matches: list[Match] = []
    for det_idx in order:
        row = iou[det_idx].copy()
        row[~gt_available] = -1.0
        best_gt = int(np.argmax(row))
        best_iou = float(row[best_gt])
        if best_iou < iou_thresh:
            continue
        gt_available[best_gt] = False
        matches.append(Match(det_index=int(det_idx), gt_index=best_gt, iou=best_iou))
    return matches
