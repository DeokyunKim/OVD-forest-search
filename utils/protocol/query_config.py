"""Metrics for the three-query-configuration comparison (joint / person-only / belongings-only).

Query configurations (identical across the base study and every fine-tuned
arm analysed here):

    (1) joint full:        "clothes,hat,gloves,bag,shoes,person"
    (2) person only:       "person"
    (3) belongings only:   "clothes,hat,gloves,bag,shoes"

This module is pure CPU/numpy logic operating on already-loaded prediction
dumps (``dump_calibration_predictions.py`` / ``dump_negative_predictions.py``
output, both research tree) and ground truth (ForestPersons val
person-visibility attributes, ForestBelongings tier-3 negative-pool
manifest). Wiring (loading files, rendering the report/figure) lives in
``analyze_ftarms.py`` (research tree).

Every "hit" definition below uses the standard COCO-style score-priority
greedy match (``utils.metrics.box_matching.greedy_match_by_score``) so a GT box
cannot be double-counted by two overlapping predictions -- this matters for
recall tables and the paired McNemar test, where hit/miss identity must be
well-defined per GT box. "Coverage" checks (reverse-fire, c-cell person
overlap, reattribution) instead ask an existence question ("does *any*
qualifying box overlap this location enough") and deliberately do not
consume the covering box, since those questions are not about assigning one
detection per GT box.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import binomtest

from utils.metrics.box_matching import greedy_match_by_score, iou_matrix

UNCLASSIFIED_BAND = "unclassified"
BANDS = ["100", "70", "40", "20"]

# (image_id, gt_index) uniquely identifies one ground-truth box for the
# purposes of a paired hit/miss comparison across query configurations.
GtKey = tuple[int, int]
PredsByImage = dict[int, tuple[np.ndarray, np.ndarray]]
DumpBox = tuple[tuple[float, float, float, float], float, str | None]


def xywh_to_xyxy(box: list[float]) -> list[float]:
    """Convert a COCO-style ``[x, y, w, h]`` box to ``[x1, y1, x2, y2]``."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def band_of_visible_ratio(value: Any) -> str:
    """Map a ForestPersons annotation's ``attributes.visible_ratio`` to a band string.

    Returns ``UNCLASSIFIED_BAND`` for any value other than the four known
    bands (100/70/40/20), e.g. a missing attribute -- never ``None``, so the
    box remains a valid (if unaggregated) match target.
    """
    if value in (100, 70, 40, 20):
        return str(value)
    return UNCLASSIFIED_BAND


def load_person_gt_by_image(
    gt_json_path: str, image_ids: set[int]
) -> dict[int, tuple[np.ndarray, list[str]]]:
    """Person GT boxes + visibility band, restricted to ``image_ids``.

    Reads ``attributes.visible_ratio`` per annotation (ForestPersons
    ``val_person_only.json`` convention: a single "person" category, unlike
    ``val.json``'s pose-subclass category names).
    """
    gt = json.load(open(gt_json_path))
    boxes_acc: dict[int, list[list[float]]] = {}
    bands_acc: dict[int, list[str]] = {}
    for a in gt["annotations"]:
        if a.get("iscrowd", 0):
            continue
        iid = a["image_id"]
        if iid not in image_ids:
            continue
        band = band_of_visible_ratio(a.get("attributes", {}).get("visible_ratio"))
        boxes_acc.setdefault(iid, []).append(xywh_to_xyxy(a["bbox"]))
        bands_acc.setdefault(iid, []).append(band)
    return {
        iid: (
            np.array(boxes_acc.get(iid, []), dtype=float).reshape(-1, 4),
            bands_acc.get(iid, []),
        )
        for iid in image_ids
    }


def load_tier3_belongings_gt(manifest_path: str) -> dict[int, list[tuple[list[float], str]]]:
    """Per-image belongings GT ``(bbox_xyxy, category)`` list from the negative-pool manifest.

    Includes the "etc" ignore category as-is (callers filtering to the five
    query classes must exclude it explicitly); ``bbox`` is stored ``xywh`` in
    the manifest.
    """
    manifest = json.load(open(manifest_path))
    out: dict[int, list[tuple[list[float], str]]] = {}
    for entry in manifest["tier3"]:
        out[entry["image_id"]] = [
            (xywh_to_xyxy(b["bbox"]), b["cat"]) for b in entry.get("belongings", [])
        ]
    return out


def boxes_from_dump(dump: dict[str, Any]) -> dict[int, list[DumpBox]]:
    """Flatten a prediction dump into ``image_id -> [(bbox_xyxy, score, class_name), ...]``.

    ``class_name`` is ``None`` for single-phrase dumps that predate the
    per-box class-name field.
    """
    out: dict[int, list[DumpBox]] = {}
    for rec in dump["images"]:
        out[rec["image_id"]] = [
            (tuple(p["bbox_xyxy"]), float(p["score"]), p.get("class_name"))
            for p in rec["predictions"]
        ]
    return out


def filter_class(
    boxes_by_image: dict[int, list[DumpBox]],
    score_thresh: float,
    target_class: str | None = None,
) -> PredsByImage:
    """Filter ``boxes_from_dump`` output to one class + score threshold.

    Args:
        boxes_by_image: Output of ``boxes_from_dump``.
        score_thresh: Minimum score to keep.
        target_class: If given, keep only boxes whose ``class_name`` equals
            it (or whose ``class_name`` is ``None``, i.e. a single-phrase
            dump where every box implicitly belongs to the one query
            phrase). If ``None``, keep every box regardless of class.

    Returns:
        ``image_id -> (boxes_xyxy (n,4), scores (n,))``, in score-descending
        input order (not resorted -- callers needing score order call
        ``greedy_match_by_score``, which sorts internally).
    """
    out: PredsByImage = {}
    for iid, entries in boxes_by_image.items():
        kept_boxes: list[tuple[float, float, float, float]] = []
        kept_scores: list[float] = []
        for box, score, cls in entries:
            if score < score_thresh:
                continue
            if target_class is not None and cls is not None and cls != target_class:
                continue
            kept_boxes.append(box)
            kept_scores.append(score)
        out[iid] = (
            np.array(kept_boxes, dtype=float).reshape(-1, 4),
            np.array(kept_scores, dtype=float),
        )
    return out


def pooled_scores(preds_by_image: PredsByImage) -> np.ndarray:
    """Flatten every score across every image into one array (for FPPI)."""
    if not preds_by_image:
        return np.zeros((0,), dtype=float)
    return np.concatenate([s for _, s in preds_by_image.values()]) if preds_by_image else np.zeros((0,))


def fppi_table(preds_by_image: PredsByImage, n_images: int, thresholds: tuple[float, ...]) -> dict[float, float]:
    """FPPI at each threshold, for a pool with zero ground truth by construction."""
    from utils.metrics.fppi import fppi_at

    scores = pooled_scores(preds_by_image)
    return {t: fppi_at(scores, n_images, t) for t in thresholds}


def class_hits(
    gt_by_image: dict[int, list[list[float]]],
    preds_by_image: PredsByImage,
    iou_thresh: float = 0.5,
) -> dict[GtKey, bool]:
    """Score-priority greedy hit/miss per GT box, keyed ``(image_id, gt_index)``.

    Args:
        gt_by_image: ``image_id -> [gt_box_xyxy, ...]`` for one class.
        preds_by_image: ``image_id -> (pred_boxes_xyxy, pred_scores)``,
            already filtered to the matching class and score threshold.
        iou_thresh: Minimum IoU for a valid greedy match.

    Returns:
        ``(image_id, gt_index) -> True`` if a score-priority greedy match
        exists at ``iou_thresh``, else ``False``. Every GT box in
        ``gt_by_image`` gets an entry.
    """
    hits: dict[GtKey, bool] = {}
    empty_boxes = np.zeros((0, 4), dtype=float)
    empty_scores = np.zeros((0,), dtype=float)
    for iid, gt_boxes in gt_by_image.items():
        gt_arr = np.array(gt_boxes, dtype=float).reshape(-1, 4)
        if gt_arr.shape[0] == 0:
            continue
        pred_boxes, pred_scores = preds_by_image.get(iid, (empty_boxes, empty_scores))
        matches = (
            greedy_match_by_score(pred_boxes, pred_scores, gt_arr, iou_thresh=iou_thresh)
            if pred_boxes.shape[0] > 0
            else []
        )
        matched_gt = {m.gt_index for m in matches}
        for j in range(gt_arr.shape[0]):
            hits[(iid, j)] = j in matched_gt
    return hits


def recall_from_hits(hits: dict[GtKey, bool]) -> tuple[int, int, float]:
    """``(n_tp, n_gt, recall)`` from a ``class_hits`` dict."""
    n_gt = len(hits)
    n_tp = sum(1 for v in hits.values() if v)
    return n_tp, n_gt, (n_tp / n_gt if n_gt else float("nan"))


def coverage_hit(
    query_box: tuple[float, float, float, float],
    candidate_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> bool:
    """Whether *any* of ``candidate_boxes`` overlaps ``query_box`` at >= ``iou_thresh``.

    Existence check, not exclusive assignment -- a candidate box may cover
    more than one query box (used for reverse-fire, c-cell person-coverage,
    and reattribution, none of which are about one-to-one GT assignment).
    """
    if candidate_boxes.shape[0] == 0:
        return False
    q = np.array([query_box], dtype=float)
    return bool(np.any(iou_matrix(q, candidate_boxes)[0] >= iou_thresh))


def reverse_fire_rate(
    person_gt_by_image: dict[int, tuple[np.ndarray, list[str]]],
    belongings_preds_by_image: PredsByImage,
    bands: list[str] = BANDS,
) -> dict[str, tuple[int, int, float]]:
    """Per band: fraction of person GT boxes covered by a belongings-class box.

    Args:
        person_gt_by_image: ``image_id -> (gt_boxes_xyxy, gt_bands)`` (real
            persons, e.g. ForestPersons val).
        belongings_preds_by_image: Belongings-class predictions (any of the
            five classes pooled), already score-filtered by the caller.
        bands: Bands to aggregate.

    Returns:
        ``band -> (n_covered, n_gt, rate)``.
    """
    empty = np.zeros((0, 4), dtype=float)
    n_covered: dict[str, int] = {b: 0 for b in bands}
    n_gt: dict[str, int] = {b: 0 for b in bands}
    for iid, (gt_boxes, gt_bands) in person_gt_by_image.items():
        cand, _ = belongings_preds_by_image.get(iid, (empty, np.zeros((0,))))
        for j, band in enumerate(gt_bands):
            if band not in n_gt:
                continue
            n_gt[band] += 1
            if coverage_hit(tuple(gt_boxes[j]), cand):
                n_covered[band] += 1
    return {b: (n_covered[b], n_gt[b], n_covered[b] / n_gt[b] if n_gt[b] else float("nan")) for b in bands}


@dataclass(frozen=True)
class McNemarResult:
    """2x2 paired hit/miss table + exact McNemar test.

    Attributes:
        a: Both configurations hit.
        b: Config A hit, config B missed.
        c: Config A missed, config B hit.
        d: Both configurations missed.
        p_value: Exact two-sided binomial test p-value on the discordant
            pairs (``b``, ``c``), ``binomtest(min(b, c), b + c, 0.5)``.
    """

    a: int
    b: int
    c: int
    d: int
    p_value: float

    @property
    def recall_a(self) -> float:
        total = self.a + self.b + self.c + self.d
        return (self.a + self.b) / total if total else float("nan")

    @property
    def recall_b(self) -> float:
        total = self.a + self.b + self.c + self.d
        return (self.a + self.c) / total if total else float("nan")


def mcnemar_2x2(hits_a: dict[GtKey, bool], hits_b: dict[GtKey, bool]) -> McNemarResult:
    """Paired 2x2 hit/miss table + exact McNemar test over the shared key set.

    Args:
        hits_a: ``key -> hit`` for configuration A (e.g. joint-caption "1").
        hits_b: ``key -> hit`` for configuration B (e.g. belongings-only "3").

    Returns:
        A ``McNemarResult``.

    Raises:
        ValueError: If the two dicts do not share the same key set.
    """
    if set(hits_a) != set(hits_b):
        raise ValueError("hits_a and hits_b must cover the same GT-box key set")
    a = b = c = d = 0
    for key in hits_a:
        ha, hb = hits_a[key], hits_b[key]
        if ha and hb:
            a += 1
        elif ha and not hb:
            b += 1
        elif not ha and hb:
            c += 1
        else:
            d += 1
    p_value = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else float("nan")
    return McNemarResult(a=a, b=b, c=c, d=d, p_value=p_value)


@dataclass(frozen=True)
class ReattributionResult:
    """Fraction of one config's tier-3 person false positives explained by another's belongings boxes.

    Attributes:
        n_fp: Number of person-FP boxes considered (score >= the caller's
            person threshold).
        n_reattributed: How many of those have a belongings-class box
            (score >= the belongings threshold) at IoU >= ``iou_thresh``.
        rate: ``n_reattributed / n_fp``.
        n_gt_true: Of the reattributed FPs, how many also overlap a real
            belongings GT box at IoU >= ``iou_thresh`` (the reattributing
            belongings box is not itself a hallucination).
        gt_true_rate: ``n_gt_true / n_reattributed``.
    """

    n_fp: int
    n_reattributed: int
    rate: float
    n_gt_true: int
    gt_true_rate: float


def reattribution(
    person_fp_boxes_by_image: dict[int, list[tuple[float, float, float, float]]],
    belongings_preds_by_image: PredsByImage,
    belongings_gt_by_image: dict[int, list[tuple[list[float], str]]],
    iou_thresh: float = 0.5,
) -> ReattributionResult:
    """See ``ReattributionResult``. Every input box population is pre-filtered by the caller."""
    empty = np.zeros((0, 4), dtype=float)
    n_fp = 0
    n_reattributed = 0
    n_gt_true = 0
    for iid, fp_boxes in person_fp_boxes_by_image.items():
        cand, _ = belongings_preds_by_image.get(iid, (empty, np.zeros((0,))))
        gt_boxes = np.array([b for b, _ in belongings_gt_by_image.get(iid, [])], dtype=float).reshape(-1, 4)
        for fp_box in fp_boxes:
            n_fp += 1
            if coverage_hit(fp_box, cand):
                n_reattributed += 1
                if coverage_hit(fp_box, gt_boxes):
                    n_gt_true += 1
    return ReattributionResult(
        n_fp=n_fp,
        n_reattributed=n_reattributed,
        rate=(n_reattributed / n_fp if n_fp else float("nan")),
        n_gt_true=n_gt_true,
        gt_true_rate=(n_gt_true / n_reattributed if n_reattributed else float("nan")),
    )


def c_cell_person_coverage(
    c_cell_keys: list[GtKey],
    gt_by_image: dict[int, list[list[float]]],
    person_preds_by_image: PredsByImage,
) -> tuple[int, int, float]:
    """Of the c-cell GT boxes, how many are covered by a person-query box (any score >= caller's threshold)."""
    empty = np.zeros((0, 4), dtype=float)
    n = len(c_cell_keys)
    n_covered = 0
    for iid, j in c_cell_keys:
        cand, _ = person_preds_by_image.get(iid, (empty, np.zeros((0,))))
        gt_box = tuple(gt_by_image[iid][j])
        if coverage_hit(gt_box, cand):
            n_covered += 1
    return n_covered, n, (n_covered / n if n else float("nan"))


def best_match_score(
    gt_box: tuple[float, float, float, float],
    cand_boxes: np.ndarray,
    cand_scores: np.ndarray,
    iou_thresh: float = 0.5,
) -> float | None:
    """Highest-score candidate box overlapping ``gt_box`` at >= ``iou_thresh``, or ``None``.

    Used for the c-cell score-band histogram: among all raw (floor-only,
    unfiltered by any recall threshold) candidate boxes, find the best
    overlap and report its score, regardless of whether it clears any
    downstream detection threshold.
    """
    if cand_boxes.shape[0] == 0:
        return None
    q = np.array([gt_box], dtype=float)
    ious = iou_matrix(q, cand_boxes)[0]
    qualifying = np.where(ious >= iou_thresh)[0]
    if qualifying.size == 0:
        return None
    return float(cand_scores[qualifying][np.argmax(cand_scores[qualifying])])


def score_band_histogram(
    scores: list[float | None], bins: list[tuple[float, float]], floor: float = 0.05
) -> dict[str, int]:
    """Bin a list of (possibly ``None``) scores into labelled ranges.

    Args:
        scores: One entry per GT box; ``None`` means no candidate box
            overlapped it at all (not even below ``floor``).
        bins: Half-open ``[lo, hi)`` ranges to report, in order.
        floor: The dump's score floor (informational, used only for the
            "no_detection" label).

    Returns:
        ``"<lo,hi)" -> count`` plus a ``f"no_detection(<{floor})"`` bucket.
    """
    labels = [f"[{lo:.2f},{hi:.2f})" for lo, hi in bins]
    out = {label: 0 for label in labels}
    out[f"no_detection(<{floor:.2f})"] = 0
    for s in scores:
        if s is None:
            out[f"no_detection(<{floor:.2f})"] += 1
            continue
        placed = False
        for (lo, hi), label in zip(bins, labels):
            if lo <= s < hi:
                out[label] += 1
                placed = True
                break
        if not placed:
            out.setdefault("other", 0)
            out["other"] += 1
    return out
