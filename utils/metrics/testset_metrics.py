"""Held-out TEST-set metrics: ignore-aware AP50 (continuous interpolation) + P@R70.

Extends ``utils.protocol.query_config`` (val/tier3 recall & FPPI tables)
with the two things the held-out ForestPersons/ForestBelongings TEST-set
study needs that the val study never computed:

1. A genuine precision-recall AP -- the val/tier3 study only ever reported
   recall-at-a-threshold and FPPI-at-a-threshold, never area under a full
   PR curve. AP50 here uses the PASCAL VOC 2010+ "all points" (continuous)
   interpolation, i.e. the precision envelope is the running max from the
   high-recall end, not COCO's 101-point discrete sampling
   (``pycocotools.cocoeval``'s convention, used elsewhere in this repo by
   ``measure_ap50_recall_at_p70.py`` (research tree)) -- the two are NOT numerically
   interchangeable, hence "AP50 (continuous interpolation)" is spelled out
   explicitly wherever it is reported.
2. ``etc``-as-ignore-region handling (repo convention, see
   ``etc_patch.py`` in the research tree, and commit a935b16): a detection whose
   single best-overlapping ground-truth box is an ``etc`` box must be
   dropped entirely from the ranking (neither a true positive nor a false
   positive), and ``etc`` boxes never contribute to any class's recall
   denominator. This mirrors the COCO ``iscrowd`` convention already used
   elsewhere in this repo, extended to ForestBelongings' own dedicated
   ignore category.

Every function here is pure CPU/numpy logic over already-loaded prediction
dumps and ground truth -- no model loading, no I/O beyond what the caller
already parsed. Wiring lives in ``analyze_testset.py`` (research tree).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.metrics.box_matching import iou_matrix
from utils.protocol.query_config import xywh_to_xyxy

PredsByImage = dict[Any, tuple[np.ndarray, np.ndarray]]
GtByImage = dict[Any, np.ndarray]

# ForestBelongings v2.4's GT category names are singular (cloth/hat/glove/
# bag/shoe); every query phrase in this project's caption vocabulary is
# plural (clothes/hat/gloves/bag/shoes -- "hat" and "bag" happen to already
# be singular). This is the single source of truth for that mapping so it
# is defined once and unit-tested, rather than re-typed ad hoc per script.
QUERY_TO_GT_CLASS: dict[str, str] = {
    "clothes": "cloth",
    "hat": "hat",
    "gloves": "glove",
    "bag": "bag",
    "shoes": "shoe",
}
GT_TO_QUERY_CLASS: dict[str, str] = {v: k for k, v in QUERY_TO_GT_CLASS.items()}
FB_BELONGINGS_GT_CLASSES: tuple[str, ...] = tuple(QUERY_TO_GT_CLASS.values())
ETC_CLASS_NAME = "etc"
PERSON_CLASS_NAME = "person"

# ForestPersons(/IR) test.json category names encode pose+visibility as
# "<pose>_<visibility>" (e.g. "sitting_70"); this is the single source of
# truth for the pose prefix, shared by both the non-IR and IR test splits
# (verified: every FP-test annotation classifies cleanly; IR-test has 25
# stray "face"-category annotations with no pose prefix, bucketed as
# "unknown" rather than dropped).
POSE_CLASSES: tuple[str, ...] = ("standing", "sitting", "lying")
UNKNOWN_POSE = "unknown"


def map_query_to_gt_class(query_class: str) -> str:
    """Map a caption query phrase (plural) to its FB v2.4 GT category name (singular).

    Args:
        query_class: One of ``clothes``/``hat``/``gloves``/``bag``/``shoes``.

    Returns:
        The matching GT category name (``cloth``/``hat``/``glove``/``bag``/``shoe``).

    Raises:
        KeyError: If ``query_class`` is not one of the five known phrases.
    """
    return QUERY_TO_GT_CLASS[query_class]


def map_gt_to_query_class(gt_class: str) -> str:
    """Inverse of ``map_query_to_gt_class``: GT category name -> caption query phrase."""
    return GT_TO_QUERY_CLASS[gt_class]


@dataclass(frozen=True)
class FbTestGt:
    """Ground truth parsed from the standard-COCO FB v2.4 ``test.json``.

    Attributes:
        id2file: ``image_id -> file_name`` for every image in the split.
        all_image_ids: Every image id, in file order.
        person_by_image: ``image_id -> person GT boxes (n,4)``.
        class_by_image: ``gt_class_name -> {image_id -> boxes (n,4)}`` for
            each of ``FB_BELONGINGS_GT_CLASSES``, ``iscrowd=1`` instances of
            that SAME class excluded (COCO same-category ignore convention,
            already established in this dataset -- see
            the research tree's ``etc_patch.py`` docstring).
        ignore_by_image: ``gt_class_name -> {image_id -> ignore boxes (n,4)}``
            for each belongings class: the union of (a) ``etc``-category
            boxes (repo convention, commit a935b16 -- an out-of-vocabulary
            belonging, never scored as a false positive for ANY of the five
            core classes) and (b) that class's OWN ``iscrowd=1`` instances
            (same-category COCO ignore).
        etc_by_image: ``image_id -> etc GT boxes (n,4)`` on their own --
            the universal ignore set (applies to ANY query class, including
            ``person``, not just the five belongings classes).
    """

    id2file: dict[int, str]
    all_image_ids: list[int]
    person_by_image: dict[int, np.ndarray]
    class_by_image: dict[str, dict[int, np.ndarray]]
    ignore_by_image: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    etc_by_image: dict[int, np.ndarray] = field(default_factory=dict)


def load_fb_test_gt(gt_json_path: str) -> FbTestGt:
    """Parse FB v2.4 ``test.json`` into person / per-class / ignore GT dicts.

    Every one of the five core belongings classes gets its own ignore set
    (``etc`` boxes, universal across classes, union its own ``iscrowd=1``
    instances); ``etc`` boxes themselves never appear in any class's GT
    dict (they are not one of the five queryable classes).

    Args:
        gt_json_path: Path to a standard-COCO ``test.json`` with categories
            ``person``/``cloth``/``hat``/``glove``/``bag``/``shoe``/``etc``.

    Returns:
        An ``FbTestGt``.
    """
    gt = json.load(open(gt_json_path))
    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}
    id2file = {im["id"]: im["file_name"] for im in gt["images"]}
    all_ids = list(id2file)

    person_acc: dict[int, list[list[float]]] = {}
    class_acc: dict[str, dict[int, list[list[float]]]] = {c: {} for c in FB_BELONGINGS_GT_CLASSES}
    class_ignore_acc: dict[str, dict[int, list[list[float]]]] = {c: {} for c in FB_BELONGINGS_GT_CLASSES}
    etc_acc: dict[int, list[list[float]]] = {}

    for a in gt["annotations"]:
        name = cat_id_to_name.get(a["category_id"])
        iid = a["image_id"]
        box = xywh_to_xyxy(a["bbox"])
        is_crowd = bool(a.get("iscrowd", 0))
        if name == PERSON_CLASS_NAME:
            if is_crowd:
                continue
            person_acc.setdefault(iid, []).append(box)
        elif name == ETC_CLASS_NAME:
            # etc is always an ignore region regardless of its own iscrowd flag.
            etc_acc.setdefault(iid, []).append(box)
        elif name in class_acc:
            if is_crowd:
                class_ignore_acc[name].setdefault(iid, []).append(box)
            else:
                class_acc[name].setdefault(iid, []).append(box)

    def _to_arr(d: dict[int, list[list[float]]]) -> dict[int, np.ndarray]:
        return {iid: np.array(v, dtype=float).reshape(-1, 4) for iid, v in d.items()}

    ignore_by_image: dict[str, dict[int, np.ndarray]] = {}
    for cls in FB_BELONGINGS_GT_CLASSES:
        merged: dict[int, list[list[float]]] = {iid: list(boxes) for iid, boxes in etc_acc.items()}
        for iid, boxes in class_ignore_acc[cls].items():
            merged.setdefault(iid, []).extend(boxes)
        ignore_by_image[cls] = _to_arr(merged)

    return FbTestGt(
        id2file=id2file,
        all_image_ids=all_ids,
        person_by_image=_to_arr(person_acc),
        class_by_image={cls: _to_arr(class_acc[cls]) for cls in FB_BELONGINGS_GT_CLASSES},
        ignore_by_image=ignore_by_image,
        etc_by_image=_to_arr(etc_acc),
    )


def namespace_image_id(dataset_tag: str, image_id: Any) -> str:
    """Prefix an image id with a dataset tag so two datasets' integer ids never collide.

    Used to pool ForestPersons-test and ForestBelongings-test images into
    one combined image set (the "deployment AP" construction) without a
    numeric ``image_id`` from one dataset silently shadowing another's.
    """
    return f"{dataset_tag}:{image_id}"


def center_inside(box_xyxy: tuple[float, float, float, float],
                   container_xyxy: tuple[float, float, float, float]) -> bool:
    """Whether ``box_xyxy``'s center point falls inside ``container_xyxy``."""
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx1, cy1, cx2, cy2 = container_xyxy
    return (cx1 <= cx <= cx2) and (cy1 <= cy <= cy2)


def is_ignore_match(
    box_xyxy: tuple[float, float, float, float],
    ignore_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> bool:
    """Whether ``box_xyxy`` should be treated as landing on an ignore region.

    Repo convention (etc-as-ignore, commit a935b16): a box counts as an
    ignore match if it overlaps ANY ignore box at IoU >= ``iou_thresh``, OR
    its center falls inside any ignore box (catches the common case where a
    small detection sits fully inside a much larger ignore box, giving low
    IoU despite total containment).

    Args:
        box_xyxy: The candidate detection box.
        ignore_boxes: ``(k, 4)`` array of ignore-region boxes (e.g. FB v2.4
            ``etc`` GT boxes for this image); may be empty.
        iou_thresh: IoU threshold for the overlap branch of the check.

    Returns:
        ``True`` if the box matches an ignore region by either criterion.
    """
    if ignore_boxes.shape[0] == 0:
        return False
    q = np.array([box_xyxy], dtype=float)
    if np.any(iou_matrix(q, ignore_boxes)[0] >= iou_thresh):
        return True
    return any(center_inside(box_xyxy, tuple(ig)) for ig in ignore_boxes)


def filter_ignored(
    preds_by_image: PredsByImage,
    ignore_by_image: dict[Any, np.ndarray],
    iou_thresh: float = 0.5,
) -> PredsByImage:
    """Drop predictions whose best match is an ignore region (see ``is_ignore_match``).

    Used before FPPI counting: an ignore-region "detection" is neither a
    real false positive nor a hit, so it must not inflate FPPI.

    Args:
        preds_by_image: ``image_id -> (boxes (n,4), scores (n,))``.
        ignore_by_image: ``image_id -> ignore_boxes (k,4)``; images absent
            from this dict are treated as having zero ignore regions.
        iou_thresh: Forwarded to ``is_ignore_match``.

    Returns:
        A new dict (input is never mutated), same keys, each value filtered
        to non-ignored boxes only.
    """
    empty_ignore = np.zeros((0, 4), dtype=float)
    out: PredsByImage = {}
    for image_id, (boxes, scores) in preds_by_image.items():
        ignore_boxes = ignore_by_image.get(image_id, empty_ignore)
        if ignore_boxes.shape[0] == 0 or boxes.shape[0] == 0:
            out[image_id] = (boxes, scores)
            continue
        keep = np.array([not is_ignore_match(tuple(b), ignore_boxes, iou_thresh) for b in boxes])
        out[image_id] = (boxes[keep], scores[keep])
    return out


@dataclass(frozen=True)
class RankedDetections:
    """Global score-sorted TP/FP assignment, with ignore-region detections dropped.

    Attributes:
        scores: Detection scores, descending, ignore-matched detections removed.
        tp: Parallel boolean array -- ``True`` where the detection at that
            rank is a true positive.
        n_gt: Total ground-truth count across every image (recall denominator).
        n_ignored: Detections dropped for landing on an ignore region.
    """

    scores: np.ndarray
    tp: np.ndarray
    n_gt: int
    n_ignored: int


def rank_detections(
    preds_by_image: PredsByImage,
    gt_by_image: GtByImage,
    ignore_by_image: dict[Any, np.ndarray] | None = None,
    iou_thresh: float = 0.5,
) -> RankedDetections:
    """Score-priority greedy TP/FP assignment across an entire dataset, ignore-aware.

    Mirrors COCO-style crowd handling: detections are visited in descending
    score order (globally, not per image); each is matched to its
    highest-IoU still-unmatched real GT box in its own image if that IoU
    clears ``iou_thresh`` (true positive, consuming that GT box). Failing
    that, if it matches an ignore region (``is_ignore_match``), it is
    dropped entirely -- not a false positive, not counted at all. Anything
    left over is a false positive.

    Args:
        preds_by_image: ``image_id -> (boxes (n,4), scores (n,))``.
        gt_by_image: ``image_id -> gt_boxes (m,4)`` for the ONE class under
            evaluation; images with no entry are treated as having zero GT
            (every surviving detection there is a false positive).
        ignore_by_image: Optional ``image_id -> ignore_boxes (k,4)``, e.g.
            FB v2.4 ``etc`` boxes. ``None`` disables ignore handling.
        iou_thresh: True-positive / ignore-match IoU threshold.

    Returns:
        A ``RankedDetections`` ready for ``average_precision_continuous``.
    """
    empty_boxes = np.zeros((0, 4), dtype=float)
    empty_ignore = np.zeros((0, 4), dtype=float)

    flat: list[tuple[Any, tuple[float, float, float, float], float]] = []
    for image_id, (boxes, scores) in preds_by_image.items():
        for box, score in zip(boxes, scores):
            flat.append((image_id, tuple(box), float(score)))
    # Stable sort so tied scores keep their original (per-image) relative order.
    flat.sort(key=lambda t: -t[2])

    gt_matched: dict[Any, np.ndarray] = {
        iid: np.zeros(boxes.shape[0], dtype=bool) for iid, boxes in gt_by_image.items()
    }
    n_gt = int(sum(boxes.shape[0] for boxes in gt_by_image.values()))

    kept_scores: list[float] = []
    kept_tp: list[bool] = []
    n_ignored = 0

    for image_id, box, score in flat:
        gt_boxes = gt_by_image.get(image_id, empty_boxes)
        matched_here = False
        if gt_boxes.shape[0] > 0:
            matched = gt_matched[image_id]
            ious = iou_matrix(np.array([box]), gt_boxes)[0]
            ious = np.where(matched, -1.0, ious)
            best_j = int(np.argmax(ious))
            if ious[best_j] >= iou_thresh:
                matched[best_j] = True
                kept_scores.append(score)
                kept_tp.append(True)
                matched_here = True
        if matched_here:
            continue

        ignore_boxes = ignore_by_image.get(image_id, empty_ignore) if ignore_by_image else empty_ignore
        if is_ignore_match(box, ignore_boxes, iou_thresh):
            n_ignored += 1
            continue

        kept_scores.append(score)
        kept_tp.append(False)

    return RankedDetections(
        scores=np.array(kept_scores, dtype=float),
        tp=np.array(kept_tp, dtype=bool),
        n_gt=n_gt,
        n_ignored=n_ignored,
    )


@dataclass(frozen=True)
class ApResult:
    """Full precision-recall curve + scalar AP for one class/config.

    Attributes:
        ap: Area under the PR curve, PASCAL VOC "all points" (continuous)
            interpolation. ``nan`` if ``n_gt == 0``.
        n_gt: Ground-truth count (recall denominator).
        n_dets: Number of (non-ignored) ranked detections.
        n_ignored: Detections dropped as ignore-region matches.
        recall: Raw (non-interpolated), non-decreasing recall at each rank.
        precision: Raw precision at each rank, same length as ``recall``.
        scores: Score threshold at each rank, descending, same length.
    """

    ap: float
    n_gt: int
    n_dets: int
    n_ignored: int
    recall: np.ndarray
    precision: np.ndarray
    scores: np.ndarray


def _voc_all_points_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """PASCAL VOC 2010+ "all points" AP: area under the precision envelope.

    The precision envelope is the running max of precision from the
    high-recall end (monotonic non-increasing as recall decreases), then AP
    is the area under that envelope -- distinct from COCO's 101-point
    discrete-recall sampling.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision_continuous(ranked: RankedDetections) -> ApResult:
    """Compute AP50 (continuous/all-points interpolation) from a ``RankedDetections``.

    Args:
        ranked: Output of ``rank_detections`` (already score-sorted descending,
            ignore-matched detections already removed).

    Returns:
        An ``ApResult``. If ``ranked.n_gt == 0``, ``ap`` is ``nan`` (no
        recall axis is defined) but the raw arrays are still returned.
    """
    tp_cum = np.cumsum(ranked.tp.astype(np.int64))
    fp_cum = np.cumsum((~ranked.tp).astype(np.int64))
    denom = np.maximum(tp_cum + fp_cum, 1)
    precision = tp_cum / denom
    if ranked.n_gt > 0:
        recall = tp_cum / ranked.n_gt
        ap = _voc_all_points_ap(recall, precision)
    else:
        recall = np.zeros_like(tp_cum, dtype=float)
        ap = float("nan")
    return ApResult(
        ap=ap, n_gt=ranked.n_gt, n_dets=int(ranked.scores.size), n_ignored=ranked.n_ignored,
        recall=recall, precision=precision, scores=ranked.scores,
    )


@dataclass(frozen=True)
class PrecisionAtRecall:
    """Precision at the first rank where recall reaches a target, or "not reached"."""

    reached: bool
    target_recall: float
    precision: float = float("nan")
    recall: float = float("nan")
    score: float = float("nan")


def precision_at_recall(result: ApResult, target_recall: float = 0.70) -> PrecisionAtRecall:
    """Precision at the first point the raw recall curve reaches ``target_recall``.

    Unlike ``measure_ap50_recall_at_p70.py`` (research tree) (which searches for the
    recall at a target PRECISION), this is the reverse question the test-set
    brief asks for: precision at a target RECALL. Uses the raw (non-envelope)
    recall/precision arrays, so this is the actual achieved precision at that
    operating point, not an optimistic interpolated one.

    Args:
        result: Output of ``average_precision_continuous``.
        target_recall: The recall level to locate (e.g. 0.70).

    Returns:
        A ``PrecisionAtRecall`` with ``reached=False`` (and NaN fields) if
        the curve's max recall never reaches ``target_recall``.
    """
    idx = np.where(result.recall >= target_recall)[0]
    if idx.size == 0:
        return PrecisionAtRecall(reached=False, target_recall=target_recall)
    i = int(idx[0])
    return PrecisionAtRecall(
        reached=True, target_recall=target_recall,
        precision=float(result.precision[i]), recall=float(result.recall[i]),
        score=float(result.scores[i]),
    )


@dataclass(frozen=True)
class RecallAtPrecision:
    """Recall at the highest-recall operating point still satisfying a precision floor.

    Attributes:
        reached: Whether ANY operating point clears ``target_precision`` (if
            even the single highest-confidence detection's precision falls
            short, this is ``False`` and ``recall`` is ``0.0``, matching
            ``measure_ap50_recall_at_p70.py`` (research tree)'s convention -- an
            unattainable precision floor yields zero usable recall, not NaN).
        target_precision: The precision floor searched for (e.g. 0.70).
        recall: The achieved recall at that operating point.
        precision: The (enveloped) precision at that operating point --
            always ``>= target_precision`` when ``reached``.
        score: The score threshold at that operating point.
    """

    reached: bool
    target_precision: float
    recall: float = 0.0
    precision: float = float("nan")
    score: float = float("nan")


def recall_at_precision(result: ApResult, target_precision: float = 0.70) -> RecallAtPrecision:
    """R@P70 -- recall at the highest recall threshold where precision >= ``target_precision``.

    This is the mirror question to ``precision_at_recall`` (precision at a
    target RECALL): here the target is on precision, and the recall AT that
    precision floor is reported.

    Definition, made precise: the PASCAL VOC "all points" precision
    ENVELOPE (the running max of raw precision from the high-recall end --
    the exact same envelope ``average_precision_continuous`` integrates
    under for AP; see ``_voc_all_points_ap``) is non-increasing as recall
    increases, so ``{recall : envelope(recall) >= target_precision}`` is
    always a single contiguous prefix from recall 0 up to some boundary --
    R@P70 is the recall AT that boundary, i.e. the highest recall
    attainable while never letting the precision envelope dip below the
    target. This mirrors ``measure_ap50_recall_at_p70.py`` (research tree)'s
    COCOeval-based convention (``ok = where(pr >= thr); rec_at =
    rec_thrs[ok.max()]``) but computed on the continuous (non-101-point)
    raw recall/precision arrays this module already produces, using the
    SAME envelope construction as the AP integral itself (rather than raw,
    un-enveloped precision, which can fluctuate non-monotonically and would
    make "the highest threshold where precision >= target" ill-defined).

    Args:
        result: Output of ``average_precision_continuous``.
        target_precision: The precision floor to search for (e.g. 0.70).

    Returns:
        A ``RecallAtPrecision``.
    """
    if result.recall.size == 0:
        return RecallAtPrecision(reached=False, target_precision=target_precision)
    # Running max of precision from the high-recall (last) end -- identical
    # construction to _voc_all_points_ap's envelope, restricted to the raw
    # (non-padded) recall/precision arrays this ApResult already carries.
    envelope = np.maximum.accumulate(result.precision[::-1])[::-1]
    ok = np.where(envelope >= target_precision)[0]
    if ok.size == 0:
        return RecallAtPrecision(reached=False, target_precision=target_precision, recall=0.0)
    idx = int(ok.max())
    return RecallAtPrecision(
        reached=True, target_precision=target_precision,
        recall=float(result.recall[idx]), precision=float(envelope[idx]),
        score=float(result.scores[idx]),
    )


def pool_preds_and_gt(
    dataset_preds: dict[str, PredsByImage],
    dataset_gt: dict[str, GtByImage],
) -> tuple[PredsByImage, GtByImage]:
    """Merge several datasets' per-image dicts into one, namespacing image ids.

    Used for the pooled "deployment AP" construction: FP-test images
    (real GT throughout) and FB-test images (real GT only where a person is
    actually annotated; every other FB-test image contributes zero GT and
    so any prediction there is a pure false positive).

    Args:
        dataset_preds: ``dataset_tag -> preds_by_image`` for every dataset to pool.
        dataset_gt: ``dataset_tag -> gt_by_image`` for every dataset to pool
            (a dataset present in ``dataset_preds`` but absent here is
            treated as contributing zero GT for every one of its images).

    Returns:
        ``(pooled_preds_by_image, pooled_gt_by_image)`` with every original
        key replaced by ``namespace_image_id(tag, image_id)``.
    """
    pooled_preds: PredsByImage = {}
    pooled_gt: GtByImage = {}
    for tag, preds in dataset_preds.items():
        for image_id, value in preds.items():
            pooled_preds[namespace_image_id(tag, image_id)] = value
    for tag, gt in dataset_gt.items():
        for image_id, boxes in gt.items():
            pooled_gt[namespace_image_id(tag, image_id)] = boxes
    return pooled_preds, pooled_gt


def pose_of_category_name(category_name: str) -> str:
    """Extract the pose prefix from a ``"<pose>_<visibility>"`` category name.

    Args:
        category_name: A ForestPersons(/IR) ``test.json`` category name, e.g.
            ``"sitting_70"``. Anything not starting with one of
            ``POSE_CLASSES`` (e.g. IR's stray ``"face"`` category, or a
            hypothetical ``"unknown_0"``) maps to ``UNKNOWN_POSE`` rather
            than raising -- these are still real people, just not
            pose-classified.

    Returns:
        One of ``POSE_CLASSES`` or ``UNKNOWN_POSE``.
    """
    prefix = category_name.split("_", 1)[0]
    return prefix if prefix in POSE_CLASSES else UNKNOWN_POSE


@dataclass(frozen=True)
class PersonPoseGt:
    """Person GT split by pose, from a pose-subclass-category ``test.json``.

    Attributes:
        overall: ``image_id -> all person GT boxes (n,4)``, poses pooled
            (including ``unknown``-pose boxes) -- the same population as
            ``utils.protocol.query_config.load_person_gt_by_image``.
        by_pose: ``pose -> {image_id -> boxes (n,4)}`` for each of
            ``POSE_CLASSES`` only (``unknown`` is NOT a key here; see
            ``unknown``).
        unknown: ``image_id -> boxes (n,4)`` for annotations whose category
            name did not classify into a known pose (e.g. IR's ``face``
            category). Real people -- must still act as an ignore region
            when scoring any of the three known poses (see
            ``rank_detections_per_pose``), not silently dropped.
    """

    overall: GtByImage
    by_pose: dict[str, GtByImage]
    unknown: GtByImage


def load_person_gt_by_pose(gt_json_path: str) -> PersonPoseGt:
    """Parse a pose-subclass-category ``test.json`` (FP or FPIR) into pose-split GT.

    Unlike ``utils.protocol.query_config.load_person_gt_by_image`` (which
    needs a ``val_person_only.json``-style single-category file plus an
    ``attributes.visible_ratio`` field FP-test happens to also carry, but
    FPIR-test's ``test_person_only.json`` does NOT), this reads pose
    directly off the category name of the un-collapsed
    ``sitting_100``/``standing_70``/... category scheme, which both the
    ForestPersons and ForestPersonsIR ``test.json`` files share -- so one
    loader covers both datasets without depending on annotation attributes.

    Args:
        gt_json_path: Path to a COCO-style ``test.json`` whose categories are
            named ``"<pose>_<visibility>"`` (plus possibly a handful of
            non-conforming categories, e.g. IR's ``face``).

    Returns:
        A ``PersonPoseGt``. ``iscrowd=1`` annotations are excluded entirely
        (consistent with ``load_person_gt_by_image``); neither test split has
        any (verified: FP-test and FPIR-test are both 100% ``iscrowd=0``).
    """
    gt = json.load(open(gt_json_path))
    cat_id_to_pose = {c["id"]: pose_of_category_name(c["name"]) for c in gt["categories"]}

    overall_acc: dict[int, list[list[float]]] = {}
    pose_acc: dict[str, dict[int, list[list[float]]]] = {p: {} for p in POSE_CLASSES}
    unknown_acc: dict[int, list[list[float]]] = {}

    for a in gt["annotations"]:
        if a.get("iscrowd", 0):
            continue
        iid = a["image_id"]
        box = xywh_to_xyxy(a["bbox"])
        overall_acc.setdefault(iid, []).append(box)
        pose = cat_id_to_pose.get(a["category_id"], UNKNOWN_POSE)
        if pose in pose_acc:
            pose_acc[pose].setdefault(iid, []).append(box)
        else:
            unknown_acc.setdefault(iid, []).append(box)

    def _to_arr(d: dict[int, list[list[float]]]) -> dict[int, np.ndarray]:
        return {iid: np.array(v, dtype=float).reshape(-1, 4) for iid, v in d.items()}

    return PersonPoseGt(
        overall=_to_arr(overall_acc),
        by_pose={p: _to_arr(pose_acc[p]) for p in POSE_CLASSES},
        unknown=_to_arr(unknown_acc),
    )


def rank_detections_per_pose(
    preds_by_image: PredsByImage,
    pose_gt_by_image: dict[str, GtByImage],
    extra_ignore_by_image: dict[Any, np.ndarray] | None = None,
    iou_thresh: float = 0.5,
) -> dict[str, RankedDetections]:
    """Per-pose ``RankedDetections``, treating every OTHER pose's GT as an ignore region.

    A person detection that best-overlaps a GT box of a DIFFERENT pose is
    correctly finding a person, just not one of the pose under evaluation --
    scoring it as a false positive for that pose would conflate "the model
    can't see this pose" with "the model can't tell poses apart", which is a
    different (and, per the task brief, not the one being asked) question.
    Such detections are therefore dropped from the ranking entirely (neither
    TP nor FP), exactly like ``etc``-ignore in ``rank_detections``.

    Args:
        preds_by_image: The SAME (person-class, already score/floor-filtered)
            detections re-used for every pose -- one caller-supplied
            prediction set, ranked once per pose against that pose's own GT.
        pose_gt_by_image: ``pose -> gt_by_image``, one entry per pose to
            evaluate (typically ``PersonPoseGt.by_pose``, i.e. exactly
            ``POSE_CLASSES``).
        extra_ignore_by_image: Optional additional universal ignore region
            per image (e.g. ``PersonPoseGt.unknown`` -- real people whose
            pose is indeterminate, or FB v2.4's ``etc``), unioned into every
            pose's cross-pose ignore set.
        iou_thresh: Forwarded to ``rank_detections``.

    Returns:
        ``pose -> RankedDetections``, ready for ``average_precision_continuous``.
    """
    empty = np.zeros((0, 4), dtype=float)
    out: dict[str, RankedDetections] = {}
    for pose, gt in pose_gt_by_image.items():
        merged: dict[Any, list[np.ndarray]] = {}
        for other_pose, other_gt in pose_gt_by_image.items():
            if other_pose == pose:
                continue
            for iid, boxes in other_gt.items():
                if boxes.shape[0] > 0:
                    merged.setdefault(iid, []).append(boxes)
        if extra_ignore_by_image:
            for iid, boxes in extra_ignore_by_image.items():
                if boxes.shape[0] > 0:
                    merged.setdefault(iid, []).append(boxes)
        ignore_by_image = {iid: np.concatenate(parts, axis=0) if parts else empty
                            for iid, parts in merged.items()}
        out[pose] = rank_detections(preds_by_image, gt, ignore_by_image=ignore_by_image,
                                     iou_thresh=iou_thresh)
    return out


def pool_all_classes_gt(fb_gt: FbTestGt) -> tuple[GtByImage, dict[Any, np.ndarray]]:
    """Class-agnostic "object" GT: person + all 5 belongings classes pooled into one set.

    Used for the class-agnostic detection-rate question ("does the model
    find the objects at all, regardless of what it calls them") -- as
    opposed to ``analyze_fb_belongings``'s per-class tables, which ask
    whether the model names each object correctly. Detections (any class
    name, from a joint-query dump) are scored against this single pooled GT
    set; a detection landing on a belongings box counts as a hit for the
    "object" class even if its own ``class_name`` was e.g. ``"person"``.

    Args:
        fb_gt: Output of ``load_fb_test_gt``.

    Returns:
        ``(gt_by_image, ignore_by_image)``: ``gt_by_image`` unions
        ``fb_gt.person_by_image`` with every one of ``FB_BELONGINGS_GT_CLASSES``'
        ``class_by_image``; ``ignore_by_image`` is the union of every
        belongings class's own ``ignore_by_image`` entry (each of which is
        already ``etc`` boxes union that class's own ``iscrowd=1``
        instances, per ``load_fb_test_gt``'s docstring -- so this union is
        exactly "etc boxes plus every belongings class's crowd instances",
        with no separate access needed to the per-class crowd sets); the same
        ``etc`` box appears in up to five of those per-class sets and is
        de-duplicated back down to one row per image.
        NOTE: any ``iscrowd=1`` PERSON annotations are not represented here
        at all (``load_fb_test_gt`` drops them outright, not exposing them
        as GT or ignore for any class -- inherited limitation, not
        introduced by this function).
    """
    merged_gt: dict[Any, list[np.ndarray]] = {}
    for iid, boxes in fb_gt.person_by_image.items():
        if boxes.shape[0] > 0:
            merged_gt.setdefault(iid, []).append(boxes)
    for cls in FB_BELONGINGS_GT_CLASSES:
        for iid, boxes in fb_gt.class_by_image[cls].items():
            if boxes.shape[0] > 0:
                merged_gt.setdefault(iid, []).append(boxes)
    gt_by_image = {iid: np.concatenate(parts, axis=0) for iid, parts in merged_gt.items()}

    merged_ignore: dict[Any, list[np.ndarray]] = {}
    for cls in FB_BELONGINGS_GT_CLASSES:
        for iid, boxes in fb_gt.ignore_by_image.get(cls, {}).items():
            if boxes.shape[0] > 0:
                merged_ignore.setdefault(iid, []).append(boxes)
    ignore_by_image = {
        iid: np.unique(np.concatenate(parts, axis=0), axis=0) for iid, parts in merged_ignore.items()
    }

    return gt_by_image, ignore_by_image
