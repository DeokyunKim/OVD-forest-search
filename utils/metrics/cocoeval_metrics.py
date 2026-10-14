"""pycocotools COCOeval wrapper: the 7-metric block for A/B-family fragments.

Computes AP50 / AP75 / AP@[.5:.95] / AR50 / AR75 / AR@[.5:.95] (all
maxDets=100, area=all) per class + macro from in-memory predictions and GT,
plus an on-dataset FPPI block. Conventions:

- COCOeval standard 101-point interpolation -- deliberately DIFFERENT from
  ``utils/metrics/testset_metrics.py``'s PASCAL VOC continuous AP50; the
  unified table prints both where both exist, with a footnote.
- Ignore regions (FB ``etc`` boxes, same-class ``iscrowd=1``) are encoded as
  ``iscrowd=1`` annotations of the evaluated category, which pycocotools
  already treats as "match leniently, never count" -- the same semantics as
  ``testset_metrics.is_ignore_match``, minus the center-inside extension
  (pycocotools uses IoU-vs-ignore only; noted in SCHEMA.md).
- FPPI here is FALSE POSITIVES PER IMAGE ON THE EVALUATED DATASET at fixed
  score thresholds (score-priority greedy at IoU 0.5, ignore-aware), NOT the
  tier3 person-free association FPPI of the unified_benchmark.md note --
  that one needs the negative-pool dumps and is reported separately when
  they exist.

Pure CPU logic -- no torch, no I/O.
"""
from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

from utils.metrics.testset_metrics import rank_detections

PredsByImage = dict[Any, tuple[np.ndarray, np.ndarray]]
GtByImage = dict[Any, np.ndarray]

SEVEN_KEYS = ("ap50", "ap75", "ap", "ar50", "ar75", "ar")
FPPI_THRESHOLDS = (0.05, 0.25, 0.45)


def _xyxy_to_xywh(box: np.ndarray) -> list[float]:
    return [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]


def _valid_mean(arr: np.ndarray) -> float | None:
    vals = arr[arr > -1]
    return float(vals.mean()) if vals.size else None


def cocoeval_block(
    preds: dict[str, PredsByImage],
    gts: dict[str, GtByImage],
    ignores: dict[str, GtByImage] | None,
    image_ids: list[Any],
) -> dict[str, Any]:
    """Run one COCOeval over every class and extract the 7-metric block.

    Args:
        preds: ``class_name -> image_id -> (boxes xyxy (n,4), scores (n,))``.
        gts: ``class_name -> image_id -> gt boxes xyxy (m,4)`` (real GT only).
        ignores: Same shape, ignore-region boxes per class (or ``None``).
        image_ids: EVERY image in the split (negatives included).

    Returns:
        ``{"per_class": {cls: {ap50..ar}}, "macro": {ap50..ar}}`` -- a value
        is ``None`` where pycocotools yields no valid entry (e.g. zero GT).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    classes = sorted(gts)
    cat_ids = {cls: k + 1 for k, cls in enumerate(classes)}
    img_id_map = {iid: k + 1 for k, iid in enumerate(image_ids)}

    anns, dets = [], []
    for cls in classes:
        cid = cat_ids[cls]
        for iid, boxes in gts[cls].items():
            if iid not in img_id_map:
                continue
            for b in boxes:
                anns.append({"id": len(anns) + 1, "image_id": img_id_map[iid],
                             "category_id": cid, "bbox": _xyxy_to_xywh(b),
                             "area": float(max(b[2] - b[0], 0) * max(b[3] - b[1], 0)),
                             "iscrowd": 0})
        for iid, boxes in (ignores or {}).get(cls, {}).items():
            if iid not in img_id_map:
                continue
            for b in boxes:
                anns.append({"id": len(anns) + 1, "image_id": img_id_map[iid],
                             "category_id": cid, "bbox": _xyxy_to_xywh(b),
                             "area": float(max(b[2] - b[0], 0) * max(b[3] - b[1], 0)),
                             "iscrowd": 1})
        for iid, (boxes, scores) in preds[cls].items():
            if iid not in img_id_map:
                continue
            for b, s in zip(boxes, scores):
                dets.append({"image_id": img_id_map[iid], "category_id": cid,
                             "bbox": _xyxy_to_xywh(b), "score": float(s)})

    empty = {k: None for k in SEVEN_KEYS}
    if not anns:
        return {"per_class": {cls: dict(empty) for cls in classes}, "macro": dict(empty)}

    gt_coco = COCO()
    gt_coco.dataset = {
        "images": [{"id": v} for v in img_id_map.values()],
        "annotations": anns,
        "categories": [{"id": cat_ids[c], "name": c} for c in classes],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco.createIndex()
        dt_coco = gt_coco.loadRes(dets) if dets else COCO()
        if not dets:
            dt_coco.dataset = {"images": gt_coco.dataset["images"], "annotations": [],
                               "categories": gt_coco.dataset["categories"]}
            dt_coco.createIndex()
        ev = COCOeval(gt_coco, dt_coco, "bbox")
        ev.evaluate()
        ev.accumulate()

    # precision: [T, R, K, A, M]; recall: [T, K, A, M]; A=0 (all), M=-1 (100)
    precision = ev.eval["precision"]
    recall = ev.eval["recall"]
    per_class: dict[str, dict[str, float | None]] = {}
    for cls in classes:
        k = cat_ids[cls] - 1
        per_class[cls] = {
            "ap50": _valid_mean(precision[0, :, k, 0, -1]),
            "ap75": _valid_mean(precision[5, :, k, 0, -1]),
            "ap": _valid_mean(precision[:, :, k, 0, -1]),
            "ar50": _valid_mean(recall[0:1, k, 0, -1]),
            "ar75": _valid_mean(recall[5:6, k, 0, -1]),
            "ar": _valid_mean(recall[:, k, 0, -1]),
        }
    macro = {}
    for key in SEVEN_KEYS:
        vals = [v[key] for v in per_class.values() if v[key] is not None]
        macro[key] = float(np.mean(vals)) if vals else None
    return {"per_class": per_class, "macro": macro}


def fppi_on_dataset(
    preds_all: PredsByImage,
    gt_all: GtByImage,
    ignore_all: GtByImage | None,
    n_images: int,
    thresholds: tuple[float, ...] = FPPI_THRESHOLDS,
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """False positives per image at fixed score thresholds (class-pooled).

    Uses ``testset_metrics.rank_detections`` (score-priority greedy,
    ignore-aware) so a box is an FP iff it matched neither GT nor ignore at
    ``iou_thresh``.
    """
    ranked = rank_detections(preds_all, gt_all, ignore_by_image=ignore_all,
                             iou_thresh=iou_thresh)
    fp_scores = ranked.scores[~ranked.tp]
    return {
        "by_threshold": {f"{t:.2f}": float((fp_scores >= t).sum()) / n_images
                         for t in thresholds},
        "iou": iou_thresh,
        "n_images": n_images,
        "note": "on-dataset FP/img (class-pooled), NOT tier3 association FPPI",
    }
