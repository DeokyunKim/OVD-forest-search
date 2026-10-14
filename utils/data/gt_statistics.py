"""Ground-truth statistics the spec calls "essential for interpretation".

Spec: the vocabulary specification §1.

Per V6 class, from the annotations alone (no model involved):

===================================  ===========================================
mean IoU with ``person`` boxes       quantifies spatial overlap
mean box area as a fraction of the   isolates the small-object effect
image
mean instances per image             density effect
===================================  ===========================================

Overlap definition
------------------
For a non-``person`` box, the reported overlap is the IoU with the **best
matching** ``person`` box in the same image (max over that image's person
boxes; 0.0 when the image has none). A mean over all pairs would be dominated
by the number of people in frame rather than by whether belongings sit on a
person, so the per-box max is the quantity that answers "how much does this
class overlap people". Both the mean over all boxes and the mean restricted
to boxes that overlap some person at all (IoU > 0) are reported, because the
two answer different questions and diverge when a class appears in
person-free frames.

``iscrowd=1`` annotations (the ``etc`` ignore regions and the source's own
small-box ignore flags) are excluded from every statistic: they are not
scored, so they are not part of the evaluated population.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from utils.protocol.vocabulary import CLASS_BY_ID, PERSON_CLASS, V6_CLASSES


@dataclass(frozen=True)
class ClassStatistics:
    """Per-class ground-truth statistics on one split."""

    class_name: str
    n_boxes: int
    n_images_with_class: int
    #: Mean over all boxes of the class of max-IoU with a person box in the
    #: same image (0.0 where the image holds no person box).
    mean_iou_with_person: float
    #: Same, restricted to boxes whose max-IoU with a person box is > 0.
    mean_iou_with_person_when_overlapping: float
    frac_boxes_overlapping_person: float
    #: Box area / (image width * height), averaged over boxes.
    mean_box_area_fraction: float
    median_box_area_fraction: float
    #: Instances per image, averaged over **all** images of the split
    #: (negatives included) and over images that contain the class.
    mean_instances_per_image: float
    mean_instances_per_image_when_present: float


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two COCO ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def compute_class_statistics(data: Mapping) -> dict[str, ClassStatistics]:
    """Compute :class:`ClassStatistics` for every V6 class in ``data``."""
    image_wh = {im["id"]: (float(im["width"]), float(im["height"])) for im in data["images"]}
    n_images = len(data["images"])

    by_image: dict[int, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        name = CLASS_BY_ID.get(ann["category_id"])
        if name is None:  # etc / anything outside V6
            continue
        by_image[ann["image_id"]][name].append([float(v) for v in ann["bbox"]])

    ious: dict[str, list[float]] = defaultdict(list)
    areas: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(list)

    for image_id, per_class in by_image.items():
        width, height = image_wh[image_id]
        image_area = width * height
        persons = per_class.get(PERSON_CLASS, [])
        for name, boxes in per_class.items():
            counts[name].append(len(boxes))
            for box in boxes:
                areas[name].append((box[2] * box[3]) / image_area if image_area > 0 else 0.0)
                if name == PERSON_CLASS:
                    others = [p for p in persons if p is not box]
                    ious[name].append(max((iou_xywh(box, p) for p in others), default=0.0))
                else:
                    ious[name].append(max((iou_xywh(box, p) for p in persons), default=0.0))

    out: dict[str, ClassStatistics] = {}
    for name in V6_CLASSES:
        class_ious = ious.get(name, [])
        overlapping = [v for v in class_ious if v > 0.0]
        class_counts = counts.get(name, [])
        out[name] = ClassStatistics(
            class_name=name,
            n_boxes=len(class_ious),
            n_images_with_class=len(class_counts),
            mean_iou_with_person=_mean(class_ious),
            mean_iou_with_person_when_overlapping=_mean(overlapping),
            frac_boxes_overlapping_person=len(overlapping) / len(class_ious) if class_ious else 0.0,
            mean_box_area_fraction=_mean(areas.get(name, [])),
            median_box_area_fraction=_median(areas.get(name, [])),
            mean_instances_per_image=(sum(class_counts) / n_images) if n_images else 0.0,
            mean_instances_per_image_when_present=_mean([float(c) for c in class_counts]),
        )
    return out


@dataclass(frozen=True)
class CooccurrenceReport:
    """How often a class shares an image with ``person`` at all.

    The IoU statistics are uninterpretable without this: an IoU of ~0 can mean
    "the objects are in the same frame but never overlap" or "the objects are
    never in the same frame". Only the first is a statement about geometry.
    """

    n_images: int
    n_images_with_person: int
    #: class name -> number of images holding both that class and ``person``.
    n_images_with_person_and_class: Mapping[str, int]
    #: class name -> number of images holding that class at all.
    n_images_with_class: Mapping[str, int]


def compute_cooccurrence(data: Mapping) -> CooccurrenceReport:
    """Count images in which each belongings class shares a frame with ``person``."""
    per_image: dict[int, set[str]] = defaultdict(set)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        name = CLASS_BY_ID.get(ann["category_id"])
        if name is not None:
            per_image[ann["image_id"]].add(name)

    with_person = 0
    both: dict[str, int] = {n: 0 for n in V6_CLASSES if n != PERSON_CLASS}
    present: dict[str, int] = {n: 0 for n in V6_CLASSES}
    for classes in per_image.values():
        for name in classes:
            present[name] += 1
        if PERSON_CLASS in classes:
            with_person += 1
            for name in classes - {PERSON_CLASS}:
                both[name] += 1
    return CooccurrenceReport(
        n_images=len(data["images"]),
        n_images_with_person=with_person,
        n_images_with_person_and_class=both,
        n_images_with_class=present,
    )


def overlap_ranking(stats: Mapping[str, ClassStatistics]) -> tuple[str, ...]:
    """Belongings classes ordered by descending mean IoU with ``person``."""
    belongings = [n for n in V6_CLASSES if n != PERSON_CLASS]
    return tuple(sorted(belongings, key=lambda n: stats[n].mean_iou_with_person, reverse=True))


def statistics_to_json(stats: Mapping[str, ClassStatistics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: asdict(v) for k, v in stats.items()}, indent=2))
