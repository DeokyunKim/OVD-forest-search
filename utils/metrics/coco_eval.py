"""Per-class COCO metrics under the frozen protocol.

Spec: the frozen-hyperparameter specification §6, the vocabulary specification §7.

Scoring rules that are consequences of the spec, not choices made here:

- **Only the condition's scored classes are evaluated.** A condition that does
  not query ``hat`` has no ``hat`` detections, so scoring ``hat`` would report
  a guaranteed 0.0 that reads like a measurement. Unqueried classes are absent
  from the fragment, not zero.
- **Distractor detections are discarded**, because no ground truth exists for
  them (spec §4). Discarding them *after* the 100-per-image cap is what makes
  the absorption-vs-stealing mechanism observable: a distractor that outranks a
  ``person`` box removes that box from the budget, which is the effect being
  measured.
- **``iscrowd=1`` annotations stay in the GT** and are handled by pycocotools'
  own ignore logic; they are never converted to background.

``AR`` and ``AR@100`` coincide here because the protocol freezes maxDets at
100. Both names are emitted so the §7 checklist is satisfied literally rather
than by argument, and ``AR50``/``AR75`` are added because a single IoU-averaged
recall hides whether a miss is a localisation failure or a true miss.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Mapping, Sequence

from utils.runner.logging_utils import get_logger
from utils.protocol.eval_protocol import EVAL_PROTOCOL, EvalProtocol
from utils.runner.fragment import ClassMetrics
from utils.protocol.vocabulary import CATEGORY_ID

logger = get_logger(__name__)


@dataclass(frozen=True)
class DetectionRecord:
    """One scored detection, already mapped to a V6 ground-truth class."""

    image_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    score: float

    def as_coco(self) -> dict:
        return {"image_id": self.image_id, "category_id": self.category_id,
                "bbox": [float(v) for v in self.bbox_xywh], "score": float(self.score)}


def _index_of(values: Sequence[float], target: float) -> int:
    for idx, value in enumerate(values):
        if abs(value - target) < 1e-6:
            return idx
    raise ValueError(f"{target} not in {list(values)}")


def evaluate_per_class(
    coco_gt,
    detections: Sequence[DetectionRecord],
    scored_classes: Sequence[str],
    detections_per_class: Mapping[str, int],
    protocol: EvalProtocol = EVAL_PROTOCOL,
) -> dict[str, ClassMetrics]:
    """Run COCOeval once and pull out the §7 per-class block.

    Args:
        coco_gt: A loaded ``pycocotools.coco.COCO`` over the full test set.
        detections: All kept detections, distractors already dropped.
        scored_classes: V6 class names this condition queried.
        detections_per_class: Raw detection counts per class, so that
            "the model said nothing" stays distinguishable from "the model was
            wrong" even when AP is 0.

    Returns:
        class name -> :class:`ClassMetrics`. ``None`` fields mean unmeasured.
    """
    import numpy as np
    from pycocotools.cocoeval import COCOeval

    category_ids = [CATEGORY_ID[name] for name in scored_classes]
    gt_counts = {
        name: len(coco_gt.getAnnIds(catIds=[CATEGORY_ID[name]], iscrowd=False))
        for name in scored_classes
    }

    if not detections:
        # Nothing was returned anywhere. AP/AR are genuinely 0 (the GT exists
        # and nothing matched it), but the detection count is what tells a
        # reader this was silence rather than mislocalisation.
        return {
            name: ClassMetrics(AP=0.0, AP50=0.0, AP75=0.0, AR=0.0, AR100=0.0,
                               AR50=0.0, AR75=0.0,
                               n_ground_truth=gt_counts[name], n_detections=0)
            for name in scored_classes
        }

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes([d.as_coco() for d in detections])
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.catIds = category_ids
        evaluator.params.maxDets = [1, 10, protocol.max_detections_per_image]
        evaluator.evaluate()
        evaluator.accumulate()

    iou_values = list(evaluator.params.iouThrs)
    max_det_index = _index_of([float(m) for m in evaluator.params.maxDets],
                              float(protocol.max_detections_per_image))
    i50, i75 = _index_of(iou_values, 0.5), _index_of(iou_values, 0.75)

    # precision: [IoU, recall, category, area, maxDets]
    # recall:    [IoU, category, area, maxDets]
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]

    def _mean(values) -> float | None:
        values = values[values > -1]
        return float(np.mean(values)) if values.size else None

    out: dict[str, ClassMetrics] = {}
    for position, name in enumerate(scored_classes):
        p_all = precision[:, :, position, 0, max_det_index]
        r_all = recall[:, position, 0, max_det_index]
        out[name] = ClassMetrics(
            AP=_mean(p_all),
            AP50=_mean(precision[i50, :, position, 0, max_det_index]),
            AP75=_mean(precision[i75, :, position, 0, max_det_index]),
            AR=_mean(r_all),
            AR100=_mean(r_all),
            AR50=_mean(recall[i50: i50 + 1, position, 0, max_det_index]),
            AR75=_mean(recall[i75: i75 + 1, position, 0, max_det_index]),
            n_ground_truth=gt_counts[name],
            n_detections=int(detections_per_class.get(name, 0)),
        )
    return out


def cap_detections_per_image(
    detections: Sequence[tuple[str, float, tuple[float, float, float, float]]],
    max_detections: int,
) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Keep the ``max_detections`` highest-scoring detections of one image.

    The cap is applied across **all** queried terms including distractors,
    before distractors are dropped — that ordering is what lets a distractor
    displace a ``person`` box, which is the absorption/stealing mechanism the
    C axis is designed to observe. Capping per class, or after the drop, would
    silently suppress the effect.
    """
    return sorted(detections, key=lambda d: d[1], reverse=True)[:max_detections]


def count_empty_frame_false_positives(
    counts_on_empty_images: Sequence[int],
) -> tuple[int, int, float | None]:
    """``(n_empty_images, n_false_positives, per_image)``.

    Every detection on an image whose ground truth is empty is a false
    positive by construction. With no empty frames the rate is ``None``
    (undefined), never 0.0.
    """
    n_images = len(counts_on_empty_images)
    total = int(sum(counts_on_empty_images))
    return n_images, total, (total / n_images if n_images else None)
