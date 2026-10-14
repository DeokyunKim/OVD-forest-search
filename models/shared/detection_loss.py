"""Hungarian set-prediction loss for OmDet-Turbo.

WHY THIS EXISTS
---------------
``OmDetTurboForObjectDetection.forward`` explicitly raises
``NotImplementedError("Training is not implemented yet")`` when ``labels`` is
passed - HuggingFace ships inference only. To fine-tune the model at all we
must reconstruct the set-prediction objective outside the model, from the two
tensors its forward *does* return:

* ``decoder_class_logits`` - ``(batch, num_queries, num_classes)`` cosine
  similarities between query features and the projected class embeddings,
  scaled by ``1/0.07``; ``sigmoid`` of these is exactly what the processor's
  post-processing treats as the detection score, so a sigmoid focal loss is
  the matching objective (there is no background logit - unmatched queries are
  simply all-negative).
* ``decoder_coord_logits`` - ``(batch, num_queries, 4)`` boxes already through
  ``sigmoid``, in normalised ``cxcywh``.

This is the standard DETR/DINO objective (focal classification + L1 + GIoU)
restricted to the final decoder layer. Auxiliary per-layer supervision is
deliberately omitted: the model only surfaces the last layer's predictions
through its public output, and adding it would require patching the decoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchvision.ops import generalized_box_iou

Target = dict[str, Tensor]


@dataclass(frozen=True)
class LossWeights:
    """Relative weights of the three set-prediction terms (DINO defaults)."""

    classification: float = 2.0
    bbox_l1: float = 5.0
    giou: float = 2.0


@dataclass(frozen=True)
class FocalParams:
    """Sigmoid focal loss parameters."""

    alpha: float = 0.25
    gamma: float = 2.0


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert normalised ``cxcywh`` boxes to ``xyxy`` (new tensor)."""
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height],
        dim=-1,
    )


def sigmoid_focal_loss(logits: Tensor, targets: Tensor, focal: FocalParams) -> Tensor:
    """Element-wise sigmoid focal loss (no reduction)."""
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** focal.gamma)
    alpha_t = focal.alpha * targets + (1 - focal.alpha) * (1 - targets)
    return alpha_t * loss


@torch.no_grad()
def hungarian_match(
    class_logits: Tensor,
    pred_boxes: Tensor,
    targets: list[Target],
    weights: LossWeights,
    focal: FocalParams,
) -> list[tuple[Tensor, Tensor]]:
    """Match queries to ground-truth boxes, one assignment per image.

    Args:
        class_logits: ``(batch, num_queries, num_classes)``.
        pred_boxes: ``(batch, num_queries, 4)`` normalised ``cxcywh``.
        targets: Per-image dicts with ``labels`` ``(n,)`` and ``boxes`` ``(n,4)``.
        weights: Term weights, reused as matching costs.
        focal: Focal parameters for the classification cost.

    Returns:
        ``(query_index, target_index)`` LongTensor pairs, one per image.
    """
    batch_size, num_queries, _ = class_logits.shape
    prob = class_logits.flatten(0, 1).sigmoid()  # (batch*queries, classes)
    out_boxes = pred_boxes.flatten(0, 1)

    target_labels = torch.cat([t["labels"] for t in targets])
    target_boxes = torch.cat([t["boxes"] for t in targets])

    if target_labels.numel() == 0:
        # Every image in the batch is a pure negative (zero GT boxes):
        # nothing to match, every query becomes an all-negative
        # classification target. Guarded explicitly because
        # ``cost.view(batch, queries, -1)`` below cannot infer ``-1``
        # from a 0-element tensor. Mixed batches (some images empty) need
        # no guard: ``split`` handles size-0 chunks and
        # ``linear_sum_assignment`` returns empty assignments for them.
        empty = torch.zeros(0, dtype=torch.long, device=class_logits.device)
        return [(empty, empty) for _ in targets]

    neg_cost = (1 - focal.alpha) * (prob**focal.gamma) * (-(1 - prob + 1e-8).log())
    pos_cost = focal.alpha * ((1 - prob) ** focal.gamma) * (-(prob + 1e-8).log())
    cost_class = pos_cost[:, target_labels] - neg_cost[:, target_labels]

    cost_l1 = torch.cdist(out_boxes, target_boxes, p=1)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_boxes), box_cxcywh_to_xyxy(target_boxes))

    cost = weights.classification * cost_class + weights.bbox_l1 * cost_l1 + weights.giou * cost_giou
    cost = cost.view(batch_size, num_queries, -1).cpu()

    sizes = [len(t["labels"]) for t in targets]
    indices = [
        linear_sum_assignment(chunk[i]) for i, chunk in enumerate(cost.split(sizes, dim=-1))
    ]
    device = class_logits.device
    return [
        (
            torch.as_tensor(query_idx, dtype=torch.long, device=device),
            torch.as_tensor(target_idx, dtype=torch.long, device=device),
        )
        for query_idx, target_idx in indices
    ]


def set_prediction_loss(
    class_logits: Tensor,
    pred_boxes: Tensor,
    targets: list[Target],
    weights: LossWeights = LossWeights(),
    focal: FocalParams = FocalParams(),
) -> dict[str, Tensor]:
    """Compute the matched detection loss.

    Args:
        class_logits: ``(batch, num_queries, num_classes)``.
        pred_boxes: ``(batch, num_queries, 4)`` normalised ``cxcywh``.
        targets: Per-image dicts with ``labels`` and normalised ``cxcywh`` ``boxes``.
        weights: Term weights.
        focal: Focal parameters.

    Returns:
        Dict with ``loss_class``, ``loss_bbox``, ``loss_giou`` (each already
        multiplied by its weight) and their sum under ``loss_detection``.
    """
    indices = hungarian_match(class_logits, pred_boxes, targets, weights, focal)
    device = class_logits.device

    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for src, _ in indices])
    matched_labels = torch.cat([t["labels"][tgt] for t, (_, tgt) in zip(targets, indices)])
    matched_boxes = torch.cat([t["boxes"][tgt] for t, (_, tgt) in zip(targets, indices)])

    num_boxes = torch.as_tensor(max(len(matched_labels), 1), dtype=torch.float32, device=device)

    class_targets = torch.zeros_like(class_logits)
    class_targets[batch_idx, src_idx, matched_labels] = 1.0
    loss_class = sigmoid_focal_loss(class_logits, class_targets, focal).sum() / num_boxes

    src_boxes = pred_boxes[batch_idx, src_idx]
    loss_bbox = F.l1_loss(src_boxes, matched_boxes, reduction="sum") / num_boxes
    giou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(matched_boxes)))
    loss_giou = (1.0 - giou).sum() / num_boxes

    weighted = {
        "loss_class": weights.classification * loss_class,
        "loss_bbox": weights.bbox_l1 * loss_bbox,
        "loss_giou": weights.giou * loss_giou,
    }
    weighted["loss_detection"] = weighted["loss_class"] + weighted["loss_bbox"] + weighted["loss_giou"]
    return weighted
