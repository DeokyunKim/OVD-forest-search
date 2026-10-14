"""OWLv2 under the frozen protocol, read out per channel.

``post_process_object_detection`` max-reduces over the query axis and returns
one label per box -- the forced choice this benchmark must avoid, since a box
that fires on two terms has to be credited to both. The fix: run **one** image
forward, then read the logits out channel by channel through a single-query
slice. A max over one channel is the identity, so each term's scores come back
unreduced, and image encoding -- which dominates OWLv2's cost -- still happens
once per image rather than once per term.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from utils.runner.detector import TermDetection
from utils.protocol.eval_protocol import EVAL_PROTOCOL, EvalProtocol
from utils.runner.logging_utils import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = get_logger(__name__)


@dataclass(frozen=True)
class Owlv2Prepared:
    """Processor output for one image plus the target size the read-out needs.

    ``target_size`` is ``(height, width)`` of the *original* image, kept as a
    plain tuple so the prepared item carries no reference to the decoded PIL
    image and no device tensor (the host-to-device copy happens on the
    consumer thread).
    """

    inputs: object
    target_size: tuple[int, int]



class _SingleChannelOutputs:
    """A view of OWLv2 outputs exposing exactly one query channel.

    Slicing the logits to ``[..., q:q+1]`` makes the processor's max over the
    query axis an identity, which is what turns a forced-choice read-out into
    a per-channel one.

    This is the *reference* implementation. It is correct but quadratic in
    practice — at ``score_threshold=0.0`` the processor's ``scores > threshold``
    keeps every patch (sigmoid is always positive), so an 85-term condition
    materialises ~85 x 3,600 detections per image and re-derives the boxes 85
    times. :class:`Owlv2Detector` uses the vectorised path below instead and
    the regression tests asserts the two agree exactly.
    """

    def __init__(self, outputs, channel: int) -> None:
        self.logits = outputs.logits[..., channel: channel + 1]
        self.pred_boxes = outputs.pred_boxes


class Owlv2Detector:
    """OWLv2 with an explicit per-channel read-out (see module docstring)."""

    def __init__(self, variant: str = "large", device: str = "cuda:0",
                 protocol: EvalProtocol = EVAL_PROTOCOL,
                 checkpoint_override: str | None = None,
                 cache_dir: str | None = None,
                 max_output_detections: int | None = None,
                 v6_adapter_dir: str | None = None) -> None:
        """``max_output_detections`` overrides how many detections the read-out
        returns. It must exceed the protocol cap for the budget-window sweep,
        which compares windows above and below 100; leaving it at the cap made
        windows >100 unreachable for OWLv2. It does not change any threshold —
        the protocol governs scoring, this governs how much is emitted.

        ``v6_adapter_dir`` switches the arm from zero-shot to fine-tuned: spec
        §2's LoRA is injected at the ViT targets, the V6 checkpoint is loaded
        and its manifest is re-verified against the live model. Nothing else
        changes — the read-out above and the protocol are the same objects the
        zero-shot arm uses, which is what §6 requires. The adapter is never
        merged into the base weights (see :mod:`utils.runner.adapter_eval`)."""
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        from utils.metrics.text_similarity import OWLV2_CHECKPOINTS

        if variant not in OWLV2_CHECKPOINTS:
            raise ValueError(f"unknown OWLv2 variant {variant!r}")
        checkpoint = OWLV2_CHECKPOINTS[variant]
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self.protocol = protocol
        self.model_name = "owlv2"
        self.checkpoint_name = f"{variant}-zeroshot" if checkpoint_override is None else variant
        self._torch = torch
        self._processor = Owlv2Processor.from_pretrained(checkpoint, **kwargs)
        weights = checkpoint_override if checkpoint_override is not None else checkpoint
        self._model = Owlv2ForObjectDetection.from_pretrained(weights, **kwargs)
        self.v6_checkpoint = None
        if v6_adapter_dir is not None:
            from utils.runner.adapter_eval import attach_owlv2_adapter, ft_checkpoint_name

            self.v6_checkpoint = attach_owlv2_adapter(self._model, v6_adapter_dir)
            self.checkpoint_name = ft_checkpoint_name(variant, self.v6_checkpoint)
        self._device = torch.device(device)
        self._model.to(self._device).eval()
        self.max_output_detections = (max_output_detections
                                      or protocol.max_detections_per_image)

    def prepare(self, image: "PILImage", terms: Sequence[str]) -> "Owlv2Prepared":
        """The CPU half of the forward: the processor call, nothing else.

        This is the ~0.35 s/image numpy+PIL path that dominates OWLv2's CPU
        cost. Split out verbatim so :mod:`utils.runner.prefetch` can run it on a
        worker thread while the GPU works on the previous image. The processor
        is called with the same code and the same arguments as before; only the
        host-to-device copy moved to the consumer side, where it belongs.
        ``use_fast`` is deliberately NOT set: the fast processor is documented
        to change outputs.
        """
        inputs = self._processor(text=[list(terms)], images=image, return_tensors="pt")
        return Owlv2Prepared(inputs=inputs, target_size=tuple(image.size[::-1]))

    def _forward_prepared(self, prepared: "Owlv2Prepared"):
        torch = self._torch
        inputs = prepared.inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([prepared.target_size], device=self._device)
        return outputs, target_sizes

    def _forward(self, image: "PILImage", terms: Sequence[str]):
        return self._forward_prepared(self.prepare(image, terms))

    def predict_reference(self, image: "PILImage",
                          terms: Sequence[str]) -> list[TermDetection]:
        """Reference read-out: one processor call per query channel.

        Kept as the correctness oracle for :meth:`predict`. Do not use it for
        a census run — see :class:`_SingleChannelOutputs`.
        """
        outputs, target_sizes = self._forward(image, terms)
        grounded = getattr(self._processor, "post_process_grounded_object_detection", None)
        out: list[TermDetection] = []
        for channel, term in enumerate(terms):
            sliced = _SingleChannelOutputs(outputs, channel)
            if grounded is not None:
                result = grounded(outputs=sliced, threshold=self.protocol.score_threshold,
                                  target_sizes=target_sizes, text_labels=[[term]])[0]
            else:
                result = self._processor.post_process_object_detection(
                    outputs=sliced, threshold=self.protocol.score_threshold,
                    target_sizes=target_sizes)[0]
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            for box, score in zip(boxes, scores):
                out.append(TermDetection(
                    term=term, score=float(score),
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                ))
        return out

    def predict(self, image: "PILImage", terms: Sequence[str]) -> list[TermDetection]:
        """Vectorised per-channel read-out, provably identical to the reference.

        The processor computes ``scores = sigmoid(max(logits, dim=-1))`` and
        derives boxes from ``pred_boxes`` alone. On a single-channel slice the
        max is the identity, so per-channel scores are just
        ``sigmoid(logits[..., q])`` and the boxes do not depend on ``q`` at all
        — they can be derived once instead of once per term.

        The protocol keeps only the 100 highest-scoring detections per image
        across all terms, so taking ``topk`` over the flattened
        *(patch x term)* score matrix here yields exactly the set the runner's
        cap would have selected, without materialising the ~300k intermediate
        detections that ``score_threshold=0.0`` implies for an 85-term
        condition. the regression tests asserts the equality
        against :meth:`predict_reference` on real images.
        """
        return self.predict_prepared(self.prepare(image, terms), terms)

    def predict_prepared(self, prepared: "Owlv2Prepared",
                         terms: Sequence[str]) -> list[TermDetection]:
        """:meth:`predict` on an already-preprocessed input (see :meth:`prepare`).

        Identical statements in an identical order; only the processor call has
        already happened, on another thread.
        """
        import torch
        from transformers.image_transforms import center_to_corners_format
        from transformers.models.owlv2.image_processing_owlv2 import _scale_boxes

        outputs, target_sizes = self._forward_prepared(prepared)
        # scores: (n_patches, n_terms) -- no reduction over the term axis.
        scores = torch.sigmoid(outputs.logits[0])
        boxes = _scale_boxes(center_to_corners_format(outputs.pred_boxes), target_sizes)[0]

        n_patches, n_terms = scores.shape
        keep = min(self.max_output_detections, scores.numel())
        flat_scores, flat_index = torch.topk(scores.reshape(-1), keep)
        patch_index = torch.div(flat_index, n_terms, rounding_mode="floor")
        term_index = flat_index % n_terms

        selected = boxes[patch_index].detach().cpu().numpy()
        flat_scores = flat_scores.detach().cpu().numpy()
        term_index = term_index.detach().cpu().numpy()
        return [
            TermDetection(term=terms[int(t)], score=float(s),
                          bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for b, s, t in zip(selected, flat_scores, term_index)
        ]

    def postprocessing_notes(self) -> dict:
        return {
            "v6_checkpoint": self.v6_checkpoint,
            "readout": "per_channel_vectorised",
            "readout_detail": ("one image forward; per-term scores are "
                               "sigmoid(logits[..., q]) with NO max over the query "
                               "axis, so no box is forced onto a single label. Boxes "
                               "are derived once (they do not depend on the query) and "
                               "the protocol's 100-per-image cap is applied as a topk "
                               "over the flattened (patch x term) score matrix, which "
                               "selects exactly the set the unvectorised read-out plus "
                               "the cap would select"),
            "max_detections_per_image": self.protocol.max_detections_per_image,
            "max_output_detections": self.max_output_detections,
            "nms": "model_default_unchanged",
        }


