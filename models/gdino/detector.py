"""Grounding DINO under the frozen protocol, read out per channel.

Its detectron2-style ``inference`` already takes ``topk`` over the flattened
*(query x class)* axis, so one box can be returned under several classes. That
is per-channel by construction and needs no intervention: the model default
``select_box_nums_for_evaluation`` is left untouched and recorded.

No forced choice. A box that fires on two terms is credited to both, because the
interference this benchmark measures lives exactly there. A post-processing step
that max-reduces over the query axis would hide the effect and do it silently --
the numbers would still look plausible.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from utils.runner.detector import TermDetection
from utils.protocol.eval_protocol import EVAL_PROTOCOL, EvalProtocol
from utils.runner.logging_utils import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = get_logger(__name__)


class GroundingDinoDetector:
    """Grounding DINO under the frozen protocol.

    The joint caption is rendered by the condition, so every term of the
    condition is present in one caption and the terms compete exactly as the
    experiment intends.

    Args:
        v6_adapter_dir: A V6 LoRA checkpoint directory (``ckpt_final_step5667``)
            written by ``train.py --model gdino``. The base model is built
            exactly as for zero-shot, spec §2's LoRA is injected at the targets
            of the scope the checkpoint records, the adapter is loaded and the
            manifest is re-verified against the live model. **Every other line
            of this class is shared with the zero-shot path**, which is what
            makes ZS and FT comparable under §6: the read-out, the protocol
            values and the post-processing cannot drift between the two arms
            because there is only one copy of them.

    The adapter is deliberately **not merged** into the base weights: merging
    has a recorded history on this project of producing run-to-run different
    results with no log evidence (``CLAUDE.md``). See
    :mod:`utils.runner.adapter_eval`.
    """

    def __init__(self, variant: str = "swinb_zeroshot", device: str = "cuda:0",
                 protocol: EvalProtocol = EVAL_PROTOCOL,
                 checkpoint_override: str | None = None,
                 v6_adapter_dir: str | None = None) -> None:
        from models.gdino.loader import GroundingDinoAdapter

        self.protocol = protocol
        self.model_name = "gdino"
        self.checkpoint_name = variant
        # score_thresh 0.0: §6 keeps every detection; AP is rank-based.
        self._adapter = GroundingDinoAdapter(
            variant=variant, device=device, score_thresh=protocol.score_threshold,
            checkpoint_override=checkpoint_override,
        )
        self.v6_checkpoint = None
        if v6_adapter_dir is not None:
            from utils.runner.adapter_eval import attach_gdino_adapter, ft_checkpoint_name

            self.v6_checkpoint = attach_gdino_adapter(self._adapter._model,
                                                      v6_adapter_dir)
            self.checkpoint_name = ft_checkpoint_name(variant, self.v6_checkpoint)

    def predict(self, image: "PILImage", terms: Sequence[str]) -> list[TermDetection]:
        return self.predict_prepared(self.prepare(image, terms), terms)

    def prepare(self, image: "PILImage", terms: Sequence[str]) -> dict:
        """CPU-side preprocessing only (see :mod:`utils.runner.prefetch`)."""
        return self._adapter.prepare(image, list(terms))

    def predict_prepared(self, prepared: dict,
                         terms: Sequence[str]) -> list[TermDetection]:
        detections = self._adapter.predict_prepared(prepared, list(terms))
        return [TermDetection(term=d.class_name, score=d.score, bbox_xyxy=d.bbox_xyxy)
                for d in detections]

    def postprocessing_notes(self) -> dict:
        model = self._adapter._model
        return {
            "v6_checkpoint": self.v6_checkpoint,
            "readout": "per_channel_by_construction",
            "readout_detail": ("detectron2-style inference takes topk over the flattened "
                               "(query x class) axis, so one box may be returned under "
                               "several terms; no argmax is applied"),
            "select_box_nums_for_evaluation": getattr(
                model, "select_box_nums_for_evaluation", None),
            "nms": "model_default_unchanged",
        }


