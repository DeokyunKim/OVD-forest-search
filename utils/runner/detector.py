"""What a detector is, and how one is built.

Spec: the frozen-hyperparameter specification §6.

The single most important property of this module: **no forced choice.** A box
that fires on two terms must be credited to both, because the interference
this study measures lives exactly there. A post-processing step that
max-reduces over the query axis would hide the effect being measured and would
do so silently — the numbers would look plausible.

Grounding DINO
    Its detectron2-style ``inference`` already takes ``topk`` over the
    flattened *(query x class)* axis (``groundingdino_dt.py``: ``prob.view(bs,
    -1)`` then ``labels = topk_indexes % n_classes``), so one box can be
    returned under several classes. That is per-channel by construction and
    needs no intervention; the model default ``select_box_nums_for_evaluation``
    is left untouched per §6 and recorded.

OWLv2
    ``post_process_object_detection`` max-reduces over the query axis and
    returns one label per box — the forced choice this study must avoid. The
    fix is the one already proven in the retired
    ``probe_query_wording.py`` (research tree): run **one** image forward, then read
    the logits out channel by channel through a single-query slice. A max over
    one channel is the identity, so each term's scores come back unreduced.
    Image encoding, which dominates OWLv2's cost, still happens once.


The four architectures live in ``models.*``; this module holds only
the contract they satisfy and the factory that reaches them. The imports inside
``build_detector`` are deliberate: each model needs its own interpreter, so a
module-level import of all four would make this file unimportable in every
environment. In the GLIP venv, importing OWLv2's detector raises before it can
say anything useful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

from utils.protocol.eval_protocol import EVAL_PROTOCOL, EvalProtocol
from utils.runner.logging_utils import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = get_logger(__name__)


@dataclass(frozen=True)
class TermDetection:
    """One detection attributed to one queried term (never to an argmax label)."""

    term: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]


class ConditionDetector(Protocol):
    """Predict detections for an ordered term list, per channel.

    A detector MAY additionally implement the two-phase pair
    ``prepare(image, terms) -> prepared`` / ``predict_prepared(prepared, terms)``,
    splitting the CPU preprocessing from the GPU forward so the runner can
    overlap them (:mod:`utils.runner.prefetch`). The contract is strict:
    ``predict(image, terms)`` MUST equal ``predict_prepared(prepare(image,
    terms), terms)`` bit for bit — the split is a scheduling change, never an
    arithmetic one. A detector that does not implement the pair is driven
    through ``predict`` and only its image loading is overlapped.
    """

    model_name: str
    checkpoint_name: str

    def predict(self, image: "PILImage", terms: Sequence[str]) -> list[TermDetection]:
        ...

    def postprocessing_notes(self) -> dict:
        ...



def build_detector(model: str, variant: str, device: str,
                   protocol: EvalProtocol = EVAL_PROTOCOL,
                   checkpoint_override: str | None = None,
                   cache_dir: str | None = None,
                   max_output_detections: int | None = None,
                   v6_adapter_dir: str | None = None,
                   config_override: str | None = None) -> "ConditionDetector":
    """Exact-match dispatch; an unknown model name is never guessed at.

    ``v6_adapter_dir`` switches the arm from zero-shot to fine-tuned. It is the
    ONLY difference between the two arms: the same detector class, the same
    ``EVAL_PROTOCOL`` and the same post-processing run either way. Zero-shot and
    fine-tuned have to be scored identically, and one implementation is the only
    way to guarantee that.

    All four models are dispatched here. One factory, because two are two
    places for the protocol default to drift.
    """
    if model == "gdino":
        from models.gdino.detector import GroundingDinoDetector
        return GroundingDinoDetector(variant=variant, device=device, protocol=protocol,
                                     checkpoint_override=checkpoint_override,
                                     v6_adapter_dir=v6_adapter_dir)
    if model == "owlv2":
        from models.owlv2.detector import Owlv2Detector
        return Owlv2Detector(variant=variant, device=device, protocol=protocol,
                             checkpoint_override=checkpoint_override, cache_dir=cache_dir,
                             max_output_detections=max_output_detections,
                             v6_adapter_dir=v6_adapter_dir)
    if model == "glip":
        from models.glip.detector import build_glip_detector
        return build_glip_detector(variant=variant, device=device, protocol=protocol,
                                   checkpoint_override=checkpoint_override,
                                   config_override=config_override,
                                   max_output_detections=max_output_detections,
                                   v6_adapter_dir=v6_adapter_dir)
    if model == "omdet":
        from models.omdet.detector import build_omdet_detector
        # OmDet-Turbo's decoder emits a fixed query set, so there is no larger
        # candidate pool for the protocol cap to select from and the flag has
        # nothing to widen. Refused rather than dropped: a caller who passed it
        # would otherwise believe a cap had been raised.
        if max_output_detections is not None:
            raise ValueError(
                "max_output_detections is not applicable to OmDet-Turbo: its "
                "decoder emits a fixed number of queries.")
        return build_omdet_detector(variant=variant, device=device, protocol=protocol,
                                    checkpoint_override=checkpoint_override,
                                    cache_dir=cache_dir,
                                    v6_adapter_dir=v6_adapter_dir)
    raise ValueError(
        f"unknown model {model!r}; expected one of 'gdino', 'owlv2', 'glip', 'omdet'")
