"""The frozen evaluation protocol — identical for ZS and FT, all models.

Spec: the frozen-hyperparameter specification §6, the vocabulary specification §7.

    "여기가 어긋나면 모든 비교가 무효가 된다."
    (If this drifts, every comparison is void.)

There is exactly one protocol object, :data:`EVAL_PROTOCOL`, and both the
zero-shot and the fine-tuned arms take their thresholds from it. It is a
frozen dataclass so a caller cannot mutate it after the fact; a run that
needs different numbers must construct its own object and will then fail
:func:`assert_frozen_protocol`, which every evaluation entry point calls.

Visualisation thresholds (the 0.25/0.35 used in demos) must never reach this
object — spec §6 calls that out explicitly as an AP-distorting mistake.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence


class ProtocolViolationError(ValueError):
    """An evaluation was configured with values that are not the frozen ones."""


@dataclass(frozen=True)
class EvalProtocol:
    """Post-processing and scoring settings applied identically to ZS and FT.

    Attributes:
        score_threshold: Keep every detection; AP is rank-based (§6).
        max_detections_per_image: COCO standard 100 (§6).
        nms: ``"model_default"`` — each model keeps its own NMS behaviour and
            it is never changed (§6). Recorded per model in the fragment.
        box_threshold / text_threshold: GroundingDINO-specific, both 0.0 at
            evaluation time (§6).
        deterministic: Decoding must be deterministic (§6).
        iou_thresholds: 0.5 / 0.75 / 0.5:0.95 (§6).
    """

    score_threshold: float = 0.0
    max_detections_per_image: int = 100
    nms: str = "model_default"
    box_threshold: float = 0.0
    text_threshold: float = 0.0
    deterministic: bool = True
    iou_thresholds: tuple[float, ...] = field(
        default_factory=lambda: (0.5, 0.75)
    )
    iou_sweep: tuple[float, float, float] = (0.5, 0.95, 0.05)

    def as_record(self) -> dict:
        """The post-processing block recorded in every fragment (§7)."""
        return {
            "score_threshold": self.score_threshold,
            "max_detections_per_image": self.max_detections_per_image,
            "nms": self.nms,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "deterministic": self.deterministic,
            "iou_thresholds": list(self.iou_thresholds),
            "iou_sweep": list(self.iou_sweep),
        }


#: The single frozen instance. Import this; do not build another.
EVAL_PROTOCOL = EvalProtocol()

#: Values that a demo/visualisation config would carry. Present so the guard
#: can name the mistake it is catching rather than only reporting a mismatch.
VISUALISATION_THRESHOLDS: tuple[float, ...] = (0.25, 0.35)


def assert_frozen_protocol(protocol: EvalProtocol) -> None:
    """Raise unless ``protocol`` is field-for-field the frozen one.

    Called by every evaluation entry point, for both checkpoints, so a ZS/FT
    threshold divergence cannot survive to a results table.
    """
    if protocol != EVAL_PROTOCOL:
        diffs = {
            name: (getattr(protocol, name), getattr(EVAL_PROTOCOL, name))
            for name in EVAL_PROTOCOL.__dataclass_fields__
            if getattr(protocol, name) != getattr(EVAL_PROTOCOL, name)
        }
        hint = ""
        if any(v[0] in VISUALISATION_THRESHOLDS for v in diffs.values()):
            hint = (" A visualisation threshold (0.25/0.35) reached the evaluation "
                    "config; spec §6 forbids this.")
        raise ProtocolViolationError(
            f"evaluation protocol deviates from the frozen spec: {diffs}.{hint}"
        )


#: Metrics that must be present per class in every fragment (spec §7).
REQUIRED_PER_CLASS_METRICS: tuple[str, ...] = ("AP", "AP50", "AP75", "AR", "AR100")

#: Fragment-level fields required alongside the per-class metrics (spec §7).
REQUIRED_FRAGMENT_FIELDS: tuple[str, ...] = (
    "condition_id",
    "query_string",          # the verbatim rendered caption
    "model",
    "checkpoint",            # "zeroshot" | "ft_v6" | ...
    "n_images",              # census gate
    "per_class",
    "empty_frame_false_positives_per_image",
    "detection_count_distribution",
    "postprocessing",        # EvalProtocol.as_record()
)


def missing_fragment_fields(fragment: Mapping) -> tuple[str, ...]:
    """Required §7 fields absent from ``fragment``, in declaration order."""
    return tuple(f for f in REQUIRED_FRAGMENT_FIELDS if f not in fragment)


def missing_per_class_metrics(per_class: Mapping[str, Mapping]) -> Mapping[str, tuple[str, ...]]:
    """Per class, the required §7 metrics that are absent.

    A metric present with value ``None`` counts as present-and-unmeasured; the
    caller distinguishes ``None`` (unmeasured) from ``0.0`` (measured zero).
    """
    return {
        name: tuple(m for m in REQUIRED_PER_CLASS_METRICS if m not in metrics)
        for name, metrics in per_class.items()
        if any(m not in metrics for m in REQUIRED_PER_CLASS_METRICS)
    }


def with_overrides(**overrides) -> EvalProtocol:
    """Build a deviating protocol. Only for tests that assert the guard fires."""
    return replace(EVAL_PROTOCOL, **overrides)


def census_gate(n_images: int, expected: int) -> None:
    """Refuse a fragment that did not cover the full evaluation set.

    Spec §5 forbids per-condition partial evaluation: every condition is
    compared on the same census.
    """
    if n_images != expected:
        raise ProtocolViolationError(
            f"fragment covers {n_images} images, expected the full census {expected}; "
            "spec §5 forbids per-condition partial evaluation"
        )


def iou_sweep_values(protocol: EvalProtocol = EVAL_PROTOCOL) -> Sequence[float]:
    """The 0.50:0.95:0.05 sweep as explicit values."""
    start, stop, step = protocol.iou_sweep
    n = int(round((stop - start) / step)) + 1
    return tuple(round(start + i * step, 2) for i in range(n))
