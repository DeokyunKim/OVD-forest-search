"""Result fragments: one condition x one checkpoint x one model.

Spec: the vocabulary specification §7.

Every field the spec lists is required, and :func:`write_fragment` refuses to
write a fragment that is missing one or that did not cover the full census.
The census gate is the important one: spec §5 forbids per-condition partial
evaluation, so a fragment that saw fewer than the full test set must not
reach disk at all — a short fragment on disk is indistinguishable from a
complete one once it is in a table.

Two distinctions the schema keeps explicit, because collapsing either of them
destroys the finding:

``None`` vs ``0.0``
    ``None`` means unmeasured. A metric that is genuinely zero is ``0.0``.
    Renderers print ``_미측정_`` for ``None``.
``n_detections == 0`` vs ``recall == 0``
    Silence (the model returned nothing for this class) is not the same
    failure as mislocalisation (it returned boxes in the wrong place). The
    detection-count distribution is stored per condition so the two can be
    told apart after the fact.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from utils.runner.logging_utils import get_logger
from utils.protocol.eval_protocol import (
    EVAL_PROTOCOL,
    EvalProtocol,
    ProtocolViolationError,
    assert_frozen_protocol,
    census_gate,
    missing_fragment_fields,
    missing_per_class_metrics,
)

logger = get_logger(__name__)

SCHEMA_VERSION = "v6.1"


@dataclass(frozen=True)
class ClassMetrics:
    """Spec §7 per-class block. ``None`` means unmeasured, never zero."""

    AP: float | None
    AP50: float | None
    AP75: float | None
    AR: float | None
    AR100: float | None
    AR50: float | None = None
    AR75: float | None = None
    n_ground_truth: int = 0
    n_detections: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DetectionCountDistribution:
    """Detections per image, after the protocol's 100-per-image cap."""

    total: int
    mean_per_image: float
    median_per_image: float
    p95_per_image: float
    max_per_image: int
    n_images_with_zero_detections: int
    histogram: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EmptyFrameFalsePositives:
    """False positives on images whose ground truth is empty.

    ``n_empty_images`` is the denominator; a condition with no empty frames
    reports ``None`` for the rate rather than 0.0.
    """

    n_empty_images: int
    n_false_positives: int
    per_image: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def build_fragment(
    *,
    condition_id: str,
    query_string: str,
    model: str,
    checkpoint: str,
    n_images: int,
    per_class: Mapping[str, ClassMetrics],
    empty_frame: EmptyFrameFalsePositives,
    detection_counts: DetectionCountDistribution,
    text_cosines: Mapping[str, float] | None,
    protocol: EvalProtocol = EVAL_PROTOCOL,
    extra: Mapping | None = None,
) -> dict:
    """Assemble a §7-complete fragment. Does not write it."""
    assert_frozen_protocol(protocol)
    return {
        "schema_version": SCHEMA_VERSION,
        "condition_id": condition_id,
        "query_string": query_string,
        "model": model,
        "checkpoint": checkpoint,
        "n_images": n_images,
        "per_class": {name: m.as_dict() for name, m in per_class.items()},
        "empty_frame_false_positives_per_image": empty_frame.as_dict(),
        "detection_count_distribution": detection_counts.as_dict(),
        "text_embedding_cosines": dict(text_cosines) if text_cosines else None,
        "postprocessing": protocol.as_record(),
        **(dict(extra) if extra else {}),
    }


def validate_fragment(fragment: Mapping, expected_images: int) -> None:
    """Raise unless the fragment is §7-complete and covers the full census."""
    missing = missing_fragment_fields(fragment)
    if missing:
        raise ProtocolViolationError(f"fragment is missing required §7 fields: {missing}")
    gaps = missing_per_class_metrics(fragment["per_class"])
    if gaps:
        raise ProtocolViolationError(f"per-class metrics incomplete: {gaps}")
    census_gate(fragment["n_images"], expected_images)
    if not fragment["query_string"]:
        raise ProtocolViolationError("query_string is empty; §7 requires the verbatim query")


def write_fragment(fragment: Mapping, path: Path, expected_images: int) -> Path:
    """Validate, then write. A fragment that fails validation never lands."""
    validate_fragment(fragment, expected_images)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(fragment, indent=2))
    tmp.rename(path)  # atomic: a reader never sees a half-written fragment
    logger.info("wrote fragment %s", path)
    return path


def read_fragment(path: Path) -> dict:
    return json.loads(path.read_text())


def summarise_detection_counts(counts_per_image: Sequence[int]) -> DetectionCountDistribution:
    """Build the §7 detection-count block from per-image detection counts."""
    if not counts_per_image:
        raise ValueError("no per-image counts; refusing to fabricate a distribution")
    ordered = sorted(counts_per_image)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    p95 = ordered[min(n - 1, int(round(0.95 * (n - 1))))]
    histogram: dict[str, int] = {}
    for bucket in ((0, 0), (1, 1), (2, 5), (6, 10), (11, 25), (26, 50), (51, 100)):
        low, high = bucket
        key = f"{low}" if low == high else f"{low}-{high}"
        histogram[key] = sum(1 for c in ordered if low <= c <= high)
    return DetectionCountDistribution(
        total=sum(ordered),
        mean_per_image=sum(ordered) / n,
        median_per_image=float(median),
        p95_per_image=float(p95),
        max_per_image=ordered[-1],
        n_images_with_zero_detections=sum(1 for c in ordered if c == 0),
        histogram=histogram,
    )
