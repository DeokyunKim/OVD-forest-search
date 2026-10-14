"""Detection and adapter types shared across the models.

Named `detection_types` rather than `types`: the two evaluation runners live in
this directory, Python puts a script's own directory first on `sys.path`, and a
module called `types.py` here shadows the standard library for the whole
process. It surfaced as `transformers` failing to import `_weakrefset`, which
names nothing about the real cause.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One predicted box in absolute pixel coordinates.

    Attributes:
        bbox_xyxy: ``(x1, y1, x2, y2)`` in absolute pixel coordinates.
        score: Model confidence in ``[0, 1]``.
        class_name: Raw class label emitted by the model. Callers are
            responsible for mapping this back to the canonical COCO name
            (see ``srcs.ovdbench.coco_utils.match_class_name``) -- adapters
            are allowed to emit their own vocabulary spelling verbatim.
    """

    bbox_xyxy: BBoxXYXY
    score: float
    class_name: str


@runtime_checkable
class ModelAdapter(Protocol):
    """Common interface every OVD model wrapper implements.

    Implementations own model loading (in ``__init__``) and expose a single
    ``predict`` call that takes a PIL image plus the open-vocabulary prompt
    list to query with, and returns raw detections in the model's own label
    spelling.
    """

    name: str
    variant: str

    def predict(self, image: "PILImage", class_names: list[str]) -> list[Detection]:
        """Run one forward pass and return detections for ``image``.

        Args:
            image: A ``PIL.Image.Image`` opened in RGB mode.
            class_names: The open-vocabulary prompt list to detect against
                (typically the 80 COCO class names, in canonical order).

        Returns:
            A list of ``Detection`` objects with pixel-space boxes.
        """
        ...


# Forward-reference only used for the Protocol annotation above; avoids a
# hard PIL import for callers that only need the dataclasses/Protocol.
try:  # pragma: no cover - trivial typing convenience
    from PIL.Image import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = "PIL.Image.Image"  # type: ignore[assignment]
