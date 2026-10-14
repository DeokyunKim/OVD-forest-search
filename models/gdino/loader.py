"""GroundingDINO adapter for the ovdbench interface.

Wraps this project's own detectron2-style GroundingDINO loader so the model
under study can be benchmarked with exactly the same harness and query set as
the third-party detectors. Variants map to entries in ``configs/default.yaml``;
``base`` uses the zero-branch (pretrained) model rather than any adapted
checkpoint.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from utils.protocol.detection_types import Detection
from utils.runner.logging_utils import get_logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = get_logger(__name__)

# variant -> (config variant key, checkpoint override or None for zero-branch base)
# variant -> config variant key. The checkpoint comes from configs/models.yaml,
# not from this table: one home for a path, so it cannot go stale in one place
# and not the other.
_VARIANTS: dict[str, str] = {
    "swinb_zeroshot": "swinb_dt_base",
}


class GroundingDinoAdapter:
    """Runs this repository's GroundingDINO through the ovdbench contract."""

    name = "groundingdino"

    def __init__(self, variant: str = "base", device: str = "cuda:0",
                 score_thresh: float = 0.0, config: str | None = None,
                 checkpoint_override: str | None = None) -> None:
        """``checkpoint_override`` swaps only the weights, keeping the variant's
        architecture config -- used to walk intermediate training checkpoints
        for the collapse-onset trajectory. The alpha cross-check in
        ``models/gdino/variants.py`` still applies, so an override pointing at a
        checkpoint trained with a different alpha fails loudly."""
        import torch

        from utils.runner.paths import REPO_ROOT
        from models.gdino.variants import load_variant
        from utils.data.preprocessing_manifest import EVAL_MAX_SIZE, EVAL_SHORT_EDGE
        from utils.runner.config import load_config
        from utils.runner.repo_bridge import ensure_external_repo

        if variant not in _VARIANTS:
            raise ValueError(f"Unknown GroundingDINO variant '{variant}', "
                             f"expected {list(_VARIANTS)}")
        self.variant = variant
        self.score_thresh = score_thresh
        cfg = load_config(config or REPO_ROOT / "configs" / "models.yaml")
        ensure_external_repo(cfg.external_repo)
        spec = cfg.variants[_VARIANTS[variant]]
        if checkpoint_override is not None:
            spec = dataclasses.replace(spec, checkpoint=checkpoint_override)
        logger.info("Loading GroundingDINO variant=%s checkpoint=%s",
                    variant, spec.checkpoint)
        self._model = load_variant(spec, device, base_checkpoint=cfg.base_checkpoint)
        self._torch = torch
        import detectron2.data.transforms as T
        self._aug = T.ResizeShortestEdge(short_edge_length=EVAL_SHORT_EDGE,
                                        max_size=EVAL_MAX_SIZE)

    def prepare(self, image: "PILImage", class_names: list[str]) -> dict:
        """The CPU half of :meth:`predict`: decode, resize, build batched input.

        Split out verbatim so a caller can run it on a worker thread while the
        GPU is busy with the previous image (``utils/runner/prefetch.py``). The
        statements and their order are unchanged, so ``prepare`` followed by
        ``predict_prepared`` is the same computation as ``predict``.
        """
        arr = np.asarray(image.convert("RGB"))
        h, w = arr.shape[:2]
        resized = self._aug.get_transform(arr).apply_image(arr)
        return {
            "image": self._torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1))),
            "height": h, "width": w,
            "captions": " . ".join(class_names) + " .",
        }

    def predict(self, image: "PILImage", class_names: list[str]) -> list[Detection]:
        return self.predict_prepared(self.prepare(image, class_names), class_names)

    def predict_prepared(self, inputs: dict, class_names: list[str]) -> list[Detection]:
        """The GPU half of :meth:`predict`, on an already-preprocessed input."""
        with self._torch.no_grad():
            (out,) = self._model([inputs])
        inst = out["instances"].to("cpu")
        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        classes = inst.pred_classes.numpy()
        dets = []
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
            if s < self.score_thresh:
                continue
            name = class_names[int(c)] if 0 <= int(c) < len(class_names) else class_names[0]
            dets.append(Detection(bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                                  score=float(s), class_name=name))
        return dets
