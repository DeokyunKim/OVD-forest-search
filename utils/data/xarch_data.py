"""The V6 train/val splits as plain records, for the GLIP and OmDet trainers.

Spec: the frozen-hyperparameter specification §4.0 (data policy) and §5
(resolution). Vocabulary and category ids: ``utils/protocol/vocabulary.py``.

Why not ``utils/data/dataset_register.py``
---------------------------------------
That module registers the split **with detectron2**, which only the Grounding
DINO arm uses. GLIP is mmdetection and OmDet-Turbo is plain HuggingFace, and
neither can consume a detectron2 ``DatasetCatalog`` entry. So the same json is
read here into a framework-neutral record, and the three invariants that module
enforces are enforced again — because each has already gone wrong once:

1. **Category order is caption order.** ``gt_label`` indexes directly into the
   caption's term list on both arms (GLIP through ``tokens_positive[label]``,
   OmDet through the class-list column of ``decoder_class_logits``). If the two
   orders ever disagree the model trains ``person`` boxes against the word
   ``clothes`` and nothing errors.
2. **Zero-annotation images are kept.** §4.0 forbids removing images, and the
   V6 train split deliberately carries negatives.
3. **``file_name`` is already absolute**, so no root is joined on.

``iscrowd`` handling
--------------------
Annotations with ``iscrowd != 0`` are dropped from supervision. This is not a
new decision: the external repo's ``DetrDatasetMapper`` — which the Grounding
DINO arm trains through — drops every ``iscrowd != 0`` annotation, so dropping
them here is what makes the three arms train on the *same* boxes. It removes
the 277 ``etc`` regions (category 7, outside the queried vocabulary by
construction) and the 436 V6-category crowd regions, 713 of 174,054
annotations. The image itself is always kept, so those pixels still act as
unlabelled background exactly as they do for Grounding DINO.
"""
from __future__ import annotations

from utils.runner.paths import paths

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from utils.runner.logging_utils import get_logger
from utils.protocol.train_caption import training_class_names
from utils.protocol.vocabulary import ETC_CATEGORY_ID, ETC_CATEGORY_NAME

logger = get_logger(__name__)

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / 'pyproject.toml').is_file())
# Through the config, not assembled from a literal: the evaluation side
# was routed this way already, and leaving the training side on a hard
# path is how a repository evaluates fine and cannot train.
V6_DATASET_DIR = paths().dataset("benchmark")

SPLITS: tuple[str, ...] = ("train", "val", "test")

#: Multi-scale training resize, in short-edge pixels. Identical to the Grounding
#: DINO trainer's ``TRAIN_SHORT_EDGES`` so the two arms see the same
#: augmentation distribution; §5 fixes GLIP's evaluation resolution at 800/1333,
#: the same as GDINO's. OmDet-Turbo ignores this — its processor forces 640x640.
TRAIN_SHORT_EDGES: tuple[int, ...] = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
EVAL_SHORT_EDGE: int = 800
EVAL_MAX_SIZE: int = 1333


class DatasetContractError(RuntimeError):
    """The json on disk is not the dataset the trainer expects."""


@dataclass(frozen=True)
class V6Record:
    """One image and its supervision, in absolute pixel coordinates.

    ``labels`` are contiguous indices into :func:`training_class_names`, which
    is also the caption's term order. ``boxes_xyxy`` may be empty: §4.0 forbids
    dropping negatives.
    """

    image_id: int
    image_path: str
    width: int
    height: int
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    labels: tuple[int, ...]
    source: str

    @property
    def n_boxes(self) -> int:
        return len(self.labels)


def _assert_category_contract(categories: Sequence[dict]) -> dict[int, int]:
    """Map COCO category id -> contiguous index, and verify caption order.

    Returns:
        ``{category_id: contiguous_index}`` for the six trained classes only.
        ``etc`` is absent from the map, which is what makes its annotations
        unmappable rather than silently folded into class 6.
    """
    names = training_class_names()
    by_id = {int(c["id"]): str(c["name"]) for c in categories}
    expected = {index + 1: name for index, name in enumerate(names)}
    expected[ETC_CATEGORY_ID] = ETC_CATEGORY_NAME
    if by_id != expected:
        raise DatasetContractError(
            f"category table {by_id} does not match the frozen vocabulary "
            f"{expected}. The label index is used directly as a caption term "
            f"position, so a mismatch trains one class's boxes against another "
            f"class's word with no error anywhere")
    return {index + 1: index for index in range(len(names))}


def load_v6_records(split: str, json_path: Path | None = None,
                    max_images: int | None = None) -> list[V6Record]:
    """Read one split into :class:`V6Record` objects.

    Args:
        split: ``train`` / ``val`` / ``test``.
        json_path: Override, for tests.
        max_images: Cap for smoke runs **only**. §4.0 forbids reporting any
            result from a subsampled training set, so callers must log it.

    Raises:
        DatasetContractError: category table or ``file_name`` contract broken.
    """
    if split not in SPLITS:
        raise DatasetContractError(f"unknown split {split!r}; expected {SPLITS}")
    path = json_path or (V6_DATASET_DIR / f"{split}.json")
    payload = json.loads(Path(path).read_text())
    id_to_index = _assert_category_contract(payload["categories"])

    by_image: dict[int, list[dict]] = {}
    n_crowd = 0
    for ann in payload["annotations"]:
        if int(ann.get("iscrowd", 0)) != 0:
            n_crowd += 1
            continue
        category = int(ann["category_id"])
        if category not in id_to_index:
            # Only reachable for a non-crowd `etc`; the vocabulary module
            # guarantees `etc` is written as iscrowd=1, so this is a contract
            # breach rather than something to skip quietly.
            raise DatasetContractError(
                f"annotation {ann.get('id')} has category_id {category}, which is "
                f"outside the six trained classes and is not marked iscrowd")
        by_image.setdefault(int(ann["image_id"]), []).append(ann)

    records: list[V6Record] = []
    for image in payload["images"]:
        file_name = str(image["file_name"])
        if not file_name.startswith("/"):
            raise DatasetContractError(
                f"file_name {file_name!r} is not absolute; utils/data/dataset_build.py "
                f"resolves every path at build time and no root is joined on here")
        boxes: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        for ann in by_image.get(int(image["id"]), []):
            x, y, w, h = (float(v) for v in ann["bbox"])
            if w <= 1.0 or h <= 1.0:
                continue
            boxes.append((x, y, x + w, y + h))
            labels.append(id_to_index[int(ann["category_id"])])
        records.append(V6Record(
            image_id=int(image["id"]), image_path=file_name,
            width=int(image["width"]), height=int(image["height"]),
            boxes_xyxy=tuple(boxes), labels=tuple(labels),
            source=str(image.get("source", "")),
        ))
        if max_images is not None and len(records) >= max_images:
            break

    n_empty = sum(1 for r in records if r.n_boxes == 0)
    logger.info("v6 %s: %d images (%d with zero boxes, kept per §4.0), %d boxes, "
                "%d iscrowd annotations dropped (matches DetrDatasetMapper)",
                split, len(records), n_empty, sum(r.n_boxes for r in records), n_crowd)
    if max_images is not None:
        logger.warning("max_images=%d: this is a SMOKE subset and §4.0 forbids "
                       "reporting any result trained on it", max_images)
    return records


def probe_records(split: str = "val", n_images: int = 64) -> list[V6Record]:
    """A fixed, evenly spaced held-out subset for the collapse probe.

    Evenly spaced rather than the first N, for the same reason the Grounding
    DINO trainer gives: the split is ForestPersons followed by
    ForestBelongings, so a prefix is all-person and could not detect a collapse
    across the six V6 words at all.
    """
    records = load_v6_records(split)
    if len(records) < n_images:
        raise DatasetContractError(
            f"{split} split has {len(records)} images, need {n_images}")
    stride = len(records) // n_images
    chosen = [records[i * stride] for i in range(n_images)]
    logger.info("collapse probe subset: %d held-out %s images, stride %d "
                "(fixed across seeds)", len(chosen), split, stride)
    return chosen
