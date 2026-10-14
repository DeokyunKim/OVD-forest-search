"""Register the V6 ForestPersons + ForestBelongings splits with detectron2.

Spec: the frozen-hyperparameter specification §4.0 (data policy) and
``utils/protocol/train_caption.py`` (the class-name order that becomes the caption).

Three things are enforced here rather than assumed, because each has already
gone wrong somewhere in this project's history:

1. **``file_name`` is already absolute.** ``utils/data/dataset_build.py`` resolves
   every path at build time, so the detectron2 ``image_root`` must be ``"/"``.
   Passing a real root would silently produce ``"<root>" + "/data" + "/..."`` and every
   image read would fail *at the first training step*, not at registration.
   :func:`assert_absolute_file_names` checks the json instead.

2. **The contiguous class order is the caption order.** detectron2 maps COCO
   category ids to contiguous ids by *sorted id*, and ``gt_classes`` indexes
   directly into the caption's ``.``-separated terms
   (``groundingdino_dt.py::recover_to_cls_logits``). If those two orders ever
   disagree the model trains ``person`` boxes against the word ``clothes`` and
   nothing errors. :func:`assert_class_order` compares the registered order to
   :func:`utils.protocol.train_caption.training_class_names` element by element.

3. **``filter_empty`` is False.** §4.0 forbids removing images to balance
   classes; the V6 train split deliberately carries zero-annotation negatives.
   :func:`load_v6_dataset_dicts` hard-codes ``filter_empty=False`` so a caller
   cannot pass the detectron2 default (``True``) by omission.

``etc`` (category id 7) stays in the json as an ``iscrowd=1`` ignore region.
It is *not* a caption term: the external repo's ``DetrDatasetMapper`` drops
every ``iscrowd != 0`` annotation, so ``etc`` boxes never become supervision
and never become background either. :func:`assert_class_order` asserts exactly
this shape (six trained names, then ``etc``).
"""
from __future__ import annotations

from utils.runner.paths import paths

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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

#: ``file_name`` is absolute in the built json, so detectron2's image root is
#: the filesystem root. This is the same convention the external repo's own
#: ``forest_joint_fpfbir_v3_6class.py`` uses.
IMAGES_ROOT = "/"

SPLITS: tuple[str, ...] = ("train", "val", "test")

#: Registered dataset names. Prefixed so they cannot collide with the external
#: repo's ``forest_joint_*`` registrations, which point at different json files.
DATASET_NAME_TEMPLATE = "v6_fp_fb_{split}"


class DatasetRegistrationError(RuntimeError):
    """The dataset on disk is not the dataset the trainer expects."""


@dataclass(frozen=True)
class V6DatasetInfo:
    """What was registered, verified against the json that produced it."""

    name: str
    split: str
    json_path: Path
    image_root: str
    n_images: int
    n_annotations: int
    #: Contiguous class index -> class name, in detectron2's own order.
    thing_classes: tuple[str, ...]
    #: The six names that become caption terms, in caption order.
    trained_class_names: tuple[str, ...]

    def describe(self) -> str:
        return (f"{self.name}: {self.n_images} images, {self.n_annotations} "
                f"annotations, classes {list(self.thing_classes)} "
                f"(trained: {list(self.trained_class_names)})")


def split_json_path(split: str) -> Path:
    if split not in SPLITS:
        raise DatasetRegistrationError(
            f"unknown split {split!r}; expected one of {list(SPLITS)}")
    return V6_DATASET_DIR / f"{split}.json"


def dataset_name(split: str) -> str:
    if split not in SPLITS:
        raise DatasetRegistrationError(
            f"unknown split {split!r}; expected one of {list(SPLITS)}")
    return DATASET_NAME_TEMPLATE.format(split=split)


def assert_absolute_file_names(images: Sequence[Mapping], json_path: Path,
                               sample: int = 200) -> None:
    """Refuse a json whose ``file_name`` is relative.

    ``image_root="/"`` only works because the builder resolved every path. A
    relative name would be joined onto ``"/"`` and the failure would surface as
    a missing-file error inside a dataloader worker, thousands of steps into a
    run.
    """
    if not images:
        raise DatasetRegistrationError(f"{json_path} has no images")
    step = max(1, len(images) // sample)
    for image in list(images)[::step]:
        name = image.get("file_name")
        if not name or not str(name).startswith("/"):
            raise DatasetRegistrationError(
                f"{json_path}: file_name {name!r} is not absolute, but the V6 "
                f"registration uses image_root={IMAGES_ROOT!r}. Rebuild with "
                f"utils/data/dataset_build.py, which resolves paths at build time")


def assert_class_order(thing_classes: Sequence[str],
                       dataset_id_to_contiguous_id: Mapping[int, int]) -> None:
    """The contiguous order must be the caption order, then ``etc``.

    ``gt_classes`` (a contiguous index) is used directly as a column index into
    the per-term logits the caption produces, so a permutation here silently
    swaps supervision between words.
    """
    expected = training_class_names()
    prefix = list(thing_classes)[:len(expected)]
    if prefix != expected:
        raise DatasetRegistrationError(
            f"registered class order {prefix} != training caption order "
            f"{expected}. gt_classes indexes the caption's terms directly, so "
            f"this would train each class against the wrong word")

    trailing = list(thing_classes)[len(expected):]
    if trailing not in ([], [ETC_CATEGORY_NAME]):
        raise DatasetRegistrationError(
            f"unexpected extra categories {trailing}; only "
            f"{ETC_CATEGORY_NAME!r} (an iscrowd=1 ignore region) may follow "
            f"the six trained classes")

    for name, expected_id in zip(expected, range(1, len(expected) + 1)):
        contiguous = dataset_id_to_contiguous_id.get(expected_id)
        if contiguous != expected.index(name):
            raise DatasetRegistrationError(
                f"category id {expected_id} ({name!r}) maps to contiguous id "
                f"{contiguous}, expected {expected.index(name)}")
    if trailing:
        etc_contiguous = dataset_id_to_contiguous_id.get(ETC_CATEGORY_ID)
        if etc_contiguous != len(expected):
            raise DatasetRegistrationError(
                f"{ETC_CATEGORY_NAME!r} maps to contiguous id {etc_contiguous}, "
                f"expected {len(expected)}")


def register_v6_split(split: str, name: str | None = None) -> V6DatasetInfo:
    """Register one V6 split and verify it. Idempotent.

    Returns:
        A :class:`V6DatasetInfo` built from detectron2's own metadata, not from
        the arguments — so a registration that silently resolved to a different
        json is visible in the returned record.
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    json_path = split_json_path(split)
    if not json_path.is_file():
        raise DatasetRegistrationError(
            f"{json_path} is missing; build it with utils/data/dataset_build.py")

    payload = json.loads(json_path.read_text())
    assert_absolute_file_names(payload["images"], json_path)

    registered = name or dataset_name(split)
    if registered not in DatasetCatalog.list():
        register_coco_instances(registered, {}, str(json_path), IMAGES_ROOT)

    metadata = MetadataCatalog.get(registered)
    if not hasattr(metadata, "thing_classes"):
        # register_coco_instances defers metadata to the first load; force it.
        DatasetCatalog.get(registered)
        metadata = MetadataCatalog.get(registered)

    if Path(metadata.json_file) != json_path:
        raise DatasetRegistrationError(
            f"dataset {registered!r} is already registered against "
            f"{metadata.json_file}, not {json_path}")

    thing_classes = tuple(metadata.thing_classes)
    assert_class_order(thing_classes, metadata.thing_dataset_id_to_contiguous_id)

    info = V6DatasetInfo(
        name=registered,
        split=split,
        json_path=json_path,
        image_root=IMAGES_ROOT,
        n_images=len(payload["images"]),
        n_annotations=len(payload["annotations"]),
        thing_classes=thing_classes,
        trained_class_names=tuple(training_class_names()),
    )
    logger.info("registered %s", info.describe())
    return info


def load_v6_dataset_dicts(split: str, name: str | None = None) -> list[dict]:
    """Dataset dicts for ``split`` with ``filter_empty`` forced to False.

    §4.0 forbids removing images. detectron2's ``get_detection_dataset_dicts``
    defaults ``filter_empty=True``, which would silently drop every
    zero-annotation negative — the exact filtering the spec bans — so the flag
    is not exposed as a parameter here.
    """
    from detectron2.data import get_detection_dataset_dicts

    info = register_v6_split(split, name=name)
    dicts = get_detection_dataset_dicts(names=info.name, filter_empty=False)
    if len(dicts) != info.n_images:
        raise DatasetRegistrationError(
            f"{info.name}: loaded {len(dicts)} dataset dicts from a json with "
            f"{info.n_images} images — images were filtered out, which spec "
            f"§4.0 forbids")
    return dicts
