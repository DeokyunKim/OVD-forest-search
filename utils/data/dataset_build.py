"""Build the V6 ForestPersons + ForestBelongings COCO splits.

Spec: the vocabulary specification §5 (data policy),
the frozen-hyperparameter specification §4.0.

Policy enforced here (not optional):

- **No image is removed.** Zero-annotation negatives are carried through
  verbatim -- class balancing by image removal and difficulty filtering are
  both forbidden by §4.0. The negative count is reported, not acted on.
- **No subsampling.** Every image of every source split is emitted.
- ForestPersonsIR is not a source (user decision 3, 2026-08-08).

Every emitted file is checked by :func:`verify_v6_json` before it is written,
so a mapping regression cannot reach disk.
"""
from __future__ import annotations

from utils.runner.paths import paths

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from utils.runner.logging_utils import get_logger
from utils.protocol.vocabulary import (
    CATEGORY_ID,
    CLASS_BY_ID,
    ETC_CATEGORY_ID,
    ETC_CATEGORY_NAME,
    FB_RENAME,
    FORBIDDEN_CATEGORY_NAMES,
    FP_POSE_SUBCLASSES,
    V6_CLASSES,
    build_categories,
)

logger = get_logger(__name__)

PERSON_SUPERCATEGORY = "person"


class UnexpectedCategoryError(ValueError):
    """A source json holds a category this builder refuses to guess about."""


class VerificationError(ValueError):
    """A built json failed the V6 verification gate."""


@dataclass(frozen=True)
class SourceSpec:
    """One source split to merge."""

    tag: str
    json_path: Path
    images_root: Path
    kind: str  # "forest_persons" | "forest_belongings"


@dataclass(frozen=True)
class SplitReport:
    """Per-split census, written verbatim into the statistics document."""

    split: str
    n_images: int
    n_annotations: int
    n_negative_images: int
    class_distribution: Mapping[str, int]
    source_image_counts: Mapping[str, int]
    source_annotation_counts: Mapping[str, int]
    #: Source-category name -> count, before any renaming/collapsing.
    pre_mapping_distribution: Mapping[str, int]
    n_iscrowd: int


def _ann_record(image_id: int, category_id: int, ann: Mapping, iscrowd: int) -> dict:
    bbox = [float(v) for v in ann["bbox"]]
    return {
        "id": 0,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": float(ann.get("area", bbox[2] * bbox[3])),
        "iscrowd": iscrowd,
    }


def _image_record(new_id: int, im: Mapping, images_root: Path, source: str) -> dict:
    return {
        "id": new_id,
        "file_name": str((images_root / im["file_name"]).resolve()),
        "width": im["width"],
        "height": im["height"],
        "orig_id": im["id"],
        "source": source,
    }


def load_forest_persons(path: Path) -> tuple[list[dict], dict[int, list[dict]], Counter]:
    """All 13 pose/visibility subclasses collapse to ``person``.

    Raises:
        UnexpectedCategoryError: if a category is not a known pose subclass
            or carries a supercategory other than ``person``, or if an
            annotation references an id absent from ``categories``.
    """
    data = json.loads(path.read_text())
    name_by_id = {c["id"]: c["name"] for c in data["categories"]}
    unknown = set(name_by_id.values()) - set(FP_POSE_SUBCLASSES)
    if unknown:
        raise UnexpectedCategoryError(
            f"{path}: category names outside the known pose subclasses: {sorted(unknown)}"
        )
    for cat in data["categories"]:
        if cat.get("supercategory", PERSON_SUPERCATEGORY) != PERSON_SUPERCATEGORY:
            raise UnexpectedCategoryError(f"{path}: {cat!r} has supercategory != 'person'")

    pre: Counter = Counter()
    anns_by_img: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid not in name_by_id:
            raise UnexpectedCategoryError(f"{path}: annotation references unknown category {cid!r}")
        pre[name_by_id[cid]] += 1
        record = _ann_record(0, CATEGORY_ID["person"], ann, iscrowd=int(ann.get("iscrowd", 0)))
        anns_by_img.setdefault(ann["image_id"], []).append(record)
    return data["images"], anns_by_img, pre


def load_forest_belongings(path: Path) -> tuple[list[dict], dict[int, list[dict]], Counter]:
    """Singular belongings names become plurals; ``etc`` becomes an ignore region.

    ``iscrowd`` already present on a source annotation is preserved (the FB
    dataset uses it for very small boxes); ``etc`` is forced to
    ``iscrowd=1`` under :data:`ETC_CATEGORY_ID`.
    """
    data = json.loads(path.read_text())
    name_by_id = {c["id"]: c["name"] for c in data["categories"]}
    known = set(V6_CLASSES) | set(FB_RENAME) | {ETC_CATEGORY_NAME}
    unknown = set(name_by_id.values()) - known
    if unknown:
        raise UnexpectedCategoryError(f"{path}: unmapped categories {sorted(unknown)}")

    pre: Counter = Counter()
    anns_by_img: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid not in name_by_id:
            raise UnexpectedCategoryError(f"{path}: annotation references unknown category {cid!r}")
        name = name_by_id[cid]
        pre[name] += 1
        if name == ETC_CATEGORY_NAME:
            record = _ann_record(0, ETC_CATEGORY_ID, ann, iscrowd=1)
        else:
            v6_name = FB_RENAME.get(name, name)
            record = _ann_record(0, CATEGORY_ID[v6_name], ann, iscrowd=int(ann.get("iscrowd", 0)))
        anns_by_img.setdefault(ann["image_id"], []).append(record)
    return data["images"], anns_by_img, pre


_LOADERS = {
    "forest_persons": load_forest_persons,
    "forest_belongings": load_forest_belongings,
}


def build_split(split: str, sources: Sequence[SourceSpec]) -> tuple[dict, SplitReport]:
    """Merge ``sources`` into one V6 COCO json. Every image is carried through."""
    images_out: list[dict] = []
    anns_out: list[dict] = []
    dist: Counter = Counter()
    pre_total: Counter = Counter()
    src_images: Counter = Counter()
    src_anns: Counter = Counter()
    names = {**CLASS_BY_ID, ETC_CATEGORY_ID: ETC_CATEGORY_NAME}

    for spec in sources:
        loader = _LOADERS.get(spec.kind)
        if loader is None:
            raise ValueError(f"unknown source kind {spec.kind!r}")
        images, anns_by_img, pre = loader(spec.json_path)
        pre_total.update({f"{spec.tag}:{k}": v for k, v in pre.items()})
        for im in images:
            new_id = len(images_out) + 1
            for ann in anns_by_img.get(im["id"], []):
                ann["id"] = len(anns_out) + 1
                ann["image_id"] = new_id
                anns_out.append(ann)
                dist[names[ann["category_id"]]] += 1
                src_anns[spec.tag] += 1
            images_out.append(_image_record(new_id, im, spec.images_root, spec.tag))
        src_images[spec.tag] += len(images)
        logger.info("%s/%s: %d images, running total %d", split, spec.tag, len(images), len(images_out))

    data = {"images": images_out, "annotations": anns_out, "categories": build_categories()}
    with_ann = {a["image_id"] for a in anns_out}
    report = SplitReport(
        split=split,
        n_images=len(images_out),
        n_annotations=len(anns_out),
        n_negative_images=sum(1 for im in images_out if im["id"] not in with_ann),
        class_distribution=dict(dist),
        source_image_counts=dict(src_images),
        source_annotation_counts=dict(src_anns),
        pre_mapping_distribution=dict(pre_total),
        n_iscrowd=sum(1 for a in anns_out if a["iscrowd"]),
    )
    return data, report


def verify_v6_json(data: Mapping, expected_images: int | None = None) -> None:
    """Fail loudly if a built json is not a valid V6 set.

    Checks the exact failure modes the Phase 0 brief calls out: a surviving
    singular name (``cloth``/``glove``/``shoe``) or a surviving pose subclass
    in ``categories``, plus id/name agreement and the ``etc`` ignore contract.
    """
    cats = list(data["categories"])
    names = [c["name"] for c in cats]
    leaked = sorted(set(names) & FORBIDDEN_CATEGORY_NAMES)
    if leaked:
        raise VerificationError(f"forbidden category names survived the mapping: {leaked}")

    expected = {**{CATEGORY_ID[n]: n for n in V6_CLASSES}, ETC_CATEGORY_ID: ETC_CATEGORY_NAME}
    actual = {c["id"]: c["name"] for c in cats}
    if actual != expected:
        raise VerificationError(f"categories block is {actual}, expected {expected}")

    valid_ids = set(expected)
    bad = {a["category_id"] for a in data["annotations"]} - valid_ids
    if bad:
        raise VerificationError(f"annotations reference category ids outside V6: {sorted(bad)}")

    not_ignored = [a for a in data["annotations"]
                   if a["category_id"] == ETC_CATEGORY_ID and not a["iscrowd"]]
    if not_ignored:
        raise VerificationError(
            f"{len(not_ignored)} 'etc' annotations are not iscrowd=1 ignore regions"
        )

    image_ids = [im["id"] for im in data["images"]]
    if len(set(image_ids)) != len(image_ids):
        raise VerificationError("duplicate image ids")
    dangling = {a["image_id"] for a in data["annotations"]} - set(image_ids)
    if dangling:
        raise VerificationError(f"{len(dangling)} annotations point at missing images")

    if expected_images is not None and len(image_ids) != expected_images:
        raise VerificationError(
            f"census mismatch: {len(image_ids)} images, expected {expected_images}"
        )


def write_json(data: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


# --- Production source paths -------------------------------------------------
REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / 'pyproject.toml').is_file())
# Both corpora are located through configs/paths.yaml, so a reader points the
# config (or UC_DATA_ROOT) at wherever they unpacked them.
#
# Resolved when a split is built, not when this module is imported. `require`
# raises on a missing corpus, which is right when someone asks to build the
# benchmark and wrong when they ask `prepare.py --help`: a tool that cannot
# print its own usage until the data has arrived is one whose usage nobody
# reads. The raise still happens before any annotation is opened.
def out_dir() -> Path:
    """Where the built benchmark lands. Through the config, not a literal: the
    evaluation side was routed this way already, and leaving the training side
    on a hard path is how a repository evaluates fine and cannot train."""
    return paths().dataset("benchmark")

#: ForestPersonsIR is deliberately absent (user decision 3, 2026-08-08).
def sources() -> Mapping[str, tuple[SourceSpec, ...]]:
    """The two corpora's annotation and image roots, per split."""
    fp = paths().require("forest_persons")
    fb = paths().require("forest_belongings")
    return {
    "train": (
        SourceSpec("fp", fp / "annotations" / "train.json", fp / "images", "forest_persons"),
        SourceSpec("fb", fb / "annotations" / "train.json", fb / "images", "forest_belongings"),
    ),
    "val": (
        SourceSpec("fp", fp / "annotations" / "val.json", fp / "images", "forest_persons"),
        SourceSpec("fb", fb / "annotations" / "val.json", fb / "images", "forest_belongings"),
    ),
    "test": (
        SourceSpec("fp", fp / "annotations" / "test.json", fp / "images", "forest_persons"),
        SourceSpec("fb", fb / "annotations" / "test.json", fb / "images", "forest_belongings"),
    ),
}

#: Census gates asserted at build time so a source swap cannot silently move
#: the denominator. train: spec §0-1; test: spec §5 (14,980).
EXPECTED_IMAGES: Mapping[str, int] = {"train": 90_670, "val": 22_729, "test": 14_980}
