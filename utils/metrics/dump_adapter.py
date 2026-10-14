"""The SINGLE place a detection dump becomes an ``(n, 6)`` xyxy array.

Every other module in this package consumes :class:`ConditionDump` and never
parses a file or converts a box itself. There is exactly one ``xywh -> xyxy``
conversion in the codebase-facing direction (:func:`xywh_to_xyxy`) and exactly
one COCO-record reader (:func:`detections_from_coco_records`); both are pinned
by the regression tests. A second conversion site is how a
half-pixel or a width-vs-x2 error becomes an apparent difference between the
zero-shot and fine-tuned arms.

Three input forms are accepted, and they must produce identical arrays:

1. **``v6_npz.1``** — the campaign format (:func:`load_npz_dump`). A CSR-indexed
   float32 ``(N_det, 6)`` block plus its own ordered ``class_names``.
2. **rev2 JSONL** — what ``utils.runner.detection_dump.DetectionDumpWriter`` writes.
   Rows are already ``[x1, y1, x2, y2, score, class_index]`` in original image
   pixels, taken *after* the evaluation protocol (score threshold 0.0, the cap
   policy, and COCOeval's per-category ``maxDets``). Retained as a selectable
   safety net while the campaign moves over.
3. **COCO-native records** — ``DetectionRecord.as_coco()`` dicts
   (``image_id`` / ``category_id`` / ``bbox`` in xywh / ``score``). Accepted so
   a fragment-side artefact can be analysed without a re-run.

Forms 1 and 2 are pinned to each other by a reader-equivalence test, so a
result computed before the format change and one computed after are comparable
row for row. Note that the float32 storage of form 1 quantises a score that has
no float32 representation; the dump writer counts those rows and this reader
**refuses** a file whose count is non-zero (see :class:`LosslessGuard`).

Why the dump must be post-protocol
----------------------------------
The consistency check ``ΔAR ≈ PFR − NFR`` (see :mod:`.flip`) compares a
detection-set statement (layer 1) with an AP/AR statement (layer 2). It is only
meaningful when both describe the *same* detections. If the dump were taken
before the per-category ``maxDets`` cap, a residual could equally well mean
"the matching logic is wrong" or "the two layers scored different sets", and
the check would lose its diagnostic power entirely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from utils.runner.logging_utils import get_logger
from utils.runner.detection_dump import read_dump
from utils.protocol.vocabulary import CATEGORY_ID, CLASS_BY_ID, V6_CLASSES

logger = get_logger(__name__)

#: Column layout of every array this package passes around. Identical to
#: ``utils.metrics.invariance``' dump contract, which is what makes
#: that module's ``class_detections`` / ``greedy_match`` directly reusable.
COLUMNS: tuple[str, ...] = ("x1", "y1", "x2", "y2", "score", "class_index")

#: Rendered in place of ``None`` when a quantity was not measured. Never ``0``.
UNMEASURED = "_unmeasured_"


class DumpFormatError(ValueError):
    """A dump could not be read as the frozen rev2 contract."""


class DumpPairError(ValueError):
    """Two dumps were about to be compared that do not describe the same run."""


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """COCO ``[x, y, w, h]`` -> ``[x1, y1, x2, y2]``. The only conversion site.

    Args:
        boxes: ``(n, 4)`` array. Never mutated; a new array is returned.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`xywh_to_xyxy`, for round-trip tests and reporting."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


@dataclass(frozen=True)
class ConditionDump:
    """One (arm, condition) detection set, keyed by image id.

    Attributes:
        detections: ``image_id -> (n, 6)`` float array in :data:`COLUMNS`
            order. An image that produced nothing is present with an empty
            ``(0, 6)`` array — absence and silence are different facts and the
            dump writer records the difference deliberately.
        vocabulary: The class names ``class_index`` indexes into.
    """

    condition_id: str
    arm: str
    pool: str
    detections: Mapping[int, np.ndarray]
    vocabulary: tuple[str, ...] = V6_CLASSES
    #: Provenance stamp carried by ``v6_npz.1``. Checked for HOMOGENEITY across
    #: a compared pair, never against a hard-coded value: the stamp covers the
    #: detector module, so it legitimately differs per model, and a table of
    #: expected constants would flag a healthy fragment as drift.
    runner_version: str = ""

    @property
    def image_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.detections))

    def class_index(self, class_name: str) -> int:
        """Exact-name lookup. Never a prefix or first match."""
        if class_name not in self.vocabulary:
            raise KeyError(f"{class_name!r} not in vocabulary {list(self.vocabulary)}")
        return self.vocabulary.index(class_name)

    def get(self, image_id: int) -> np.ndarray:
        """One image's detections, or an empty ``(0, 6)`` array if absent."""
        return self.detections.get(image_id, np.zeros((0, 6), dtype=float))


def _as_array(rows: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(rows, dtype=float).reshape(-1, len(COLUMNS))
    if array.size and (array[:, 2] < array[:, 0]).any():
        raise DumpFormatError(
            "a dumped row has x2 < x1; the rows are probably xywh, but the rev2 "
            "contract is xyxy. Convert once, with xywh_to_xyxy, at load time.")
    return array


def cocoeval_box_geometry(boxes: np.ndarray) -> np.ndarray:
    """Re-derive the box COCOeval actually scored, from a dumped ``xyxy`` box.

    The dump stores ``xyxy``, but the scoring path is
    ``xyxy -> DetectionRecord.bbox_xywh -> as_coco() -> COCOeval``, and
    pycocotools converts back with ``x2' = x + w``. Since ``w = x2 - x1``, what
    COCOeval saw is::

        x2' = x1 + (x2 - x1)        NOT necessarily x2

    In IEEE arithmetic those differ for a few percent of float64 box pairs (by
    ~1e-13 px, which moves an IoU by ~1e-15). That is only enough to flip a
    match when an IoU sits within that distance of 0.5 — vanishingly rare, and
    measured to have flipped nothing on the 14,980-image census. Applying the
    round-trip anyway converts "verified to agree on this data" into "agrees by
    construction", at no cost.

    Precision: the round-trip is done in **float64**, because ``as_coco()``
    calls ``float(v)`` (a Python float) and pycocotools works in float64.
    Performing it in float32 would reproduce a different arithmetic than the
    one that actually ran.

    Note that for a ``v6_npz.1`` dump this is provably the identity: the
    difference of two float32 values is exact in float64, so ``x1 + (x2 - x1)``
    returns ``x2`` bit for bit (verified over 5e6 random pairs, 0 mismatches).
    It bites only for float64-valued dumps such as the JSONL form, where 3.81%
    of independent random pairs differ.

    The input is never mutated; a new array is returned.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    out = boxes.copy()
    out[:, 2] = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0])
    out[:, 3] = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1])
    return out


def _as_scored_rows(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """Dumped rows in the exact geometry COCOeval scored them in.

    Applied to the two DUMP forms (npz, rev2 JSONL), whose boxes never passed
    through ``as_coco()``. It is deliberately NOT applied to the COCO-native
    input form: there the record already carries ``w``, and
    :func:`xywh_to_xyxy` computes ``x1 + w`` — which is exactly what COCOeval
    computes, so re-deriving ``w`` from the result could only move it.
    """
    array = _as_array(rows)
    if not array.size:
        return array
    array[:, :4] = cocoeval_box_geometry(array[:, :4])
    return array


def load_condition_dump(path: Path, expect_condition: str | None = None) -> ConditionDump:
    """Read one rev2 JSONL dump file (``.jsonl`` or ``.jsonl.gz``).

    The file is produced by ``DetectionDumpWriter``, which writes exactly one
    rung per file, so a file holding several rungs, several arms or several
    pools is a merge that already went wrong upstream and is rejected here.
    """
    arms: set[str] = set()
    pools: set[str] = set()
    rungs: set[str] = set()
    detections: dict[int, np.ndarray] = {}
    for row in read_dump(Path(path)):
        image_id = int(row["image_id"])
        if image_id in detections:
            raise DumpFormatError(
                f"{path}: image_id {image_id} appears twice; one row would be "
                "silently discarded")
        prompts = row["prompts"]
        if len(prompts) != 1:
            raise DumpFormatError(
                f"{path}: row for image {image_id} holds {len(prompts)} rungs "
                f"{sorted(prompts)}; one file is one condition")
        (rung, rows), = prompts.items()
        rungs.add(rung)
        arms.add(str(row.get("arm")))
        pools.add(str(row.get("pool")))
        detections[image_id] = _as_scored_rows(rows)

    if not detections:
        raise DumpFormatError(f"{path}: no rows")
    for name, values in (("arm", arms), ("pool", pools), ("rung", rungs)):
        if len(values) > 1:
            raise DumpFormatError(f"{path}: dump spans several {name}s {sorted(values)}")
    condition_id = rungs.pop()
    if expect_condition is not None and condition_id != expect_condition:
        raise DumpFormatError(
            f"{path}: holds condition {condition_id!r}, expected {expect_condition!r}")
    return ConditionDump(condition_id=condition_id, arm=arms.pop(), pool=pools.pop(),
                         detections=detections)


#: Suffixes handled by :func:`load_dump`. ``rev2_jsonl`` remains selectable as a
#: safety net while the campaign moves to ``v6_npz.1``; both readers must
#: produce the same :class:`ConditionDump`, which
#: the regression tests pins.
JSONL_SUFFIXES: tuple[str, ...] = (".jsonl", ".jsonl.gz")
BINARY_SUFFIXES: tuple[str, ...] = (".npz",)

#: The only binary format version this reader claims to understand.
NPZ_FORMAT_VERSION = "v6_npz.1"
#: Scalar string fields carried by a ``v6_npz.1`` file. Read with ``.item()``.
NPZ_SCALAR_FIELDS: tuple[str, ...] = (
    "arm", "pool", "condition_id", "cap_policy", "format_version",
    "runner_version", "meta_json")
#: Array fields carried by a ``v6_npz.1`` file.
NPZ_ARRAY_FIELDS: tuple[str, ...] = ("data", "offsets", "image_ids", "class_names")

#: Metadata keys that may carry the float32 lossless guard. Several spellings
#: are accepted because the guard is written by another line; an unrecognised
#: spelling is reported as "not recorded", never silently read as zero.
LOSSLESS_ROW_KEYS: tuple[str, ...] = (
    "n_rows_not_exact", "dump_inexact_rows",          # spellings observed in v6_npz.1
    "n_imprecise_rows", "imprecise_rows", "n_inexact_rows", "inexact_rows",
    "lossless_n_imprecise_rows", "float32_imprecise_rows")
LOSSLESS_ULP_KEYS: tuple[str, ...] = (
    "max_relative_error", "dump_max_rel_err",         # spellings observed in v6_npz.1
    "max_ulp", "max_ulp_error", "lossless_max_ulp", "float32_max_ulp")


class LosslessGuardError(ValueError):
    """The float32 round-trip guard reported a value that is not exactly zero."""


@dataclass(frozen=True)
class LosslessGuard:
    """The float32 round-trip guard as recorded by the dump writer.

    A non-zero value is not a rounding curiosity. ``greedy_match`` orders by
    descending score, so a perturbed score changes which detections tie and
    therefore which one wins a match — and the transition matrix would be
    silently wrong. Hence :meth:`assert_ok` refuses rather than warns.
    """

    present: bool
    n_imprecise_rows: int | None
    #: The magnitude counter, under whichever name the writer used
    #: (``max_relative_error`` in ``v6_npz.1``, ``max_ulp`` in the original
    #: description). ``error_key`` records which one was found, so a table can
    #: say what it is reporting rather than implying a unit it did not measure.
    max_error: float | None
    row_key: str | None
    error_key: str | None
    source: str

    @property
    def clean(self) -> bool:
        return bool(self.present and not self.n_imprecise_rows and not self.max_error)

    def assert_ok(self, where: str) -> None:
        """Raise unless the guard is present and exactly zero on both counters."""
        if not self.present:
            logger.warning(
                "%s: no float32 lossless guard recorded (looked for %s / %s). "
                "Treating it as NOT RECORDED, never as zero — say so in the table.",
                where, list(LOSSLESS_ROW_KEYS), list(LOSSLESS_ULP_KEYS))
            return
        if self.clean:
            return
        raise LosslessGuardError(
            f"{where}: float32 lossless guard is non-zero ({self.row_key}="
            f"{self.n_imprecise_rows}, {self.error_key}={self.max_error}, "
            f"source={self.source}). Do NOT build a table from this dump. A changed "
            "score changes which detections tie under greedy_match's descending-score "
            "order, so the transition matrix would be silently wrong.")


def _first_present(mapping: Mapping, keys: tuple[str, ...]) -> tuple[str, Any] | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return key, mapping[key]
    return None


def read_lossless_guard(metadata: Mapping, source: str) -> LosslessGuard:
    """Extract the float32 guard from an npz ``meta_json`` or a fragment block.

    Nested one-level dicts are searched too, so a guard written under
    ``{"lossless_guard": {...}}`` is found without the caller knowing where the
    other line chose to put it.
    """
    candidates: list[Mapping] = [metadata]
    candidates += [v for v in metadata.values() if isinstance(v, Mapping)]
    rows = ulp = None
    for block in candidates:
        rows = rows or _first_present(block, LOSSLESS_ROW_KEYS)
        ulp = ulp or _first_present(block, LOSSLESS_ULP_KEYS)
    if rows is None and ulp is None:
        return LosslessGuard(False, None, None, None, None, source)
    return LosslessGuard(
        present=True,
        n_imprecise_rows=(int(rows[1]) if rows else None),
        max_error=(float(ulp[1]) if ulp else None),
        row_key=(rows[0] if rows else None),
        error_key=(ulp[0] if ulp else None),
        source=source,
    )


def runner_hash(runner_version: str) -> str | None:
    """The code hash out of a ``runner_version`` stamp, or ``None`` if absent.

    ``v6_npz.1`` stores a JSON blob whose ``version`` field is the hash and
    which also carries run-local settings (``prefetch_workers``,
    ``prefetch_depth``, ...). Those legitimately differ between two runs of the
    same code, so comparing the blob verbatim would reject a valid pair; only
    the hash is compared. A plain string is treated as the hash itself.
    """
    if not runner_version:
        return None
    try:
        parsed = json.loads(runner_version)
    except (json.JSONDecodeError, TypeError):
        return runner_version
    if isinstance(parsed, Mapping):
        version = parsed.get("version")
        return str(version) if version is not None else None
    return runner_version


def fragment_lossless_guard(fragment: Mapping) -> LosslessGuard:
    """The same guard as recorded on the fragment side (§7 requires both)."""
    return read_lossless_guard(fragment, "fragment")


def load_npz_dump(path: Path, expect_condition: str | None = None,
                  strict_lossless: bool = True) -> ConditionDump:
    """Read a ``v6_npz.1`` dump into the same :class:`ConditionDump` as JSONL.

    Format (frozen 2026-08-10), all arrays top level::

        data        float32 (N_det, 6) C-order  [x1, y1, x2, y2, score, class_index]
        offsets     int64   (N_img + 1,)        CSR; offsets[0] == 0
        image_ids   int64   (N_img,)            census order
        class_names <U      (n_classes,)        ORDERED
        arm / pool / condition_id / cap_policy / format_version /
        runner_version / meta_json               <U scalars, read with .item()

    Three conventions are enforced rather than assumed:

    1. Boxes are ``xyxy`` in original image pixels — validated by the shared
       ``x2 >= x1`` check, so a transposed axis order fails loudly.
    2. ``class_index`` indexes **this file's own** ``class_names``, which
       becomes the returned dump's vocabulary. :data:`V6_CLASSES` is
       deliberately not used: were its order ever to change, an old dump would
       otherwise be silently relabelled.
    3. An image with no detections is a real row (``offsets[i] == offsets[i+1]``)
       and is kept as an empty ``(0, 6)`` array, so silence stays distinct from
       absence.

    Args:
        strict_lossless: Refuse the file when its float32 guard is non-zero.
            Only set ``False`` to inspect a flagged dump, never to report one.
    """
    with np.load(path, allow_pickle=False) as archive:
        missing = [k for k in NPZ_ARRAY_FIELDS + NPZ_SCALAR_FIELDS
                   if k not in archive.files]
        if missing:
            raise DumpFormatError(
                f"{path}: not a {NPZ_FORMAT_VERSION} dump; missing {missing}. "
                f"Present: {sorted(archive.files)}")
        scalars = {k: str(archive[k].item()) for k in NPZ_SCALAR_FIELDS}
        data = np.asarray(archive["data"])
        offsets = np.asarray(archive["offsets"])
        image_ids = np.asarray(archive["image_ids"])
        class_names = tuple(str(c) for c in archive["class_names"].tolist())

    if scalars["format_version"] != NPZ_FORMAT_VERSION:
        raise DumpFormatError(
            f"{path}: format_version {scalars['format_version']!r}, this reader "
            f"implements {NPZ_FORMAT_VERSION!r}. Refusing to guess at a revision.")

    guard = read_lossless_guard(json.loads(scalars["meta_json"] or "{}"), str(path))
    if strict_lossless:
        guard.assert_ok(str(path))

    if data.ndim != 2 or data.shape[1] != len(COLUMNS):
        raise DumpFormatError(
            f"{path}: data has shape {data.shape}, expected (N_det, {len(COLUMNS)}) "
            f"in {list(COLUMNS)} order")
    if len(offsets) != len(image_ids) + 1:
        raise DumpFormatError(
            f"{path}: offsets has {len(offsets)} entries for {len(image_ids)} images; "
            "CSR requires N_img + 1")
    if len(offsets) and (int(offsets[0]) != 0 or int(offsets[-1]) != len(data)):
        raise DumpFormatError(
            f"{path}: CSR offsets run {int(offsets[0])}..{int(offsets[-1])} but data "
            f"holds {len(data)} rows; the index does not describe this array")
    if np.any(np.diff(offsets) < 0):
        raise DumpFormatError(f"{path}: CSR offsets are not non-decreasing")
    if len(set(image_ids.tolist())) != len(image_ids):
        raise DumpFormatError(f"{path}: duplicate image ids; one row would be lost")

    condition_id = scalars["condition_id"]
    if expect_condition is not None and condition_id != expect_condition:
        raise DumpFormatError(
            f"{path}: holds condition {condition_id!r}, expected {expect_condition!r}")

    # float32 -> float64 is exact, so this widening cannot move a score and
    # cannot change a tie under greedy_match.
    detections = {
        int(image_id): _as_scored_rows(
            data[int(offsets[i]):int(offsets[i + 1])].astype(np.float64))
        for i, image_id in enumerate(image_ids.tolist())
    }
    logger.info("read %s: %d images, %d detections, vocabulary %s, cap_policy %s",
                path, len(detections), len(data), list(class_names),
                scalars["cap_policy"])
    return ConditionDump(condition_id=condition_id, arm=scalars["arm"],
                         pool=scalars["pool"], detections=detections,
                         vocabulary=class_names,
                         runner_version=scalars["runner_version"])


def load_binary_dump(path: Path, expect_condition: str | None = None) -> ConditionDump:
    """Alias kept for callers written against the pre-``v6_npz.1`` name."""
    return load_npz_dump(path, expect_condition)


def load_dump(path: Path, expect_condition: str | None = None) -> ConditionDump:
    """Load a dump of either supported format, dispatching on the suffix."""
    path = Path(path)
    name = path.name
    if any(name.endswith(s) for s in JSONL_SUFFIXES):
        return load_condition_dump(path, expect_condition)
    if any(name.endswith(s) for s in BINARY_SUFFIXES):
        return load_binary_dump(path, expect_condition)
    raise DumpFormatError(
        f"{path}: unrecognised dump suffix. Known: {list(JSONL_SUFFIXES + BINARY_SUFFIXES)}. "
        "A '.partial' file is a run still in flight and must never be read as data.")


def detections_from_coco_records(
    records: Iterable[Mapping],
    image_ids: Sequence[int],
    vocabulary: tuple[str, ...] = V6_CLASSES,
) -> dict[int, np.ndarray]:
    """COCO-native detection dicts -> the ``(n, 6)`` xyxy form.

    Args:
        records: ``{"image_id", "category_id", "bbox" (xywh), "score"}`` dicts,
            exactly as ``DetectionRecord.as_coco()`` emits them.
        image_ids: The census. Every id gets an entry, empty where the model
            returned nothing, so "no row" can never be confused with "no
            detection".
    """
    per_image: dict[int, list[list[float]]] = {int(i): [] for i in image_ids}
    for record in records:
        image_id = int(record["image_id"])
        if image_id not in per_image:
            raise DumpFormatError(
                f"detection for image {image_id}, which is not in the census of "
                f"{len(per_image)} images")
        name = CLASS_BY_ID[int(record["category_id"])]
        if name not in vocabulary:
            raise DumpFormatError(f"category {record['category_id']} -> {name!r} "
                                  f"is outside the vocabulary {list(vocabulary)}")
        x1, y1, x2, y2 = xywh_to_xyxy(np.asarray([record["bbox"]], dtype=float))[0]
        per_image[image_id].append(
            [x1, y1, x2, y2, float(record["score"]), float(vocabulary.index(name))])
    return {i: _as_array(rows) for i, rows in per_image.items()}


def condition_dump_from_coco(
    condition_id: str, arm: str, pool: str,
    records: Iterable[Mapping], image_ids: Sequence[int],
    vocabulary: tuple[str, ...] = V6_CLASSES,
) -> ConditionDump:
    """:class:`ConditionDump` from COCO-native records (form 2 of the contract)."""
    return ConditionDump(
        condition_id=condition_id, arm=arm, pool=pool, vocabulary=vocabulary,
        detections=detections_from_coco_records(records, image_ids, vocabulary))


@dataclass(frozen=True)
class GroundTruth:
    """Per-image ground truth for one class, plus the ignore regions.

    ``ignore`` holds ``iscrowd`` annotations (which on this dataset are the
    out-of-vocabulary ``etc`` belongings, category 7). A detection landing on
    an ignore region is neither a true nor a false positive; counting it as an
    FP would inflate every false-alarm number by an artefact of the annotation
    policy.
    """

    class_name: str
    boxes: Mapping[int, np.ndarray]
    ignore: Mapping[int, np.ndarray]

    @property
    def n_objects(self) -> int:
        return int(sum(len(v) for v in self.boxes.values()))

    def get(self, image_id: int) -> np.ndarray:
        return self.boxes.get(image_id, np.zeros((0, 4), dtype=float))

    def get_ignore(self, image_id: int) -> np.ndarray:
        return self.ignore.get(image_id, np.zeros((0, 4), dtype=float))


def load_ground_truth(coco_json: Mapping | Path, class_name: str,
                      image_ids: Sequence[int] | None = None) -> GroundTruth:
    """Ground-truth boxes of one V6 class, in xyxy, from a COCO annotation file.

    Args:
        coco_json: A parsed COCO dict or a path to one.
        image_ids: Restrict to this census. ``None`` keeps every image in the
            file.

    The conversion goes through :func:`xywh_to_xyxy` like everything else.
    """
    data = (json.loads(Path(coco_json).read_text())
            if isinstance(coco_json, (str, Path)) else coco_json)
    category_id = CATEGORY_ID[class_name]
    wanted = None if image_ids is None else {int(i) for i in image_ids}
    boxes: dict[int, list[list[float]]] = {}
    ignore: dict[int, list[list[float]]] = {}
    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        if wanted is not None and image_id not in wanted:
            continue
        box = xywh_to_xyxy(np.asarray([ann["bbox"]], dtype=float))[0].tolist()
        if ann.get("iscrowd"):
            ignore.setdefault(image_id, []).append(box)
        elif int(ann["category_id"]) == category_id:
            boxes.setdefault(image_id, []).append(box)
    ids = sorted(wanted) if wanted is not None else sorted(
        {int(im["id"]) for im in data.get("images", [])} | set(boxes) | set(ignore))
    return GroundTruth(
        class_name=class_name,
        boxes={i: np.asarray(boxes.get(i, []), dtype=float).reshape(-1, 4) for i in ids},
        ignore={i: np.asarray(ignore.get(i, []), dtype=float).reshape(-1, 4) for i in ids},
    )


def load_ground_truth_by_class(coco_json: Mapping | Path, class_names: Sequence[str],
                               image_ids: Sequence[int] | None = None
                               ) -> dict[str, GroundTruth]:
    """:func:`load_ground_truth` for several classes, parsing the file once."""
    data = (json.loads(Path(coco_json).read_text())
            if isinstance(coco_json, (str, Path)) else coco_json)
    return {name: load_ground_truth(data, name, image_ids) for name in class_names}


def assert_comparable(a: ConditionDump, b: ConditionDump) -> tuple[int, ...]:
    """Return the shared image ids, refusing pairs that are not comparable.

    Comparing two arms, or two pools, would attribute a checkpoint or census
    difference to the query wording. Both are rejected rather than warned
    about: the resulting numbers look entirely plausible.
    """
    if a.arm != b.arm:
        raise DumpPairError(
            f"dumps come from different arms ({a.arm!r} vs {b.arm!r}); the "
            "difference would be a checkpoint effect, not a query effect")
    if a.pool != b.pool:
        raise DumpPairError(
            f"dumps come from different pools ({a.pool!r} vs {b.pool!r}); the "
            "image sets differ, so nothing is paired")
    if a.condition_id == b.condition_id:
        raise DumpPairError(
            f"both dumps are condition {a.condition_id!r}; there is no contrast")
    if a.vocabulary != b.vocabulary:
        raise DumpPairError("dumps use different vocabularies; class_index would "
                            "mean different things in the two arrays")
    version_a, version_b = runner_hash(a.runner_version), runner_hash(b.runner_version)
    if version_a and version_b and version_a != version_b:
        raise DumpPairError(
            f"dumps carry different runner code hashes ({version_a!r} vs {version_b!r}); "
            "the two conditions came out of different code. This is a HOMOGENEITY "
            "check within one comparison — the hash covers the detector module and so "
            "legitimately differs BETWEEN MODELS (gdino/owlv2, glip and omdet each "
            "have their own), and is therefore never compared against a hard-coded "
            "expected value.")
    ids_a, ids_b = set(a.detections), set(b.detections)
    if ids_a != ids_b:
        only_a, only_b = sorted(ids_a - ids_b)[:5], sorted(ids_b - ids_a)[:5]
        raise DumpPairError(
            f"dumps cover different images: {len(ids_a)} vs {len(ids_b)} "
            f"(only in {a.condition_id}: {only_a}..., only in {b.condition_id}: "
            f"{only_b}...). A per-object flip analysis needs the same census.")
    return tuple(sorted(ids_a))


def unmeasured(value: float | None) -> float | str:
    """``value``, or the :data:`UNMEASURED` sentinel — never a substituted 0."""
    return UNMEASURED if value is None else value
