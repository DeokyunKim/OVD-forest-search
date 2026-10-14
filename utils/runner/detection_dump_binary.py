"""The scored detections, as a compact binary (npz) census dump.

Why binary and not compressed JSONL
-----------------------------------
The JSONL dump costs 102-103 bytes per detection (measured across three models):
a detection is six numbers, and JSON spends most of that on the decimal
expansion of a ``float32`` that was widened to ``float64`` on the way out
(``0.6576232314109802`` for a value the model only ever held to 24 bits). gzip
on that text recovers only 2.78x (measured), which is why the fix is the format
and not the compressor -- the campaign would still be ~100 GB on a filesystem
that has to hold training checkpoints too.

Storing the same six numbers as ``float32`` costs 24 B/detection, and deflating
those arrays takes it to ~17-19 B/detection: 280 GB of JSONL becomes ~47 GB. It
is *lossless*, because every number in the dump originates as a ``float32``
model output. A ``float64 -> float32 -> float64`` round trip changed zero values
over 400,000 real detections sampled from the live dump, and again over
4,012,783 detections replayed here through this writer, for boxes and scores
alike.

That "originates as float32" is an assumption about the detector, not a law, so
this writer does not merely assume it. Unless ``verify_lossless=False`` it
checks the round trip of every row it writes, on the writer thread, and reports
the outcome in the dump's own metadata and in the fragment's dump record:
``n_rows_not_exact`` and ``max_relative_error``. A campaign can therefore be
audited for losslessness after the fact instead of being trusted.

Two severities, deliberately different:

* a row that rounds by a float32 ulp (``rel <= 1e-6``) is **counted, not
  fatal**. It means some value reached the dump as a Python/float64 literal --
  e.g. a post-processing multiply done in Python -- and it moves a coordinate
  by ~1e-7 relative, which cannot change which detections were scored. Killing
  a half-hour census run over that would be the instrument damaging the
  experiment.
* an error above :data:`LOSSLESS_TOLERANCE`, or a value that becomes non-finite,
  **fails the run**. That is not rounding, it is the wrong dtype (overflow to
  ``inf``), and a dump built on it is not the set COCOeval scored.

Ragged data without ragged arrays
---------------------------------
Detections per image vary (0..100 per category). Instead of an object array or
one array per image, the file is CSR-shaped: one flat ``data`` matrix plus an
``offsets`` index, so image ``i``'s detections are ``data[offsets[i]:offsets[i+1]]``
-- a slice, i.e. a view, no copying and no per-image Python object.

Off the critical path
---------------------
``utils/runner/prefetch.py`` exists because the evaluation loop was leaving the GPU
idle behind CPU work. Serialising 100 detections per image on the same thread
would hand part of that back, so :class:`BinaryDetectionDumpWriter.add` only
puts a reference on a bounded queue; the per-image work (the COCOeval maxDets
replication, the float conversion, the lossless check) runs on a writer thread.
The file itself is assembled and written once, in :meth:`commit`, after the
loop has finished -- at 24 bytes per detection a full condition is 36-61 MB in
memory, so holding it is cheaper than streaming it.

File layout (format ``v6_npz.1``)
---------------------------------
One ``.npz`` per condition, named ``<arm>__<condition_id>.npz`` where ``arm`` is
``<model>__<checkpoint>``. Arrays, all top level:

===================  ==========  ====================  ============================
key                  dtype       shape                 meaning
===================  ==========  ====================  ============================
``data``             float32     ``(N_det, 6)``        ``[x1, y1, x2, y2, score,
                                                       class_index]``, C order
``offsets``          int64       ``(N_img + 1,)``      CSR index, ``offsets[0]==0``
``image_ids``        int64       ``(N_img,)``          COCO image ids, dump order
``class_names``      ``<U``      ``(n_classes,)``      ordered vocabulary
``arm``              ``<U``      ``()``                ``<model>__<fragment label>``
``pool``             ``<U``      ``()``                image set the dump covers
``condition_id``     ``<U``      ``()``                the rung
``cap_policy``       ``<U``      ``()``                which cap produced the set
``format_version``   ``<U``      ``()``                ``"v6_npz.1"``
``runner_version``   ``<U``      ``()``                JSON of the fragment block
``meta_json``        ``<U``      ``()``                JSON of all of the above
===================  ==========  ====================  ============================

``class_index`` (column 5 of ``data``, stored as a float32 that is exact for the
integers involved) is an index **into this file's own ``class_names``**, not
into any module-level constant. Readers must map through ``class_names`` as
read from the file; a future reordering of :data:`utils.protocol.vocabulary.V6_CLASSES`
then cannot silently relabel an old dump. Boxes are ``xyxy`` in original image
pixels. Column order is fixed by :data:`DUMP_COLUMNS`.

Row order within an image is the order the runner submitted to COCOeval, minus
the detections COCOeval's per-category ``maxDets`` drops
(:func:`utils.runner.detection_dump.select_scored_detections`). Image order is census
order. Images with no detections are present in ``image_ids`` with
``offsets[i] == offsets[i+1]``: absent and empty mean different things
downstream, so silence is recorded rather than skipped.
"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from utils.runner.logging_utils import get_logger
from utils.runner.detection_dump import DUMP_COLUMNS, select_scored_detections
from utils.protocol.vocabulary import V6_CLASSES

logger = get_logger(__name__)

#: Written into every file and into the fragment's ``runner_version``. Bump on
#: any change a reader would have to know about.
DUMP_FORMAT_NPZ: str = "v6_npz.1"

#: The JSONL format this replaces, kept as a selectable value so a fragment can
#: still say which of the two it carries.
DUMP_FORMAT_JSONL: str = "rev2_jsonl"

#: Column count of ``data``; fixed by DUMP_COLUMNS.
N_COLUMNS: int = len(DUMP_COLUMNS)

#: Relative error above which a float32 round trip is a dtype fault rather than
#: rounding. float32 has a 24-bit mantissa (eps ~1.2e-7), so any true float32
#: value round trips at exactly 0 and a float64 literal at <=6e-8; 1e-6 leaves
#: an order of magnitude of headroom and still catches overflow to ``inf``.
LOSSLESS_TOLERANCE: float = 1e-6

#: Images queued ahead of the writer thread. Bounds memory and, far more
#: importantly, makes a stalled writer visible as backpressure instead of as
#: unbounded growth. One queued item is a reference to a list the runner has
#: already finished with, so this is cheap.
DEFAULT_QUEUE_SIZE: int = 256

_SENTINEL = object()

Detections = Sequence[tuple[str, float, tuple[float, float, float, float]]]


def _scalar(value: str) -> np.ndarray:
    """A 0-d unicode array, read back with ``.item()``."""
    return np.array(value, dtype=np.str_)


@dataclass(frozen=True)
class BinaryDump:
    """One dump file, read back whole.

    ``data``/``offsets``/``image_ids`` are exactly the arrays on disk; nothing
    is re-ordered or re-typed on load, so a reader can compare them with ``==``
    against what was written.
    """

    meta: Mapping[str, Any]
    class_names: tuple[str, ...]
    image_ids: np.ndarray
    offsets: np.ndarray
    data: np.ndarray

    @property
    def n_images(self) -> int:
        return int(self.image_ids.shape[0])

    @property
    def n_detections(self) -> int:
        return int(self.data.shape[0])

    def detections_at(self, index: int) -> np.ndarray:
        """The ``(n, 6)`` block for the ``index``-th image, as a view."""
        return self.data[self.offsets[index]:self.offsets[index + 1]]

    def detections_for(self, image_id: int) -> np.ndarray:
        """The ``(n, 6)`` block for a COCO image id.

        Raises:
            KeyError: the image is not in this dump. A dump is a census, so a
                missing image is a defect and not an empty result.
        """
        hits = np.flatnonzero(self.image_ids == image_id)
        if hits.size == 0:
            raise KeyError(f"image_id {image_id} is not in {self.meta.get('path')}")
        return self.detections_at(int(hits[0]))


class BinaryDetectionDumpWriter:
    """Queue-fed writer for one condition's scored detections.

    The interface mirrors :class:`utils.runner.detection_dump.DetectionDumpWriter`
    (``add`` / ``commit`` / ``abandon`` / ``as_record``) so the runner chooses a
    format and nothing else changes.

    The file is written under a ``.partial`` name and renamed only by
    :meth:`commit`, so an interrupted run cannot leave something that reads as a
    complete census dump.
    """

    def __init__(
        self,
        path: Path,
        arm: str,
        pool: str,
        rung: str,
        max_dets_per_category: int,
        cap_policy: str,
        runner_version: Mapping[str, Any] | None = None,
        class_names: Sequence[str] = V6_CLASSES,
        compress: bool = False,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        verify_lossless: bool = True,
    ) -> None:
        self.path = Path(path)
        self.arm = arm
        self.pool = pool
        self.rung = rung
        self.max_dets_per_category = max_dets_per_category
        self.cap_policy = cap_policy
        self.runner_version = dict(runner_version) if runner_version else None
        self.class_names = tuple(class_names)
        self.compress = compress
        self.verify_lossless = verify_lossless
        self._class_index = {name: i for i, name in enumerate(self.class_names)}

        self.n_rows = 0
        self.n_detections = 0
        #: Rows whose float64 -> float32 -> float64 round trip is not bitwise
        #: identical. Expected to be 0; recorded either way so the claim is
        #: auditable per condition rather than assumed campaign-wide.
        self.n_rows_not_exact = 0
        self.max_relative_error = 0.0
        self._image_ids: list[int] = []
        self._counts: list[int] = []
        self._chunks: list[np.ndarray] = []
        self._error: BaseException | None = None
        self._committed = False

        self._tmp = self.path.with_suffix(self.path.suffix + ".partial")
        self._tmp.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._drain, name="v6-dump-writer",
                                        daemon=False)
        self._thread.start()

    # -- producer side (the inference thread) --------------------------------

    def add(self, image_id: int, detections: Detections) -> None:
        """Hand one image's detections to the writer thread.

        ``detections`` must not be mutated afterwards; the runner builds a fresh
        list per image, so it never is.

        Raises:
            RuntimeError: the writer thread has failed. The failure is re-raised
                here, on the run's own thread, so a broken dump stops the run
                instead of quietly producing a short file.
        """
        self._raise_if_failed()
        self._queue.put((int(image_id), detections))

    def commit(self) -> Path:
        """Drain, assemble and publish the dump; returns the final path."""
        self._finish_thread()
        self._raise_if_failed()
        self._write(self._tmp, complete=True)
        self._tmp.rename(self.path)
        self._committed = True
        logger.info("wrote detection dump %s (%d images, %d detections, %d bytes, "
                    "format %s%s)", self.path, self.n_rows, self.n_detections,
                    self.path.stat().st_size, DUMP_FORMAT_NPZ,
                    ", compressed" if self.compress else "")
        if self.n_rows_not_exact:
            # Below the tolerance by construction (above it raises), so this is
            # a note about provenance, not a defect: some value arrived as a
            # float64 literal rather than a float32 model output.
            logger.warning("%s: %d of %d rows are not bitwise identical after the "
                           "float32 round trip (max relative error %.3e, tolerance "
                           "%.0e)", self.path.name, self.n_rows_not_exact,
                           self.n_detections, self.max_relative_error,
                           LOSSLESS_TOLERANCE)
        return self.path

    def abandon(self) -> None:
        """Stop without publishing, leaving the partial file as evidence.

        Best effort by construction: this runs while an exception is already
        propagating, so a failure to write the evidence must not replace the
        original error.
        """
        self._finish_thread()
        try:
            self._write(self._tmp, complete=False)
        except BaseException as exc:  # noqa: BLE001 - evidence is best effort
            logger.warning("could not write partial dump %s: %r", self._tmp, exc)
        logger.warning("detection dump abandoned at %s after %d images",
                       self._tmp, self.n_rows)

    def as_record(self) -> dict:
        """The block a fragment carries so the dump can be found again."""
        return {
            "path": str(self.path),
            "format": DUMP_FORMAT_NPZ,
            "compressed": bool(self.compress),
            "columns": list(DUMP_COLUMNS),
            "class_index_vocabulary": list(self.class_names),
            "arm": self.arm,
            "pool": self.pool,
            "rung": self.rung,
            "n_images": self.n_rows,
            "n_detections": self.n_detections,
            "max_dets_per_category": self.max_dets_per_category,
            "cap_policy": self.cap_policy,
            "stage": "post_protocol_as_scored_by_cocoeval",
            "bytes": self.path.stat().st_size if self.path.is_file() else None,
            "float32_round_trip": self._losslessness(),
        }

    def _losslessness(self) -> dict:
        """What the float32 packing cost, measured rather than asserted."""
        return {
            "checked": bool(self.verify_lossless),
            "n_rows_not_exact": self.n_rows_not_exact,
            "max_relative_error": self.max_relative_error,
            "tolerance": LOSSLESS_TOLERANCE,
        }

    # -- consumer side (the writer thread) -----------------------------------

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                return
            if self._error is not None:
                # Already failed: keep draining so the producer never blocks on
                # a full queue while it is on its way to raising.
                self._queue.task_done()
                continue
            try:
                image_id, detections = item
                self._accumulate(image_id, detections)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the producer
                self._error = exc
                logger.error("detection dump writer failed: %r", exc)
            finally:
                self._queue.task_done()

    def _accumulate(self, image_id: int, detections: Detections) -> None:
        scored = select_scored_detections(detections, self.max_dets_per_category)
        rows = np.empty((len(scored), N_COLUMNS), dtype=np.float64)
        for i, (name, score, (x1, y1, x2, y2)) in enumerate(scored):
            rows[i] = (x1, y1, x2, y2, score, self._class_index[name])
        # An overflow here is a real fault, but it is diagnosed below with the
        # offending value in hand; a bare RuntimeWarning would say less.
        with np.errstate(over="ignore"):
            packed = rows.astype(np.float32)
        if self.verify_lossless:
            self._check_round_trip(image_id, rows, packed)
        self._image_ids.append(image_id)
        self._counts.append(len(scored))
        self._chunks.append(packed)
        self.n_rows += 1
        self.n_detections += len(scored)

    def _check_round_trip(self, image_id: int, rows: np.ndarray,
                          packed: np.ndarray) -> None:
        """Measure the float32 round trip; raise only on a dtype fault.

        Raises:
            ValueError: a value did not survive as a finite number, or moved by
                more than :data:`LOSSLESS_TOLERANCE` relatively. Both mean the
                dumped numbers are not the numbers that were scored.
        """
        back = packed.astype(np.float64)
        finite_before = np.isfinite(rows)
        if np.any(finite_before & ~np.isfinite(back)):
            raise ValueError(
                f"float32 overflowed on image {image_id}: a finite value became "
                f"non-finite. The dump would not be the set COCOeval scored")
        # Values below 1 are judged absolutely; scores live there and a relative
        # error near zero would otherwise be unbounded.
        error = np.abs(back - rows) / np.maximum(np.abs(rows), 1.0)
        worst = float(error.max()) if error.size else 0.0
        if worst > LOSSLESS_TOLERANCE:
            index = int(np.unravel_index(int(error.argmax()), error.shape)[0])
            raise ValueError(
                f"float32 is not lossless for image {image_id}: relative error "
                f"{worst:.3e} > {LOSSLESS_TOLERANCE:.0e} at row {index} "
                f"({rows[index].tolist()} -> {back[index].tolist()}). The dumped "
                f"numbers are not the numbers COCOeval scored")
        if worst > 0.0:
            self.n_rows_not_exact += int(np.count_nonzero(np.any(back != rows, axis=1)))
            self.max_relative_error = max(self.max_relative_error, worst)

    # -- internals ------------------------------------------------------------

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"detection dump writer thread failed for {self.path}") from self._error

    def _finish_thread(self) -> None:
        if self._thread.is_alive():
            self._queue.put(_SENTINEL)
            self._thread.join()

    def _arrays(self, complete: bool) -> dict[str, np.ndarray]:
        data = (np.concatenate(self._chunks, axis=0) if self._chunks
                else np.zeros((0, N_COLUMNS), dtype=np.float32))
        offsets = np.zeros(len(self._counts) + 1, dtype=np.int64)
        np.cumsum(np.asarray(self._counts, dtype=np.int64), out=offsets[1:])
        meta = {
            "format_version": DUMP_FORMAT_NPZ,
            "arm": self.arm,
            "pool": self.pool,
            "condition_id": self.rung,
            "cap_policy": self.cap_policy,
            "columns": list(DUMP_COLUMNS),
            "class_names": list(self.class_names),
            "max_dets_per_category": self.max_dets_per_category,
            "stage": "post_protocol_as_scored_by_cocoeval",
            "n_images": int(len(self._image_ids)),
            "n_detections": int(data.shape[0]),
            "complete": bool(complete),
            "compressed": bool(self.compress),
            "float32_round_trip": self._losslessness(),
            "runner_version": self.runner_version,
        }
        return {
            "data": data,
            "offsets": offsets,
            "image_ids": np.asarray(self._image_ids, dtype=np.int64),
            "class_names": np.asarray(self.class_names, dtype=np.str_),
            "arm": _scalar(self.arm),
            "pool": _scalar(self.pool),
            "condition_id": _scalar(self.rung),
            "cap_policy": _scalar(self.cap_policy),
            "format_version": _scalar(DUMP_FORMAT_NPZ),
            "runner_version": _scalar(json.dumps(self.runner_version)),
            "meta_json": _scalar(json.dumps(meta)),
        }

    def _write(self, target: Path, complete: bool) -> None:
        arrays = self._arrays(complete)
        save = np.savez_compressed if self.compress else np.savez
        # Written through an open handle on purpose: ``np.savez`` appends
        # ``.npz`` to a *path* that does not end in it, which would defeat the
        # ``.partial`` staging.
        with open(target, "wb") as handle:
            save(handle, **arrays)


def read_binary_dump(path: Path) -> BinaryDump:
    """Read a whole dump file into memory.

    Raises:
        ValueError: the file's internal invariants do not hold (wrong format
            version, offsets that do not describe ``data``, a class index out of
            range). A dump that fails these is not a partially usable dump.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        meta = json.loads(str(handle["meta_json"].item()))
        # np.asarray, like the three lines below it: NpzFile.__getitem__ has no
        # declared return type, so iterating its result directly is unprovable
        # to a checker even though it is always an array here.
        class_names = tuple(str(name) for name in np.asarray(handle["class_names"]))
        image_ids = np.asarray(handle["image_ids"])
        offsets = np.asarray(handle["offsets"])
        data = np.asarray(handle["data"])
    meta = dict(meta)
    meta["path"] = str(path)

    version = meta.get("format_version")
    if version != DUMP_FORMAT_NPZ:
        raise ValueError(f"{path}: format_version {version!r} != {DUMP_FORMAT_NPZ!r}")
    if data.ndim != 2 or data.shape[1] != N_COLUMNS or data.dtype != np.float32:
        raise ValueError(f"{path}: data is {data.shape} {data.dtype}, expected "
                         f"(N, {N_COLUMNS}) float32")
    if offsets.shape[0] != image_ids.shape[0] + 1:
        raise ValueError(f"{path}: {offsets.shape[0]} offsets for "
                         f"{image_ids.shape[0]} images")
    if offsets[0] != 0 or offsets[-1] != data.shape[0]:
        raise ValueError(f"{path}: offsets span [{offsets[0]}, {offsets[-1]}] but "
                         f"data has {data.shape[0]} rows")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"{path}: offsets are not non-decreasing")
    if data.shape[0] and (data[:, 5].min() < 0 or data[:, 5].max() >= len(class_names)):
        raise ValueError(f"{path}: class_index out of range for {len(class_names)} "
                         f"class names")
    return BinaryDump(meta=meta, class_names=class_names, image_ids=image_ids,
                      offsets=offsets, data=data)


def iter_binary_dump(path: Path) -> Iterator[dict]:
    """Yield the JSONL dump's row dicts from a binary dump.

    Provided so a reader written against the JSONL format needs one import
    swapped and nothing else. Floats are widened back to ``float64``, which is
    exact (see the module docstring), so the rows equal the JSONL rows the same
    run would have produced.
    """
    dump = read_binary_dump(path)
    arm = str(dump.meta["arm"])
    pool = str(dump.meta["pool"])
    rung = str(dump.meta["condition_id"])
    for index, image_id in enumerate(dump.image_ids.tolist()):
        rows = dump.detections_at(index).astype(np.float64).tolist()
        for row in rows:
            row[5] = int(row[5])
        yield {"image_id": int(image_id), "arm": arm, "pool": pool,
               "prompts": {rung: rows}}
