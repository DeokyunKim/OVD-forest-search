"""Run one condition over the full test census and build its fragment.

Spec: the vocabulary specification §5/§7, the frozen-hyperparameter specification §6.

The order of operations inside :func:`run_condition` is load-bearing:

1. predict every term's detections for the image (per channel, no argmax);
2. cap to 100 detections **across all terms, distractors included**;
3. only then drop distractor detections, which have no ground truth.

Swapping 2 and 3 would let a condition keep 100 scored detections regardless
of how many distractors outranked them, which erases exactly the
absorption/stealing effect the C axis exists to measure.

Probe mode (``limit``) exists for pipeline checks and speed measurement. It
writes no fragment — spec §4.0/§5 forbid a partial evaluation reaching a
results table, so the only way to get a fragment is to run the full census.

Prefetch
--------
The three stages above are CPU, CPU, GPU. Run strictly serially, each device
idles while the other works; for OWLv2 the two halves are of comparable cost,
so about half of the wall clock was one device waiting. The loop now stages the
CPU half on worker threads (:mod:`utils.runner.prefetch`) and keeps the GPU half on
this thread.

What that is allowed to change: **when** preprocessing runs. What it must not
change: anything arithmetic. No image is batched with another, no reduction is
reordered, no processor is swapped for a faster one, and results are consumed
in the census order regardless of which worker finished first. ``prefetch=False``
restores the previous statements exactly, and every fragment records which of
the two paths produced it (``runner_version``).

The detection dump obeys the same rule for the same reason: ``writer.add`` hands
the image's detections to a writer thread and returns, so the ~6 MB/s of
serialisation does not reoccupy the CPU the prefetch just freed. It reads the
list the metrics were computed from and cannot alter it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from utils.runner.logging_utils import get_logger
from utils.protocol.cap_policy import DEFAULT_CAP_POLICY, apply_cap_policy, budget_is_shared
from utils.metrics.coco_eval import (
    DetectionRecord,
    count_empty_frame_false_positives,
    evaluate_per_class,
)
from utils.protocol.conditions import Condition
from utils.runner.detection_dump import DetectionDumpWriter
from utils.runner.detection_dump_binary import (
    DUMP_FORMAT_JSONL,
    DUMP_FORMAT_NPZ,
    BinaryDetectionDumpWriter,
)
from utils.runner.detector import ConditionDetector, TermDetection
from utils.protocol.eval_protocol import EVAL_PROTOCOL, EvalProtocol, assert_frozen_protocol
from utils.runner.fragment import (
    EmptyFrameFalsePositives,
    build_fragment,
    summarise_detection_counts,
)
from utils.runner.prefetch import (
    DEFAULT_PREFETCH_DEPTH,
    DEFAULT_PREFETCH_WORKERS,
    prefetch_ordered,
)
from utils.runner.runner_version import (
    RUNNER_PATH_PREFETCH,
    RUNNER_PATH_SERIAL,
    runner_version_record,
)
from utils.protocol.vocabulary import CATEGORY_ID

logger = get_logger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Outcome of one condition run."""

    fragment: dict | None
    n_images: int
    elapsed_seconds: float
    images_per_second: float
    is_probe: bool


def _load_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


@dataclass(frozen=True)
class _Staged:
    """One image's CPU work, done and waiting for the GPU.

    ``prepared`` distinguishes the two payload kinds so the consumer never has
    to guess: ``True`` means ``payload`` already went through the detector's
    ``prepare``; ``False`` means it is the decoded PIL image and the detector's
    own ``predict`` still has to do its preprocessing.
    """

    record: Mapping
    payload: object
    prepared: bool


def _stage_fn(detector: ConditionDetector,
              terms: Sequence[str]) -> Callable[[Mapping], _Staged]:
    """Build the callable that runs off the critical path, per image.

    A detector that implements the two-phase contract gets its preprocessing
    staged too; one that does not gets only its image loading staged, which is
    always safe because loading is what the runner itself does.
    """
    prepare = getattr(detector, "prepare", None)
    if prepare is None:
        def stage_loaded(record: Mapping) -> _Staged:
            return _Staged(record, _load_image(record["file_name"]), False)
        return stage_loaded

    # Distinct names, not two definitions of one: the branch above returns, so
    # only one of these is ever built. Sharing the name reads as a redefinition
    # to anything that does not model the early return -- a static checker, and
    # a person skimming.
    def stage_prepared(record: Mapping) -> _Staged:
        image = _load_image(record["file_name"])
        return _Staged(record, prepare(image, terms), True)
    return stage_prepared


def _serial_stream(records: Sequence[Mapping]) -> Iterator[_Staged]:
    """The pre-prefetch path, statement for statement: load, then predict."""
    for record in records:
        yield _Staged(record, _load_image(record["file_name"]), False)


def _predict_staged(detector: ConditionDetector, staged: _Staged,
                    terms: Sequence[str]) -> list[TermDetection]:
    if staged.prepared:
        return detector.predict_prepared(staged.payload, terms)
    return detector.predict(staged.payload, terms)


#: Serialisations ``run_condition`` can write. ``npz`` is the default: at 24
#: bytes per detection against JSONL's ~143 it is what makes the census
#: campaign fit on disk, and it is lossless because every dumped number is a
#: ``float32`` model output to begin with (see
#: :mod:`utils.runner.detection_dump_binary`). ``jsonl`` is retained for readers not
#: yet ported; whichever ran is recorded in the fragment's ``runner_version``.
DUMP_FORMATS: tuple[str, ...] = (DUMP_FORMAT_NPZ, DUMP_FORMAT_JSONL)

DEFAULT_DUMP_FORMAT: str = DUMP_FORMAT_NPZ

_DUMP_SUFFIX: Mapping[str, str] = {DUMP_FORMAT_NPZ: ".npz", DUMP_FORMAT_JSONL: ".jsonl"}

#: Whether each format compresses when the caller does not say. ``npz`` does:
#: deflate takes the measured campaign from ~66 GB to ~47 GB, and it costs ~2 s
#: once per condition (a run of half an hour or more) plus 7% on read. ``jsonl``
#: does not, because gzipping it changes the file name and every reader with it,
#: and that format is only retained for readers not yet ported.
_DUMP_COMPRESS_DEFAULT: Mapping[str, bool] = {DUMP_FORMAT_NPZ: True,
                                              DUMP_FORMAT_JSONL: False}


def dump_compression_default(dump_format: str) -> bool:
    """Whether ``dump_format`` compresses when ``dump_compress`` is ``None``."""
    if dump_format not in _DUMP_COMPRESS_DEFAULT:
        raise ValueError(f"unknown dump format {dump_format!r}; expected one of "
                         f"{DUMP_FORMATS}")
    return _DUMP_COMPRESS_DEFAULT[dump_format]


def dump_path_for(dump_dir: Path, model: str, checkpoint: str, condition_id: str,
                  is_probe: bool = False, compress: bool = False,
                  dump_format: str = DEFAULT_DUMP_FORMAT) -> Path:
    """Where one condition's detection dump lands.

    A probe dump carries ``.probe`` in its name: it covers only the first
    ``limit`` images, and a short dump that is indistinguishable from a census
    dump is exactly the failure ``write_fragment``'s census gate exists to
    prevent.

    The extension names the format, so a directory of dumps cannot become a mix
    of two serialisations distinguishable only by opening them. ``compress``
    adds ``.gz`` for JSONL; for ``npz`` it selects deflate *inside* the
    container, which is why it does not change the name.
    """
    if dump_format not in _DUMP_SUFFIX:
        raise ValueError(f"unknown dump format {dump_format!r}; expected one of "
                         f"{DUMP_FORMATS}")
    stem = f"{model}__{checkpoint}__{condition_id}"
    probe = ".probe" if is_probe else ""
    gz = ".gz" if (compress and dump_format == DUMP_FORMAT_JSONL) else ""
    return dump_dir / f"{stem}{probe}{_DUMP_SUFFIX[dump_format]}{gz}"


def _open_dump(condition: Condition, detector: ConditionDetector,
               dump_dir: Path | None, dump_pool: str | None,
               dump_label: str | None,
               protocol: EvalProtocol, is_probe: bool, compress: bool,
               dump_format: str, cap_policy: str,
               runner_version: Mapping,
               ) -> DetectionDumpWriter | BinaryDetectionDumpWriter | None:
    if dump_dir is None:
        return None
    if not dump_pool:
        raise ValueError(
            "dump_pool is required when dumping detections: it records WHICH "
            "image set the dump covers, and rev2's merge refuses to merge dumps "
            "from different pools. Merging them silently would compare two "
            "different censuses")
    # The FRAGMENT LABEL names the dump, not the checkpoint. Two different
    # adapters can carry the same checkpoint name -- a retrained seed 1 is still
    # `swin-tiny-ft-seed1-step5667` -- so naming dumps by checkpoint let one run
    # silently overwrite another's raw detections while both fragments looked
    # fine. The label is what distinguishes the runs, so it is what distinguishes
    # their dumps.
    path = dump_path_for(Path(dump_dir), detector.model_name,
                         dump_label or detector.checkpoint_name,
                         condition.condition_id,
                         is_probe=is_probe, compress=compress,
                         dump_format=dump_format)
    # The arm names the RUN, using the same string that names the file, so
    # `<arm>__<condition>.npz` stays true and a dump identifies itself without
    # reference to its path. The checkpoint name does not distinguish two runs
    # of the same model, seed and step; the label does.
    arm = f"{detector.model_name}__{dump_label or detector.checkpoint_name}"
    if dump_format == DUMP_FORMAT_JSONL:
        return DetectionDumpWriter(
            path=path, arm=arm, pool=str(dump_pool), rung=condition.condition_id,
            max_dets_per_category=protocol.max_detections_per_image,
            compress=compress,
        )
    return BinaryDetectionDumpWriter(
        path=path, arm=arm, pool=str(dump_pool), rung=condition.condition_id,
        max_dets_per_category=protocol.max_detections_per_image,
        cap_policy=cap_policy, runner_version=runner_version, compress=compress,
    )


def run_condition(
    condition: Condition,
    detector: ConditionDetector,
    coco_gt,
    image_records: Sequence[Mapping],
    text_cosines: Mapping[str, float] | None = None,
    protocol: EvalProtocol = EVAL_PROTOCOL,
    limit: int | None = None,
    log_every: int = 500,
    cap_policy: str = DEFAULT_CAP_POLICY,
    prefetch: bool = True,
    prefetch_workers: int = DEFAULT_PREFETCH_WORKERS,
    prefetch_depth: int = DEFAULT_PREFETCH_DEPTH,
    dump_dir: Path | None = None,
    dump_pool: str | None = None,
    dump_label: str | None = None,
    dump_compress: bool | None = None,
    dump_format: str = DEFAULT_DUMP_FORMAT,
) -> RunResult:
    """Evaluate ``condition`` over ``image_records``.

    Args:
        image_records: COCO image dicts with absolute ``file_name``.
        limit: Probe mode. Processes only the first ``limit`` images and
            returns ``fragment=None``.
        cap_policy: ``per_category`` (the COCO standard, and the default) hands
            every scored-class detection to COCOeval, which applies its own
            per-category ``maxDets``. ``global_per_image`` keeps only the top
            100 across all terms first, which makes the queried terms zero-sum
            and is retained solely as a deployment-view measurement. See
            the cap-policy note.
        prefetch: Overlap the CPU stages with the GPU forward. ``False`` is the
            escape hatch: it restores the pre-prefetch statements exactly and
            is recorded in the fragment as the serial runner path.
        prefetch_workers: Threads running the CPU stage.
        prefetch_depth: Upper bound on staged images held in memory.
        dump_dir: Write the scored detections here, one file per condition.
            ``None`` disables the dump. The dump is derived from the same list
            COCOeval is given and cannot change a metric.
        dump_pool: The image set the dump covers, recorded in the dump.
            Required whenever ``dump_dir`` is set.
        dump_compress: Compress the dump: gzip for JSONL, deflate inside the
            container for ``npz``. ``None`` takes the format's own default
            (on for ``npz``, off for JSONL) -- see
            the dump-format note for the measured trade.
        dump_format: ``v6_npz.1`` (default, binary, ~6x smaller) or
            ``rev2_jsonl``. Recorded in the fragment's ``runner_version`` so a
            fragment states which format its dump is in, and states ``None``
            when it has no dump at all.
    """
    assert_frozen_protocol(protocol)
    if dump_format not in DUMP_FORMATS:
        raise ValueError(f"unknown dump format {dump_format!r}; expected one of "
                         f"{DUMP_FORMATS}")
    if dump_compress is None:
        dump_compress = dump_compression_default(dump_format)
    records = list(image_records[:limit] if limit else image_records)
    is_probe = limit is not None

    empty_image_ids = {
        im["id"] for im in records
        if not coco_gt.getAnnIds(imgIds=[im["id"]], iscrowd=None)
    }

    detections: list[DetectionRecord] = []
    counts_per_image: list[int] = []
    counts_on_empty: list[int] = []
    per_class_counts: dict[str, int] = {name: 0 for name in condition.scored_classes}

    two_phase = hasattr(detector, "prepare") and hasattr(detector, "predict_prepared")
    runner_path = RUNNER_PATH_PREFETCH if prefetch else RUNNER_PATH_SERIAL
    # Built before the loop, not after it, so the identical block goes into the
    # dump file and into the fragment. A dump found on its own then carries the
    # provenance of the run that wrote it, without a join back to the fragment.
    runner_version = runner_version_record(
        detector.model_name, runner_path,
        prefetch_workers=prefetch_workers if prefetch else None,
        prefetch_depth=prefetch_depth if prefetch else None,
        detector_two_phase=two_phase and prefetch,
        dump_format=dump_format if dump_dir is not None else None,
    )
    logger.info("%s: runner path %s (workers=%s depth=%s, detector two-phase=%s, "
                "dump=%s)", condition.condition_id, runner_path,
                prefetch_workers if prefetch else "-",
                prefetch_depth if prefetch else "-", two_phase,
                runner_version["dump_format"] or "off")

    if prefetch:
        # ``prefetch_ordered`` yields in census order by construction; the
        # detections list below is therefore assembled in exactly the order the
        # serial loop produced, which is what makes fragments comparable.
        stream: Iterator[_Staged] = prefetch_ordered(
            records, _stage_fn(detector, condition.terms),
            workers=prefetch_workers, depth=prefetch_depth)
    else:
        stream = _serial_stream(records)

    writer = _open_dump(condition, detector, dump_dir, dump_pool, dump_label, protocol,
                        is_probe, dump_compress, dump_format, cap_policy,
                        runner_version)

    start = time.monotonic()
    try:
        for index, staged in enumerate(stream, start=1):
            record = staged.record
            raw = _predict_staged(detector, staged, condition.terms)

            # The cap policy decides whether the queried terms compete for one
            # budget. Distractors are dropped by the policy in the correct order.
            kept = apply_cap_policy(
                [(d.term, d.score, d.bbox_xyxy) for d in raw],
                condition.term_to_gt_class, cap_policy,
                protocol.max_detections_per_image,
            )
            # Counts describe what was SUBMITTED for scoring (scored classes only).
            counts_per_image.append(len(kept))
            if record["id"] in empty_image_ids:
                counts_on_empty.append(len(kept))

            for term, score, (x1, y1, x2, y2) in kept:
                gt_class = condition.term_to_gt_class[term]
                per_class_counts[gt_class] = per_class_counts.get(gt_class, 0) + 1
                detections.append(DetectionRecord(
                    image_id=record["id"],
                    category_id=CATEGORY_ID[gt_class],
                    bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                    score=score,
                ))

            if writer is not None:
                # Read-only: the scored set is derived from ``kept``, which has
                # already been appended to ``detections`` above unchanged. For
                # the binary writer this call is a queue put -- the maxDets
                # replication and the float packing happen on the writer
                # thread, so serialisation does not re-enter the critical path
                # the prefetch was added to clear.
                writer.add(record["id"],
                           [(condition.term_to_gt_class[t], s, b) for t, s, b in kept])

            if log_every and index % log_every == 0:
                rate = index / (time.monotonic() - start)
                logger.info("%s: %d/%d images (%.2f img/s)",
                            condition.condition_id, index, len(records), rate)
    except BaseException:
        if writer is not None:
            writer.abandon()
        raise
    finally:
        stream.close()  # joins the prefetch workers; a no-op for the serial path

    elapsed = time.monotonic() - start
    dump_record = None
    if writer is not None:
        writer.commit()
        dump_record = writer.as_record()
        # Lift the lossless-guard telemetry INTO runner_version, so it reaches
        # the fragment. Left only in the npz it would sit in a file nobody
        # opens while building a table -- the same "recorded beside, not
        # inside" failure that let A1_singleton_hat claim a dump it lacked.
        _rt = (dump_record or {}).get("float32_round_trip") or {}
        n_inexact = int(_rt.get("n_rows_not_exact", 0) or 0)
        max_rel = float(_rt.get("max_relative_error", 0.0) or 0.0)
        runner_version["dump_inexact_rows"] = n_inexact
        runner_version["dump_max_rel_err"] = max_rel
        if n_inexact:
            # Do not abort: losing a half-hour census to a 2.6e-8 rounding is
            # the instrument damaging the experiment. But this must be loud --
            # a changed value can reorder greedy_match's score-descending ties.
            logger.error(
                "%s: float32 dump packing was NOT lossless on %d row(s), max "
                "relative error %.3e. Detection VALUES differ from what "
                "COCOeval scored; score-tie ordering in greedy_match may "
                "differ. Recorded in the fragment as dump_inexact_rows.",
                condition.condition_id, n_inexact, max_rel)
    rate = len(records) / elapsed if elapsed > 0 else 0.0

    if is_probe:
        logger.info("PROBE %s: %d images in %.1fs (%.2f img/s); no fragment written",
                    condition.condition_id, len(records), elapsed, rate)
        return RunResult(fragment=None, n_images=len(records), elapsed_seconds=elapsed,
                         images_per_second=rate, is_probe=True)

    per_class = evaluate_per_class(
        coco_gt, detections, condition.scored_classes, per_class_counts, protocol)
    n_empty, n_fp, fp_rate = count_empty_frame_false_positives(counts_on_empty)

    notes = detector.postprocessing_notes()
    fragment = build_fragment(
        condition_id=condition.condition_id,
        query_string=condition.caption,
        model=detector.model_name,
        checkpoint=detector.checkpoint_name,
        n_images=len(records),
        per_class=per_class,
        empty_frame=EmptyFrameFalsePositives(
            n_empty_images=n_empty, n_false_positives=n_fp, per_image=fp_rate),
        detection_counts=summarise_detection_counts(counts_per_image),
        text_cosines=text_cosines,
        protocol=protocol,
        extra={
            "axis": condition.axis,
            "subgroup": condition.subgroup,
            "vocabulary_status": condition.vocabulary_status,
            "terms": list(condition.terms),
            "distractor_terms": list(condition.distractor_terms),
            "scored_classes": list(condition.scored_classes),
            "readout": notes,
            "cap_policy": cap_policy,
            "budget_is_shared": budget_is_shared(cap_policy),
            "elapsed_seconds": elapsed,
            "images_per_second": rate,
            # Where the detections behind these metrics live. Both the numbers
            # and the dump come from one list, so a downstream check such as
            # dAR ~= PFR - NFR is comparing like with like.
            "detection_dump": dump_record,
            # Which code produced these numbers, as a lookup rather than a
            # guess from launch timestamps: a content hash of the modules that
            # can change a number, plus the name of the loop path taken and the
            # dump format written (``None`` when no dump was). The prefetch is a
            # scheduling change and is expected to leave every number bit
            # identical -- this is what lets that be checked.
            "runner_version": runner_version,
        },
    )
    return RunResult(fragment=fragment, n_images=len(records), elapsed_seconds=elapsed,
                     images_per_second=rate, is_probe=False)


def load_test_set(path: Path):
    """Load the V6 test set as ``(COCO, image_records)``."""
    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(path))
    records = [coco.loadImgs(i)[0] for i in sorted(coco.getImgIds())]
    return coco, records
