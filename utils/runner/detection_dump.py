"""Stream the detections COCOeval scores to disk, in the rev2 dump format.

Why after the protocol, not before
----------------------------------
The consistency check this dump exists for is ``ΔAR ≈ PFR − NFR``: it only
holds if the dumped detections and the fragment's AP/AR describe *the same
set*. So the dump is taken after every stage that can remove a detection from
scoring:

1. the score threshold (§6 freezes it at 0.0, so it removes nothing);
2. the cap policy (``global_per_image`` keeps the top 100 across all terms;
   ``per_category`` keeps everything and lets COCOeval cap);
3. **COCOeval's own per-category ``maxDets``**, which is applied here rather
   than inherited, because pycocotools applies it internally and would
   otherwise score fewer detections than the dump contains.

Step 3 replicates ``COCOeval.evaluateImg``: per image and per category, sort by
descending score with a *stable* sort and keep the first ``maxDets``. Stability
matters only for tied scores, and tied scores are interchangeable for AP, so
the dump is the scored set either way — but replicating the rule exactly means
nobody has to reason about that.

This module only reads what the runner already computed. It cannot change a
number: the list handed to COCOeval is not touched, only copied from.

Format (consumed by ``utils/metrics/invariance.py``)
--------------------------------------------------------------
One JSON object per line, one line per image::

    {"image_id": 12, "arm": "owlv2__large-zeroshot", "pool": "<test.json>",
     "prompts": {"<condition_id>": [[x1, y1, x2, y2, score, class_index], ...]}}

``class_index`` indexes :data:`utils.protocol.vocabulary.V6_CLASSES` — the same index
``rev2_metrics.class_detections`` filters on (``dets[:, 5]``), with the score in
column 4 and the ``xyxy`` box in columns 0..3, in original image pixels.
``merge_rung_dumps`` merges one file per condition into
``{image_id: {condition_id: dets}}``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from utils.runner.logging_utils import get_logger
from utils.protocol.vocabulary import V6_CLASSES

logger = get_logger(__name__)

#: Column layout of one dumped detection row.
DUMP_COLUMNS: tuple[str, ...] = ("x1", "y1", "x2", "y2", "score", "class_index")

CLASS_INDEX: Mapping[str, int] = {name: i for i, name in enumerate(V6_CLASSES)}


def select_scored_detections(
    detections: Sequence[tuple[str, float, tuple[float, float, float, float]]],
    max_dets_per_category: int,
) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """The subset COCOeval scores: per-category top ``max_dets_per_category``.

    Args:
        detections: ``(class_name, score, bbox_xyxy)`` for ONE image, already
            capped by the cap policy and with distractors dropped.

    Returns:
        A new list in the input order, minus the detections COCOeval would drop.
        The input is never mutated.
    """
    keep: set[int] = set()
    by_class: dict[str, list[int]] = {}
    for index, (name, _score, _box) in enumerate(detections):
        by_class.setdefault(name, []).append(index)
    for indices in by_class.values():
        # Stable, descending by score -- pycocotools' mergesort on -score.
        ordered = sorted(indices, key=lambda i: -detections[i][1])
        keep.update(ordered[:max_dets_per_category])
    return [d for i, d in enumerate(detections) if i in keep]


class DetectionDumpWriter:
    """Append-only JSONL writer, committed atomically on a clean close.

    The file is built under a ``.partial`` name and renamed only by
    :meth:`commit`. A run that dies half way therefore leaves something that
    cannot be mistaken for a complete census dump — the same rule
    ``write_fragment`` applies to fragments.
    """

    def __init__(self, path: Path, arm: str, pool: str, rung: str,
                 max_dets_per_category: int, compress: bool = False) -> None:
        self.path = path
        self.arm = arm
        self.pool = pool
        self.rung = rung
        self.max_dets_per_category = max_dets_per_category
        self.compress = compress
        self.n_rows = 0
        self.n_detections = 0
        self._tmp = path.with_suffix(path.suffix + ".partial")
        self._tmp.parent.mkdir(parents=True, exist_ok=True)
        self._fh = (gzip.open(self._tmp, "wt", encoding="utf-8") if compress
                    else open(self._tmp, "w", encoding="utf-8"))

    def add(self, image_id: int,
            detections: Sequence[tuple[str, float, tuple[float, float, float, float]]]
            ) -> None:
        """Write one image's row. Images with no detections still get a row.

        An absent row and an empty row mean different things downstream
        (``assert_uniform_rungs`` treats a missing image as a ragged ladder),
        so silence is recorded explicitly as ``[]``.
        """
        scored = select_scored_detections(detections, self.max_dets_per_category)
        rows = [[float(x1), float(y1), float(x2), float(y2), float(score),
                 CLASS_INDEX[name]]
                for name, score, (x1, y1, x2, y2) in scored]
        self._fh.write(json.dumps({
            "image_id": int(image_id), "arm": self.arm, "pool": self.pool,
            "prompts": {self.rung: rows},
        }) + "\n")
        self.n_rows += 1
        self.n_detections += len(rows)

    def commit(self) -> Path:
        """Close and publish the dump; returns the final path."""
        self._fh.close()
        self._tmp.rename(self.path)
        logger.info("wrote detection dump %s (%d images, %d detections, %d bytes)",
                    self.path, self.n_rows, self.n_detections, self.path.stat().st_size)
        return self.path

    def abandon(self) -> None:
        """Close without publishing. The ``.partial`` file is left as evidence."""
        if not self._fh.closed:
            self._fh.close()
        logger.warning("detection dump abandoned at %s after %d images",
                       self._tmp, self.n_rows)

    def as_record(self) -> dict:
        """The block a fragment carries so the dump can be found again."""
        return {
            "path": str(self.path),
            "format": "rev2_jsonl" + (".gz" if self.compress else ""),
            "columns": list(DUMP_COLUMNS),
            "class_index_vocabulary": list(V6_CLASSES),
            "arm": self.arm,
            "pool": self.pool,
            "rung": self.rung,
            "n_images": self.n_rows,
            "n_detections": self.n_detections,
            "max_dets_per_category": self.max_dets_per_category,
            "stage": "post_protocol_as_scored_by_cocoeval",
            "bytes": self.path.stat().st_size if self.path.is_file() else None,
        }


def read_dump(path: Path) -> Iterable[dict]:
    """Yield the parsed rows of a dump, transparently handling ``.gz``."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            if line.strip():
                yield json.loads(line)
