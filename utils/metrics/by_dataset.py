"""Score a condition on ForestPersons and ForestBelongings separately.

The benchmark merges the two corpora into one split, and a single AP over the
merge answers neither question the paper asks. "Can it still find people" and
"can it find what they dropped" are different questions, and pooling them
weights one against the other by nothing more meaningful than how many frames
each corpus contributed -- 10,553 against 4,427 here, so ForestPersons quietly
sets the tone of every pooled number.

This re-scores from the raw detection dump rather than running inference again.
The dump holds every box the model produced; restricting the ground truth and
the detections to one corpus and calling the same COCO evaluation is exactly
what a separate run would compute, at no GPU cost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from utils.metrics.coco_eval import DetectionRecord, evaluate_per_class
from utils.protocol.eval_protocol import EVAL_PROTOCOL, EvalProtocol
from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)

#: The tag `utils.data.dataset_build` writes onto every merged image.
SOURCES: Mapping[str, str] = {"fp": "ForestPersons", "fb": "ForestBelongings"}


@dataclass(frozen=True)
class Split:
    """One corpus, and the images of the merged benchmark that came from it."""

    key: str
    name: str
    image_ids: frozenset[int]

    def __len__(self) -> int:
        return len(self.image_ids)


def splits_of(test_json: Path) -> list[Split]:
    """The corpora present in a merged benchmark, in a fixed order."""
    coco = json.loads(test_json.read_text(encoding="utf-8"))
    by_source: dict[str, set[int]] = {}
    for image in coco["images"]:
        source = image.get("source")
        if source is None:
            raise ValueError(
                f"{test_json} has an image with no `source` tag, so its corpus "
                f"cannot be told from another's. Rebuild it with "
                f"prepare.py, which writes the tag.")
        by_source.setdefault(source, set()).add(image["id"])
    return [Split(key, SOURCES.get(key, key), frozenset(ids))
            for key, ids in sorted(by_source.items())]


def score_split(coco_gt, detections: Sequence[DetectionRecord],
                split: Split, scored_classes: Sequence[str],
                category_of: Mapping[str, int],
                protocol: EvalProtocol = EVAL_PROTOCOL) -> dict:
    """AP/AR for one corpus, using the same evaluation as the pooled number.

    Both sides are restricted: the ground truth to this corpus's images, and the
    detections likewise. Restricting only one would score this corpus's boxes
    against the other's annotations and produce a number that looks like a
    result.
    """
    kept = [d for d in detections if d.image_id in split.image_ids]
    # Counts are per corpus, not carried over from the pooled set: they are what
    # keeps "the model said nothing here" distinguishable from "the model was
    # wrong here" when AP comes out 0.
    name_of = {cid: name for name, cid in category_of.items()}
    counts: dict[str, int] = {}
    for detection in kept:
        name = name_of[detection.category_id]
        counts[name] = counts.get(name, 0) + 1

    subset = _restrict(coco_gt, split.image_ids)
    if not subset.getAnnIds():
        logger.warning("%s: no annotations in this corpus for the queried "
                       "classes; reporting it as unmeasured rather than 0.0",
                       split.name)
        return {"dataset": split.name, "n_images": len(split),
                "per_class": None, "note": "no ground truth in this corpus"}

    per_class = evaluate_per_class(subset, kept, list(scored_classes), counts,
                                   protocol=protocol)
    return {
        "dataset": split.name,
        "source_tag": split.key,
        "n_images": len(split),
        "n_detections": len(kept),
        "per_class": {name: metrics.as_dict() if hasattr(metrics, "as_dict")
                      else dict(metrics.__dict__)
                      for name, metrics in per_class.items()},
    }


def _restrict(coco_gt, image_ids: frozenset[int]):
    """A COCO object holding only these images and their annotations."""
    from pycocotools.coco import COCO

    subset = COCO()
    subset.dataset = {
        "images": [im for im in coco_gt.dataset["images"] if im["id"] in image_ids],
        "annotations": [a for a in coco_gt.dataset["annotations"]
                        if a["image_id"] in image_ids],
        "categories": coco_gt.dataset["categories"],
    }
    subset.createIndex()
    return subset


def _json_load(path: Path) -> dict:
    import json as _json
    return _json.loads(path.read_text(encoding="utf-8"))


def score_by_dataset(model: str, variant: str, conditions: Sequence[str],
                     protocol: EvalProtocol = EVAL_PROTOCOL) -> Path:
    """Re-score every condition's dump per corpus; write one JSON, return its path.

    A condition whose dump is missing is recorded as missing. It is not skipped
    quietly: the caller asked for these conditions, and a file that answers for
    fewer of them without saying so is the kind of gap that gets averaged over
    later.
    """
    import json as _json
    from pycocotools.coco import COCO

    from utils.runner.paths import paths
    from utils.metrics.dump_adapter import load_dump

    test_json = paths().require("benchmark") / "test.json"
    splits = splits_of(test_json)
    coco_gt = COCO(str(test_json))
    dumps = paths().output("detection_dumps")

    fragments = paths().output("fragments")

    results: list[dict] = []
    missing: list[str] = []
    for condition in conditions:
        # The dump's own name is built from the CHECKPOINT, the fragment's from
        # its LABEL, and the two are not the same string. Guessing one from the
        # other is what made every condition report as missing; the fragment
        # records the path it wrote, so that is what gets read.
        fragment = fragments / f"{model}__{variant}__{condition}.json"
        path = dumps / f"{model}__{variant}__{condition}.npz"
        if fragment.is_file():
            recorded = (_json_load(fragment).get("detection_dump") or {}).get("path")
            if recorded:
                path = Path(recorded)
        if not path.is_file():
            missing.append(condition)
            continue
        dump = load_dump(path, expect_condition=condition)
        # The dump stores a class INDEX into its own vocabulary, so the
        # names must come from the same file: a vocabulary taken from
        # anywhere else would relabel every box without failing.
        vocabulary = list(dump.vocabulary)
        # The dump stores a class INDEX into its own vocabulary; the names must
        # come from the same file, because a vocabulary taken from anywhere else
        # would relabel every box without failing. The index is then mapped to
        # the benchmark's category id BY NAME -- position is not a contract, and
        # two files agreeing on order today is not a reason to rely on it.
        vocabulary = list(dump.vocabulary)
        category_of = {c["name"]: c["id"] for c in coco_gt.dataset["categories"]}
        unknown = [n for n in vocabulary if n not in category_of]
        if unknown:
            raise KeyError(
                f"{path.name} queries {unknown}, which the benchmark does not "
                f"annotate; scoring them would compare boxes against nothing.")

        detections: list[DetectionRecord] = []
        counts: dict[str, int] = {}
        for image_id in dump.image_ids:
            rows = dump.get(image_id)
            # `if rows:` on a numpy array asks for its truth value, which is
            # ambiguous for more than one element.
            if rows is None:
                continue
            for row in rows:
                name = vocabulary[int(row[5])]
                x0, y0, x1, y1 = (float(row[0]), float(row[1]),
                                  float(row[2]), float(row[3]))
                detections.append(DetectionRecord(
                    image_id=int(image_id), category_id=category_of[name],
                    bbox_xywh=(x0, y0, x1 - x0, y1 - y0), score=float(row[4])))
                counts[name] = counts.get(name, 0) + 1

        results.append({
            "condition": condition,
            "splits": [score_split(coco_gt, detections, split, vocabulary,
                                   category_of, protocol) for split in splits],
        })

    out = paths().output("analysis") / f"per_dataset__{model}__{variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps({
        "model": model, "variant": variant,
        "corpora": {s.key: {"name": s.name, "n_images": len(s)} for s in splits},
        "conditions_requested": list(conditions),
        "conditions_missing_a_dump": missing,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if missing:
        logger.warning("%d condition(s) had no dump and are listed as missing: %s",
                       len(missing), ", ".join(missing))
    return out
