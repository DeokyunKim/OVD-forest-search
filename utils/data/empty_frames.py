"""False-alarm rates on GT-absent frames, per class, over a threshold sweep.

Why this exists — a genuine §6/§7 conflict in the spec
------------------------------------------------------
Spec §7 requires "empty-frame false positives per image", and spec §4 makes it
the number the absorption claim rests on. Spec §6 freezes
``score_threshold = 0.0`` with ``max_detections_per_image = 100``.

**Those two requirements cannot both be satisfied in one inference pass.** At
threshold 0.0 a sigmoid score is never actually zero and GroundingDINO returns
200 candidates, so *every* image emits exactly 100 detections and the
false-positive count per frame is the constant 100.0 for every condition and
every model. Measured, not argued: the first census fragment
(``C1_v6_plus_nat``, GDINO) has all 14,980 images in the 51-100 bucket and a
per-image rate of exactly 100.0.

§6 is right for AP — AP is rank-based and needs the full ranked list. But a
false-alarm rate is inherently a function of the threshold an operator deploys
at. The resolution keeps both: **AP/AR stay at threshold 0.0 exactly as §6
freezes them, and the false-alarm rate is reported as a threshold-sweep
curve.** No single operating point is chosen, so nothing arbitrary enters.

The denominator
---------------
The denominator is **per class**, not "frames with no annotations at all":

    denominator(c) = every frame whose ground truth contains no instance of c

The absorption hypothesis asks whether a distractor takes boxes that would
have become ``person`` false positives, so ``person``'s denominator is the
4,407 frames with no person — not the 187 frames that are empty of
everything. Using the 187 would shrink the denominator 24-fold and the effect
would not be resolvable. The denominators differ sharply by class on this
census (person 4,407 · clothes 12,496 · bag 14,301 · hat 14,417 · shoes
14,484 · gloves 14,541), so every rate is stored with its own ``n_images``.

The 187 all-empty frames are retained as a **separate, narrower** statistic
(``all_empty_frames``). The two definitions answer different questions and are
never mixed.

Note that the union of the six per-class denominators is the entire census, so
producing these curves requires a full-census inference pass. That pass dumps
its detections, which makes every threshold-dependent metric recomputable
offline without ever running inference again.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)

#: Operating points swept for the false-alarm curve. 0.0 is retained so the
#: degenerate protocol value stays visible beside the useful ones.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)

#: Detections below this score are dropped from the dump. It sits an order of
#: magnitude below the lowest useful sweep point so the curve is unaffected.
DUMP_SCORE_FLOOR: float = 0.005


@dataclass(frozen=True)
class FalseAlarmPoint:
    """False alarms for one class at one threshold, on that class's denominator."""

    class_name: str
    threshold: float
    #: Frames with no ground-truth instance of ``class_name``. The denominator.
    n_images: int
    n_false_positives: int
    false_positives_per_image: float
    n_images_with_any_false_positive: int
    frac_images_with_any_false_positive: float


def gt_absent_images(coco_gt, category_id: int, image_ids: Sequence[int]) -> set[int]:
    """Images holding no non-crowd ground truth of ``category_id``."""
    present = {
        ann["image_id"]
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds(catIds=[category_id], iscrowd=False))
    }
    return {i for i in image_ids if i not in present}


def false_alarm_curve(
    detections_by_image: Mapping[int, Sequence[tuple[str, float]]],
    denominator_image_ids: Sequence[int],
    terms_for_class: Sequence[str],
    class_name: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[FalseAlarmPoint]:
    """False-alarm curve for one class on its own GT-absent denominator.

    Args:
        detections_by_image: image id -> ``(term, score)`` pairs.
        denominator_image_ids: the frames lacking this class's ground truth.
        terms_for_class: query terms that map to this class (one term normally,
            but a B-axis paraphrase maps its variant term to the class).
    """
    ids = list(denominator_image_ids)
    if not ids:
        raise ValueError(f"{class_name}: empty denominator; refusing to fabricate a rate")
    wanted = set(terms_for_class)
    points: list[FalseAlarmPoint] = []
    for threshold in thresholds:
        total = frames_hit = 0
        for image_id in ids:
            hits = sum(1 for term, score in detections_by_image.get(image_id, ())
                       if term in wanted and score >= threshold)
            total += hits
            if hits:
                frames_hit += 1
        points.append(FalseAlarmPoint(
            class_name=class_name, threshold=threshold, n_images=len(ids),
            n_false_positives=total, false_positives_per_image=total / len(ids),
            n_images_with_any_false_positive=frames_hit,
            frac_images_with_any_false_positive=frames_hit / len(ids),
        ))
    return points


@dataclass(frozen=True)
class BudgetDecomposition:
    """How the per-image detection budget splits between scored terms and distractors.

    Threshold-independent: it is computed from the capped top-100 lists, so the
    §6/§7 conflict does not touch it. This is the cleanest available evidence
    for absorption — a distractor that takes budget is a distractor that
    removed a scored-class box from the ranked list.
    """

    condition_id: str
    n_detections_total: int
    n_scored_class: int
    n_distractor: int
    frac_scored_class: float
    frac_distractor: float
    per_term_share: Mapping[str, float]


def decompose_budget(fragment: Mapping,
                     term_to_gt_class: Mapping[str, str | None]) -> BudgetDecomposition:
    """Split a fragment's detection budget into scored-class and distractor shares.

    Reads only fields already present in every census fragment, so it needs no
    extra inference.
    """
    total = int(fragment["detection_count_distribution"]["total"])
    scored = sum(int(v["n_detections"]) for v in fragment["per_class"].values())
    distractor = total - scored
    return BudgetDecomposition(
        condition_id=fragment["condition_id"],
        n_detections_total=total,
        n_scored_class=scored,
        n_distractor=distractor,
        frac_scored_class=scored / total if total else 0.0,
        frac_distractor=distractor / total if total else 0.0,
        per_term_share={name: (int(v["n_detections"]) / total if total else 0.0)
                        for name, v in fragment["per_class"].items()},
    )


def build_record(
    *,
    condition_id: str,
    query_string: str,
    model: str,
    checkpoint: str,
    n_census_images: int,
    per_class_curves: Mapping[str, Sequence[FalseAlarmPoint]],
    all_empty_curve: Sequence[FalseAlarmPoint],
    n_all_empty_images: int,
    budget: BudgetDecomposition | None,
) -> dict:
    return {
        "schema_version": "v6-falsealarm.2",
        "condition_id": condition_id,
        "query_string": query_string,
        "model": model,
        "checkpoint": checkpoint,
        "n_census_images": n_census_images,
        "denominator_definition": (
            "per class: every frame whose ground truth contains no instance of "
            "that class. Denominators differ by class and each curve carries its "
            "own n_images."
        ),
        "per_class_false_alarm_curves": {
            name: [asdict(p) for p in points] for name, points in per_class_curves.items()
        },
        "all_empty_frames": {
            "definition": ("the narrower statistic: frames with NO ground truth of any "
                           "class. A subset of every per-class denominator; reported "
                           "separately and never mixed with them."),
            "n_images": n_all_empty_images,
            "curve": [asdict(p) for p in all_empty_curve],
        },
        "budget_decomposition": asdict(budget) if budget else None,
        "note": ("Supplementary to the census fragment; AP/AR are unaffected and "
                 "remain at score_threshold=0.0 per spec §6. The census fragment's "
                 "empty_frame_false_positives_per_image is structurally 100.0 under "
                 "that threshold and must not be read as a false-alarm rate."),
    }


def write_record(record: Mapping, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.rename(path)
    logger.info("wrote false-alarm record %s", path)
    return path


def score_quantiles(scores: Sequence[float]) -> dict[str, float]:
    """Quantiles of detection scores, for calibration context."""
    if not scores:
        return {}
    ordered = sorted(scores)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]

    return {"min": ordered[0], "p50": q(0.5), "p90": q(0.9), "p99": q(0.99),
            "max": ordered[-1]}
