"""Layer 0 and layer 1 — statements that need no ground truth.

Contents
--------
* :func:`exact_match_rate` and :func:`permutation_identity` — layer 0. For a
  dual encoder that scores each query channel independently (OWLv2), permuting
  the query list cannot change a per-class score, so a non-zero difference is a
  harness bug rather than a model property. For a fusion model (Grounding DINO,
  GLIP, OmDet-Turbo) the same measurement is the quantity of interest.
* :func:`stability_pair` — M1 (PI-F1), M3 (location jitter) and M4 (score rank
  correlation) for one class between two conditions.
* :func:`noise_floor_normalised` — the S_norm control.

M1 and M3 are reported side by side and never merged. PI-F1 counts *which*
objects were found; jitter measures *where* the ones found in both ended up.
A prompt that finds the same people half a box to the left and a prompt that
finds different people are opposite failures, and a single "stability" number
cannot distinguish them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from utils.runner.logging_utils import get_logger
from utils.metrics.dump_adapter import ConditionDump, assert_comparable
from utils.metrics.matching import (
    IOU_MATCH,
    THRESHOLDS,
    class_detections,
    matched_ious,
    pairwise_pi_f1,
)

logger = get_logger(__name__)

#: Below this the noise floor is treated as unusable rather than divided by.
FLOOR_EPSILON = 1e-9


@dataclass(frozen=True)
class StabilityPair:
    """M1 / M3 / M4 for one class between exactly two conditions."""

    target_class: str
    q0: str
    q1: str
    arm: str
    score_threshold: float
    iou_threshold: float
    n_images: int
    #: Images where NEITHER condition detected the class. They score 1.0 by
    #: rev2's convention, which is what makes a silent model look invariant, so
    #: the count is always carried alongside the value.
    n_images_both_empty: int
    n_images_one_empty: int
    pi_f1: float | None
    #: M2. rev2's convention: images where BOTH conditions are silent have an
    #: undefined 0/0 CV and are excluded, counted separately.
    count_cv: float | None
    n_images_count_cv_defined: int
    #: M3. ``1 - mean(IoU)`` over matched pairs, micro-averaged over pairs.
    location_jitter: float | None
    mean_matched_iou: float | None
    n_matched_pairs: int
    #: M4. Per-image Spearman rho over matched pairs' scores, then averaged.
    score_rank_correlation: float | None
    n_images_rank_defined: int
    n_images_rank_undefined: int
    n_dets_q0: int
    n_dets_q1: int


def _image_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    rho = float(spearmanr(x, y).statistic)
    return None if np.isnan(rho) else rho


def stability_pair(q0: ConditionDump, q1: ConditionDump, target_class: str,
                   score_threshold: float = THRESHOLDS[0],
                   iou_threshold: float = IOU_MATCH) -> StabilityPair:
    """M1, M3 and M4 for one class over the shared census.

    PI-F1 is macro over images (rev2's convention, so the two lines' numbers
    are the same quantity); jitter is micro over matched pairs, so a
    thirty-box image does not weigh the same as a one-box image; the rank
    correlation is macro over the images where it is defined, because a
    correlation cannot be pooled across images with different score scales.
    """
    image_ids = assert_comparable(q0, q1)
    index = q0.class_index(target_class)

    f1s: list[float] = []
    cvs: list[float] = []
    rhos: list[float] = []
    iou_sum, iou_n = 0.0, 0
    both_empty = one_empty = 0
    rank_undefined = 0
    n_a = n_b = 0

    for image_id in image_ids:
        a = class_detections(q0.get(image_id), index, score_threshold)
        b = class_detections(q1.get(image_id), index, score_threshold)
        n_a += len(a)
        n_b += len(b)
        counts = np.asarray([len(a), len(b)], dtype=float)
        if counts.mean() > 0:
            cvs.append(float(counts.std() / counts.mean()))
        if len(a) == 0 and len(b) == 0:
            both_empty += 1
            f1s.append(1.0)
            rank_undefined += 1
            continue
        if len(a) == 0 or len(b) == 0:
            one_empty += 1
            f1s.append(0.0)
            rank_undefined += 1
            continue
        f1s.append(pairwise_pi_f1(q0.get(image_id), q1.get(image_id), index,
                                  score_threshold, iou_threshold))
        pairs = matched_ious(a, b, iou_threshold)
        for ai, bj, iou in pairs:
            iou_sum += iou
            iou_n += 1
        rho = _image_spearman(np.asarray([a[ai, 4] for ai, _, _ in pairs]),
                              np.asarray([b[bj, 4] for _, bj, _ in pairs]))
        if rho is None:
            rank_undefined += 1
        else:
            rhos.append(rho)

    mean_iou = (iou_sum / iou_n) if iou_n else None
    return StabilityPair(
        target_class=target_class, q0=q0.condition_id, q1=q1.condition_id, arm=q0.arm,
        score_threshold=score_threshold, iou_threshold=iou_threshold,
        n_images=len(image_ids), n_images_both_empty=both_empty,
        n_images_one_empty=one_empty,
        pi_f1=float(np.mean(f1s)) if f1s else None,
        count_cv=float(np.mean(cvs)) if cvs else None,
        n_images_count_cv_defined=len(cvs),
        location_jitter=(1.0 - mean_iou) if mean_iou is not None else None,
        mean_matched_iou=mean_iou, n_matched_pairs=iou_n,
        score_rank_correlation=float(np.mean(rhos)) if rhos else None,
        n_images_rank_defined=len(rhos), n_images_rank_undefined=rank_undefined,
        n_dets_q0=n_a, n_dets_q1=n_b,
    )


def noise_floor_normalised(pi_f1: float | None, floor: float | None) -> float | None:
    """``S_norm = 1 - PI-F1 / PI-F1_floor``. ``None`` when the floor is missing.

    The floor comes from the **permutation** condition: the same vocabulary in
    a different order, which changes nothing about what was asked. Instability
    at that level is sequence noise, so an ``S_norm`` near 0 means the observed
    contrast is indistinguishable from it and must NOT be reported as a real
    composition effect.

    A missing floor returns ``None``, which callers render as
    ``_unmeasured_``. It is never estimated: an invented floor would make every
    normalised number look decisive.
    """
    if pi_f1 is None or floor is None:
        return None
    if abs(floor) < FLOOR_EPSILON:
        logger.warning("noise floor PI-F1 is %.3g; S_norm would divide by ~0 and is "
                       "left unmeasured", floor)
        return None
    return 1.0 - pi_f1 / floor


@dataclass(frozen=True)
class ExactMatch:
    """Layer 0 — how often two conditions returned literally the same set."""

    q0: str
    q1: str
    n_images: int
    n_identical_images: int
    exact_match_rate: float | None
    max_abs_score_difference: float | None
    max_box_difference: float | None
    identical: bool


def _sorted_rows(dets: np.ndarray) -> np.ndarray:
    if not len(dets):
        return dets.reshape(-1, 6)
    order = np.lexsort((dets[:, 3], dets[:, 2], dets[:, 1], dets[:, 0],
                        -dets[:, 4], dets[:, 5]))
    return dets[order]


def exact_match_rate(q0: ConditionDump, q1: ConditionDump,
                     class_name: str | None = None,
                     score_threshold: float = 0.0,
                     atol: float = 0.0) -> ExactMatch:
    """Bit-level agreement between two conditions' detection sets.

    Args:
        class_name: Restrict to one class, or ``None`` for every class the two
            dumps share.
        atol: Tolerance for "identical". ``0.0`` means exactly equal, which is
            what the OWLv2 permutation null model requires.
    """
    image_ids = assert_comparable(q0, q1)
    indices = ([q0.class_index(class_name)] if class_name
               else list(range(len(q0.vocabulary))))
    identical_images = 0
    max_score = 0.0
    max_box = 0.0
    measured = False
    for image_id in image_ids:
        same = True
        for index in indices:
            a = _sorted_rows(class_detections(q0.get(image_id), index, score_threshold))
            b = _sorted_rows(class_detections(q1.get(image_id), index, score_threshold))
            if len(a) != len(b):
                same = False
                continue
            if not len(a):
                continue
            measured = True
            max_score = max(max_score, float(np.abs(a[:, 4] - b[:, 4]).max()))
            max_box = max(max_box, float(np.abs(a[:, :4] - b[:, :4]).max()))
            if max(np.abs(a[:, :5] - b[:, :5]).max(), 0.0) > atol:
                same = False
        identical_images += int(same)
    n = len(image_ids)
    return ExactMatch(
        q0=q0.condition_id, q1=q1.condition_id, n_images=n,
        n_identical_images=identical_images,
        exact_match_rate=(identical_images / n) if n else None,
        max_abs_score_difference=max_score if measured else None,
        max_box_difference=max_box if measured else None,
        identical=(identical_images == n and n > 0),
    )


def permutation_identity(q0: ConditionDump, q1: ConditionDump,
                         classes: Sequence[str] | None = None,
                         score_threshold: float = 0.0) -> Mapping[str, ExactMatch]:
    """Per-class layer-0 null model for a permutation pair.

    For a dual encoder every returned value must be exactly identical; a
    non-zero ``max_abs_score_difference`` there is a harness bug and is logged
    at error level rather than being reported as a model property.
    """
    names = list(classes) if classes else list(q0.vocabulary)
    out: dict[str, ExactMatch] = {}
    for name in names:
        result = exact_match_rate(q0, q1, name, score_threshold)
        out[name] = result
        if result.max_abs_score_difference not in (None, 0.0):
            logger.warning(
                "permutation %s vs %s changed %s scores by up to %.3g — for a "
                "dual-encoder model this is a harness bug, not a finding",
                q0.condition_id, q1.condition_id, name,
                result.max_abs_score_difference)
    return out
