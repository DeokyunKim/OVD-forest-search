"""Layer 1.5 — per-object transitions between two query conditions.

This is the layer the study exists for. Layer 1 says the output *moved*; layer
2 says the aggregate AP/AR did or did not change. Neither can say what happened
to an individual person in the forest, which is the deployment-relevant fact.

Every ground-truth object of the target class gets a three-valued state under
each condition (:func:`~.matching.assign_gt_states`):

======  ==========================================================
state   meaning
======  ==========================================================
``P``   detected as the target class at IoU >= tau
``O``   detected, but claimed by another queried class
``M``   not detected at all
======  ==========================================================

and the pair of states is the transition:

=========  =====================  =====================  ==============
q0 \\ q1    ``P``                  ``O``                  ``M``
=========  =====================  =====================  ==============
``P``      retained (M18)         **stolen (M17)**       lost
``M``      **fired**              --                     unchanged
``O``      recovered              re-stolen              lost-from-O
=========  =====================  =====================  ==============

``O`` is a first-class state, and that is the methodological point. The
positive-congruent-training literature this extends (Yan et al., *Positive-
Congruent Training*, CVPR 2021) defines negative/positive flip rates over a
BINARY correct/incorrect outcome for image classification; a detector's
"incorrect" splits into *taken by a competing label* and *not found*, which
have opposite remedies. Yan et al. report that two ResNet152s trained from
different initialisations reach the same 78.3% accuracy while disagreeing on
10.4% of validation samples; the quantity here is that phenomenon carried onto
the detection and query axes, and it is an EXTENSION, not a reuse.

Definitions (N = number of ground-truth objects of the target class):

    NFR  = |{g : s0(g) = P, s1(g) != P}| / N
    PFR  = |{g : s0(g) != P, s1(g) = P}| / N
    SR   = |{g : s0(g) = P, s1(g) = O}| / N          (a component of NFR)
    NFR_cond = |{P -> not P}| / |{s0 = P}|

The identity ``recall(q1) - recall(q0) = PFR - NFR`` is exact by construction
(both sides are ``(|s1=P| - |s0=P|) / N``) and is asserted internally by
:mod:`.consistency`; the interesting version of the check compares ``PFR-NFR``
against the ``ΔAR`` a fragment reports from COCOeval.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from utils.runner.logging_utils import get_logger
from utils.metrics.dump_adapter import ConditionDump, GroundTruth, assert_comparable
from utils.metrics.matching import (
    IOU_MATCH,
    STATE_M,
    STATE_O,
    STATE_P,
    STATES,
    assign_gt_states,
    iou_matrix,
)

logger = get_logger(__name__)

#: Both flip rates at or above this, with |Δrecall| under
#: :data:`CHURN_DELTA_TOLERANCE`, is "hidden churn": the totals agree and the
#: individuals do not. In search and rescue that reads as "a different person
#: is found each time", so it is flagged rather than left for a reader to spot.
CHURN_RATE_FLOOR = 0.05
CHURN_DELTA_TOLERANCE = 0.01


@dataclass(frozen=True)
class RetainedStats:
    """M18 — what happened to the objects that stayed detected (P -> P).

    Box preservation is reported with its **5th percentile** as well as its
    centre: a mean IoU of 0.95 with a 5th percentile of 0.55 means one object
    in twenty moved enough to change which person a responder walks towards.
    """

    n: int
    iou_mean: float | None
    iou_median: float | None
    iou_p5: float | None
    delta_score_mean: float | None
    delta_score_median: float | None
    delta_score_abs_mean: float | None
    frac_score_increased: float | None
    frac_score_decreased: float | None
    spearman_rho: float | None
    spearman_n: int


@dataclass(frozen=True)
class FlipResult:
    """The full layer-1.5 result for one (target class, q0, q1)."""

    target_class: str
    q0: str
    q1: str
    arm: str
    iou_threshold: float
    score_threshold: float
    n_images: int
    n_gt: int
    #: ``"P->O"`` style keys over :data:`STATES` squared; every cell present.
    transitions: Mapping[str, int]
    n_p_q0: int
    n_p_q1: int
    negative_flip_rate: float | None
    positive_flip_rate: float | None
    stealing_rate: float | None
    negative_flip_rate_conditional: float | None
    positive_flip_rate_conditional: float | None
    recall_q0: float | None
    recall_q1: float | None
    delta_recall: float | None
    #: ``other class index -> count`` for the P -> O cell: who took them.
    stolen_by: Mapping[str, int]
    retained: RetainedStats
    hidden_churn: bool
    note: str


def _transition_key(s0: str, s1: str) -> str:
    return f"{s0}->{s1}"


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    """Spearman rho, or ``None`` where it is undefined (n < 2, or no variance)."""
    from scipy.stats import spearmanr

    if len(x) < 2:
        return None, len(x)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        # Zero variance in a rank vector: rho is 0/0, not 0.0.
        return None, len(x)
    rho = float(spearmanr(x, y).statistic)
    return (None if np.isnan(rho) else rho), len(x)


def _retained_stats(ious: Sequence[float], s0: Sequence[float],
                    s1: Sequence[float]) -> RetainedStats:
    n = len(ious)
    if n == 0:
        return RetainedStats(0, None, None, None, None, None, None, None, None, None, 0)
    iou = np.asarray(ious, dtype=float)
    a, b = np.asarray(s0, dtype=float), np.asarray(s1, dtype=float)
    delta = b - a
    rho, rho_n = _spearman(a, b)
    return RetainedStats(
        n=n,
        iou_mean=float(iou.mean()),
        iou_median=float(np.median(iou)),
        iou_p5=float(np.percentile(iou, 5)),
        delta_score_mean=float(delta.mean()),
        delta_score_median=float(np.median(delta)),
        delta_score_abs_mean=float(np.abs(delta).mean()),
        frac_score_increased=float((delta > 0).mean()),
        frac_score_decreased=float((delta < 0).mean()),
        spearman_rho=rho,
        spearman_n=rho_n,
    )


def flip_analysis(q0: ConditionDump, q1: ConditionDump, gt: GroundTruth,
                  target_class: str, iou_threshold: float = IOU_MATCH,
                  score_threshold: float = 0.0) -> FlipResult:
    """M16 / M17 / M18 for one target class between two conditions.

    Args:
        q0: The reference condition (e.g. ``person`` alone).
        q1: The comparison condition (e.g. ``person . clothes``).
        gt: Ground truth of ``target_class`` over the same census.

    Rates are ``None`` — never ``0.0`` — when their denominator is empty. A
    census with no ground truth of the class produces no flip rates at all,
    and a conditional rate whose denominator is 0 stays ``None`` while the
    unconditional one is still defined.
    """
    image_ids = assert_comparable(q0, q1)
    target_index = q0.class_index(target_class)

    counts: Counter[str] = Counter()
    stolen_by: Counter[str] = Counter()
    ious: list[float] = []
    scores0: list[float] = []
    scores1: list[float] = []
    n_gt = 0

    for image_id in image_ids:
        boxes = gt.get(image_id)
        if not len(boxes):
            continue
        a = assign_gt_states(q0.get(image_id), boxes, target_index,
                             iou_threshold, score_threshold)
        b = assign_gt_states(q1.get(image_id), boxes, target_index,
                             iou_threshold, score_threshold)
        n_gt += len(boxes)
        for j in range(len(boxes)):
            s0, s1 = a.states[j], b.states[j]
            counts[_transition_key(s0, s1)] += 1
            if s0 == STATE_P and s1 == STATE_O:
                index = b.other_class[j]
                stolen_by[q1.vocabulary[index] if index is not None else "?"] += 1
            if s0 == STATE_P and s1 == STATE_P:
                ious.append(float(iou_matrix(a.det_box[j:j + 1], b.det_box[j:j + 1])[0, 0]))
                scores0.append(float(a.det_score[j]))
                scores1.append(float(b.det_score[j]))

    transitions = {_transition_key(s0, s1): int(counts[_transition_key(s0, s1)])
                   for s0 in STATES for s1 in STATES}
    n_p_q0 = sum(v for k, v in transitions.items() if k.startswith(f"{STATE_P}->"))
    n_p_q1 = sum(v for k, v in transitions.items() if k.endswith(f"->{STATE_P}"))
    n_not_p_q0 = n_gt - n_p_q0

    def rate(numerator: int, denominator: int) -> float | None:
        return (numerator / denominator) if denominator else None

    lost = sum(transitions[_transition_key(STATE_P, s)] for s in (STATE_O, STATE_M))
    gained = sum(transitions[_transition_key(s, STATE_P)] for s in (STATE_O, STATE_M))
    nfr = rate(lost, n_gt)
    pfr = rate(gained, n_gt)
    recall0, recall1 = rate(n_p_q0, n_gt), rate(n_p_q1, n_gt)
    delta_recall = None if (recall0 is None or recall1 is None) else recall1 - recall0

    churn = bool(
        delta_recall is not None and abs(delta_recall) <= CHURN_DELTA_TOLERANCE
        and nfr is not None and pfr is not None
        and nfr >= CHURN_RATE_FLOOR and pfr >= CHURN_RATE_FLOOR)
    note = ""
    if n_gt == 0:
        note = ("no ground truth of this class in the census; every rate is "
                "unmeasured, which is not the same as 0.0")
    elif churn:
        note = (f"HIDDEN CHURN: |Δrecall| = {abs(delta_recall):.4f} but NFR = {nfr:.4f} "
                f"and PFR = {pfr:.4f}. The totals agree and the individuals do not — "
                "the model finds a different set of people under each wording.")
        logger.warning("%s vs %s (%s): %s", q0.condition_id, q1.condition_id,
                       target_class, note)

    return FlipResult(
        target_class=target_class, q0=q0.condition_id, q1=q1.condition_id, arm=q0.arm,
        iou_threshold=iou_threshold, score_threshold=score_threshold,
        n_images=len(image_ids), n_gt=n_gt, transitions=transitions,
        n_p_q0=n_p_q0, n_p_q1=n_p_q1,
        negative_flip_rate=nfr, positive_flip_rate=pfr,
        stealing_rate=rate(transitions[_transition_key(STATE_P, STATE_O)], n_gt),
        negative_flip_rate_conditional=rate(lost, n_p_q0),
        positive_flip_rate_conditional=rate(gained, n_not_p_q0),
        recall_q0=recall0, recall_q1=recall1, delta_recall=delta_recall,
        stolen_by=dict(stolen_by),
        retained=_retained_stats(ious, scores0, scores1),
        hidden_churn=churn, note=note,
    )
