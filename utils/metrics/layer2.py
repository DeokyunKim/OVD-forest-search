"""Layer 2 — the AP/AR statements, read from fragments.

Nothing here recomputes COCO accuracy. Every value is read from a §7 result
fragment with :func:`utils.metrics.analysis.metric`, which does an **exact** key
lookup and returns ``None`` for an absent class rather than 0.0.

Already implemented elsewhere and imported rather than rewritten:

============  =========================================================
metric        implementation
============  =========================================================
M8 ``I(c)``   ``utils.metrics.analysis.compute_interference``
M8 ``M(c)``   ``utils.metrics.analysis.compute_interference``
M9 residual   ``utils.metrics.analysis.compute_interference`` (additivity)
M13 (layer2)  ``utils.metrics.analysis.axis_c_mechanism`` (absorption/stealing
              from ΔAP/ΔAR; the detection-level version is in
              :mod:`.layer3`)
============  =========================================================

New here: the generic pairwise ΔAP/ΔAR between any two conditions (M6 — the
existing helpers only compare against the fixed A4/A1 baselines), and the AR−AP
gap (M7).

Layer 1 and layer 2 are always reported together. The case where layer 1 sees a
large effect and layer 2 sees none is not a contradiction to be resolved — it
is the evidence that aggregate accuracy hides per-object churn, and dropping
either half removes the methodological contribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from utils.metrics.analysis import (  # noqa: F401  (re-exported on purpose)
    InterferenceResult,
    axis_b_points,
    axis_c_mechanism,
    compute_interference,
    load_fragments,
)
from utils.metrics.analysis import metric as fragment_metric

#: The §7 keys every fragment carries per class.
METRIC_KEYS: tuple[str, ...] = ("AP", "AP50", "AP75", "AR", "AR100", "AR50", "AR75")


@dataclass(frozen=True)
class DeltaAccuracy:
    """M6 — per-class accuracy difference between two conditions.

    ``None`` for a key the class does not carry in one of the fragments (it
    was not queried there). A missing measurement never becomes 0.0.
    """

    target_class: str
    q0: str
    q1: str
    values_q0: Mapping[str, float | None]
    values_q1: Mapping[str, float | None]
    deltas: Mapping[str, float | None]


def delta_accuracy(fragment_q0: Mapping, fragment_q1: Mapping, target_class: str,
                   keys: Sequence[str] = METRIC_KEYS) -> DeltaAccuracy:
    """ΔAP / ΔAP50 / ΔAP75 / ΔAR for one class between two fragments."""
    a = {k: fragment_metric(fragment_q0, target_class, k) for k in keys}
    b = {k: fragment_metric(fragment_q1, target_class, k) for k in keys}
    return DeltaAccuracy(
        target_class=target_class,
        q0=fragment_q0["condition_id"], q1=fragment_q1["condition_id"],
        values_q0=a, values_q1=b,
        deltas={k: (b[k] - a[k]) if (a[k] is not None and b[k] is not None) else None
                for k in keys},
    )


@dataclass(frozen=True)
class AccuracyGap:
    """M7 — ``G_c(q) = AR_c(q) - AP_c(q)``.

    A large gap means the boxes are being found and then ranked or localised
    badly: recall is available that precision is not converting. A gap that
    collapses after fine-tuning (clothes: 65.1 -> 23.0 on the reference
    zero-shot/fine-tuned pair) says the model stopped proposing what it cannot
    score, which is a different change from "it got better".
    """

    target_class: str
    condition_id: str
    ap: float | None
    ar: float | None
    gap: float | None
    ap_key: str
    ar_key: str


def accuracy_gap(fragment: Mapping, target_class: str,
                 ap_key: str = "AP", ar_key: str = "AR") -> AccuracyGap:
    """The AR−AP gap for one class in one fragment. Exact keys, no prefixes."""
    ap = fragment_metric(fragment, target_class, ap_key)
    ar = fragment_metric(fragment, target_class, ar_key)
    return AccuracyGap(
        target_class=target_class, condition_id=fragment["condition_id"],
        ap=ap, ar=ar,
        gap=(ar - ap) if (ap is not None and ar is not None) else None,
        ap_key=ap_key, ar_key=ar_key,
    )


@dataclass(frozen=True)
class WorstMeanBest:
    """M10 — the spread of one class's accuracy over a set of conditions.

    In a high-risk domain the **worst** value is the deployment number: an
    operator does not get to choose the wording that happened to work.
    """

    target_class: str
    metric_key: str
    n_conditions: int
    worst: float | None
    mean: float | None
    best: float | None
    argworst: str | None
    argbest: str | None
    per_condition: Mapping[str, float]


def worst_mean_best(fragments: Mapping[str, Mapping], target_class: str,
                    metric_key: str = "AP") -> WorstMeanBest:
    """Worst / mean / best of one metric over the conditions that queried the class."""
    values = {cid: fragment_metric(f, target_class, metric_key)
              for cid, f in fragments.items()}
    values = {cid: v for cid, v in values.items() if v is not None}
    if not values:
        return WorstMeanBest(target_class, metric_key, 0, None, None, None, None, None, {})
    lo = min(values, key=lambda c: values[c])
    hi = max(values, key=lambda c: values[c])
    return WorstMeanBest(
        target_class=target_class, metric_key=metric_key, n_conditions=len(values),
        worst=values[lo], mean=sum(values.values()) / len(values), best=values[hi],
        argworst=lo, argbest=hi, per_condition=dict(values),
    )
