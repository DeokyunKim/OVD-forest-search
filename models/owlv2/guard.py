"""In-loop collapse detection for OWLv2 — pre-registered, and model-appropriate.

Thresholds and their measured zero-shot baselines are pre-registered in
the training risk register §9 and must not be tuned against an
observed training curve.

Why the shared metrics do not transfer
--------------------------------------
``utils/train/collapse_guard.py`` watches ``dup_score_rate`` and
``labels_per_box``. Both were designed for a **joint-caption** detector: one
caption carrying six words is encoded once, so the words can stop being
distinguishable and every one of them then lands on the same box with the same
score. That is a real, observed Grounding DINO failure.

OWLv2 cannot fail that way. It takes an ordered **query list**, encodes each
term as its own sequence through the CLIP text tower, and the class head scores
every (patch, term) pair independently — ``models/owlv2/detector.py`` reads the
score matrix out per channel with no reduction over the term axis. Two
consequences:

1. ``labels_per_box`` is a property of the read-out, not of the model: the
   per-channel read-out attributes the same patch box to every term whose score
   survives the cap, so it sits near the vocabulary size **when the model is
   perfectly healthy**.
2. ``dup_score_rate`` counts boxes whose terms score within 1e-4 of each other.
   Independent channels over continuous logits essentially never tie, so it
   reads ~0 **whatever happens** — including under a genuine failure.

Wiring those two into an OWLv2 run would produce a guard that is armed, logs
reassuring numbers, and cannot fire. That is worse than no guard.

What replaces them
------------------
Four signals, each with a measured zero-shot baseline (§9 of the risk
register):

``text_embedding_drift``
    Max absolute change of the six frozen query embeddings against the
    reference taken before step 1. Spec §2 freezes the CLIP text tower, and
    §2's own note makes the permutation/composition invariance regression check
    a requirement. Zero-shot OWLv2's ``person`` AP was **0.580014 to six
    decimals across every A-axis condition**; that identity exists precisely
    because the text tower is frozen and the channels are independent. This is
    the mechanism behind that identity, measured directly and cheaply, instead
    of re-running an AP sweep every 250 steps.

``permutation_drift``
    Max absolute change of the per-channel score matrix when the query list is
    reversed and the permutation undone. The A-axis identity is a *behavioural*
    claim about the composed model, and a text tower can be frozen while
    something downstream still makes the score of ``person`` depend on which
    other words were queried. This measures the claim itself.

``channel_entropy``
    Normalised entropy of each patch's score profile across the six terms,
    averaged. It is the structural analogue of ``dup_score_rate``: it reaches
    exactly **1.0** when every term scores identically on every patch — the
    "all six words are the same word" failure — while remaining well defined on
    independent channels.

``distinct_box_rate`` / ``score_std``
    Box degeneracy and score-distribution degeneracy. PaliGemma2 and Qwen-VL
    produced constant box scores on this project, which makes AP undefined
    rather than merely bad, and a constant-box detector reads as "working" on
    every loss curve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from utils.runner.logging_utils import get_logger
from utils.train.collapse_guard import (
    ACTION_ABORT,
    ACTION_ALARM,
    ACTION_OK,
    CollapseGuard,
    CollapseThresholds,
)

logger = get_logger(__name__)

#: Measured on ``google/owlv2-large-patch14-ensemble`` **before any training**,
#: on the fixed 64-image held-out probe subset, under the same bf16 autocast the
#: trainer uses. Two repeats agreed to every printed digit (repeat jitter 0.0 on
#: all five signals), so every threshold below sits outside the measurement
#: noise by construction.
#: Source: ``measure_v6_owlv2_baseline.py`` (research tree) ->
#: ``outputs/v6/train_gates/owlv2_zeroshot_baseline.json``.
ZERO_SHOT_BASELINE: Mapping[str, float] = {
    "channel_entropy": 0.6023459881544113,
    "score_std": 0.0022044479940177553,
    "distinct_box_rate": 0.5787500000000001,
    "text_embedding_drift": 0.0,
    "permutation_drift": 0.0,
}


@dataclass(frozen=True)
class Owlv2Thresholds:
    """Pre-registered. See the training risk register §9.

    Every bound is an absolute number chosen from a measured zero-shot
    baseline, **not** a factor applied to a baseline re-measured at run time. A
    threshold derived from a fresh baseline would relax itself exactly when the
    baseline is already unhealthy.
    """

    #: Collapse direction is *up*: 1.0 means every term scores identically on
    #: every patch. Zero-shot baseline 0.602346 (64 held-out val images, bf16,
    #: repeat jitter 0.0). Alarm at 0.80 = halfway to total collapse,
    #: headroom x1.328; abort at 0.90, x1.494.
    channel_entropy_alarm: float = 0.80
    channel_entropy_abort: float = 0.90
    #: Collapse direction is *down* for the next two.
    #: Zero-shot baseline 0.00220445; a constant score matrix gives 0.
    #: Alarm at 5e-4 (headroom x4.409), abort at 2e-4 (x11.02). Generous on
    #: purpose: fine-tuning legitimately *raises* the spread, and only the
    #: downward direction is a failure.
    score_std_alarm: float = 5e-4
    score_std_abort: float = 2e-4
    #: Zero-shot baseline 0.578750 distinct boxes per 100 emitted detections;
    #: full box collapse gives 0.01. Alarm at 0.30 (headroom x1.929), abort at
    #: 0.10 (x5.788). Not placed just under the baseline because concentrating
    #: detections on fewer regions is a legitimate effect of fine-tuning on a
    #: six-class domain.
    distinct_box_rate_alarm: float = 0.30
    distinct_box_rate_abort: float = 0.10
    #: Spec §2 freezes the text tower, so the correct value is exactly 0.
    #: Measured 0.0 with 0.0 repeat jitter, so there is no multiplicative
    #: headroom to state: any non-zero reading is a real change.
    text_embedding_drift_alarm: float = 0.0
    text_embedding_drift_abort: float = 1e-4
    #: Independent channels make the score matrix permutation-equivariant.
    #: Measured **exactly 0.0** on the zero-shot checkpoint — the mechanism
    #: behind ``person`` AP = 0.580014 being identical across every A-axis
    #: condition. Alarm 1e-6 sits above a bitwise-zero measurement whose repeat
    #: jitter is also 0.0.
    permutation_drift_alarm: float = 1e-6
    permutation_drift_abort: float = 1e-3
    #: Shared with the GDINO guard so the two runs are comparable.
    loss_class_alarm: float = 0.20
    loss_class_alarm_after_step: int = 2_000
    zero_grad_norm_steps_abort: int = 50
    probe_every_steps: int = 250
    probe_images: int = 64
    #: Images used for the permutation check (two forwards each).
    permutation_images: int = 8

    def as_record(self) -> dict:
        return asdict(self)

    def as_shared_thresholds(self) -> CollapseThresholds:
        """The subset the shared guard evaluates: loss, NaN, dead gradient.

        ``dup_score_rate`` and ``labels_per_box`` are pushed out of reach
        (``inf``) rather than left at their GDINO values: on a per-channel
        read-out ``labels_per_box`` sits near the vocabulary size when the model
        is healthy, so the inherited 3.0 alarm would fire on every probe of a
        perfectly good run and train the reader to ignore the guard.
        """
        return CollapseThresholds(
            dup_score_rate_alarm=float("inf"),
            dup_score_rate_abort=float("inf"),
            labels_per_box_alarm=float("inf"),
            loss_class_alarm=self.loss_class_alarm,
            loss_class_alarm_after_step=self.loss_class_alarm_after_step,
            zero_grad_norm_steps_abort=self.zero_grad_norm_steps_abort,
            probe_every_steps=self.probe_every_steps,
            probe_images=self.probe_images,
        )


@dataclass(frozen=True)
class Owlv2ProbeMetrics:
    """One probe's measurement of OWLv2 health."""

    step: int
    n_images: int
    n_boxes: int
    channel_entropy: float
    score_std: float
    distinct_box_rate: float
    text_embedding_drift: float
    permutation_drift: float
    loss_class_ema: float | None = None

    def as_record(self) -> dict:
        return asdict(self)


def normalised_channel_entropy(scores) -> float:
    """Mean normalised entropy of each patch's score profile over the terms.

    Args:
        scores: ``(..., n_patches, n_terms)`` non-negative per-channel scores
            (sigmoid of the class logits).

    Returns:
        A float in ``[0, 1]``. **1.0 is total collapse** — every term scores
        identically on every patch, the OWLv2 analogue of one box carrying all
        six words at one score. A single queried term makes the quantity
        undefined, which is raised on rather than reported as 0 or 1.
    """
    import torch

    if scores.shape[-1] < 2:
        raise ValueError(
            "normalised channel entropy needs at least two queried terms; with "
            "one term every patch is trivially 'uniform' and the metric would "
            "read 1.0 on a healthy model")
    scores = scores.float().clamp_min(0.0)
    total = scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probabilities = scores / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    n_terms = scores.shape[-1]
    return float((entropy / torch.log(torch.tensor(float(n_terms)))).mean())


def distinct_box_rate(boxes, decimals: int = 3) -> float:
    """Fraction of the emitted detections that carry a *distinct* box.

    Args:
        boxes: ``(n_detections, 4)`` in absolute pixels.
        decimals: Rounding before the uniqueness test, so that floating-point
            noise does not count as a distinct box.

    A detector that has collapsed onto one region returns the same rectangle
    under every term and every rank, so this falls towards ``1 / n_detections``.
    """
    import torch

    if boxes.shape[0] == 0:
        raise ValueError("no detections; refusing to report a health metric")
    rounded = torch.round(boxes.float() * (10 ** decimals)) / (10 ** decimals)
    unique = torch.unique(rounded, dim=0)
    return float(unique.shape[0]) / float(boxes.shape[0])


def max_abs_drift(current, reference) -> float:
    """Max absolute elementwise difference, as a plain float."""
    import torch

    if current.shape != reference.shape:
        raise ValueError(f"shape mismatch {tuple(current.shape)} vs "
                         f"{tuple(reference.shape)}; a drift measured across "
                         f"different shapes is not a drift")
    return float(torch.max(torch.abs(current.float() - reference.float())))


@dataclass
class Owlv2CollapseGuard:
    """Pre-registered OWLv2 thresholds plus the shared loss/gradient checks.

    Composes :class:`~utils.train.collapse_guard.CollapseGuard` rather than
    subclassing it: the shared guard keeps sole ownership of the NaN, dead-
    gradient and ``loss_class`` rules (identical for every model), while the
    four OWLv2 signals are evaluated here. Both write into the **same** alarm
    list, so ``ALARM.md`` carries one ordered history.
    """

    thresholds: Owlv2Thresholds = field(default_factory=Owlv2Thresholds)
    armed: bool = True
    trajectory_path: Path | None = None
    history: list[Owlv2ProbeMetrics] = field(default_factory=list)
    _shared: CollapseGuard = field(init=False)

    def __post_init__(self) -> None:
        self._shared = CollapseGuard(
            thresholds=self.thresholds.as_shared_thresholds(),
            armed=self.armed, trajectory_path=None)

    # --- shared, cheap, every step -------------------------------------------

    @property
    def alarms(self) -> list[str]:
        return self._shared.alarms

    def observe_loss(self, step: int, loss: float) -> str:
        return self._shared.observe_loss(step, loss)

    def observe_grad_norm(self, step: int, grad_norm: float) -> str:
        return self._shared.observe_grad_norm(step, grad_norm)

    def write_alarm_file(self, path: Path, context: Mapping | None = None) -> None:
        self._shared.write_alarm_file(path, context)

    def should_probe(self, step: int) -> bool:
        return step > 0 and step % self.thresholds.probe_every_steps == 0

    # --- OWLv2 probe ----------------------------------------------------------

    def describe(self) -> str:
        """Startup line. Armed state and every threshold, per risk register §5."""
        state = "ARMED" if self.armed else "DISARMED (no aborts will fire)"
        t = self.thresholds
        return (
            f"CollapseGuard[owlv2] {state} | "
            f"channel_entropy alarm>{t.channel_entropy_alarm} "
            f"abort>{t.channel_entropy_abort} | "
            f"score_std alarm<{t.score_std_alarm} abort<{t.score_std_abort} | "
            f"distinct_box_rate alarm<{t.distinct_box_rate_alarm} "
            f"abort<{t.distinct_box_rate_abort} | "
            f"text_embedding_drift alarm>{t.text_embedding_drift_alarm} "
            f"abort>{t.text_embedding_drift_abort} | "
            f"permutation_drift alarm>{t.permutation_drift_alarm} "
            f"abort>{t.permutation_drift_abort} | "
            f"loss_class alarm>{t.loss_class_alarm} after step "
            f"{t.loss_class_alarm_after_step} | grad-norm 0 for "
            f"{t.zero_grad_norm_steps_abort} steps aborts | probe every "
            f"{t.probe_every_steps} steps on {t.probe_images} images "
            f"({t.permutation_images} permuted). "
            f"dup_score_rate / labels_per_box are NOT used: OWLv2 scores each "
            f"query channel independently, so both are structurally incapable "
            f"of firing (see models/owlv2/guard.py)")

    def evaluate(self, metrics: Owlv2ProbeMetrics) -> str:
        """Return :data:`ACTION_OK`, :data:`ACTION_ALARM` or :data:`ACTION_ABORT`."""
        self.history.append(metrics)
        self._append_trajectory(metrics)

        t = self.thresholds
        reasons: list[str] = []
        abort = False

        def note(condition_abort: bool, condition_alarm: bool, text: str) -> None:
            nonlocal abort
            if condition_abort:
                abort = True
                reasons.append(text)
            elif condition_alarm:
                reasons.append(text)

        note(metrics.channel_entropy > t.channel_entropy_abort,
             metrics.channel_entropy > t.channel_entropy_alarm,
             f"channel_entropy {metrics.channel_entropy:.6f} > "
             f"{t.channel_entropy_abort if metrics.channel_entropy > t.channel_entropy_abort else t.channel_entropy_alarm} "
             f"(1.0 = every term scores identically)")
        note(metrics.score_std < t.score_std_abort,
             metrics.score_std < t.score_std_alarm,
             f"score_std {metrics.score_std:.6f} < "
             f"{t.score_std_abort if metrics.score_std < t.score_std_abort else t.score_std_alarm} "
             f"(score distribution degenerating to a constant)")
        note(metrics.distinct_box_rate < t.distinct_box_rate_abort,
             metrics.distinct_box_rate < t.distinct_box_rate_alarm,
             f"distinct_box_rate {metrics.distinct_box_rate:.6f} < "
             f"{t.distinct_box_rate_abort if metrics.distinct_box_rate < t.distinct_box_rate_abort else t.distinct_box_rate_alarm} "
             f"(boxes collapsing onto one region)")
        note(metrics.text_embedding_drift > t.text_embedding_drift_abort,
             metrics.text_embedding_drift > t.text_embedding_drift_alarm,
             f"text_embedding_drift {metrics.text_embedding_drift:.3e} > "
             f"{t.text_embedding_drift_abort if metrics.text_embedding_drift > t.text_embedding_drift_abort else t.text_embedding_drift_alarm} "
             f"(spec §2 freezes the CLIP text tower; the correct value is 0)")
        note(metrics.permutation_drift > t.permutation_drift_abort,
             metrics.permutation_drift > t.permutation_drift_alarm,
             f"permutation_drift {metrics.permutation_drift:.3e} > "
             f"{t.permutation_drift_abort if metrics.permutation_drift > t.permutation_drift_abort else t.permutation_drift_alarm} "
             f"(per-channel scores must not depend on query order)")

        if (metrics.loss_class_ema is not None
                and metrics.step >= t.loss_class_alarm_after_step
                and metrics.loss_class_ema > t.loss_class_alarm):
            reasons.append(f"loss_class EMA {metrics.loss_class_ema:.4f} > "
                           f"{t.loss_class_alarm} after step "
                           f"{t.loss_class_alarm_after_step}")

        action = ACTION_ABORT if abort else (ACTION_ALARM if reasons else ACTION_OK)
        if reasons:
            message = f"step {metrics.step}: " + "; ".join(reasons)
            self.alarms.append(message)
            logger.warning("COLLAPSE %s -- %s", action.upper(), message)
        if not self.armed and action == ACTION_ABORT:
            logger.warning("guard is DISARMED; abort suppressed")
            return ACTION_ALARM
        return action

    def assert_healthy_baseline(self, metrics: Owlv2ProbeMetrics) -> None:
        """Refuse to start if the *zero-shot* model already trips a threshold.

        The thresholds are pre-registered against a measured healthy baseline.
        If the untrained model does not satisfy them, either the baseline was
        measured on a different configuration or the probe is broken — and in
        both cases every later probe is uninterpretable.
        """
        from models.shared.lora_spec_v6line import SpecViolationError

        action = self.evaluate(metrics)
        if action != ACTION_OK:
            raise SpecViolationError(
                f"the zero-shot baseline probe already trips a pre-registered "
                f"OWLv2 threshold: {self.alarms[-1]}. Thresholds come from "
                f"the training risk register §9 and are not to be "
                f"relaxed here; investigate the probe or the checkpoint")

    def _append_trajectory(self, metrics: Owlv2ProbeMetrics) -> None:
        if self.trajectory_path is None:
            return
        import json

        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics.as_record()) + "\n")


__all__ = [
    "ACTION_ABORT",
    "ACTION_ALARM",
    "ACTION_OK",
    "Owlv2CollapseGuard",
    "Owlv2ProbeMetrics",
    "Owlv2Thresholds",
    "distinct_box_rate",
    "max_abs_drift",
    "normalised_channel_entropy",
]


def summarise_thresholds(thresholds: Owlv2Thresholds,
                         baselines: Mapping[str, float] = ZERO_SHOT_BASELINE,
                         ) -> Sequence[str]:
    """Human-readable threshold/baseline/headroom table, for the run log.

    ``baselines`` are the numbers measured on the zero-shot checkpoint and
    recorded in the risk register; the headroom column is the factor between
    the measured value and the alarm bound, in whichever direction collapse
    moves the metric.
    """
    rows = [
        ("channel_entropy", baselines.get("channel_entropy"),
         thresholds.channel_entropy_alarm, "up"),
        ("score_std", baselines.get("score_std"), thresholds.score_std_alarm, "down"),
        ("distinct_box_rate", baselines.get("distinct_box_rate"),
         thresholds.distinct_box_rate_alarm, "down"),
        ("text_embedding_drift", baselines.get("text_embedding_drift"),
         thresholds.text_embedding_drift_abort, "up"),
        ("permutation_drift", baselines.get("permutation_drift"),
         thresholds.permutation_drift_alarm, "up"),
    ]
    lines = ["pre-registered OWLv2 thresholds vs measured zero-shot baseline "
             "(the training risk register §9)"]
    for name, baseline, alarm, direction in rows:
        if baseline is None:
            lines.append(f"  {name:<22} baseline _미측정_")
            continue
        if baseline == 0.0:
            # A multiplicative headroom is undefined against an exactly-zero
            # baseline, and printing "xinf" would read as "very safe" when the
            # truth is the opposite: the bound is absolute and tight.
            headroom = (f"absolute bound {alarm:g} above a bitwise-zero baseline "
                        f"(repeat jitter 0.0)")
        elif direction == "up":
            headroom = f"headroom x{alarm / baseline:.3f}"
        else:
            headroom = f"headroom x{baseline / alarm:.3f}" if alarm else "headroom inf"
        lines.append(f"  {name:<22} baseline {baseline:<12.8g} alarm {alarm:<10.6g} "
                     f"collapse moves it {direction:<4} {headroom}")
    return lines
