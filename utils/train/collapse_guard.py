"""In-loop collapse detection for Phase 2 training.

Thresholds are pre-registered in the training risk register §5 and
must not be tuned against an observed curve.

What it detects
---------------
Not divergence. No GroundingDINO run in this repository has ever diverged or
produced NaN. The failure is **joint-caption phrase-discrimination collapse**:
every class name lands on the same box with an identical score, so the model
becomes unable to tell the six V6 words apart while its loss curve still looks
ordinary.

    metric              healthy      collapsed
    dup_score_rate      ~0.002       1.0000
    labels_per_box      1.2 - 1.9    6.0  (= vocabulary size)
    score_spread        > 0          0.0
    loss_class (end)    ~0.004       ~0.278

Every full-scale Swin-B arm that adapted the frozen regions collapsed, with
onset at iteration 2,000-4,000 of a batch-2 run — optimizer step ~500-1,000.
Under the frozen spec's 5,667 steps that is 9-18% in, so a 250-step probe
cadence catches it within one interval.

Why in-process
--------------
The retired system detected collapse with an out-of-process watcher that read
checkpoints after the fact. That is the single cheapest thing that design got
wrong: a run could burn hours after collapsing. This guard runs inside the
training loop and can abort.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)

#: Scores within this of each other count as duplicated.
SCORE_EPSILON: float = 1e-4

ACTION_OK = "ok"
ACTION_ALARM = "alarm"
ACTION_ABORT = "abort"


@dataclass(frozen=True)
class CollapseThresholds:
    """Pre-registered. See the training risk register §5."""

    dup_score_rate_alarm: float = 0.01
    dup_score_rate_abort: float = 0.5
    labels_per_box_alarm: float = 3.0
    loss_class_alarm: float = 0.20
    loss_class_alarm_after_step: int = 2_000
    zero_grad_norm_steps_abort: int = 50
    probe_every_steps: int = 250
    probe_images: int = 64

    def as_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProbeMetrics:
    """One probe's measurement of discrimination health."""

    step: int
    n_boxes: int
    dup_score_rate: float
    labels_per_box: float
    mean_score_spread: float
    loss_class_ema: float | None = None

    def as_record(self) -> dict:
        return asdict(self)


def measure_discrimination(
    detections_by_image: Mapping[int, Sequence[tuple[str, float, tuple]]],
    epsilon: float = SCORE_EPSILON,
) -> tuple[float, float, float, int]:
    """``(dup_score_rate, labels_per_box, mean_score_spread, n_boxes)``.

    A box is *duplicated* when the words it carries score within ``epsilon`` of
    each other — the signature of the model assigning one box to every class at
    the same confidence. Boxes carrying a single word are counted in the
    denominator but can never be duplicated, so a healthy model with mostly
    single-word boxes scores near zero.
    """
    n_boxes = 0
    n_duplicated = 0
    total_labels = 0
    total_spread = 0.0
    for detections in detections_by_image.values():
        grouped: dict[tuple, list[float]] = {}
        for term, score, box in detections:
            key = tuple(round(float(v), 4) for v in box)
            grouped.setdefault(key, []).append(float(score))
        for scores in grouped.values():
            n_boxes += 1
            total_labels += len(scores)
            spread = max(scores) - min(scores)
            total_spread += spread
            if len(scores) > 1 and spread < epsilon:
                n_duplicated += 1
    if n_boxes == 0:
        raise ValueError("no boxes in the probe; refusing to report a health metric")
    return (n_duplicated / n_boxes, total_labels / n_boxes,
            total_spread / n_boxes, n_boxes)


@dataclass
class CollapseGuard:
    """Evaluates pre-registered thresholds and records a trajectory.

    ``armed`` is explicit and logged: a guard that exists but is disarmed is
    worse than none, because it produces false confidence.
    """

    thresholds: CollapseThresholds = field(default_factory=CollapseThresholds)
    armed: bool = True
    trajectory_path: Path | None = None
    history: list[ProbeMetrics] = field(default_factory=list)
    _zero_grad_streak: int = 0
    alarms: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Startup log line. Both the armed state and the thresholds."""
        state = "ARMED" if self.armed else "DISARMED (no aborts will fire)"
        return (f"CollapseGuard {state} | "
                f"dup_score_rate alarm>{self.thresholds.dup_score_rate_alarm} "
                f"abort>{self.thresholds.dup_score_rate_abort} | "
                f"labels_per_box alarm>{self.thresholds.labels_per_box_alarm} | "
                f"loss_class alarm>{self.thresholds.loss_class_alarm} "
                f"after step {self.thresholds.loss_class_alarm_after_step} | "
                f"probe every {self.thresholds.probe_every_steps} steps on "
                f"{self.thresholds.probe_images} images")

    def should_probe(self, step: int) -> bool:
        return step > 0 and step % self.thresholds.probe_every_steps == 0

    def evaluate(self, metrics: ProbeMetrics) -> str:
        """Return :data:`ACTION_OK`, :data:`ACTION_ALARM` or :data:`ACTION_ABORT`."""
        self.history.append(metrics)
        self._append_trajectory(metrics)

        action = ACTION_OK
        reasons: list[str] = []
        t = self.thresholds

        if metrics.dup_score_rate > t.dup_score_rate_abort:
            action = ACTION_ABORT
            reasons.append(f"dup_score_rate {metrics.dup_score_rate:.4f} > "
                           f"{t.dup_score_rate_abort} (collapse confirmed)")
        elif metrics.dup_score_rate > t.dup_score_rate_alarm:
            action = ACTION_ALARM
            reasons.append(f"dup_score_rate {metrics.dup_score_rate:.4f} > "
                           f"{t.dup_score_rate_alarm}")

        if metrics.labels_per_box > t.labels_per_box_alarm:
            action = ACTION_ABORT if action == ACTION_ABORT else ACTION_ALARM
            reasons.append(f"labels_per_box {metrics.labels_per_box:.3f} > "
                           f"{t.labels_per_box_alarm}")

        if (metrics.loss_class_ema is not None
                and metrics.step >= t.loss_class_alarm_after_step
                and metrics.loss_class_ema > t.loss_class_alarm):
            action = ACTION_ABORT if action == ACTION_ABORT else ACTION_ALARM
            reasons.append(f"loss_class EMA {metrics.loss_class_ema:.4f} > "
                           f"{t.loss_class_alarm} after step "
                           f"{t.loss_class_alarm_after_step}")

        if reasons:
            message = f"step {metrics.step}: " + "; ".join(reasons)
            self.alarms.append(message)
            logger.warning("COLLAPSE %s -- %s", action.upper(), message)
        if not self.armed and action == ACTION_ABORT:
            logger.warning("guard is DISARMED; abort suppressed")
            return ACTION_ALARM
        return action

    def observe_loss(self, step: int, loss: float) -> str:
        """NaN/Inf abort. Cheap, every step, no probe needed."""
        if loss != loss or loss in (float("inf"), float("-inf")):
            message = f"step {step}: loss is {loss}"
            self.alarms.append(message)
            logger.error("COLLAPSE ABORT -- %s", message)
            return ACTION_ABORT if self.armed else ACTION_ALARM
        return ACTION_OK

    def observe_grad_norm(self, step: int, grad_norm: float) -> str:
        """Abort on a sustained zero gradient — silent no-learning."""
        if grad_norm == 0.0:
            self._zero_grad_streak += 1
        else:
            self._zero_grad_streak = 0
        if self._zero_grad_streak >= self.thresholds.zero_grad_norm_steps_abort:
            message = (f"step {step}: gradient norm has been 0.0 for "
                       f"{self._zero_grad_streak} consecutive steps")
            self.alarms.append(message)
            logger.error("COLLAPSE ABORT -- %s", message)
            return ACTION_ABORT if self.armed else ACTION_ALARM
        return ACTION_OK

    def _append_trajectory(self, metrics: ProbeMetrics) -> None:
        if self.trajectory_path is None:
            return
        self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trajectory_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics.as_record()) + "\n")

    def write_alarm_file(self, path: Path, context: Mapping | None = None) -> None:
        """Leave a human-readable marker beside the checkpoints."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# COLLAPSE ALARM", "",
                 "Pre-registered thresholds: see the training risk register §5.",
                 "", "## Triggers", ""]
        lines += [f"- {alarm}" for alarm in self.alarms]
        if context:
            lines += ["", "## Context", ""]
            lines += [f"- {k}: {v}" for k, v in context.items()]
        path.write_text("\n".join(lines))
        logger.error("wrote %s", path)
