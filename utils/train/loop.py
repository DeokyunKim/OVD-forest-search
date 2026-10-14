"""The training loop, once, for all four architectures.

Four copies of this loop existed, one per model, and they agreed on everything
that matters: zero the gradients, accumulate `accum` micro-batches, clip, step,
step the schedule, ask the collapse guard, log, checkpoint. What actually
differs between the architectures is smaller than the loop around it:

* how one micro-batch becomes a loss -- different collators, different forward
  signatures, and OmDet-Turbo needs a Hungarian set-prediction loss the others
  do not;
* how the in-training probe is run;
* how a checkpoint is written.

Those three are what a model supplies. Everything else is here, so a change to
the schedule or the guard cannot reach three models and miss the fourth.

The operation order is load-bearing and is preserved exactly: the loss is
divided by the accumulation factor *before* `backward`, the running totals are
taken from detached scalars *after* it, and the EMA folds in the class loss for
the step that has just finished. Reordering any of those changes the numbers on
architectures whose backward is deterministic, which is the property the
eight-step gate depends on.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)

#: Smoothing for the class-loss EMA the collapse guard reads. Fixed, not a
#: knob: the guard's thresholds were calibrated against this value.
EMA_BETA = 0.98


@dataclass
class StepResult:
    """What one micro-batch produced.

    The two running-total terms are supplied by the model rather than derived
    here, because the architectures do not agree on where the sum happens.
    Grounding DINO accumulates float32 tensors and converts once at the end;
    the others convert each term to a Python float and accumulate in float64.
    Those give different last digits, so the choice belongs to whichever model
    made it and cannot be normalised away without changing its numbers.
    """

    #: The tensor to backpropagate. Not yet divided by the accumulation factor.
    loss: Any
    #: Added to the running loss total. Float or 0-d tensor -- the model's call.
    loss_scalar: Any
    #: Added to the running class-loss total, on the same terms.
    loss_class: Any


class TrainableModel(Protocol):
    """What the loop needs from a model, and nothing more."""

    def training_step(self, batch: Any) -> StepResult:
        """One micro-batch to a loss. Device placement belongs here: the shapes
        differ per architecture and the loop must not know them."""

    def clip_parameters(self) -> list:
        """Parameters passed to gradient clipping."""

    def probe(self, step: int, loss_class_ema: float | None) -> Any | None:
        """In-training health probe as a :class:`ProbeMetrics`, or None if this
        model has none. Raising is how an unmeasurable probe is reported."""

    def save_checkpoint(self, output_dir: Path, step: int, tag: str) -> Path:
        """Write a checkpoint and return where it went."""


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def cycle(loader: Any) -> Iterator:
    """Endless batches. The loop counts optimizer steps, not epochs."""
    while True:
        for batch in loader:
            yield batch


@dataclass
class LoopConfig:
    n_steps: int
    accumulation: int
    max_grad_norm: float
    log_every: int
    checkpoint_every: int
    seed: int


def run_training_loop(
    model: TrainableModel,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    guard: Any,
    config: LoopConfig,
    output_dir: Path,
    *,
    torch: Any,
    on_first_backward: Callable[[], None] | None = None,
) -> tuple[int, list[float]]:
    """Run the loop. Returns `(exit_code, per-step seconds)`.

    `torch` is passed in rather than imported: this module is imported by the
    four model environments, and each has its own torch. Taking it as an
    argument keeps this file free of a top-level import that would bind it to
    whichever one happened to be installed first.
    """
    from utils.train.collapse_guard import ACTION_ABORT, ACTION_OK

    data_iter = cycle(loader)
    train_log = output_dir / "train_log.jsonl"
    step_times: list[float] = []
    loss_class_ema: float | None = None
    exit_code = 0
    first_backward_checked = on_first_backward is None

    for step in range(config.n_steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        loss_class_total = 0.0

        for _ in range(config.accumulation):
            result = model.training_step(next(data_iter))
            (result.loss / config.accumulation).backward()
            loss_total += result.loss_scalar
            loss_class_total += result.loss_class

        if not first_backward_checked:
            on_first_backward()
            first_backward_checked = True

        grad_norm = torch.nn.utils.clip_grad_norm_(model.clip_parameters(),
                                                   config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        step_index = step + 1
        loss_value = float(loss_total) / config.accumulation
        loss_class_value = float(loss_class_total) / config.accumulation
        grad_norm_value = float(grad_norm)
        loss_class_ema = (loss_class_value if loss_class_ema is None
                          else EMA_BETA * loss_class_ema
                          + (1 - EMA_BETA) * loss_class_value)
        elapsed = time.perf_counter() - started
        step_times.append(elapsed)

        action = guard.observe_loss(step_index, loss_value)
        if action == ACTION_OK:
            action = guard.observe_grad_norm(step_index, grad_norm_value)
        if action == ACTION_OK and guard.should_probe(step_index):
            action = _probe(model, guard, step_index, loss_class_ema)

        append_jsonl(train_log, {
            "step": step_index, "loss": loss_value, "loss_class": loss_class_value,
            "loss_class_ema": loss_class_ema, "grad_norm": grad_norm_value,
            "lr_lora": optimizer.param_groups[0]["lr"], "sec_per_step": elapsed,
        })
        if step_index % config.log_every == 0 or step_index == config.n_steps:
            recent = step_times[-config.log_every:]
            logger.info("step %d/%d | loss %.4f | loss_class %.4f | grad_norm %.3f "
                        "| lr %.3e | %.2f s/step", step_index, config.n_steps,
                        loss_value, loss_class_value, grad_norm_value,
                        optimizer.param_groups[0]["lr"], sum(recent) / len(recent))

        if action == ACTION_ABORT:
            guard.write_alarm_file(output_dir / "ALARM.md", context={
                "step": step_index, "seed": config.seed,
                "output_dir": str(output_dir)})
            model.save_checkpoint(output_dir, step_index, f"aborted_step{step_index}")
            logger.error("ABORT at step %d; stopping the run", step_index)
            exit_code = 2
            break

        if (step_index % config.checkpoint_every == 0
                and step_index != config.n_steps):
            model.save_checkpoint(output_dir, step_index, f"step{step_index}")

    return exit_code, step_times


def _probe(model: TrainableModel, guard: Any, step_index: int,
           loss_class_ema: float | None) -> str:
    """Run the health probe and let the guard judge it.

    A probe that cannot be measured is an abort, not a skip: the guard exists to
    notice collapse, and a guard that silently stops looking is worse than none.
    """
    from utils.train.collapse_guard import ACTION_ABORT, ACTION_OK

    try:
        metrics = model.probe(step_index, loss_class_ema)
    except (RuntimeError, ValueError) as error:
        guard.alarms.append(f"step {step_index}: probe failed: {error}")
        logger.error("COLLAPSE ABORT -- probe unmeasurable at step %d: %s",
                     step_index, error)
        return ACTION_ABORT
    if metrics is None:
        return ACTION_OK
    action = guard.evaluate(metrics)
    # `as_record`, not named fields: OWLv2's probe returns its own metrics type
    # with its own field names, and a log line spelled for the other three
    # raises the first time it fires -- 250 steps in, long after any short gate
    # has finished.
    logger.info("step %d probe: %s", step_index, metrics.as_record())
    return action
