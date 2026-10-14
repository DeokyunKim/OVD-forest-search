"""The frozen training schedule — single definition site.

Spec: the frozen-hyperparameter specification §1/§3/§4.1 (as amended §8).

The step count changed on 2026-08-08 from 8,000 to 5,667. The spec's own
epoch check made 8,000 steps 1.41 epochs on a 90,670-image training set,
outside the 3-15 range the spec names; the user fixed the budget at exactly
one epoch of the full training set instead. **The same value applies to every
model** — a per-model step count would break the equal-budget control.

One epoch is still outside the spec's 3-15 range. That is a deliberate user
decision, recorded here and in the spec's change log so it reaches the
limitations section rather than being quietly normalised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

#: Image count of ``datasets/v6_fp_fb/train.json``. The step count is derived
#: from this, so a dataset rebuild that moves it must move the schedule too.
TRAIN_SET_SIZE: int = 90_670

EFFECTIVE_BATCH_SIZE: int = 16

#: ceil(90_670 / 16) = 5_667. One full epoch, every model.
TOTAL_STEPS: int = 5_667

WARMUP_FRACTION: float = 0.03
MAX_GRAD_NORM: float = 1.0

#: Per-device batch sizes; grad accumulation is derived to reach 16 (§4.1).
def _per_device_batch_sizes() -> Mapping[str, int]:
    """From configs/models.yaml, so the split is visible where the run is described."""
    from utils.runner.model_registry import per_device_batches
    return per_device_batches()


PER_DEVICE_BATCH_SIZE: Mapping[str, int] = _per_device_batch_sizes()
#: All four divide EFFECTIVE_BATCH_SIZE exactly at world_size 1
#: (accum 8 / 4 / 8 / 2). ``glip``/``omdet`` merged 2026-08-10 from the xarch
#: line's measurements, not from §4.1 (which names only gdino and owlv2):
#:   glip  batch 2 -> 4.483 s/step,  7.74 GiB
#:         batch 4 -> 4.815 s/step, 14.45 GiB  (slower AND double the memory:
#:         already GPU-bound, so a larger batch only adds padding waste)
#:   omdet batch 8 -> 0.250 s/step,  5.11 GiB (11% of a 48 GiB card)

SEEDS: tuple[int, ...] = (0, 1, 2)
PRECISION: str = "bf16"
GRADIENT_CHECKPOINTING: bool = True
USE_REENTRANT: bool = False
EARLY_STOPPING: bool = False
#: Final step, never a best-validation checkpoint (§4.1).
CHECKPOINT_SELECTION: str = "final_step"


@dataclass(frozen=True)
class LoraConfig:
    """§1, identical across models."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    use_rslora: bool = False


LORA = LoraConfig()

#: §3 learning rates.
LR_LORA: float = 1e-4
LR_FULL_HEAD: float = 1e-5
LR_SAMPLING_OFFSETS_LORA: float = 1e-5
WEIGHT_DECAY: float = 0.01
WEIGHT_DECAY_NORM_BIAS: float = 0.0
ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
ADAM_EPS: float = 1e-8
SCHEDULER: str = "cosine"
#: Final LR as a fraction of peak (§3).
FINAL_LR_FRACTION: float = 0.01


def grad_accumulation_steps(model: str) -> int:
    """Accumulation needed to reach :data:`EFFECTIVE_BATCH_SIZE` for ``model``.

    Raises:
        KeyError: on an unknown model (never guess a batch size).
        ValueError: if the per-device batch does not divide the effective batch.
    """
    per_device = PER_DEVICE_BATCH_SIZE[model]
    if EFFECTIVE_BATCH_SIZE % per_device:
        raise ValueError(
            f"per-device batch {per_device} for {model!r} does not divide the "
            f"effective batch {EFFECTIVE_BATCH_SIZE}"
        )
    return EFFECTIVE_BATCH_SIZE // per_device


def warmup_steps() -> int:
    """Linear warmup length: 3% of the total (§3)."""
    return round(TOTAL_STEPS * WARMUP_FRACTION)


def epochs() -> float:
    """Epoch equivalent of the frozen budget, for the limitations section."""
    return TOTAL_STEPS * EFFECTIVE_BATCH_SIZE / TRAIN_SET_SIZE


@dataclass(frozen=True)
class BatchPlan:
    """A verified decomposition of the effective batch (spec §4.1)."""

    model: str
    per_device_batch: int
    grad_accumulation: int
    world_size: int

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch * self.grad_accumulation * self.world_size

    def describe(self) -> str:
        """The line spec §4.1 requires in the log."""
        return (f"effective batch {self.effective_batch} = "
                f"batch {self.per_device_batch} x accum {self.grad_accumulation} "
                f"x world {self.world_size}")


def plan_batch(model: str, world_size: int,
               effective_batch: int = EFFECTIVE_BATCH_SIZE) -> BatchPlan:
    """Derive ``grad_accumulation`` from the GPU count, or refuse.

    Spec §4.1 fixes the effective batch at 16 and the per-device batch per
    model. The accumulation factor is therefore determined, not chosen — and it
    only exists for GPU counts that divide the remainder exactly. Three GPUs
    cannot produce an effective batch of 16 for GDINO (16 / 2 / 3 is not an
    integer), so this refuses instead of silently rounding to 15 or 18 and
    breaking the equal-budget control across runs.

    Raises:
        KeyError: unknown model (a per-device batch is never guessed).
        ValueError: the decomposition is not exact.
    """
    per_device = PER_DEVICE_BATCH_SIZE[model]
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    denominator = per_device * world_size
    if effective_batch % denominator:
        raise ValueError(
            f"cannot hit effective batch {effective_batch} with per-device batch "
            f"{per_device} on {world_size} GPU(s): {effective_batch} / "
            f"{denominator} is not an integer. Use a GPU count that divides "
            f"{effective_batch // per_device}")
    return BatchPlan(model=model, per_device_batch=per_device,
                     grad_accumulation=effective_batch // denominator,
                     world_size=world_size)
