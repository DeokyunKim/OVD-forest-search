"""Training-side helpers shared by the GLIP and OmDet-Turbo trainers.

Companion of :mod:`models.shared.lora_spec` (placement) — this module holds the
things a *trainer* needs and that the V6 line's Grounding DINO trainer solves
inside ``train.py --model gdino``: reduced-precision kernel workarounds, the
spec §3 parameter-group split, and the §2 injection gate.

**Nothing in** ``utils/{lora_spec,collapse_guard,training_schedule,checkpoint,
train_caption,eval_protocol,coco_eval,conditions}.py`` **is modified.** They are
owned by the V6 line; this module imports them read-only. ``models/gdino/lora``
is imported only for :func:`~models.gdino.lora.make_lr_lambda` and
:func:`~models.gdino.lora.describe_schedule`, which are pure functions of the
frozen schedule and carry no Grounding DINO specifics.

Why the parameter-group split is re-implemented here
----------------------------------------------------
``gdino_lora.build_param_group_specs`` hard-codes ``GDINO_PLACEMENT`` for the
head and reduced-LR tests. The rule it encodes — every trainable tensor is
either LoRA, reduced-LR LoRA, or a fully-trained head, and an empty role is a
failure — is spec §3 and applies to every model, so it is re-expressed here
against an arbitrary :class:`~models.shared.lora_spec_v6line.LoraPlacement` instead of being
copied per model. One deliberate difference: GLIP has **no** ``sampling_offsets``
(it is anchor-based, not deformable-attention-based), so the reduced-LR role is
required only when the placement declares ``reduced_lr_patterns``. Demanding it
unconditionally would fail GLIP for a reason §3 never states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from utils.runner.logging_utils import get_logger
from models.shared.lora_spec_v6line import LoraPlacement, SpecViolationError
from utils.train.training_schedule import (
    EFFECTIVE_BATCH_SIZE,
    LR_FULL_HEAD,
    LR_LORA,
    LR_SAMPLING_OFFSETS_LORA,
    PER_DEVICE_BATCH_SIZE,
    WEIGHT_DECAY,
    WEIGHT_DECAY_NORM_BIAS,
    BatchPlan,
)

logger = get_logger(__name__)

LORA_PARAM_MARKERS: tuple[str, ...] = ("lora_A", "lora_B")

#: The three learning rates §3 fixes.
LR_BY_ROLE: Mapping[str, float] = {
    "lora": LR_LORA,
    "lora_sampling_offsets": LR_SAMPLING_OFFSETS_LORA,
    "head": LR_FULL_HEAD,
}

#: Per-device batch sizes for the two cross-architecture models, measured on an
#: RTX 6000 Ada (48 GiB) during the smoke runs. ``utils/train/training_schedule.py``
#: is the canonical table and is owned by the V6 line, so these values live here
#: until they are merged there; :func:`plan_batch_xarch` prefers the shared table
#: the moment it gains the key and *refuses* if the two disagree, so the fallback
#: can never silently outlive the merge.
XARCH_PER_DEVICE_BATCH: Mapping[str, int] = {"glip": 2, "omdet": 8}


def plan_batch_xarch(model: str, world_size: int = 1,
                     effective_batch: int = EFFECTIVE_BATCH_SIZE) -> BatchPlan:
    """§4.1 batch decomposition for a model the shared table may not list yet.

    Raises:
        KeyError: model in neither table (a batch size is never guessed).
        ValueError: the shared and local tables disagree, or the per-device
            batch does not divide the effective batch exactly.
    """
    local = XARCH_PER_DEVICE_BATCH.get(model)
    shared = PER_DEVICE_BATCH_SIZE.get(model)
    if shared is not None and local is not None and shared != local:
        raise ValueError(
            f"per-device batch for {model!r} is {shared} in "
            f"utils/train/training_schedule.py but {local} in XARCH_PER_DEVICE_BATCH. "
            f"The shared table is canonical — delete the local entry rather than "
            f"letting two batch sizes coexist")
    per_device = shared if shared is not None else local
    if per_device is None:
        raise KeyError(
            f"no per-device batch registered for {model!r} in either "
            f"utils/train/training_schedule.py or XARCH_PER_DEVICE_BATCH")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    denominator = per_device * world_size
    if effective_batch % denominator:
        raise ValueError(
            f"cannot hit effective batch {effective_batch} with per-device batch "
            f"{per_device} on {world_size} GPU(s)")
    if shared is None:
        logger.warning("per-device batch %d for %r read from XARCH_PER_DEVICE_BATCH; "
                       "utils/train/training_schedule.py does not list it yet",
                       per_device, model)
    return BatchPlan(model=model, per_device_batch=per_device,
                     grad_accumulation=effective_batch // denominator,
                     world_size=world_size)


# --- query strings -----------------------------------------------------------

def omdet_default_task_string(class_names: Sequence[str]) -> str:
    """The task prompt ``OmDetTurboProcessor`` builds when ``task`` is omitted.

    Transcribed from ``processing_omdet_turbo.py``::

        task = ["Detect {}.".format(", ".join(text_single)) for text_single in text]

    OmDet's queried text is a *class list*, not a caption, so byte-identity
    between training and evaluation means (a) the same class list in the same
    order and (b) neither path passing ``task``. Restating the formula here
    lets a test assert the string without instantiating the processor, and the
    trainer records the result in ``run_config.json``.
    """
    return "Detect {}.".format(", ".join(class_names))


# --- reduced precision -------------------------------------------------------

def enable_bf16_modulated_deform_conv() -> bool:
    """Run mmcv's modulated deformable conv in fp32 under bf16 autocast.

    GLIP's ``dyhead_tower`` builds each ``DyConv`` around
    ``mmcv.ops.ModulatedDeformConv2d``. Under ``torch.autocast(bfloat16)`` the
    offset/mask convolutions emit bf16, ``ModulatedDeformConv2dFunction.forward``
    casts input and weight to match (``input.type_as(offset)``), and the
    compiled kernel then raises::

        RuntimeError: "modulated_deformable_im2col_gpu" not implemented for 'BFloat16'

    All 18 DCNs sit in the fusion tower that spec §2 adapts, so they cannot be
    routed around.

    This is the **same remedy, applied the same way**, as the Grounding DINO
    trainer's ``enable_bf16_deformable_attention``: that function hit
    ``"ms_deform_attn_forward_cuda" not implemented for 'BFloat16'`` and up-casts
    the kernel's inputs to fp32, runs it, and casts the output back — following
    the precedent the external file already set for fp16. Here the same rule is
    applied to bf16 by rebinding the module global that
    ``ModulatedDeformConv2d.forward`` resolves at call time, so neither mmcv nor
    mmdetection is modified on disk.

    Numerically this means the deformable convolution runs at fp32 while the rest
    of the network runs at bf16 (§4.1). Recorded in ``run_config.json`` and in
    spec §8 so it is not an undocumented deviation.

    Returns:
        True if the patch is in place (idempotent).
    """
    import torch
    from mmcv.ops import modulated_deform_conv as mdc

    if getattr(mdc.modulated_deform_conv2d, "_v6_bf16_safe", False):
        return True
    original = mdc.modulated_deform_conv2d

    def modulated_deform_conv2d(input, offset, mask, weight, bias=None, *args, **kwargs):
        # `offset` drives the dtype inside mmcv's own forward, but any of the
        # three can be the bf16 one depending on how autocast reached them, so
        # all are tested rather than assuming offset is representative.
        tensors = (input, offset, mask)
        if not any(t is not None and t.dtype is torch.bfloat16 for t in tensors):
            return original(input, offset, mask, weight, bias, *args, **kwargs)
        output = original(input.float(), offset.float(), mask.float(),
                          weight.float(), None if bias is None else bias.float(),
                          *args, **kwargs)
        return output.to(torch.bfloat16)

    modulated_deform_conv2d._v6_bf16_safe = True  # type: ignore[attr-defined]
    mdc.modulated_deform_conv2d = modulated_deform_conv2d  # type: ignore[assignment]
    logger.info("modulated deformable conv will run in fp32 under bf16 autocast "
                "(the compiled mmcv CUDA op has no BFloat16 kernel); spec §4.1 "
                "keeps bf16 everywhere else")
    return True


def force_non_reentrant_checkpointing(use_reentrant: bool = False) -> bool:
    """Make every ``torch.utils.checkpoint.checkpoint`` call non-reentrant (§4.1).

    Same failure the Grounding DINO trainer documents: a checkpointed block whose
    inputs do not require grad returns **no** gradient under the reentrant
    implementation, so adapters behind a frozen backbone would train on exactly
    zero with a perfectly ordinary loss curve. GLIP's ``VLFuse`` calls
    ``checkpoint.checkpoint`` without ``use_reentrant=`` when
    ``bbox_head.use_checkpoint`` is set, so the same rebinding is needed.
    """
    import torch.utils.checkpoint as tc

    if getattr(tc.checkpoint, "_v6_non_reentrant", False):
        return True
    original = tc.checkpoint

    def checkpoint(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", use_reentrant)
        return original(function, *args, **kwargs)

    checkpoint._v6_non_reentrant = True  # type: ignore[attr-defined]
    tc.checkpoint = checkpoint  # type: ignore[assignment]
    logger.info("gradient checkpointing forced to use_reentrant=%s (spec §4.1)",
                use_reentrant)
    return True


# --- spec §3 parameter groups ------------------------------------------------

@dataclass(frozen=True)
class ParamGroupSpec:
    """One AdamW parameter group and the §3 role it plays."""

    name: str
    role: str
    lr: float
    weight_decay: float
    param_names: tuple[str, ...]
    n_params: int

    def as_record(self) -> dict:
        return {"name": self.name, "role": self.role, "lr": self.lr,
                "weight_decay": self.weight_decay,
                "n_tensors": len(self.param_names), "n_params": self.n_params}


def is_lora_param(name: str) -> bool:
    return any(marker in name for marker in LORA_PARAM_MARKERS)


def head_prefixes(placement: LoraPlacement) -> tuple[str, ...]:
    """Module prefixes plus the bare-parameter heads §2 also fully trains.

    ``LoraPlacement.modules_to_save`` can only name **modules**, because the
    shared verifier resolves it through ``named_modules()``. GLIP's head also
    includes three bare ``nn.Parameter`` attributes (``log_scale``,
    ``bias_lang``, ``bias0``) that §2 lists as fully trained; they live in
    :data:`~models.shared.lora_spec.HEAD_PARAMETER_PREFIXES` and are joined back
    in here so no §2 head tensor is left frozen by a representational accident.
    """
    from models.shared.lora_spec import HEAD_PARAMETER_PREFIXES

    return tuple(placement.modules_to_save) + tuple(
        HEAD_PARAMETER_PREFIXES.get(placement.model, ()))


def is_head_param(name: str, placement: LoraPlacement) -> bool:
    """Does ``name`` belong to one of the placement's fully-trained heads?

    Prefix match, not substring: GLIP's heads are named
    ``bbox_head.head.bbox_pred`` etc. and a substring test on ``bbox_head``
    would swallow the entire detector including every adapter.
    """
    return any(name == prefix or name.startswith(prefix + ".")
               for prefix in head_prefixes(placement))


def is_reduced_lr_param(name: str, placement: LoraPlacement) -> bool:
    """§3: ``sampling_offsets`` LoRA runs at lr/10 (grid initialisation)."""
    return any(pattern in name for pattern in placement.reduced_lr_patterns)


def is_no_decay_param(name: str) -> bool:
    """§3: weight decay 0.0 on norm and bias parameters."""
    return name.endswith(".bias") or name == "bias" or "norm" in name.lower()


def build_param_group_specs(named_trainable: Sequence[tuple[str, int]],
                            placement: LoraPlacement) -> list[ParamGroupSpec]:
    """Assign every trainable tensor to exactly one §3 role, or refuse.

    Args:
        named_trainable: ``(name, numel)`` per parameter with ``requires_grad``.
        placement: Supplies the head prefixes and the reduced-LR patterns.

    Raises:
        SpecViolationError: a trainable tensor matches no §3 role (it would then
            train at an unspecified learning rate), or a role the placement
            declares came out empty.
    """
    buckets: dict[tuple[str, bool], list[tuple[str, int]]] = {}
    for name, numel in named_trainable:
        if is_lora_param(name):
            role = ("lora_sampling_offsets" if is_reduced_lr_param(name, placement)
                    else "lora")
        elif is_head_param(name, placement):
            role = "head"
        else:
            raise SpecViolationError(
                f"trainable parameter {name!r} is neither a LoRA parameter nor "
                f"part of a §2 detection head ({list(placement.modules_to_save)}); "
                f"spec §3 assigns it no learning rate. Refusing to train at an "
                f"unspecified LR")
        buckets.setdefault((role, is_no_decay_param(name)), []).append((name, numel))

    # `lora` and `head` are required of every model. The reduced-LR role is
    # required only where the placement declares it: GLIP is anchor-based and
    # has no `sampling_offsets` at all, so demanding it would fail GLIP for a
    # reason §3 never states.
    required = ["lora", "head"]
    if placement.reduced_lr_patterns:
        required.append("lora_sampling_offsets")
    missing = [role for role in required if not any(k[0] == role for k in buckets)]
    if missing:
        raise SpecViolationError(
            f"spec §3 parameter group(s) {missing} are empty for model "
            f"{placement.model!r}. An empty group means those modules were never "
            f"made trainable, and each of the three §3 roles has its own learning "
            f"rate — none may silently vanish")

    specs: list[ParamGroupSpec] = []
    for role in ("lora", "lora_sampling_offsets", "head"):
        for no_decay in (False, True):
            entries = buckets.get((role, no_decay))
            if not entries:
                continue
            specs.append(ParamGroupSpec(
                name=f"{role}_no_decay" if no_decay else role,
                role=role,
                lr=LR_BY_ROLE[role],
                weight_decay=WEIGHT_DECAY_NORM_BIAS if no_decay else WEIGHT_DECAY,
                param_names=tuple(n for n, _ in entries),
                n_params=sum(k for _, k in entries),
            ))
    return specs


def build_param_groups(model, placement: LoraPlacement) -> tuple[list[dict],
                                                                list[ParamGroupSpec]]:
    """``(optimizer param_groups, their specs)`` for a live model."""
    named = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    specs = build_param_group_specs(named, placement)
    by_name = dict(model.named_parameters())
    groups = [{"params": [by_name[n] for n in spec.param_names],
               "lr": spec.lr,
               "weight_decay": spec.weight_decay,
               "name": spec.name}
              for spec in specs]
    return groups, specs


def describe_param_groups(specs: Sequence[ParamGroupSpec]) -> str:
    lines = [f"optimizer parameter groups (spec §3): "
             f"{len({s.role for s in specs})} learning rates, {len(specs)} groups"]
    for spec in specs:
        lines.append(f"  {spec.name:<26} lr={spec.lr:<8g} wd={spec.weight_decay:<5g} "
                     f"tensors={len(spec.param_names):<5d} params={spec.n_params:,}")
    return "\n".join(lines)


# --- spec §2 injection gate --------------------------------------------------

def mark_trainable(model, placement: LoraPlacement) -> tuple[int, int]:
    """Freeze everything, then unfreeze LoRA and the §2 heads.

    Returns ``(n_lora_tensors, n_head_tensors)``. A zero in either position is a
    §2 violation and is raised on, because both failure modes run to completion
    and only change the numbers.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    n_lora = n_head = 0
    for name, param in model.named_parameters():
        if is_lora_param(name):
            param.requires_grad_(True)
            n_lora += 1
        elif is_head_param(name, placement):
            param.requires_grad_(True)
            n_head += 1
    if n_lora == 0:
        raise SpecViolationError(
            f"[{placement.model}] no LoRA parameter was made trainable")
    if n_head == 0:
        raise SpecViolationError(
            f"[{placement.model}] no parameter under {list(head_prefixes(placement))} "
            f"was made trainable; §2 requires the detection head fully trained")
    missing = [prefix for prefix in head_prefixes(placement)
               if not any(name == prefix or name.startswith(prefix + ".")
                          for name, _ in model.named_parameters())]
    if missing:
        raise SpecViolationError(
            f"[{placement.model}] §2 head tensors {missing} do not exist in the "
            f"model; the head would be only partially trained")
    return n_lora, n_head


def assert_group_counts(model, placement: LoraPlacement,
                        expected: Mapping[str, int], group_of) -> dict[str, int]:
    """Count live LoRA modules per §2 group and compare to the measured table.

    A total alone is not enough — 72 is also reachable as 48 + 24, so a
    transformers/mmdet upgrade that reshaped one group while preserving the sum
    would pass a total check silently.

    Args:
        group_of: ``(model_key, module_name) -> group name``.

    Raises:
        SpecViolationError: any group's live count differs from ``expected``.
    """
    observed: dict[str, int] = {name: 0 for name in expected}
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") or type(module).__name__.startswith("Lora")):
            continue
        group = group_of(placement.model, name)
        observed[group] = observed.get(group, 0) + 1
    if observed != dict(expected):
        raise SpecViolationError(
            f"[{placement.model}] per-group LoRA counts {observed} != expected "
            f"{dict(expected)} (spec §2, measured on the real checkpoint). The "
            f"same total is reachable from different group combinations, so the "
            f"per-group comparison is the one that matters")
    return observed
