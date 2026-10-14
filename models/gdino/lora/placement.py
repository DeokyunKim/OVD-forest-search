"""Spec §2 LoRA placement translated into the *external repo's* module names.

Why a translation layer is needed
---------------------------------
Spec §2's regex is written against HuggingFace / MM-Grounding-DINO naming,
where one ``encoder.layers.N`` block contains the text enhancer, the fusion
block and the deformable image attention as children, and where attention
projections are called ``query/key/value/output.dense`` and the fusion
projections ``vision_proj/text_proj/values_*_proj/out_*_proj``.

The checkpoint this project trains is built by
the checkout `UC_GDINO_REPO` names, whose module tree is the
original IDEA-Research layout:

===================================  ==========================================
spec §2 (HF naming)                  this repo
===================================  ==========================================
``encoder.layers.N.text_enhancer``   ``transformer.encoder.text_layers.N``
``encoder.layers.N.fusion_layer``    ``transformer.encoder.fusion_layers.N``
``encoder.layers.N.deformable``      ``transformer.encoder.layers.N``
``decoder.layers.N``                 ``transformer.decoder.layers.N``
``…self_attn.{query,key,value}``     ``…self_attn.linear_{q,k,v}``
``…self_attn.output.dense``          ``…self_attn.linear_o``
``…{vision,text}_proj``              ``…{v,l}_proj``
``…values_{vision,text}_proj``       ``…values_{v,l}_proj``
``…out_{vision,text}_proj``          ``…out_{v,l}_proj``
``…fc1 / …fc2``                      ``…linear1 / …linear2``
``…{sampling_offsets,               identical
   attention_weights,
   value_proj,output_proj}``
===================================  ==========================================

Applying §2's regex *verbatim* to this repo matches only 48 modules — the two
deformable attentions — because none of the renamed projections match. That is
not spec §2's scope; it is an artefact of the naming. So the placement is
translated name-for-name here, the mapping above is asserted in
the regression tests, and the four groups whose names are identical in
both layouts are cross-checked against ``GDINO_PLACEMENT`` itself so the
translation cannot drift away from the spec silently.

Attention decomposition comes first
-----------------------------------
``self_attn`` / ``ca_text`` are ``nn.MultiheadAttention`` with a *fused*
``in_proj_weight``: they expose **zero** ``nn.Linear`` children. The external
repo's ``apply_custom_attention`` replaces them with an equivalent module
holding ``linear_q/k/v/o``. Collecting targets before that call yields 0
instead of 24 per group, and the run would train a smaller model with no error
anywhere — so :func:`collect_targets` refuses when a decomposable group matches
nothing.

What is frozen
--------------
``backbone`` (Swin) and ``bert`` (the text encoder) carry **no** adapters, per
§2's common rule. Those are the two regions the retired 216-module preset
adapted, and every full-scale arm that did so collapsed.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from utils.runner.logging_utils import get_logger
from models.shared.lora_spec_v6line import GDINO_PLACEMENT, LoraPlacement, SpecViolationError
from utils.train.training_schedule import (
    FINAL_LR_FRACTION,
    LR_FULL_HEAD,
    LR_LORA,
    LR_SAMPLING_OFFSETS_LORA,
    TOTAL_STEPS,
    WEIGHT_DECAY,
    WEIGHT_DECAY_NORM_BIAS,
    warmup_steps,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoraGroup:
    """One named block of §2-targeted ``nn.Linear`` modules.

    Attributes:
        name: Group key used in logs, the injection map and the manifest.
        pattern: Full-dotted-name regex in *this repo's* naming.
        expected: Module count measured on the live Swin-B model. A regex that
            matches nothing fails silently otherwise.
        spec_suffixes: The §2 regex alternatives this group implements, so the
            translation is auditable rather than asserted.
        needs_decomposition: True when the modules only exist after
            ``apply_custom_attention``.
        is_attention: False for feed-forward blocks (``fc1``/``fc2`` in §2).
    """

    name: str
    pattern: str
    expected: int
    spec_suffixes: tuple[str, ...]
    needs_decomposition: bool = False
    is_attention: bool = True

    @property
    def regex(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


#: Every group spec §2's regex names, in this repo's module naming.
GDINO_LORA_GROUPS: tuple[LoraGroup, ...] = (
    LoraGroup(
        name="fusion_bi_attn",
        pattern=(r"^transformer\.encoder\.fusion_layers\.\d+\.attn\."
                 r"(v_proj|l_proj|values_v_proj|values_l_proj|out_v_proj|out_l_proj)$"),
        expected=36,
        spec_suffixes=("vision_proj", "text_proj", "values_vision_proj",
                       "values_text_proj", "out_vision_proj", "out_text_proj"),
    ),
    LoraGroup(
        name="text_enhancer_self_attn",
        pattern=r"^transformer\.encoder\.text_layers\.\d+\.self_attn\.linear_[qkvo]$",
        expected=24,
        spec_suffixes=("query", "key", "value", "out_proj"),
        needs_decomposition=True,
    ),
    LoraGroup(
        name="text_enhancer_ffn",
        pattern=r"^transformer\.encoder\.text_layers\.\d+\.linear[12]$",
        expected=12,
        spec_suffixes=("fc1", "fc2"),
        is_attention=False,
    ),
    LoraGroup(
        name="encoder_image_deform_attn",
        pattern=(r"^transformer\.encoder\.layers\.\d+\.self_attn\."
                 r"(sampling_offsets|attention_weights|value_proj|output_proj)$"),
        expected=24,
        spec_suffixes=("sampling_offsets", "attention_weights", "value_proj",
                       "output_proj"),
    ),
    LoraGroup(
        name="encoder_image_ffn",
        pattern=r"^transformer\.encoder\.layers\.\d+\.linear[12]$",
        expected=12,
        spec_suffixes=("fc1", "fc2"),
        is_attention=False,
    ),
    LoraGroup(
        name="decoder_self_attn",
        pattern=r"^transformer\.decoder\.layers\.\d+\.self_attn\.linear_[qkvo]$",
        expected=24,
        spec_suffixes=("query", "key", "value", "out_proj"),
        needs_decomposition=True,
    ),
    LoraGroup(
        name="decoder_cross_attn_text",
        pattern=r"^transformer\.decoder\.layers\.\d+\.ca_text\.linear_[qkvo]$",
        expected=24,
        spec_suffixes=("query", "key", "value", "out_proj"),
        needs_decomposition=True,
    ),
    LoraGroup(
        name="decoder_cross_attn_image",
        pattern=(r"^transformer\.decoder\.layers\.\d+\.cross_attn\."
                 r"(sampling_offsets|attention_weights|value_proj|output_proj)$"),
        expected=24,
        spec_suffixes=("sampling_offsets", "attention_weights", "value_proj",
                       "output_proj"),
    ),
    LoraGroup(
        name="decoder_ffn",
        pattern=r"^transformer\.decoder\.layers\.\d+\.linear[12]$",
        expected=12,
        spec_suffixes=("fc1", "fc2"),
        is_attention=False,
    ),
)

GROUPS_BY_NAME: Mapping[str, LoraGroup] = {g.name: g for g in GDINO_LORA_GROUPS}

#: Named scopes. ``spec2`` is every module class §2's regex names (192).
#: ``spec2_attention`` drops the three feed-forward groups (156), matching §2's
#: prose rule ("융합·디코더 **어텐션**에 LoRA") rather than its regex. The
#: difference is recorded in the injection map and the manifest so a run can
#: never be mistaken for the other.
SCOPES: Mapping[str, tuple[str, ...]] = {
    "spec2": tuple(g.name for g in GDINO_LORA_GROUPS),
    "spec2_attention": tuple(g.name for g in GDINO_LORA_GROUPS if g.is_attention),
}
DEFAULT_SCOPE = "spec2"

#: Allowed regions in this repo's naming. ``lora_spec.GDINO_PLACEMENT`` lists
#: only ``(transformer.)(encoder|decoder).layers`` because HF puts the enhancer
#: sub-blocks *inside* ``encoder.layers.N``; here they are siblings, so the two
#: sibling stacks are named explicitly. Nothing outside the feature enhancer
#: and the decoder is admitted.
REPO_ALLOWED_PREFIXES: tuple[str, ...] = (
    "transformer.encoder.layers",
    "transformer.encoder.text_layers",
    "transformer.encoder.fusion_layers",
    "transformer.decoder.layers",
)


def build_placement(scope: str = DEFAULT_SCOPE) -> LoraPlacement:
    """A :class:`LoraPlacement` for ``scope`` in this repo's naming.

    Frozen prefixes, ``modules_to_save`` and the reduced-LR patterns are taken
    **from** ``GDINO_PLACEMENT`` rather than restated, so a future spec edit
    propagates instead of diverging.
    """
    groups = scope_groups(scope)
    return LoraPlacement(
        model=GDINO_PLACEMENT.model,
        target_regex="|".join(f"({g.pattern})" for g in groups),
        frozen_prefixes=GDINO_PLACEMENT.frozen_prefixes,
        allowed_prefixes=REPO_ALLOWED_PREFIXES,
        modules_to_save=GDINO_PLACEMENT.modules_to_save,
        reduced_lr_patterns=GDINO_PLACEMENT.reduced_lr_patterns,
    )


def scope_groups(scope: str) -> tuple[LoraGroup, ...]:
    """Groups of ``scope``. An unknown scope is never guessed at."""
    if scope not in SCOPES:
        raise SpecViolationError(
            f"unknown LoRA scope {scope!r}; expected one of {sorted(SCOPES)}")
    return tuple(GROUPS_BY_NAME[name] for name in SCOPES[scope])


def expected_module_count(scope: str = DEFAULT_SCOPE) -> int:
    return sum(g.expected for g in scope_groups(scope))


def expected_group_counts(scope: str = DEFAULT_SCOPE) -> dict[str, int]:
    return {g.name: g.expected for g in scope_groups(scope)}


# --- target collection -------------------------------------------------------

@dataclass(frozen=True)
class TargetLinear:
    """One ``nn.Linear`` to be replaced by a LoRA layer."""

    name: str
    group: str
    parent: object
    attr: str
    linear: object


def collect_targets(model, scope: str = DEFAULT_SCOPE) -> list[TargetLinear]:
    """Walk ``named_modules()`` and collect the §2 targets of ``scope``.

    Must be called **after** ``apply_custom_attention`` — the decomposable
    groups expose zero ``nn.Linear`` children before it, and a silently empty
    group is precisely the failure this function refuses to allow.

    Raises:
        SpecViolationError: any group matched a different number of modules
            than measured on the live Swin-B model, or a target landed in a
            frozen region.
    """
    import torch.nn as nn

    groups = scope_groups(scope)
    compiled = [(g, g.regex) for g in groups]
    found: list[TargetLinear] = []

    for parent_name, parent in model.named_modules():
        for attr, child in parent.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full = f"{parent_name}.{attr}" if parent_name else attr
            for group, pattern in compiled:
                if pattern.match(full):
                    found.append(TargetLinear(name=full, group=group.name,
                                              parent=parent, attr=attr,
                                              linear=child))
                    break

    _assert_no_frozen_targets(found)
    _assert_group_counts(found, groups)
    return found


def _assert_no_frozen_targets(found: Sequence[TargetLinear]) -> None:
    bad = [t.name for t in found
           if any(_under(t.name, p) for p in GDINO_PLACEMENT.frozen_prefixes)]
    if bad:
        raise SpecViolationError(
            f"{len(bad)} target(s) sit in a region spec §2 freezes, e.g. "
            f"{bad[:5]}. Adapting the Swin backbone or BERT is the retired "
            f"216-module preset's failure")


def _assert_group_counts(found: Sequence[TargetLinear],
                         groups: Sequence[LoraGroup]) -> None:
    observed: dict[str, int] = {g.name: 0 for g in groups}
    for target in found:
        observed[target.group] += 1
    problems = []
    for group in groups:
        got = observed[group.name]
        if got == group.expected:
            continue
        hint = ""
        if got == 0 and group.needs_decomposition:
            hint = (" — this group only exists after apply_custom_attention; "
                    "collect targets AFTER decomposing self_attn/ca_text")
        problems.append(f"{group.name}: expected {group.expected}, matched {got}{hint}")
    if problems:
        raise SpecViolationError(
            "LoRA target count mismatch (per group, because the same total is "
            "reachable from different group combinations):\n  - "
            + "\n  - ".join(problems))


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".") or f".{prefix}." in f".{name}."


def group_counts(found: Iterable[TargetLinear]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in found:
        counts[target.group] = counts.get(target.group, 0) + 1
    return dict(sorted(counts.items()))


# --- optimizer parameter groups (spec §3) ------------------------------------

LORA_PARAM_MARKERS: tuple[str, ...] = ("lora_A", "lora_B")

#: The three learning rates §3 fixes. The weight-decay split (0.0 for norm and
#: bias) subdivides these without adding a fourth rate.
LR_BY_ROLE: Mapping[str, float] = {
    "lora": LR_LORA,
    "lora_sampling_offsets": LR_SAMPLING_OFFSETS_LORA,
    "head": LR_FULL_HEAD,
}


@dataclass(frozen=True)
class ParamGroupSpec:
    """One AdamW parameter group, with the role it plays in §3."""

    name: str
    role: str
    lr: float
    weight_decay: float
    param_names: tuple[str, ...]
    n_params: int

    def as_record(self) -> dict:
        return {"name": self.name, "role": self.role, "lr": self.lr,
                "weight_decay": self.weight_decay, "n_tensors": len(self.param_names),
                "n_params": self.n_params}


def is_lora_param(name: str) -> bool:
    return any(marker in name for marker in LORA_PARAM_MARKERS)


def is_head_param(name: str) -> bool:
    """``bbox_embed`` / ``class_embed`` — §2's fully-trained detection heads.

    Matched by substring so the two-stage head ``transformer.enc_out_bbox_embed``
    is included: it is the same box head applied to the encoder proposals, and
    training one but not the other would be an undocumented deviation.
    """
    return any(target in name for target in GDINO_PLACEMENT.modules_to_save)


def is_reduced_lr_param(name: str) -> bool:
    """§3: ``sampling_offsets`` LoRA runs at lr/10 (grid initialisation)."""
    return any(pattern in name for pattern in GDINO_PLACEMENT.reduced_lr_patterns)


def is_no_decay_param(name: str) -> bool:
    """§3: weight decay 0.0 on norm and bias parameters."""
    return name.endswith(".bias") or name == "bias" or "norm" in name.lower()


def build_param_group_specs(named_trainable: Sequence[tuple[str, int]]) -> list[ParamGroupSpec]:
    """Assign every trainable parameter to exactly one §3 group.

    Args:
        named_trainable: ``(name, numel)`` for every parameter with
            ``requires_grad=True``. Passing names rather than tensors keeps
            this testable without building a model.

    Raises:
        SpecViolationError: a trainable parameter is neither LoRA nor a head
            (it would then be trained at an unspecified learning rate), or one
            of the three §3 roles is empty.
    """
    buckets: dict[tuple[str, bool], list[tuple[str, int]]] = {}
    for name, numel in named_trainable:
        if is_lora_param(name):
            role = "lora_sampling_offsets" if is_reduced_lr_param(name) else "lora"
        elif is_head_param(name):
            role = "head"
        else:
            raise SpecViolationError(
                f"trainable parameter {name!r} is neither a LoRA parameter nor "
                f"part of a §2 detection head; spec §3 assigns it no learning "
                f"rate. Refusing to train at an unspecified LR")
        buckets.setdefault((role, is_no_decay_param(name)), []).append((name, numel))

    missing = [role for role in LR_BY_ROLE if not any(k[0] == role for k in buckets)]
    if missing:
        raise SpecViolationError(
            f"spec §3 parameter group(s) {missing} are empty. An empty group "
            f"means the corresponding modules were never made trainable — "
            f"`sampling_offsets` LoRA and the fully-trained head both have "
            f"their own learning rate and neither may silently vanish")

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


def build_param_groups(model) -> tuple[list[dict], list[ParamGroupSpec]]:
    """``(optimizer param_groups, their specs)`` for a live model."""
    named = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    specs = build_param_group_specs(named)
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


# --- learning-rate schedule (spec §3) ----------------------------------------

def lr_scale(step: int, total_steps: int = TOTAL_STEPS,
             warmup: int | None = None,
             final_fraction: float = FINAL_LR_FRACTION) -> float:
    """Multiplier on the peak LR at optimizer ``step`` (0-based).

    Linear warmup over 3% of ``total_steps``, then cosine decay to
    ``final_fraction`` of peak at ``total_steps`` (§3).

    ``total_steps`` is the **frozen budget**, not the length of a shortened
    run: a gate run of 20 steps must see the same LR the full run sees at those
    steps, otherwise its loss curve is not comparable to the full run's.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    warmup = warmup_steps() if warmup is None else warmup
    warmup = max(0, min(warmup, total_steps))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_fraction + (1.0 - final_fraction) * cosine


def make_lr_lambda(total_steps: int = TOTAL_STEPS):
    """``LambdaLR``-compatible closure over the frozen schedule."""
    warmup = warmup_steps()

    def _lambda(step: int) -> float:
        return lr_scale(step, total_steps=total_steps, warmup=warmup)

    return _lambda


def describe_schedule(total_steps: int = TOTAL_STEPS) -> str:
    warmup = warmup_steps()
    return (f"cosine schedule: {total_steps} total steps, {warmup} linear warmup "
            f"({100 * warmup / total_steps:.1f}%), final LR = "
            f"{100 * FINAL_LR_FRACTION:.0f}% of peak "
            f"(peak: lora {LR_LORA:g}, head {LR_FULL_HEAD:g}, "
            f"sampling_offsets LoRA {LR_SAMPLING_OFFSETS_LORA:g})")
