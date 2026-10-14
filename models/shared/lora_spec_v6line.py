"""Spec §2 LoRA placement, as a rule + per-model regex — verified after injection.

Spec: the frozen-hyperparameter specification §1 (LoRA hyperparameters) and §2
(where LoRA goes).

The common rule of §2:

    "텍스트 인코더 완전 동결. 이미지 백본 동결.
     융합·디코더 어텐션에 LoRA. 검출 헤드는 전체 학습."

Why this module exists
----------------------
The retired system's ``zil_full216`` preset **violates §2**, measured from its
own ``injection_map.json``:

===========================  =========  ==================================
group                        modules    §2
===========================  =========  ==================================
``swin_backbone_attn``          48      image backbone — must be frozen
``bert_self_attn``              48      text encoder — must be frozen
other five groups              120      fusion/decoder — allowed
===========================  =========  ==================================

**96 of 216 modules (44.4%) sit in regions §2 freezes**, and the same file
records ``lora_alpha = 8.0`` where §1 requires **32**. Inheriting that preset
would silently train a different model than the spec describes, and the only
symptom would be different numbers.

So Phase 2 does not reuse any preset. It builds targets from the §2 regex and
then **counts the surviving modules with** ``named_modules()``. Trusting a
config has already failed twice on this project: alpha was silently halved
once, and the 216 violation surfaced only under exactly this kind of walk.

Extending to new models
-----------------------
Placement is expressed as data (:class:`LoraPlacement`), not as per-model
branches, so adding GLIP-T (73 injectable modules) or OmDet-Turbo (68) is a new
:class:`LoraPlacement` entry plus its regex — no changes to the trainer or the
verifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


class SpecViolationError(RuntimeError):
    """LoRA landed somewhere spec §2 freezes, or a required head is not trained.

    Raised before any optimizer step. A freeze violation runs to completion
    without erroring and only changes the numbers, so it must be caught here.
    """


@dataclass(frozen=True)
class LoraPlacement:
    """Where LoRA may and may not go, for one model.

    Attributes:
        model: Key used by the trainer.
        target_regex: Spec §2's regex, verbatim. Modules whose *name* matches
            are LoRA targets.
        frozen_prefixes: Dotted prefixes that must contain **zero** LoRA
            modules. This is the assertion that catches the 216-preset class
            of bug.
        allowed_prefixes: LoRA modules may only appear under these.
        modules_to_save: Fully-trained modules (detection heads).
        reduced_lr_patterns: Substrings whose LoRA gets lr/10 (§3:
            ``sampling_offsets`` has a grid initialisation).
    """

    model: str
    target_regex: str
    frozen_prefixes: tuple[str, ...]
    allowed_prefixes: tuple[str, ...]
    modules_to_save: tuple[str, ...]
    reduced_lr_patterns: tuple[str, ...] = field(default_factory=tuple)

    def matches(self, module_name: str) -> bool:
        return re.search(self.target_regex, module_name) is not None


#: Grounding DINO. Regex transcribed from §2 verbatim; do not "tidy" it.
GDINO_PLACEMENT = LoraPlacement(
    model="gdino",
    target_regex=(
        r"(encoder|decoder)\.layers\.\d+\..*"
        r"(query|key|value|out_proj|output\.dense|fc1|fc2|"
        r"value_proj|output_proj|sampling_offsets|attention_weights|"
        r"vision_proj|text_proj|values_vision_proj|values_text_proj|"
        r"out_vision_proj|out_text_proj)$"
    ),
    # The two regions the retired 216 preset wrongly adapted.
    frozen_prefixes=("backbone", "text_backbone", "bert"),
    allowed_prefixes=("transformer.encoder.layers", "transformer.decoder.layers",
                      "encoder.layers", "decoder.layers"),
    modules_to_save=("bbox_embed", "class_embed"),
    reduced_lr_patterns=("sampling_offsets",),
)

#: OWLv2. §2's per-model table puts LoRA on the **ViT image encoder** and
#: freezes the CLIP **text** tower. That is not a contradiction of the common
#: rule: OWLv2 is a dual-encoder with no fusion or decoder stack, so the
#: per-model table governs, and the frozen-text-tower requirement is what makes
#: its permutation invariance testable after fine-tuning (§2's own note).
OWLV2_PLACEMENT = LoraPlacement(
    model="owlv2",
    target_regex=(
        r"vision_model\.encoder\.layers\.\d+\..*"
        r"(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
    ),
    frozen_prefixes=("owlv2.text_model", "text_model", "text_projection"),
    allowed_prefixes=("owlv2.vision_model.encoder.layers", "vision_model.encoder.layers"),
    modules_to_save=("box_head", "class_head"),
)

PLACEMENTS: Mapping[str, LoraPlacement] = {
    GDINO_PLACEMENT.model: GDINO_PLACEMENT,
    OWLV2_PLACEMENT.model: OWLV2_PLACEMENT,
}


def get_placement(model: str) -> LoraPlacement:
    """Exact-match lookup; an unknown model is never guessed at."""
    return PLACEMENTS[model]


def select_targets(module_names: Iterable[str], placement: LoraPlacement) -> list[str]:
    """Names matching §2's regex **and** outside every frozen region.

    The frozen filter is applied here as well as in verification, so a regex
    that is accidentally too broad cannot inject into a frozen region in the
    first place.
    """
    out = []
    for name in module_names:
        if not placement.matches(name):
            continue
        if any(_under(name, prefix) for prefix in placement.frozen_prefixes):
            continue
        out.append(name)
    return out


def _under(name: str, prefix: str) -> bool:
    """Is ``name`` at or below dotted ``prefix``? Segment-wise, not substring."""
    return name == prefix or name.startswith(prefix + ".") or f".{prefix}." in f".{name}."


@dataclass(frozen=True)
class InjectionReport:
    """What actually got adapted, counted from ``named_modules()``."""

    model: str
    n_lora_modules: int
    n_in_frozen_regions: int
    modules_in_frozen_regions: tuple[str, ...]
    n_outside_allowed: int
    modules_outside_allowed: tuple[str, ...]
    per_region_counts: Mapping[str, int]
    trainable_params: int
    total_params: int
    trainable_pct: float
    modules_to_save_found: tuple[str, ...]
    modules_to_save_missing: tuple[str, ...]

    @property
    def is_compliant(self) -> bool:
        return (self.n_in_frozen_regions == 0
                and self.n_outside_allowed == 0
                and not self.modules_to_save_missing)


#: Class names that are LoRA *containers*, never adapted layers. PEFT's
#: ``LoraModel`` is the tuner root that ``get_peft_model`` wraps the whole
#: network in; it is one node, so counting it turns an exact 60 into "61, one
#: outside the allowed regions" and fails a gate that should have passed.
_LORA_CONTAINER_CLASS_NAMES: frozenset[str] = frozenset({"LoraModel"})


def _peft_lora_layer_types() -> tuple[type, ...]:
    """The PEFT ``LoraLayer`` family, or ``()`` where PEFT is not installed.

    Looked up lazily: the Grounding DINO venv has no PEFT at all (it injects the
    external repo's own ``lora.Linear``), and importing at module scope would
    make this module unimportable there.
    """
    try:
        from peft.tuners.lora import LoraLayer
    except Exception:  # pragma: no cover - depends on the venv
        return ()
    return (LoraLayer,)


def is_lora_module(module) -> bool:
    """True for a LoRA-adapted **layer**, false for a LoRA container.

    Two families must count:

    * PEFT's ``LoraLayer`` subclasses (``lora.Linear``, ``lora.Conv2d``, ...),
      recognised by ``isinstance`` rather than by name;
    * loralib-style layers such as the external Grounding DINO repo's
      ``modules.lora.Linear``, which is named ``Linear`` and can only be
      recognised by its ``lora_A`` attribute.

    One thing must **not** count: the tuner root. The previous predicate
    accepted any class whose name started with ``Lora``, which matched PEFT's
    ``LoraModel`` wrapper — a single extra node that reported as one LoRA
    module sitting outside every allowed region, i.e. a spurious §2 violation.
    """
    if type(module).__name__ in _LORA_CONTAINER_CLASS_NAMES:
        return False
    layer_types = _peft_lora_layer_types()
    if layer_types and isinstance(module, layer_types):
        return True
    return hasattr(module, "lora_A")


def _is_lora_module(module) -> bool:
    """Backwards-compatible private alias of :func:`is_lora_module`."""
    return is_lora_module(module)


def verify_injection(model, placement: LoraPlacement,
                     trainable_params: int | None = None,
                     total_params: int | None = None) -> InjectionReport:
    """Walk the **live** model and count where LoRA actually landed.

    This is the check that catches the retired 216 preset. It reads
    ``named_modules()`` rather than any config or preset file, because the
    config has been wrong before while looking correct.
    """
    lora_names: list[str] = []
    for name, module in model.named_modules():
        if _is_lora_module(module):
            lora_names.append(name)

    in_frozen = [n for n in lora_names
                 if any(_under(n, p) for p in placement.frozen_prefixes)]
    outside = [n for n in lora_names
               if not any(_under(n, p) for p in placement.allowed_prefixes)]

    region_counts: dict[str, int] = {}
    for name in lora_names:
        for prefix in placement.allowed_prefixes:
            if _under(name, prefix):
                region_counts[prefix] = region_counts.get(prefix, 0) + 1
                break
        else:
            region_counts["<outside allowed>"] = region_counts.get("<outside allowed>", 0) + 1

    all_names = [n for n, _ in model.named_modules()]
    found, missing = [], []
    for target in placement.modules_to_save:
        hits = [n for n in all_names if target in n]
        (found if hits else missing).append(target)

    if trainable_params is None or total_params is None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

    return InjectionReport(
        model=placement.model,
        n_lora_modules=len(lora_names),
        n_in_frozen_regions=len(in_frozen),
        modules_in_frozen_regions=tuple(sorted(in_frozen)[:20]),
        n_outside_allowed=len(outside),
        modules_outside_allowed=tuple(sorted(outside)[:20]),
        per_region_counts=region_counts,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=100.0 * trainable_params / total_params if total_params else 0.0,
        modules_to_save_found=tuple(found),
        modules_to_save_missing=tuple(missing),
    )


@dataclass(frozen=True)
class ScalingReport:
    """LoRA ``scaling`` read from the live modules, not from any config.

    PEFT stores ``scaling = lora_alpha / r`` per adapter. §1 fixes r=16 and
    alpha=32, so every adapter must read **exactly 2.0**. The retired preset
    recorded ``lora_alpha = 8.0`` (scaling 0.5), and this project has already
    had one incident where alpha was silently halved while the config looked
    correct — so the value is read from ``module.scaling``, never inferred.
    """

    n_modules_checked: int
    observed_scalings: tuple[float, ...]
    expected_scaling: float
    mismatched: tuple[tuple[str, float], ...]

    @property
    def is_compliant(self) -> bool:
        return self.n_modules_checked > 0 and not self.mismatched


def verify_lora_scaling(model, expected_r: int = 16, expected_alpha: int = 32,
                        tolerance: float = 1e-6) -> ScalingReport:
    """Read ``module.scaling`` off every LoRA module and check alpha/r.

    Args:
        expected_r / expected_alpha: §1's frozen values (16 / 32 -> scaling 2.0).

    Returns:
        A :class:`ScalingReport`. ``n_modules_checked == 0`` is itself a
        failure: it means nothing was adapted, or the adapters do not expose
        ``scaling`` and the check silently passed over them.
    """
    expected = expected_alpha / expected_r
    observed: list[float] = []
    mismatched: list[tuple[str, float]] = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if scaling is None:
            continue
        # PEFT keeps a dict keyed by adapter name; older versions use a float.
        values = list(scaling.values()) if isinstance(scaling, dict) else [scaling]
        for value in values:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            observed.append(value)
            if abs(value - expected) > tolerance:
                mismatched.append((name, value))
    return ScalingReport(
        n_modules_checked=len(observed),
        observed_scalings=tuple(sorted(set(observed))),
        expected_scaling=expected,
        mismatched=tuple(mismatched[:20]),
    )


def assert_scaling_compliant(report: ScalingReport) -> None:
    """Raise unless every adapter reads alpha/r = 2. Call before step 1."""
    if report.n_modules_checked == 0:
        raise SpecViolationError(
            "no LoRA module exposed a `scaling` attribute; alpha/r could not be "
            "verified. Refusing to train on an unverifiable adapter configuration")
    if report.mismatched:
        raise SpecViolationError(
            f"LoRA scaling must be alpha/r = {report.expected_scaling} (§1: r=16, "
            f"alpha=32) but {len(report.mismatched)} module(s) differ, e.g. "
            f"{list(report.mismatched[:5])}. Observed values: "
            f"{list(report.observed_scalings)}. The retired preset used "
            f"lora_alpha=8.0 (scaling 0.5); this project has already had alpha "
            f"silently halved once")


def assert_compliant(report: InjectionReport) -> None:
    """Raise unless the injection satisfies spec §2. Call before step 1."""
    problems: list[str] = []
    if report.n_in_frozen_regions:
        problems.append(
            f"{report.n_in_frozen_regions} LoRA module(s) landed in regions spec §2 "
            f"freezes, e.g. {list(report.modules_in_frozen_regions[:5])}. This is the "
            f"retired zil_full216 failure (96/216 modules in frozen regions)")
    if report.n_outside_allowed:
        problems.append(
            f"{report.n_outside_allowed} LoRA module(s) outside the allowed regions, "
            f"e.g. {list(report.modules_outside_allowed[:5])}")
    if report.modules_to_save_missing:
        problems.append(
            f"modules_to_save not found in the model: "
            f"{list(report.modules_to_save_missing)} — the detection head would not "
            f"be fully trained, which §2 requires")
    if problems:
        raise SpecViolationError(
            f"[{report.model}] spec §2 violation; refusing to train.\n  - "
            + "\n  - ".join(problems))


def format_report(report: InjectionReport) -> str:
    """One-block log line set, per §2's requirement to record the counts."""
    lines = [
        f"[{report.model}] LoRA injection verified from named_modules()",
        f"  LoRA modules injected     : {report.n_lora_modules}",
        f"  in frozen regions         : {report.n_in_frozen_regions} (must be 0)",
        f"  outside allowed regions   : {report.n_outside_allowed} (must be 0)",
        f"  modules_to_save found     : {list(report.modules_to_save_found)}",
        f"  trainable params          : {report.trainable_params:,} "
        f"/ {report.total_params:,} = {report.trainable_pct:.4f}%",
    ]
    for region, count in sorted(report.per_region_counts.items()):
        lines.append(f"    {region}: {count}")
    return "\n".join(lines)
