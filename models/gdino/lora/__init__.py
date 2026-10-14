"""Everything LoRA for Grounding DINO: where it goes, and what it is.

`placement` is ours -- which linears get an adapter, how the parameter groups
and the schedule are built. The other three are the adapter itself, carried
here because the checkout they used to be imported from never published them.

Why the layer is in our tree
----------------------------
Both entry points imported ``Linear`` and ``apply_custom_attention`` from
``groundingdino`` in the checkout `UC_GDINO_REPO` names. Those files exist in
the checkout on the machine this work was done on and in none of that fork's
eighteen commits -- the whole LoRA layer, seven files, is untracked there. A
reader who clones the fork gets a tree that imports fine and has no LoRA in it,
and both entry points die on `ImportError` with nothing to act on.

Carrying it is the only fix that does not wait on someone else committing
something. Documenting which files to reconstruct would leave the published
adapters unusable, since an adapter is a state dict keyed to these exact
modules.

What is here, and what is not
-----------------------------
``layer.py`` and ``multi_head_attention.py`` are byte-identical to the
checkout's ``modules/lora.py`` (sha256 ``7f523728bfded1e9``) and
``modules/multi_head_attention.py`` (``a082b1e85950d5c8``). ``utils.py``
carries the two functions this repository calls out of ``util/lora_utils.py``
(``dac8d1e1640cef28``), whose remaining contents need detectron2 and a
task-memory module nothing here uses. The fork's ``lora_pool.py`` is not here
because nothing imports it: it belongs to the pooled-adapter comparison, not to
this recipe.

Copying, not re-implementing, is deliberate. The published adapters were
trained by these bytes; a cleaner rewrite of the same idea would load the same
state dict and could still not be the same computation.

Licensing
---------
``layer.py`` is Microsoft's LoRA reference implementation under the MIT
licence, header intact. The other two are adapted from DitHub (Cappellino et
al., NeurIPS 2025) by the fork's authors and carry that attribution in their
own headers. See NOTICE.
"""
from __future__ import annotations

from .layer import Linear, LoRALayer
from .multi_head_attention import MultiHeadAttention
from .placement import (
    DEFAULT_SCOPE,
    FINAL_LR_FRACTION,
    GDINO_LORA_GROUPS,
    GROUPS_BY_NAME,
    LORA_PARAM_MARKERS,
    LR_BY_ROLE,
    LR_FULL_HEAD,
    LR_LORA,
    LR_SAMPLING_OFFSETS_LORA,
    REPO_ALLOWED_PREFIXES,
    SCOPES,
    TOTAL_STEPS,
    WEIGHT_DECAY,
    WEIGHT_DECAY_NORM_BIAS,
    LoraGroup,
    ParamGroupSpec,
    TargetLinear,
    build_param_group_specs,
    build_param_groups,
    build_placement,
    collect_targets,
    describe_param_groups,
    describe_schedule,
    expected_group_counts,
    expected_module_count,
    group_counts,
    is_head_param,
    is_lora_param,
    is_no_decay_param,
    is_reduced_lr_param,
    lr_scale,
    make_lr_lambda,
    scope_groups,
)
from .utils import apply_custom_attention, mark_only_lora_as_trainable

__all__ = [
    # the adapter
    "Linear", "LoRALayer", "MultiHeadAttention",
    "apply_custom_attention", "mark_only_lora_as_trainable",
    # where it goes
    "DEFAULT_SCOPE", "FINAL_LR_FRACTION", "GDINO_LORA_GROUPS", "GROUPS_BY_NAME",
    "LORA_PARAM_MARKERS", "LR_BY_ROLE", "LR_FULL_HEAD", "LR_LORA",
    "LR_SAMPLING_OFFSETS_LORA", "REPO_ALLOWED_PREFIXES", "SCOPES", "TOTAL_STEPS",
    "WEIGHT_DECAY", "WEIGHT_DECAY_NORM_BIAS",
    "LoraGroup", "ParamGroupSpec", "TargetLinear",
    "build_param_group_specs", "build_param_groups", "build_placement",
    "collect_targets", "describe_param_groups", "describe_schedule",
    "expected_group_counts", "expected_module_count", "group_counts",
    "is_head_param", "is_lora_param", "is_no_decay_param",
    "is_reduced_lr_param", "lr_scale", "make_lr_lambda", "scope_groups",
]
