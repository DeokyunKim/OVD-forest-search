"""LoRA placement for GLIP and OmDet-Turbo, derived from spec §2's common rule.

    텍스트 인코더 완전 동결. 이미지 백본 동결.
    융합·디코더 어텐션에 LoRA. 검출 헤드는 전체 학습.

Spec §2 shipped per-model tables for GDINO and OWLv2 only: GLIP was listed
"ZS 전용" (corrected 2026-08-09, see the lora_feasibility_glip_omdet.md note)
and OmDet-Turbo was absent entirely. The two placements here are **derived from
the common rule**, and every count in the §2 tables was read off
``named_modules()`` of the real checkpoints, not guessed.

**This module extends** :mod:`models.shared.lora_spec_v6line` **without modifying it.** That
file is owned by the V6 training line; ``LoraPlacement`` is expressive enough
for both models, so no change to the shared file was needed --- only two new
instances and a lookup that falls back to the shared registry.

Derivation, model by model
--------------------------
**GLIP** — fusion lives inside the detection head, not in a separate encoder:
``VLFusionModule`` interleaves ``VLFuse`` (the ``BiMultiHeadAttention`` that
mixes vision and language) with a ``BertEncoderLayer`` (which re-contextualises
the language side) and a ``DyConv`` (vision only), six times.

* ``b_attn.attn.*`` — 36 modules. The cross-modal attention proper.
* ``attention.self.* / intermediate.dense / output.dense`` — 36 modules, the
  in-tower ``BertEncoderLayer``. These are **not** the frozen text encoder:
  ``language_model`` is a separate module that stays frozen. They are the exact
  analogue of Grounding DINO's ``text_enhancer``, which §2's own GDINO regex
  adapts (it matches ``encoder.layers.N.text_enhancer.self_attn.query`` etc.),
  so including them follows the existing precedent rather than inventing one.
* Requires ``early_fuse=True``. Under ``early_fuse=False`` (GLIP-T(A)) neither
  group exists and there is nothing to adapt --- asserted at build time.

**OmDet-Turbo** — ``decoder.layers.*`` only. Its ``encoder`` is a *vision-only*
hybrid encoder (no text enters it), so unlike GDINO's feature enhancer it is
frozen. Within the decoder layer, and despite the name, ``self_attn`` is the
cross-modal one: ``OmDetTurboDeformableTransformerDecoderLayer.forward``
concatenates ``task_features`` (text) onto the q/k/v sequence. ``cross_attn`` is
multi-scale deformable attention over vision features and is adapted as the
decoder-attention half of the rule, matching GDINO where
``sampling_offsets``/``attention_weights``/``value_proj``/``output_proj`` are
all in §2's regex.

What is deliberately **not** adapted
-------------------------------------
Standard LoRA wraps ``nn.Linear`` only. These sit inside the fusion path but
are convolutional or otherwise ineligible, and are left frozen:

======  =====================================  =====
model   modules                                count
======  =====================================  =====
GLIP    ``dyhead_tower.*.{attnconv.1,offset}``    12
GLIP    ``ModulatedDeformConv2d`` in DyConv       18
GLIP    ``dyhead_tower.*.relu.fc.*`` (DyReLU)     12
OmDet   ``decoder.channel_projection_layers``      3
======  =====================================  =====

All are vision-branch modules, which the common rule freezes anyway, so nothing
the rule asks for is lost. Recorded because "we could not adapt it" and "the
rule says not to" are different statements.

Detection heads, measured
-------------------------
Spec §2 says the detection head trains fully. GDINO taught that this can be
vacuous — its ``class_embed`` is a parameter-free ``ContrastiveEmbed``. So the
tensors that actually receive gradients were counted:

======  =========================================================  ===========
model   fully-trained modules                                      parameters
======  =========================================================  ===========
GLIP    ``bbox_pred``, ``centerness`` (both ``nn.Conv2d``),            198,924
        ``dot_product_projection_text``, ``scales.{0..4}``,
        ``log_scale``, ``bias_lang``, ``bias0``
OmDet   ``decoder_bbox_head.{0..5}`` (3 Linear each),                1,847,580
        ``encoder_bbox_head``, ``decoder_class_head.{0..5}``,
        ``encoder_class_head``
======  =========================================================  ===========

GLIP has **no learnable classification head**: class scores are a dot product
between region features and the projected text embedding, so
``dot_product_projection_text`` is the only classification-side tensor and the
head total is small. OmDet carries six bbox heads and six class heads (deep
supervision, one per decoder layer), which is why its trainable fraction is
nearly double GLIP's despite having fewer LoRA modules.

``modules_to_save`` for GLIP is expressed as parameter prefixes rather than PEFT
``modules_to_save``, because two of its head modules are ``nn.Conv2d`` inside an
mmdet detector that is adapted in place with ``inject_adapter_in_model`` (the
wrapper form would hide mmdet's ``train_step``).
"""
from __future__ import annotations

from typing import Mapping

from models.shared.lora_spec_v6line import PLACEMENTS as SHARED_PLACEMENTS
from models.shared.lora_spec_v6line import LoraPlacement

#: GLIP-T(C). Fusion is inside ``bbox_head.head.dyhead_tower``; requires
#: ``early_fuse=True`` (asserted by the trainer before injection).
GLIP_PLACEMENT = LoraPlacement(
    model="glip",
    target_regex=(
        r"^bbox_head\.head\.dyhead_tower\.\d+\."
        r"(b_attn\.attn\.(v_proj|l_proj|values_v_proj|values_l_proj|"
        r"out_v_proj|out_l_proj)|"
        r"attention\.self\.(query|key|value)|attention\.output\.dense|"
        r"intermediate\.dense|output\.dense)$"
    ),
    # `language_model` is GLIP's text encoder and stays frozen; the BERT layers
    # inside the tower are fusion and are adapted (see module docstring).
    frozen_prefixes=("language_model", "backbone", "neck"),
    allowed_prefixes=("bbox_head.head.dyhead_tower",),
    # Modules only. `log_scale`, `bias_lang` and `bias0` are bare `nn.Parameter`
    # attributes of `VLFusionModule`, not modules, so they cannot appear here:
    # the shared `verify_injection` looks `modules_to_save` up in
    # `named_modules()` and would (correctly) report them missing. They are
    # listed separately in :data:`HEAD_PARAMETER_PREFIXES` and are trained.
    modules_to_save=(
        "bbox_head.head.bbox_pred",
        "bbox_head.head.centerness",
        "bbox_head.head.dot_product_projection_text",
        "bbox_head.head.scales",
    ),
)

#: OmDet-Turbo. ``encoder`` is vision-only and frozen; only ``decoder.layers.*``
#: is adapted. ``self_attn`` there is the cross-modal attention.
OMDET_PLACEMENT = LoraPlacement(
    model="omdet",
    target_regex=(
        r"^decoder\.layers\.\d+\..*"
        r"(query|key|value|out_proj|"
        r"value_proj|output_proj|sampling_offsets|attention_weights|"
        r"linear1|linear2)$"
    ),
    frozen_prefixes=("language_backbone", "vision_backbone", "encoder"),
    allowed_prefixes=("decoder.layers",),
    modules_to_save=(
        "decoder.decoder_bbox_head",
        "decoder.encoder_bbox_head",
        "decoder.decoder_class_head",
        "decoder.encoder_class_head",
    ),
    # Same grid-initialisation argument as GDINO's (§3): lr/10.
    reduced_lr_patterns=("sampling_offsets",),
)

#: Fully-trained detection-head tensors that are bare ``nn.Parameter``
#: attributes rather than modules, so ``LoraPlacement.modules_to_save`` (which
#: the shared verifier resolves through ``named_modules()``) cannot carry them.
#: Spec §2's GLIP row lists all of ``bbox_pred``, ``centerness``,
#: ``dot_product_projection_text``, ``scales``, ``log_scale``, ``bias_lang`` and
#: ``bias0`` as fully trained; the split here is a representational detail of
#: this codebase, not a narrowing of the spec — the trainer unfreezes both sets
#: and the 198,924-parameter head total in §2 includes these three.
HEAD_PARAMETER_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "glip": ("bbox_head.head.log_scale", "bbox_head.head.bias_lang",
             "bbox_head.head.bias0"),
    "omdet": (),
}

#: Expected module counts, measured on the real checkpoints. The trainer
#: asserts against these so a transformers/mmdet upgrade that reshapes a model
#: fails loudly instead of quietly adapting a different set.
EXPECTED_LORA_MODULES: Mapping[str, int] = {"glip": 72, "omdet": 60}

#: Per-group expected counts — a total that happens to match while the groups
#: shifted would otherwise pass.
EXPECTED_GROUP_COUNTS: Mapping[str, Mapping[str, int]] = {
    "glip": {"vlfuse_cross_attn": 36, "in_tower_bert": 36},
    "omdet": {"fusion_self_attn": 24, "deformable_cross_attn": 24, "decoder_ffn": 12},
}

#: Fusion-path modules that standard LoRA cannot wrap. Recorded, not adapted.
UNADAPTABLE_FUSION_MODULES: Mapping[str, Mapping[str, int]] = {
    "glip": {"dyconv_attnconv_offset_conv2d": 12,
             "modulated_deform_conv2d": 18,
             "dyrelu_fc_linear_vision_branch": 12},
    "omdet": {"decoder_channel_projection_conv2d": 3},
}

XARCH_PLACEMENTS: Mapping[str, LoraPlacement] = {
    GLIP_PLACEMENT.model: GLIP_PLACEMENT,
    OMDET_PLACEMENT.model: OMDET_PLACEMENT,
}


def get_placement(model: str) -> LoraPlacement:
    """Exact-match lookup over this module's placements, then the shared ones.

    An unknown model is never guessed at — ``KeyError`` propagates.
    """
    if model in XARCH_PLACEMENTS:
        return XARCH_PLACEMENTS[model]
    return SHARED_PLACEMENTS[model]


def group_of(model: str, module_name: str) -> str:
    """Which expected group a LoRA target belongs to, for the count assertion."""
    if model == "glip":
        return "vlfuse_cross_attn" if ".b_attn." in module_name else "in_tower_bert"
    if model == "omdet":
        if ".self_attn." in module_name:
            return "fusion_self_attn"
        if ".cross_attn." in module_name:
            return "deformable_cross_attn"
        return "decoder_ffn"
    raise KeyError(model)
