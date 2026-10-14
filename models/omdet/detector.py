"""OmDet-Turbo under the frozen V6 protocol, read out **per channel**.

Spec: the frozen-hyperparameter specification §6, the vocabulary specification §7.

Companion of :mod:`utils.runner.detector` (GDINO / OWLv2). It lives in its own
module so the GDINO/OWLv2 arm can keep being edited independently; nothing in
:mod:`utils.runner.detector` is modified. :class:`~utils.runner.detector.TermDetection`
is imported from there so both arms hand the runner the same record type.

Read-out audit (transformers ``models/omdet_turbo/processing_omdet_turbo.py``)
-----------------------------------------------------------------------------
OmDet-Turbo is **per channel by construction**, like Grounding DINO and unlike
OWLv2:

``compute_score(logits)`` returns ``sigmoid(logits)`` of shape
*(num_queries, num_classes)* with **no reduction over the class axis**, plus a
``classes`` index vector that simply enumerates the class axis. Then
``_post_process_boxes_for_image`` takes ``scores.flatten(0, 1).topk(...)`` over
that flattened *(query x class)* matrix and recovers the class with
``labels[topk_indices]``. One query box can therefore be emitted under several
class names, each with **its own** score. Finally ``batched_nms`` is offset by
label, so it never suppresses the same box under a different term either.

That is exactly the property this study needs, so no override is applied. It is
verified empirically rather than assumed --- the regression tests
asserts that boxes fire on several near-synonymous terms and that no
multi-term box carries a repeated score (a repeated score would mean a max was
being copied across the term axis).

Model-default post-processing (§6: "each model keeps its own NMS behaviour")
----------------------------------------------------------------------------
``nms_threshold`` is left at the processor's own signature default and
recorded verbatim in the fragment; ``max_num_det`` is likewise left at the
model default (``None`` -> ``num_queries`` entries kept from the flattened
matrix, the direct analogue of GDINO's ``select_box_nums_for_evaluation``).
Only ``threshold`` is set, to the protocol's ``score_threshold`` (0.0).
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Sequence

from utils.runner.logging_utils import get_logger
from utils.runner.detector import TermDetection
from utils.protocol.eval_protocol import EVAL_PROTOCOL, EvalProtocol

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = get_logger(__name__)

#: variant -> HuggingFace checkpoint id. Exact-match; never guessed.
OMDET_CHECKPOINTS: dict[str, str] = {
    "swin-tiny": "omlab/omdet-turbo-swin-tiny-hf",
}


def processor_nms_default() -> float:
    """The processor's own ``nms_threshold`` default, read from its signature.

    Recorded in the fragment so "model_default" is a number in the record and
    not a promise: if a ``transformers`` upgrade changes the default, the
    fragments of the two eras differ visibly instead of silently.
    """
    from transformers import AutoProcessor  # noqa: F401  (import cost only)
    from transformers.models.omdet_turbo.processing_omdet_turbo import (
        OmDetTurboProcessor,
    )

    signature = inspect.signature(
        OmDetTurboProcessor.post_process_grounded_object_detection)
    return float(signature.parameters["nms_threshold"].default)


class OmDetTurboDetector:
    """OmDet-Turbo with the per-channel read-out described in the module docstring.

    Args:
        variant: Key of :data:`OMDET_CHECKPOINTS`.
        device: Always ``"cuda:0"`` with the GPU pinned via
            ``CUDA_VISIBLE_DEVICES``. Non-zero ordinals have a recorded
            history of silently corrupting this model (``CLAUDE.md``).
        protocol: The frozen protocol; only ``score_threshold`` is passed to
            the processor.
        checkpoint_override: Local weights (a fine-tuned checkpoint) to load
            instead of the hub id.
        cache_dir: HuggingFace cache root.
        v6_adapter_dir: A V6 LoRA checkpoint directory
            (``ckpt_final_step5667``) written by ``train_v6_omdet.py`` (research tree).
            The base model is built exactly as for zero-shot, spec §2's LoRA is
            injected at the same targets, the adapter is loaded and the
            manifest is re-verified against the live model. **Every other line
            of this class is shared with the zero-shot path**, which is what
            makes ZS and FT comparable under §6: the read-out, the protocol
            values and the post-processing cannot drift between the two arms
            because there is only one copy of them.

    The adapter is deliberately **not merged** into the base weights. Merging a
    PEFT adapter on this model has a recorded history of producing run-to-run
    different results (``CLAUDE.md``); an unmerged adapter computes
    ``base(x) + scaling * B(A(x))`` explicitly and has no such failure mode.
    """

    def __init__(self, variant: str = "swin-tiny", device: str = "cuda:0",
                 protocol: EvalProtocol = EVAL_PROTOCOL,
                 checkpoint_override: str | None = None,
                 cache_dir: str | None = None,
                 v6_adapter_dir: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor, OmDetTurboForObjectDetection

        if variant not in OMDET_CHECKPOINTS:
            raise ValueError(
                f"unknown OmDet-Turbo variant {variant!r}; "
                f"expected one of {sorted(OMDET_CHECKPOINTS)}")
        checkpoint = OMDET_CHECKPOINTS[variant]
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}

        self.protocol = protocol
        self.model_name = "omdet"
        self.checkpoint_name = (f"{variant}-zeroshot" if checkpoint_override is None
                                else variant)
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(checkpoint, **kwargs)
        weights = checkpoint_override if checkpoint_override is not None else checkpoint
        model = OmDetTurboForObjectDetection.from_pretrained(
            weights, low_cpu_mem_usage=False, **kwargs)
        _materialise_meta_buffers(model)
        _recompute_timm_relative_position_index(model)
        self.v6_checkpoint = (None if v6_adapter_dir is None
                              else _attach_v6_adapter(model, v6_adapter_dir))
        if self.v6_checkpoint is not None:
            self.checkpoint_name = (
                f"{variant}-ft-seed{self.v6_checkpoint['seed']}"
                f"-step{self.v6_checkpoint['step']}")
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._nms_threshold = processor_nms_default()
        logger.info("OmDet-Turbo %s loaded on %s (nms_threshold=%.2f, model default)",
                    variant, device, self._nms_threshold)

    def predict(self, image: "PILImage", terms: Sequence[str]) -> list[TermDetection]:
        """One forward pass; every (box, term) pair the model emits is kept.

        No argmax and no max over the term axis: the processor already returns
        one row per *(query, class)* cell that survives its topk, so a box that
        fires on two terms comes back twice with two different scores.
        """
        torch = self._torch
        term_list = list(terms)
        inputs = self._processor(images=image, text=term_list,
                                 return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        result = self._processor.post_process_grounded_object_detection(
            outputs,
            text_labels=term_list,
            threshold=self.protocol.score_threshold,
            nms_threshold=self._nms_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result["text_labels"]
        return [
            TermDetection(
                term=str(label), score=float(score),
                bbox_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            )
            for b, score, label in zip(boxes, scores, labels)
        ]

    def postprocessing_notes(self) -> dict:
        return {
            "v6_checkpoint": self.v6_checkpoint,
            "readout": "per_channel_by_construction",
            "readout_detail": (
                "transformers' OmDetTurboProcessor computes sigmoid(logits) over the "
                "(query x class) matrix with no reduction, then takes topk over that "
                "flattened matrix and recovers the class from the flat index, so one "
                "box may be returned under several terms each with its own score; "
                "batched_nms is offset by label and therefore never merges a box "
                "across terms. No argmax is applied and none is bypassed"),
            "nms": "model_default_unchanged",
            "nms_threshold": self._nms_threshold,
            "max_num_det": None,
            "max_num_det_detail": (
                "processor default: keeps num_queries entries of the flattened "
                "(query x class) matrix, the analogue of GDINO's "
                "select_box_nums_for_evaluation"),
            "decoder_num_queries": getattr(self._model.config, "num_queries", None),
            "max_detections_per_image": self.protocol.max_detections_per_image,
        }


def _attach_v6_adapter(model, adapter_dir: str) -> dict:
    """Inject spec §2 LoRA, load a V6 checkpoint, and re-verify it. Never merges.

    The manifest is not trusted as a description — it is *checked against the
    live model*: §2 compliance, per-group injection counts, the alpha/r
    fingerprint and the training caption. A value reconstructed at load time and
    never compared to what was stored is indistinguishable from a wrong one, and
    this project has already had alpha silently halved once.

    Returns:
        Provenance recorded verbatim into the fragment, so a fine-tuned
        fragment can never be mistaken for a zero-shot one.
    """
    from pathlib import Path

    from peft import LoraConfig, inject_adapter_in_model

    from utils.train.checkpoint import (
        load_adapter_weights,
        read_manifest,
        verify_loaded_checkpoint,
    )
    from models.shared.lora_spec_v6line import select_targets
    from models.shared.lora_spec import (
        EXPECTED_GROUP_COUNTS,
        EXPECTED_LORA_MODULES,
        get_placement,
        group_of,
    )
    from utils.protocol.train_caption import training_caption
    from models.shared.train import assert_group_counts

    directory = Path(adapter_dir)
    manifest = read_manifest(directory)
    if manifest.model != "omdet":
        raise ValueError(
            f"{directory} holds a {manifest.model!r} checkpoint, not an OmDet one")
    placement = get_placement("omdet")
    targets = select_targets([n for n, _ in model.named_modules()], placement)
    if len(targets) != EXPECTED_LORA_MODULES["omdet"]:
        raise ValueError(
            f"§2 regex selected {len(targets)} modules, expected "
            f"{EXPECTED_LORA_MODULES['omdet']}")
    inject_adapter_in_model(
        LoraConfig(r=manifest.lora_r, lora_alpha=manifest.lora_alpha,
                   lora_dropout=manifest.lora_dropout, bias="none",
                   target_modules=targets),
        model)
    n_loaded = load_adapter_weights(model, directory, strict=True)
    verify_loaded_checkpoint(model, manifest, placement,
                             expected_caption=training_caption())
    per_group = assert_group_counts(model, placement,
                                    EXPECTED_GROUP_COUNTS["omdet"], group_of)
    logger.info("V6 adapter loaded from %s: seed %d, step %d/%d, %d tensors, "
                "per-group %s (NOT merged)", directory, manifest.seed,
                manifest.step, manifest.total_steps, n_loaded, per_group)
    return {
        "adapter_dir": str(directory),
        "seed": manifest.seed,
        "step": manifest.step,
        "total_steps": manifest.total_steps,
        "lora_r": manifest.lora_r,
        "lora_alpha": manifest.lora_alpha,
        "scaling": manifest.scaling,
        "n_lora_modules": manifest.n_lora_modules,
        "per_group_modules": per_group,
        "trainable_params": manifest.trainable_params,
        "trainable_pct": manifest.trainable_pct,
        "training_caption": manifest.training_caption,
        "spec_version": manifest.spec_version,
        "tensors_loaded": n_loaded,
        "merged_into_base_weights": False,
    }


def _materialise_meta_buffers(model) -> None:
    """Zero-fill Swin ``attn_mask`` buffers left on the ``meta`` device.

    Same defect and same justification as
    ``srcs.ovdbench.adapters.omdet_turbo._materialize_stale_swin_attn_mask_buffers``:
    ``SwinLayer.forward`` recomputes the mask it actually uses every pass, so
    the stored buffer's value is dead weight and only its existence matters.
    Duplicated rather than imported so the V6 arm has no dependency on the
    retired ovdbench package.
    """
    import torch

    n_fixed = 0
    for module in model.modules():
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.is_meta:
                module._buffers[name] = torch.zeros(buf.shape, dtype=buf.dtype)
                n_fixed += 1
    if n_fixed:
        logger.info("materialised %d meta-device Swin buffer(s)", n_fixed)


def _recompute_timm_relative_position_index(model) -> None:
    """Recompute timm ``WindowAttention.relative_position_index`` buffers.

    ``reset_parameters()`` only fills this ``persistent=False`` buffer when the
    projection weight is not a meta tensor, which is not the case under
    ``from_pretrained`` on this checkpoint's timm-backed backbone. The buffer
    is then left as uninitialised ``torch.empty`` garbage that indexes
    ``relative_position_bias_table`` out of bounds. ``_init_buffers`` is a pure
    function of the checkpoint-fixed window size, so calling it here is exactly
    what construction would have done. Same repair as
    ``srcs.ovdbench.adapters.omdet_turbo``; duplicated so the V6 arm has no
    dependency on the retired ovdbench package.
    """
    try:
        from timm.models.swin_transformer import WindowAttention
    except ImportError:  # timm-free install: nothing to repair
        return

    n_fixed = 0
    for module in model.modules():
        if isinstance(module, WindowAttention):
            module._init_buffers()
            n_fixed += 1
    if n_fixed:
        logger.info("recomputed %d timm relative_position_index buffer(s)", n_fixed)


def build_omdet_detector(variant: str = "swin-tiny", device: str = "cuda:0",
                         protocol: EvalProtocol = EVAL_PROTOCOL,
                         checkpoint_override: str | None = None,
                         cache_dir: str | None = None,
                         v6_adapter_dir: str | None = None) -> OmDetTurboDetector:
    return OmDetTurboDetector(variant=variant, device=device, protocol=protocol,
                              checkpoint_override=checkpoint_override,
                              cache_dir=cache_dir, v6_adapter_dir=v6_adapter_dir)
