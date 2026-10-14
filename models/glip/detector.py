"""GLIP under the frozen V6 protocol, read out **per channel**.

Spec: the frozen-hyperparameter specification §5/§6, the vocabulary specification §7.

Companion of :mod:`utils.runner.detector` (GDINO / OWLv2) and
:mod:`models.omdet.detector`. Nothing in those modules is modified;
:class:`~utils.runner.detector.TermDetection` is imported so all arms hand the
runner the same record type.

Implementation
--------------
GLIP is the mmdetection 3.x re-implementation
(``mmdet/models/detectors/glip.py`` + ``atss_vlfusion_head.py``), which needs
its own interpreter --- ``~/venvs/uc-glip/bin/python``. Feasibility,
module census and the reason the frozen spec's "GLIP is ZS-only" clause is
only true of the ``early_fuse=False`` variant are in
the lora_feasibility_glip_omdet.md note.

The caption is **not** rebuilt by mmdet. mmdet's own
``to_plain_text_prompts`` would join the terms as ``"person. clothes. ..."``,
which is a different string from the frozen V6 caption
(``"person . clothes . ... ."``, ``utils.protocol.vocabulary.render_caption``) that
every other arm is evaluated on and that §7 records verbatim. Instead this
module passes the frozen caption through unchanged and supplies the explicit
``tokens_positive`` character spans that GLIP's
``get_tokens_positive_and_prompts`` accepts, so the term-to-span assignment is
computed from the same string that is recorded.

Read-out audit (``mmdet/models/dense_heads/atss_vlfusion_head.py``)
-------------------------------------------------------------------
GLIP is **per channel by construction**, like Grounding DINO and OmDet-Turbo,
and unlike OWLv2:

* ``convert_grounding_to_cls_scores`` builds a *(num_priors, num_terms)* score
  matrix by averaging ``sigmoid(cls_logits)`` over each term's token span.
  There is no reduction over the term axis.
* ``filter_scores_and_topk`` then sorts the flattened *(prior x term)* matrix
  and returns ``keep_idxs, labels = topk_idxs.unbind(dim=1)`` --- one prior can
  therefore come back under several terms, each with **its own** score.
* ``_bbox_post_process`` uses mmdet's ``batched_nms``, which offsets boxes by
  label, so NMS never merges a box across terms either.

No override is applied and none is needed. It is verified empirically rather
than assumed --- the regression tests asserts that boxes fire on
several near-synonymous terms and that no multi-term box carries a repeated
score.

Protocol overrides applied to ``test_cfg`` (spec §6)
----------------------------------------------------
``score_thr`` 0.05 -> ``protocol.score_threshold`` (0.0) and ``max_per_img``
-> ``protocol.max_detections_per_image`` (100). Both are protocol values that
§6 forces to be identical across models and across ZS/FT. Everything else ---
``nms_pre=1000`` and ``nms=dict(type='nms', iou_threshold=0.6)`` --- is GLIP's
own behaviour and is left untouched and recorded, per §6's
"each model keeps its own NMS behaviour".
"""
from __future__ import annotations

import os

from utils.runner.paths import paths

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING, Sequence

from utils.runner.logging_utils import get_logger
from utils.runner.detector import TermDetection
from utils.protocol.eval_protocol import EVAL_PROTOCOL, EvalProtocol

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = get_logger(__name__)

class _Variants(Mapping):
    """``variant -> (mmdet config, checkpoint)``, from configs/models.yaml.

    Resolved on lookup rather than on import: the mmdet root comes from an
    environment variable, and baking an unset one into a path at import time
    would produce a plausible path before anyone could act on it. It would also
    make importing this module require an mmdet checkout, which the detector
    factory and the fingerprint scan do without ever building GLIP.
    """

    def _settings(self):
        from utils.runner.model_registry import model_settings
        return model_settings("glip")

    def __getitem__(self, variant: str) -> tuple[str, str]:
        settings = self._settings()
        if variant != settings.variant:
            raise KeyError(variant)
        return (settings.mmdet_config_path,
                str(paths().checkpoint_root / "glip" / settings.checkpoint))

    def __iter__(self):
        from utils.runner.model_registry import default_variants
        return iter([default_variants()["glip"]])

    def __len__(self) -> int:
        return 1


#: ``tiny_c`` is GLIP-T(C): obj365+goldg pretraining, ``early_fuse=True``.
#: Exact-match against the config's declared variant; never guessed.
GLIP_VARIANTS: Mapping[str, tuple[str, str]] = _Variants()

#: mmdet's ``create_positive_map`` allocates this many token columns, matching
#: the language model's ``max_tokens``. A term whose span starts beyond it
#: would silently receive an all-zero positive map and then a NaN score, so the
#: span builder refuses instead. All 32 V6 captions fit (the longest, C5, is
#: 205 BERT tokens).
MAX_TEXT_TOKENS = 256


@dataclass(frozen=True)
class _Prompt:
    """A frozen caption with the char spans of each term inside it."""

    caption: str
    terms: tuple[str, ...]
    tokens_positive: tuple[tuple[tuple[int, int], ...], ...]


def build_tokens_positive(caption: str, terms: Sequence[str]) -> _Prompt:
    """Locate each term inside ``caption`` as a single character span.

    The V6 caption is ``" . ".join(terms) + " ."`` with terms guaranteed
    distinct (:class:`utils.protocol.conditions.Condition` rejects duplicates), so
    each term occurs at a well-defined offset. The scan is positional rather
    than a global ``find`` so that a term which is a substring of a later term
    (``person`` inside ``missing person``) cannot be matched to the wrong span.

    Raises:
        ValueError: If a term is not found at the expected position, i.e. the
            caption was not rendered by
            :func:`utils.protocol.vocabulary.render_caption` from these terms.
    """
    spans: list[tuple[tuple[int, int], ...]] = []
    cursor = 0
    for term in terms:
        start = caption.find(term, cursor)
        if start < 0:
            raise ValueError(
                f"term {term!r} not found in caption {caption!r} at or after "
                f"offset {cursor}; the caption was not rendered from these terms")
        end = start + len(term)
        spans.append(((start, end),))
        cursor = end
    return _Prompt(caption=caption, terms=tuple(terms), tokens_positive=tuple(spans))


class GlipDetector:
    """GLIP with the per-channel read-out described in the module docstring.

    Args:
        variant: Key of :data:`GLIP_VARIANTS`.
        device: Always ``"cuda:0"`` with the GPU pinned via
            ``CUDA_VISIBLE_DEVICES``.
        protocol: The frozen protocol. ``score_threshold`` and
            ``max_detections_per_image`` are pushed into mmdet's ``test_cfg``.
        config_override / checkpoint_override: Replace the variant's entries.
        max_output_detections: Raise mmdet's ``max_per_img`` above the protocol
            cap so more raw detections reach the cap policy. Same knob, and the
            same reason, as OWLv2's ``max_output_detections``: GLIP's own
            ``max_per_img`` default is **exactly 100**, i.e. it coincides with
            the protocol cap, so under ``cap_policy=per_category`` (which allows
            100 *per category*) GLIP can never fill more than one category's
            budget, and in the C conditions the distractors eat into the same
            100 before they are dropped. Leaving this ``None`` keeps GLIP's
            model default; it does not change any threshold, only how much the
            model is allowed to emit before the protocol scores it.
    """

    def __init__(self, variant: str = "tiny_c", device: str = "cuda:0",
                 protocol: EvalProtocol = EVAL_PROTOCOL,
                 config_override: str | None = None,
                 checkpoint_override: str | None = None,
                 max_output_detections: int | None = None,
                 v6_adapter_dir: str | None = None) -> None:
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg
        from mmengine.config import Config
        from mmengine.dataset import Compose

        if variant not in GLIP_VARIANTS and (config_override is None or
                                             checkpoint_override is None):
            raise ValueError(
                f"unknown GLIP variant {variant!r}; expected one of "
                f"{sorted(GLIP_VARIANTS)} or an explicit config+checkpoint override")
        default_config, default_checkpoint = (GLIP_VARIANTS[variant]
                                     if variant in GLIP_VARIANTS else (None, None))
        config = config_override or default_config
        checkpoint = checkpoint_override or default_checkpoint

        cfg = Config.fromfile(config)
        self.model_default_test_cfg = {
            "score_thr": cfg.model.test_cfg.score_thr,
            "nms_pre": cfg.model.test_cfg.nms_pre,
            "nms": dict(cfg.model.test_cfg.nms),
            "max_per_img": cfg.model.test_cfg.max_per_img,
            "min_bbox_size": cfg.model.test_cfg.min_bbox_size,
        }
        # Spec §6 -- protocol values, identical for every model and for ZS/FT.
        cfg.model.test_cfg.score_thr = protocol.score_threshold
        cfg.model.test_cfg.max_per_img = (max_output_detections
                                          or protocol.max_detections_per_image)
        self.max_output_detections = cfg.model.test_cfg.max_per_img

        self.protocol = protocol
        self.model_name = "glip"
        self.checkpoint_name = (f"{variant}-zeroshot" if checkpoint_override is None
                                else variant)
        self._model = init_detector(cfg, checkpoint, device=device, palette="none")
        self._model.eval()

        pipeline_cfg = get_test_pipeline_cfg(cfg.copy())
        pipeline_cfg = [t for t in pipeline_cfg if t["type"] != "LoadAnnotations"]
        pipeline_cfg[0]["type"] = "mmdet.LoadImageFromNDArray"
        self._pipeline = Compose(pipeline_cfg)
        self._resize = next(
            (dict(t) for t in pipeline_cfg if "Resize" in str(t["type"])), None)
        self.v6_checkpoint = (None if v6_adapter_dir is None
                              else _attach_v6_adapter(self._model, v6_adapter_dir))
        if self.v6_checkpoint is not None:
            self.checkpoint_name = (
                f"{variant}-ft-seed{self.v6_checkpoint['seed']}"
                f"-step{self.v6_checkpoint['step']}")
            self._model.eval()
        self._prompt_cache: _Prompt | None = None
        logger.info("GLIP %s loaded on %s (config=%s)", variant, device, config)

    def _prompt_for(self, terms: Sequence[str]) -> _Prompt:
        """Cache the caption/span build; conditions repeat it 14,980 times."""
        from utils.protocol.vocabulary import render_caption

        if self._prompt_cache is None or self._prompt_cache.terms != tuple(terms):
            self._prompt_cache = build_tokens_positive(
                render_caption(tuple(terms)), terms)
        return self._prompt_cache

    def predict(self, image: "PILImage", terms: Sequence[str]) -> list[TermDetection]:
        """One forward pass; every (box, term) pair the head emits is kept.

        No argmax and no max over the term axis --- see the module docstring's
        read-out audit. Labels are positional indices into ``terms``, so a term
        is never recovered by string matching against a decoded phrase.
        """
        import numpy as np
        import torch

        prompt = self._prompt_for(terms)
        # mmcv/mmdet consume BGR; PIL gives RGB.
        array = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        data = self._pipeline(dict(img=array, img_id=0, text=prompt.caption,
                                   custom_entities=False))
        sample = data["data_samples"]
        # Explicit char spans -> GLIP skips its own caption rebuild and NER.
        sample.tokens_positive = [list(span) for span in prompt.tokens_positive]
        batch = {"inputs": [data["inputs"]], "data_samples": [sample]}

        with torch.no_grad():
            result = self._model.test_step(batch)[0]

        instances = result.pred_instances
        boxes = instances.bboxes.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy()

        n_terms = len(prompt.terms)
        out: list[TermDetection] = []
        for box, score, label in zip(boxes, scores, labels):
            index = int(label)
            if not 0 <= index < n_terms:
                raise ValueError(
                    f"GLIP returned label index {index} for a {n_terms}-term query; "
                    "the positive map and the term list are out of step")
            out.append(TermDetection(
                term=prompt.terms[index], score=float(score),
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            ))
        return out

    def postprocessing_notes(self) -> dict:
        test_cfg = self._model.test_cfg
        return {
            "v6_checkpoint": self.v6_checkpoint,
            "readout": "per_channel_by_construction",
            "readout_detail": (
                "mmdet's convert_grounding_to_cls_scores averages sigmoid(cls_logits) "
                "over each term's token span into a (prior x term) matrix with no "
                "reduction over the term axis; filter_scores_and_topk sorts that "
                "flattened matrix and unbinds the flat index into (prior, term), so "
                "one prior may be returned under several terms each with its own "
                "score; batched_nms is offset by label and never merges a box across "
                "terms. No argmax is applied and none is bypassed"),
            "caption_source": "utils.protocol.vocabulary.render_caption (frozen V6 caption)",
            "tokens_positive": "explicit char spans of each term in the frozen caption",
            "nms": "model_default_unchanged",
            "nms_cfg": dict(test_cfg.nms),
            "nms_pre": test_cfg.nms_pre,
            "min_bbox_size": test_cfg.min_bbox_size,
            "score_thr_applied": test_cfg.score_thr,
            "max_per_img_applied": test_cfg.max_per_img,
            "max_detections_per_image": self.protocol.max_detections_per_image,
            "model_default_test_cfg": self.model_default_test_cfg,
            "protocol_overrides": ["score_thr", "max_per_img"],
            "input_resize": self._resize,
            "max_text_tokens": MAX_TEXT_TOKENS,
        }


def _attach_v6_adapter(model, adapter_dir: str) -> dict:
    """Inject spec §2 LoRA, load a V6 checkpoint, and re-verify it. Never merges.

    Mirror of :func:`models.omdet.detector._attach_v6_adapter`. The manifest is
    not trusted as a description — it is checked against the live model: §2
    compliance, per-group injection counts, the alpha/r fingerprint and the
    training caption.

    GLIP's ``modules_to_save`` covers only the four head *modules*; the three
    bare head ``Parameter`` s (``log_scale``, ``bias_lang``, ``bias0``) are
    restored by the adapter state dict itself, which stores parameters by name.

    Returns:
        Provenance recorded verbatim into the fragment, so a fine-tuned fragment
        can never be mistaken for a zero-shot one.
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
    if manifest.model != "glip":
        raise ValueError(
            f"{directory} holds a {manifest.model!r} checkpoint, not a GLIP one")
    placement = get_placement("glip")
    targets = select_targets([n for n, _ in model.named_modules()], placement)
    if len(targets) != EXPECTED_LORA_MODULES["glip"]:
        raise ValueError(
            f"§2 regex selected {len(targets)} modules, expected "
            f"{EXPECTED_LORA_MODULES['glip']}")
    inject_adapter_in_model(
        LoraConfig(r=manifest.lora_r, lora_alpha=manifest.lora_alpha,
                   lora_dropout=manifest.lora_dropout, bias="none",
                   target_modules=targets),
        model)
    n_loaded = load_adapter_weights(model, directory, strict=True)
    verify_loaded_checkpoint(model, manifest, placement,
                             expected_caption=training_caption())
    per_group = assert_group_counts(model, placement,
                                    EXPECTED_GROUP_COUNTS["glip"], group_of)
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


def build_glip_detector(variant: str = "tiny_c", device: str = "cuda:0",
                        protocol: EvalProtocol = EVAL_PROTOCOL,
                        config_override: str | None = None,
                        checkpoint_override: str | None = None,
                        max_output_detections: int | None = None,
                        v6_adapter_dir: str | None = None) -> GlipDetector:
    return GlipDetector(variant=variant, device=device, protocol=protocol,
                        config_override=config_override,
                        checkpoint_override=checkpoint_override,
                        max_output_detections=max_output_detections,
                        v6_adapter_dir=v6_adapter_dir)
