"""Measure semantic distance with each model's own text encoder (axis B).

Spec: the vocabulary specification §3.

    "의미 거리는 선언하지 않고 측정한다."
    (Semantic distance is measured, not declared.)

The B axis must not be reported as a synonym/non-synonym dichotomy. Instead,
each ``person`` variant gets a cosine against ``person`` **computed with the
encoder of the model being evaluated**, and ΔAP is plotted against that
cosine. Grounding DINO uses BERT; OWLv2 uses the CLIP text tower. The two
disagree, and that disagreement is part of what explains model-to-model
differences — so a cosine is always stored with the model it came from and
is never shared between models.

Two pooling conventions are recorded per encoder:

``mean_subtoken``
    Mean of the contextual hidden states of the term's own sub-tokens,
    special tokens excluded. This is the representation a phrase-grounding
    model actually contrasts region features against, so it is the primary
    number.
``pooled``
    The encoder's own sentence-level output (BERT ``[CLS]`` pooler; CLIP's
    projected EOS embedding). Reported alongside because it is the
    conventional "sentence embedding" and readers will expect it.

Four encoders are defined, one per evaluated model, and a cosine is always
stored with the model it came from:

======  ==================  ===========================================
model   encoder             ``pooled`` convention
======  ==================  ===========================================
gdino   BERT                ``[CLS]`` pooler (``tanh(W·CLS)``)
owlv2   CLIP text tower     ``text_projection(EOS)``
glip    BERT (GLIP's own)   ``[CLS]`` hidden state — see below
omdet   CLIP text tower     ``EOS @ text_projection``
======  ==================  ===========================================

The exact convention used for a row is recorded in
:attr:`TermCosines.pooling_convention` so the ``pooled`` column is never
compared across encoders that do not mean the same thing by it. Rows written
before that field existed load with an empty string (unrecorded), which is not
the same as a convention named "".

Weights provenance — every encoder uses its own detector's weights
------------------------------------------------------------------
Spec §3 asks for "the text encoder of the model being evaluated". A hub
checkpoint of the same *architecture* is not that: BERT and CLIP text towers
both move during detector pretraining. If one model were read out with vanilla
weights and another with its own, a gap between them could not be told apart
from a provenance artefact — and gdino/glip share an architecture, so the
confound would land exactly where the axis-B comparison is made.

All four therefore use their own detector checkpoint, recorded per row in
:attr:`TermCosines.weights_source`:

======  =========================================  =========================
model   weights                                    how
======  =========================================  =========================
gdino   ``groundingdino_swinb_cogcoor.pth``        ``bert.*`` lifted into a
                                                   hub BERT skeleton
glip    ``glip_tiny_c_mmdet-*.pth``                same, different prefix
owlv2   ``owlv2-large-patch14-ensemble``           197 text tensors ship in
                                                   the model's own file
omdet   ``omdet-turbo-swin-tiny-hf``               197 text tensors ship in
                                                   the model's own file
======  =========================================  =========================

The two CLIP towers were **verified** (2026-08-09) to carry their text weights
in the detector checkpoint itself, so ``from_pretrained`` never falls back to a
vanilla CLIP. Only the two BERT towers need the manual lift, because a hub BERT
is what the architecture id resolves to.

``BertTextEncoder(variant=None)`` reproduces the superseded vanilla-BERT
numbers for the side-by-side table in the v6_text_cosines.md note. It is not
used by the pipeline and it logs a warning.

GLIP notes
----------
GLIP's checkpoint carries **no pooler** (``add_pooling_layer=False``), so its
``pooled`` would be the ``[CLS]`` last-hidden-state — which for a one-word
input barely moves with the word, putting every GLIP ``pooled`` cosine at
~1.0. That is a degenerate quantity, not a measurement, so GLIP rows record
``cosine_pooled = None``. ``None`` here means "not measurable with this
encoder", which this project keeps distinct from a measured value, exactly as
it keeps ``_미측정_`` distinct from ``0``.
"""
from __future__ import annotations

from utils.runner.paths import paths

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)

#: bert-base-uncased is cached in the user HF cache, not in ``HF_HOME``;
#: every GDINO path in this repo relies on that.
GDINO_TEXT_ENCODER = "bert-base-uncased"
#: Grounding DINO's BERT is *its own*, updated during GDINO pretraining. The
#: weights come out of the detection checkpoint under this prefix; the hub id
#: above supplies only the architecture and tokenizer.
GDINO_STATE_DICT_PREFIX = "bert."
GDINO_CHECKPOINTS: Mapping[str, str] = {
    "swinb": (str(paths().checkpoint_root / "gdino") + "/"
              "models--ShilongLiu--GroundingDINO/snapshots/"
              "a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swinb_cogcoor.pth"),
}

#: Where a row's encoder weights came from. Recorded per row so a reader can
#: tell a model-to-model difference from a weights-provenance artefact.
OWN_CHECKPOINT = "own_checkpoint"
VANILLA_HUB = "vanilla_hub"
OWLV2_CHECKPOINTS: Mapping[str, str] = {
    "base": "google/owlv2-base-patch16-ensemble",
    "large": "google/owlv2-large-patch14-ensemble",
}

#: GLIP's text tower is bert-base-uncased *architecture* with GLIP's own
#: weights; the weights come out of the detection checkpoint under this prefix.
GLIP_TEXT_ENCODER = "bert-base-uncased"
GLIP_STATE_DICT_PREFIX = "language_model.language_backbone.body.model."
GLIP_CHECKPOINTS: Mapping[str, str] = {
    "tiny_c": str(paths().checkpoint_root / "glip" / "glip_tiny_c_mmdet-2fc427dd.pth"),
}

OMDET_CHECKPOINTS: Mapping[str, str] = {
    "swin-tiny": "omlab/omdet-turbo-swin-tiny-hf",
}


@dataclass(frozen=True)
class TermCosines:
    """Cosines of one variant term against its reference term, one encoder.

    ``pooling_convention`` defaults to the empty string so artefacts written
    before the field existed still load; empty means "unrecorded", which is not
    the same as a convention whose name happens to be empty.
    """

    reference_term: str
    term: str
    model: str
    encoder: str
    encoder_checkpoint: str
    cosine_mean_subtoken: float
    cosine_pooled: float | None
    n_subtokens: int
    pooling_convention: str = ""
    weights_source: str = ""


def _cosine(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def _load_prefixed_weights(model, weights_path: str, prefix: str) -> int:
    """Load ``prefix``-scoped tensors from a detection checkpoint into ``model``.

    Shared by the GDINO and GLIP text encoders: both are BERT architectures
    whose real weights live inside a detector ``.pth`` rather than on the hub.

    Returns:
        How many tensors were transferred.

    Raises:
        ValueError: If the prefix matches nothing, or if any tensor fails to
            map. A silent partial load would leave part of the tower at its
            hub initialisation, which is exactly the bug this function exists
            to remove.
    """
    import torch

    raw = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = raw.get("model", raw.get("state_dict", raw))
    own = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not own:
        raise ValueError(
            f"{weights_path} holds no keys under {prefix!r}; the checkpoint "
            "layout changed and the encoder must be re-audited")
    result = model.load_state_dict(own, strict=False)
    # `position_ids` is a buffer some transformers versions no longer register;
    # any *other* unexpected or missing key means the layout drifted.
    unexpected = [k for k in result.unexpected_keys if not k.endswith("position_ids")]
    missing = [k for k in result.missing_keys if not k.endswith("position_ids")]
    if unexpected or missing:
        raise ValueError(
            f"{weights_path}: text weights did not map cleanly under {prefix!r}: "
            f"missing={missing} unexpected={unexpected}")
    return len(own)


class BertTextEncoder:
    """Grounding DINO's text tower: BERT architecture, **GDINO's own weights**.

    Corrected 2026-08-09. This class previously loaded the vanilla
    ``bert-base-uncased`` hub weights, which are not Grounding DINO's encoder —
    GDINO's BERT moved during its pretraining. Spec §3 asks for "the text
    encoder of the model being evaluated", and with vanilla weights a
    gdino-vs-glip cosine gap could not be told apart from a
    weights-provenance artefact, since GLIP's tower is the same architecture.

    ``variant=None`` restores the old vanilla behaviour. It exists only so the
    superseded numbers can be reproduced for the side-by-side table in
    the v6_text_cosines.md note; it is not used by the pipeline.

    The GDINO checkpoint carries ``bert.pooler.*``, so ``pooled`` keeps its
    original meaning (the ``[CLS]`` pooler) and only the weights change.
    """

    model_name = "gdino"
    encoder = "bert"
    pooling_convention = "bert_cls_pooler"

    def __init__(self, checkpoint: str = GDINO_TEXT_ENCODER, device: str = "cuda:0",
                 variant: str | None = "swinb",
                 checkpoint_override: str | None = None) -> None:
        import os

        import torch
        from transformers import AutoModel, AutoTokenizer

        # bert-base-uncased lives in the *user* cache; HF_HOME points at
        # a cache directory that does not hold it.
        hf_home = os.environ.pop("HF_HOME", None)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModel.from_pretrained(checkpoint)
        finally:
            if hf_home is not None:
                os.environ["HF_HOME"] = hf_home

        weights_path = checkpoint_override or (
            GDINO_CHECKPOINTS[variant] if variant is not None else None)
        if weights_path is None:
            self.checkpoint = checkpoint
            self.weights_source = VANILLA_HUB
            logger.warning(
                "BertTextEncoder is using VANILLA %s weights, not Grounding DINO's "
                "own BERT; this reproduces the superseded numbers only", checkpoint)
        else:
            n = _load_prefixed_weights(self.model, weights_path, GDINO_STATE_DICT_PREFIX)
            self.checkpoint = f"{weights_path} ({checkpoint} architecture)"
            self.weights_source = OWN_CHECKPOINT
            logger.info("loaded %d GDINO BERT tensors from %s", n, weights_path)

        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def embed(self, term: str) -> tuple["object", "object", int]:
        """Return ``(mean_subtoken, pooled, n_subtokens)`` for ``term``."""
        import torch

        batch = self.tokenizer(term, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**batch)
        mask = batch["special_tokens_mask"] if "special_tokens_mask" in batch else None
        if mask is None:
            ids = batch["input_ids"][0].tolist()
            special = set(self.tokenizer.all_special_ids)
            keep = [i for i, t in enumerate(ids) if t not in special]
        else:
            keep = [i for i, m in enumerate(mask[0].tolist()) if m == 0]
        hidden = out.last_hidden_state[0][keep]
        pooled = out.pooler_output[0] if out.pooler_output is not None else hidden.mean(0)
        return hidden.mean(0), pooled, len(keep)


class Owlv2TextEncoder:
    """OWLv2's CLIP text tower, frozen and eval-mode."""

    model_name = "owlv2"
    encoder = "clip_text"
    pooling_convention = "clip_projected_eos"
    #: Verified 2026-08-09: the OWLv2 checkpoint file itself carries 197
    #: ``owlv2.text_model.*`` / ``owlv2.text_projection`` tensors, so
    #: ``from_pretrained`` loads OWLv2's own text tower. There is no vanilla
    #: CLIP fallback anywhere on this path.
    weights_source = OWN_CHECKPOINT

    def __init__(self, variant: str = "large", device: str = "cuda:0",
                 cache_dir: str | None = None) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if variant not in OWLV2_CHECKPOINTS:
            raise ValueError(f"unknown OWLv2 variant {variant!r}")
        checkpoint = OWLV2_CHECKPOINTS[variant]
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self.processor = Owlv2Processor.from_pretrained(checkpoint, **kwargs)
        detection = Owlv2ForObjectDetection.from_pretrained(checkpoint, **kwargs)
        self.clip = detection.owlv2
        self.checkpoint = f"{checkpoint} ({variant})"
        self.device = torch.device(device)
        self.clip.to(self.device).eval()

    def embed(self, term: str) -> tuple["object", "object", int]:
        import torch

        tok = self.processor.tokenizer(term, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_out = self.clip.text_model(**tok)
            pooled = self.clip.text_projection(text_out.pooler_output)[0]
        ids = tok["input_ids"][0].tolist()
        special = set(self.processor.tokenizer.all_special_ids)
        keep = [i for i, t in enumerate(ids) if t not in special]
        hidden = text_out.last_hidden_state[0][keep]
        return hidden.mean(0), pooled, len(keep)


class GlipTextEncoder:
    """GLIP's text tower: ``bert-base-uncased`` architecture, GLIP's own weights.

    The weights are lifted out of the GLIP detection checkpoint (see the module
    docstring for why the vanilla hub weights would be the wrong choice). The
    checkpoint has no pooler, so ``pooled`` is the ``[CLS]`` last-hidden-state.
    """

    model_name = "glip"
    encoder = "bert_glip"
    pooling_convention = "cls_hidden_state (checkpoint has no pooler)"
    weights_source = OWN_CHECKPOINT
    #: The checkpoint has no pooler, so ``pooled`` degenerates: the ``[CLS]``
    #: hidden state of a one-word input barely moves with the word and every
    #: cosine lands at ~1.0. Recording that as a number invites someone to read
    #: it, so it is recorded as ``None`` -- unmeasurable, which this project
    #: has always kept distinct from a measured value.
    pooled_is_meaningful = False

    def __init__(self, variant: str = "tiny_c", device: str = "cpu",
                 checkpoint_override: str | None = None) -> None:
        import os

        import torch
        from transformers import AutoTokenizer, BertModel

        if variant not in GLIP_CHECKPOINTS and checkpoint_override is None:
            raise ValueError(f"unknown GLIP variant {variant!r}")
        weights_path = checkpoint_override or GLIP_CHECKPOINTS[variant]

        # bert-base-uncased lives in the *user* cache, as for BertTextEncoder.
        hf_home = os.environ.pop("HF_HOME", None)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(GLIP_TEXT_ENCODER)
            model = BertModel.from_pretrained(GLIP_TEXT_ENCODER, add_pooling_layer=False)
        finally:
            if hf_home is not None:
                os.environ["HF_HOME"] = hf_home

        n = _load_prefixed_weights(model, weights_path, GLIP_STATE_DICT_PREFIX)
        logger.info("loaded %d GLIP BERT tensors from %s", n, weights_path)

        self.model = model
        self.checkpoint = f"{weights_path} ({GLIP_TEXT_ENCODER} architecture)"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def embed(self, term: str) -> tuple["object", "object", int]:
        """Return ``(mean_subtoken, cls_hidden, n_subtokens)`` for ``term``."""
        import torch

        batch = self.tokenizer(term, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**batch)
        ids = batch["input_ids"][0].tolist()
        special = set(self.tokenizer.all_special_ids)
        keep = [i for i, t in enumerate(ids) if t not in special]
        hidden = out.last_hidden_state[0]
        return hidden[keep].mean(0), hidden[0], len(keep)


class OmDetTextEncoder:
    """OmDet-Turbo's CLIP text tower, frozen and eval-mode.

    ``pooled`` reproduces OmDet's own *class* embedding path
    (``OmDetTurboLanguageBackbone.forward(encode_type="class")``): the hidden
    state at the EOS position -- located, as CLIP does, by ``argmax`` over the
    token ids -- projected through ``text_projection``. That is the vector the
    detector actually cosine-matches query features against, so it is the
    faithful analogue of OWLv2's projected EOS embedding.
    """

    model_name = "omdet"
    encoder = "clip_text"
    pooling_convention = "clip_projected_eos (OmDet class-embedding path)"
    #: Verified 2026-08-09: the OmDet-Turbo checkpoint file itself carries 197
    #: ``language_backbone.model.text_model.*`` / ``language_backbone
    #: .text_projection`` tensors, so this is OmDet's own tower, not vanilla CLIP.
    weights_source = OWN_CHECKPOINT

    def __init__(self, variant: str = "swin-tiny", device: str = "cpu",
                 cache_dir: str | None = None) -> None:
        import torch

        # Imported here, not at module scope: OmDet-Turbo landed in
        # transformers 4.45, and this module must stay importable in the older
        # ZiRaGroundingDINO venv (transformers 4.43.3) where the GDINO and GLIP
        # encoders are perfectly usable. A bare ImportError at that line reads
        # like a broken module rather than a missing optional stack, so it is
        # re-raised with the fix in the message.
        try:
            from transformers import AutoProcessor, OmDetTurboForObjectDetection
        except ImportError as exc:  # pragma: no cover - venv-dependent
            raise ImportError(
                "OmDetTurboForObjectDetection is not available in this "
                "interpreter (needs transformers>=4.45). Run the OmDet encoder "
                "under ~/venvs/uc-omdet; the gdino/glip/owlv2 encoders in this "
                "module do not need it."
            ) from exc

        if variant not in OMDET_CHECKPOINTS:
            raise ValueError(f"unknown OmDet-Turbo variant {variant!r}")
        checkpoint = OMDET_CHECKPOINTS[variant]
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self.processor = AutoProcessor.from_pretrained(checkpoint, **kwargs)
        detection = OmDetTurboForObjectDetection.from_pretrained(
            checkpoint, low_cpu_mem_usage=False, **kwargs)
        backbone = detection.language_backbone
        self.text_model = backbone.model
        self.text_projection = backbone.text_projection
        self.checkpoint = f"{checkpoint} ({variant})"
        self.device = torch.device(device)
        self.text_model.to(self.device).eval()
        self.text_projection.data = self.text_projection.data.to(self.device)

    def embed(self, term: str) -> tuple["object", "object", int]:
        import torch

        tok = self.processor.tokenizer(term, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.text_model(**tok)
            hidden = out.last_hidden_state[0]
            eos = int(tok["input_ids"][0].argmax())
            pooled = hidden[eos] @ self.text_projection
        ids = tok["input_ids"][0].tolist()
        special = set(self.processor.tokenizer.all_special_ids)
        keep = [i for i, t in enumerate(ids) if t not in special]
        return hidden[keep].mean(0), pooled, len(keep)


def compare_terms(encoder, reference_term: str,
                  variants: Sequence[str]) -> list[TermCosines]:
    """Cosine of every variant against ``reference_term`` under one encoder."""
    ref_mean, ref_pooled, _ = encoder.embed(reference_term)
    rows: list[TermCosines] = []
    for term in variants:
        mean_vec, pooled_vec, n_sub = encoder.embed(term)
        rows.append(TermCosines(
            reference_term=reference_term,
            term=term,
            model=encoder.model_name,
            encoder=encoder.encoder,
            encoder_checkpoint=encoder.checkpoint,
            cosine_mean_subtoken=_cosine(mean_vec, ref_mean),
            cosine_pooled=(_cosine(pooled_vec, ref_pooled)
                           if getattr(encoder, "pooled_is_meaningful", True) else None),
            n_subtokens=n_sub,
            pooling_convention=getattr(encoder, "pooling_convention", ""),
            weights_source=getattr(encoder, "weights_source", ""),
        ))
    return rows


def write_cosines(rows: Sequence[TermCosines], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=2))


def load_cosines(path: Path) -> list[TermCosines]:
    return [TermCosines(**r) for r in json.loads(path.read_text())]


def cosine_lookup(rows: Sequence[TermCosines]) -> Mapping[tuple[str, str, str], float]:
    """``(model, reference_term, term) -> cosine_mean_subtoken``, exact keys only."""
    return {(r.model, r.reference_term, r.term): r.cosine_mean_subtoken for r in rows}
