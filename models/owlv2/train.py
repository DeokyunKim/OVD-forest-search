"""OWLv2-large Phase 2 training machinery: build, gate, data, health probe.

Spec: the frozen-hyperparameter specification §1 (LoRA), §2 (placement — the
per-model OWLv2 table), §3 (optimizer), §4.0/§4.1 (data policy and loop),
§5 (resolution), §6 (evaluation, untouched here).
Placement data: ``models/shared/lora_spec_v6line.py::OWLV2_PLACEMENT``.
Pre-registered thresholds: the training risk register §9.

The CLI and the loop live in ``train_v6_owlv2.py`` (research tree); everything that can
be asserted without a GPU lives here so the regression tests can
reach it.

Three things this model needs that Grounding DINO does not
----------------------------------------------------------
1. **An ordered query list, not a caption.** OWLv2 encodes each term as its own
   CLIP sequence. There is no string to compare byte for byte, so the identity
   that has to hold between training and evaluation is over the *token* stream:
   :func:`assert_query_identity` runs the training path and the evaluation path
   through the same processor and compares ``input_ids`` and ``attention_mask``
   element by element. A reordered list is a different experiment and would not
   raise anywhere else.
2. **A reconstructed loss.** ``Owlv2ForObjectDetection.forward`` takes no
   ``labels``; HuggingFace ships inference only. The set-prediction objective
   is rebuilt outside the model by ``models/shared/detection_loss.py`` from
   ``logits`` *(batch, patches, terms)* and ``pred_boxes`` *(batch, patches, 4)*
   — the same module the OmDet-Turbo arm uses, so the two arms optimise the
   same objective.
3. **Padded-square box normalisation.** ``Owlv2ImageProcessor`` pads to a
   ``(0,0)``-anchored square *before* resizing, and the model's boxes are
   normalised to that square — ``_scale_boxes`` multiplies by ``max(H, W)``,
   not by ``H`` and ``W`` separately. Normalising targets per axis instead
   would train every non-square image against systematically wrong boxes and
   the loss would still fall.

Resolution: a recorded spec deviation
-------------------------------------
§5's table says OWLv2 is ``960 x 960``. That is the value for
``owlv2-base-patch16-ensemble``. The checkpoint this arm uses,
``google/owlv2-large-patch14-ensemble``, ships a processor fixed at
**1008 x 1008** (``preprocessor_config.json``: ``size = {1008, 1008}``;
1008 / 14 = 72 patches, 5,184 tokens). The resolution is a property of the
checkpoint, not a free parameter, so it cannot be set to 960 without resizing
the position embeddings. §5's operative requirement — *"각 모델의 ZS와 FT가
서로 같은 해상도를 쓰는 것만 보장한다"* — is met: the zero-shot arm
(``models/owlv2/detector.py``) calls the same processor with the same defaults.
Recorded in ``run_config.json`` and in the risk register rather than silently
normalised.
"""
from __future__ import annotations

import os
from pathlib import Path

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from utils.runner.logging_utils import get_logger
from utils.protocol.eval_protocol import EVAL_PROTOCOL
from models.shared.lora_spec_v6line import (
    OWLV2_PLACEMENT,
    SpecViolationError,
    assert_compliant,
    assert_scaling_compliant,
    format_report,
    is_lora_module,
    select_targets,
    verify_injection,
    verify_lora_scaling,
)
from models.owlv2.guard import (
    Owlv2ProbeMetrics,
    Owlv2Thresholds,
    distinct_box_rate,
    max_abs_drift,
    normalised_channel_entropy,
)
from utils.protocol.train_caption import training_class_names
from utils.train.training_schedule import LORA
from utils.data.xarch_data import V6Record

logger = get_logger(__name__)

MODEL_KEY = "owlv2"
CHECKPOINT_ID = "google/owlv2-large-patch14-ensemble"
#: The cache that actually holds the checkpoint (offline; see CLAUDE.md).
#: HuggingFace cache. `HF_HOME` is the standard variable and is honoured first;
#: the fallback keeps a run working when it is unset.
#: `cache_dir` wants the HUB directory, not HF_HOME. They differ by one path
#: component and the difference is invisible until an offline load fails with
#: "can't load image processor", which names the model rather than the path.
HF_CACHE = (os.environ.get("HF_HUB_CACHE")
            or (os.path.join(os.environ["HF_HOME"], "hub")
                if os.environ.get("HF_HOME") else None)
            or str(Path.home() / ".cache" / "huggingface" / "hub"))

#: 24 ViT blocks x {q,k,v,out_proj,fc1,fc2}. Measured on the real checkpoint,
#: not derived from the config, because a config can be wrong while looking
#: right (this project has had that failure twice).
EXPECTED_LORA_MODULES: int = 144
#: 1008 = 72 patches x 14. Fixed by the checkpoint's processor (see docstring).
INPUT_RESOLUTION: tuple[int, int] = (1008, 1008)
N_PATCHES: int = (INPUT_RESOLUTION[0] // 14) ** 2

#: §2 freezes these even though they are neither the text tower nor the ViT.
#: Recorded so "which heads train" is answerable from the code, not from
#: someone's memory: OWLv2's objectness head plays no part in text-conditioned
#: detection scoring (``post_process_object_detection`` reads ``logits`` only),
#: and §2's per-model table names exactly two heads.
FROZEN_HEADS: tuple[str, ...] = ("objectness_head", "layer_norm")


# --- query list (the OWLv2 analogue of caption byte-identity) ----------------

def training_query_list() -> list[str]:
    """The ordered term list sent to the processor during training.

    OWLv2 takes a **list**, so the training/evaluation contract is over the
    order and the exact strings, not over a rendered caption. There is exactly
    one source: :func:`utils.protocol.train_caption.training_class_names`, which is the
    ``A4_full_v6`` condition's term tuple — the same object
    ``eval.py`` hands to :class:`~utils.runner.detector.
    Owlv2Detector`.
    """
    return training_class_names()


def assert_query_identity(processor, terms: Sequence[str] | None = None) -> dict:
    """Token-level identity between the training and the evaluation query list.

    The GDINO arm compares caption *bytes*; OWLv2 has no caption, so the
    equivalent is the tokenised query batch. Both paths call the processor the
    same way — training through :class:`Owlv2Collator`, evaluation through
    ``models/owlv2/detector.py::Owlv2Detector._forward`` — so this drives the real
    call and compares tensors.

    Raises:
        SpecViolationError: the two paths disagree in order, content, tokens or
            attention mask.
    """
    import torch

    train_terms = list(terms) if terms is not None else training_query_list()
    eval_terms = list(training_class_names())
    if train_terms != eval_terms:
        raise SpecViolationError(
            f"training query list {train_terms} != evaluation query list "
            f"{eval_terms}. OWLv2 scores channel q against term q, so a "
            f"reordering trains one class's boxes against another class's word "
            f"and nothing errors")

    train_encoding = processor(text=[train_terms], return_tensors="pt")
    eval_encoding = processor(text=[eval_terms], return_tensors="pt")
    for key in ("input_ids", "attention_mask"):
        if key not in train_encoding or key not in eval_encoding:
            raise SpecViolationError(f"processor returned no {key!r}")
        if not torch.equal(train_encoding[key], eval_encoding[key]):
            raise SpecViolationError(
                f"tokenised query {key} differs between the training and the "
                f"evaluation path")
    return {
        "terms": train_terms,
        "n_terms": len(train_terms),
        "input_ids_shape": list(train_encoding["input_ids"].shape),
        "input_ids_sha_prefix": train_encoding["input_ids"].flatten()[:16].tolist(),
    }


# --- data --------------------------------------------------------------------

class Owlv2V6Dataset:
    """V6 images with boxes normalised to the **padded square**.

    ``Owlv2ImageProcessor`` pads to ``max(H, W)`` anchored at ``(0, 0)`` and
    then resizes to 1008; the model's boxes live in that square's normalised
    coordinates. Dividing ``x`` by ``W`` and ``y`` by ``H`` — the usual DETR
    convention — is therefore wrong here for every non-square image, and wrong
    in a way that still produces a falling loss.

    Horizontal flip is the only augmentation, for the reason the OmDet arm
    gives: the processor's output size is fixed by the model (§5), so a scale
    jitter would mean re-implementing the processor.
    """

    def __init__(self, records: Sequence[V6Record], train: bool) -> None:
        self._records = list(records)
        self._train = train

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict:
        import torch
        from PIL import Image

        record = self._records[index]
        with Image.open(record.image_path) as raw:
            image = raw.convert("RGB")
        boxes = np.asarray(record.boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if self._train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                x1 = record.width - boxes[:, 2]
                x2 = record.width - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x1, x2
        square = float(max(record.width, record.height))
        if len(boxes):
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / square
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / square
            bw = (boxes[:, 2] - boxes[:, 0]) / square
            bh = (boxes[:, 3] - boxes[:, 1]) / square
            norm = np.stack([cx, cy, bw, bh], axis=1).clip(0.0, 1.0)
        else:
            norm = np.zeros((0, 4), dtype=np.float32)
        return {
            "image": image,
            "boxes": torch.from_numpy(norm.astype(np.float32)),
            "labels": torch.as_tensor(record.labels, dtype=torch.long),
            "image_id": record.image_id,
            "orig_size": (record.height, record.width),
        }


class Owlv2Collator:
    """Batch into processor inputs plus set-prediction targets.

    The query list is repeated once per image, which is how the processor
    expects a batch (``text=[[...], [...]]``) and what makes ``logits`` come
    back as *(batch, patches, terms)* with column ``q`` belonging to term ``q``.
    """

    def __init__(self, processor, terms: Sequence[str]) -> None:
        self._processor = processor
        self.terms = list(terms)

    def __call__(self, batch: list[dict]) -> tuple[dict, list[dict], list[tuple]]:
        images = [item["image"] for item in batch]
        encoding = self._processor(images=images, text=[self.terms] * len(images),
                                   return_tensors="pt")
        targets = [{"boxes": item["boxes"], "labels": item["labels"]} for item in batch]
        sizes = [item["orig_size"] for item in batch]
        return dict(encoding), targets, sizes


# --- model build + §2 gate ---------------------------------------------------

@dataclass
class BuiltModel:
    """A gated, LoRA-injected model plus the evidence that it is compliant."""

    model: Any
    report: Any
    scaling_report: Any
    placement: Any
    head_summary: Mapping[str, Any]


def load_base_model(cache_dir: str = HF_CACHE):
    """``(processor, model)`` from the offline hub cache."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(CHECKPOINT_ID, cache_dir=cache_dir)
    model = Owlv2ForObjectDetection.from_pretrained(CHECKPOINT_ID, cache_dir=cache_dir)
    return processor, model


def inject_lora(model) -> Any:
    """Inject §1 LoRA at the §2 targets, **in place**. Returns the placement.

    ``inject_adapter_in_model``, never ``get_peft_model``: the wrapper form
    inserts a ``LoraModel`` root node and clones every ``modules_to_save``
    module, which inflates the parameter denominator (measured on OmDet:
    116.2M -> 118.1M) and therefore corrupts the trainable-% figure that goes
    into the spec §2 table.
    """
    from peft import LoraConfig, inject_adapter_in_model

    targets = select_targets([n for n, _ in model.named_modules()], OWLV2_PLACEMENT)
    if len(targets) != EXPECTED_LORA_MODULES:
        raise SpecViolationError(
            f"§2 regex selected {len(targets)} modules but the count measured on "
            f"{CHECKPOINT_ID} is {EXPECTED_LORA_MODULES} (24 ViT blocks x 6 "
            f"projections). Refusing to train a different model than the spec "
            f"describes")
    inject_adapter_in_model(
        LoraConfig(r=LORA.r, lora_alpha=LORA.lora_alpha,
                   lora_dropout=LORA.lora_dropout, bias=LORA.bias,
                   target_modules=targets),
        model)
    return OWLV2_PLACEMENT


def describe_trainable_heads(model) -> dict:
    """Measure which detection heads actually carry trainable parameters.

    §2 says "box head 전체 학습 / class(대조) head 전체 학습". On Grounding DINO
    the second clause turned out to be **vacuously true** — ``ContrastiveEmbed``
    has no parameters — and on GLIP the class score is a dot product. So this is
    measured rather than assumed, and the numbers go into the §2 table.
    """
    summary: dict[str, Any] = {}
    for head in list(OWLV2_PLACEMENT.modules_to_save) + list(FROZEN_HEADS):
        module = getattr(model, head, None)
        if module is None or not hasattr(module, "named_parameters"):
            summary[head] = {"present": False}
            continue
        params = list(module.named_parameters())
        summary[head] = {
            "present": True,
            "n_tensors": len(params),
            "n_params": sum(p.numel() for _, p in params),
            "n_trainable_params": sum(p.numel() for _, p in params if p.requires_grad),
            "spec_role": ("fully_trained" if head in OWLV2_PLACEMENT.modules_to_save
                          else "frozen (not named by spec §2)"),
        }
    return summary


def count_lora_modules(model) -> int:
    """Live LoRA layer count, from ``named_modules()``."""
    return sum(1 for _, module in model.named_modules() if is_lora_module(module))


def build_lora_model(device: str, cache_dir: str = HF_CACHE,
                     gradient_checkpointing: bool = True) -> tuple[Any, BuiltModel]:
    """Build OWLv2-large, inject §2 LoRA, run the §2 gate. Never trains on failure."""
    from models.shared.train import mark_trainable

    processor, model = load_base_model(cache_dir)
    placement = inject_lora(model)
    n_lora, n_head = mark_trainable(model, placement)
    logger.info("trainable tensors: %d LoRA, %d detection-head (§2 heads: %s)",
                n_lora, n_head, list(placement.modules_to_save))

    if gradient_checkpointing:
        # `use_reentrant=False` is not a style choice. Every ViT block sits
        # behind a frozen patch embedding, so under the reentrant implementation
        # none of the checkpointed inputs requires grad, the block runs under
        # no_grad, and all 144 adapters receive exactly zero gradient behind a
        # perfectly ordinary loss curve. The first-backward gate in the trainer
        # is the second line of defence for the same failure.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("gradient checkpointing ENABLED with use_reentrant=False (§4.1)")

    model.to(device)

    report = verify_injection(model, placement)
    logger.info("%s", format_report(report))
    assert_compliant(report)
    if report.n_lora_modules != EXPECTED_LORA_MODULES:
        raise SpecViolationError(
            f"{report.n_lora_modules} live LoRA modules, expected "
            f"{EXPECTED_LORA_MODULES}")
    text_tower = [n for n in _lora_module_names(model) if ".text_model." in f".{n}."]
    if text_tower:
        raise SpecViolationError(
            f"{len(text_tower)} LoRA module(s) under the CLIP text tower, e.g. "
            f"{text_tower[:3]}. §2 freezes it, and freezing it is what makes the "
            f"permutation-invariance regression check meaningful")
    logger.info("§2 gate: %d LoRA modules, ALL in the ViT image tower, "
                "0 under owlv2.text_model (frozen CLIP text encoder)",
                report.n_lora_modules)

    scaling_report = verify_lora_scaling(model, expected_r=LORA.r,
                                         expected_alpha=LORA.lora_alpha)
    assert_scaling_compliant(scaling_report)
    logger.info("LoRA scaling verified on %d live modules: %s (alpha/r = %.4f)",
                scaling_report.n_modules_checked,
                list(scaling_report.observed_scalings), scaling_report.expected_scaling)

    head_summary = describe_trainable_heads(model)
    for name, info in head_summary.items():
        logger.info("head %-16s %s", name, info)

    built = BuiltModel(model=model, report=report, scaling_report=scaling_report,
                       placement=placement, head_summary=head_summary)
    return processor, built


def _lora_module_names(model) -> list[str]:
    return [n for n, m in model.named_modules() if is_lora_module(m)]


# --- first-backward gates ----------------------------------------------------

def first_backward_gradient_gate(model) -> None:
    """Assert the **vision tower's** ``lora_B`` tensors received gradient.

    ``lora_B`` is zero-initialised, so ``d loss / d lora_A`` is exactly zero at
    step 1 by construction while ``d loss / d lora_B`` is not. Checking
    ``lora_B`` therefore separates "LoRA has not moved yet" (expected) from
    "this block returned no gradient at all".

    OWLv2's checkpointing is checked explicitly rather than assumed to behave
    like Grounding DINO's. Every ViT block sits behind a **frozen** patch
    embedding, so nothing entering a checkpointed block requires grad; under
    torch's *reentrant* implementation — which ``gradient_checkpointing_enable``
    selects unless told otherwise — the block then runs under ``no_grad`` and
    all 144 adapters would train on exact zeros behind a perfectly ordinary
    loss curve. That is the failure this gate caught on Grounding DINO.
    """
    import torch

    total = alive = 0
    dead: list[str] = []
    for name, param in model.named_parameters():
        if "lora_B" not in name or not param.requires_grad:
            continue
        if "vision_model" not in name:
            raise SpecViolationError(
                f"LoRA parameter {name!r} is outside the ViT image tower; §2 puts "
                f"OWLv2's adapters there and nowhere else")
        total += 1
        if param.grad is not None and torch.count_nonzero(param.grad) > 0:
            alive += 1
        else:
            dead.append(name)
    if total == 0:
        raise SpecViolationError("no vision-tower lora_B parameter found to check")
    if dead:
        raise SpecViolationError(
            f"{len(dead)} of {total} vision-tower `lora_B` tensors received no "
            f"gradient after the first backward, e.g. {dead[:5]}. Every ViT block "
            f"sits behind a frozen patch embedding, so reentrant gradient "
            f"checkpointing returns no gradient at all and those adapters would "
            f"train on zeros behind a plausible loss curve")
    logger.info("gradient gate: %d/%d vision-tower lora_B tensors received "
                "gradient through the checkpointed blocks", alive, total)


def head_gradient_gate(model) -> None:
    """Assert the §2 fully-trained heads received gradient too.

    §2 names two heads. On Grounding DINO the class head turned out to be
    parameter-free, so "class head fully trained" was vacuously true; on OWLv2
    both heads carry parameters, which makes the claim checkable — so it is
    checked instead of restated.
    """
    import torch

    per_head: dict[str, list[int]] = {}
    for name, param in model.named_parameters():
        head = name.split(".")[0]
        if head not in OWLV2_PLACEMENT.modules_to_save or not param.requires_grad:
            continue
        counts = per_head.setdefault(head, [0, 0])
        counts[0] += 1
        if param.grad is not None and torch.count_nonzero(param.grad) > 0:
            counts[1] += 1
    missing = [h for h in OWLV2_PLACEMENT.modules_to_save if h not in per_head]
    if missing:
        raise SpecViolationError(f"§2 head(s) {missing} have no trainable tensor")
    dead = [head for head, (_, alive) in per_head.items() if alive == 0]
    if dead:
        raise SpecViolationError(
            f"§2 head(s) {dead} received zero gradient after the first backward; "
            f"'fully trained' would be false while the run still completed")
    logger.info("head gradient gate: %s",
                {head: f"{alive}/{total} tensors with gradient"
                 for head, (total, alive) in per_head.items()})


# --- health probe ------------------------------------------------------------

def text_query_embeddings(model, processor, terms: Sequence[str], device: str):
    """The frozen query embeddings, in fp32 and without autocast.

    Deterministic by construction: same weights, same tokens, no dropout in
    ``eval``. So the *correct* drift against a reference taken before step 1 is
    exactly 0.0, and any non-zero value means the text tower moved.
    """
    import torch

    encoding = processor(text=[list(terms)], return_tensors="pt").to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model.owlv2.get_text_features(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"])
    finally:
        model.train(was_training)
    # transformers >= 5 returns the whole `BaseModelOutputWithPooling` and
    # overwrites `pooler_output` with the projected embedding; older versions
    # return the tensor directly. Both are accepted rather than pinned, but the
    # result is asserted to be a tensor so a future change cannot silently make
    # the drift check compare two objects that are always equal.
    embeds = getattr(outputs, "pooler_output", outputs)
    if not isinstance(embeds, torch.Tensor):
        raise TypeError(
            f"get_text_features returned {type(outputs).__name__} with no tensor "
            f"`pooler_output`; the frozen-text-tower check cannot be measured")
    return embeds.float().detach().clone()


def _scaled_boxes(pred_boxes, sizes, device):
    import torch
    from transformers.image_transforms import center_to_corners_format
    from transformers.models.owlv2.image_processing_owlv2 import _scale_boxes

    target_sizes = torch.tensor([[h, w] for h, w in sizes], device=device)
    return _scale_boxes(center_to_corners_format(pred_boxes), target_sizes)


@dataclass(frozen=True)
class ProbeConfig:
    """How the health probe is measured. Fixed across seeds and steps."""

    batch_size: int = 4
    permutation_images: int = 8
    max_detections: int = EVAL_PROTOCOL.max_detections_per_image
    use_bf16: bool = True


def measure_health(model, processor, records: Sequence[V6Record], device: str,
                   terms: Sequence[str], reference_text_embeds,
                   config: ProbeConfig, step: int = 0,
                   loss_class_ema: float | None = None) -> Owlv2ProbeMetrics:
    """Measure the four pre-registered OWLv2 signals on a fixed probe subset.

    Args:
        reference_text_embeds: The embeddings taken before step 1. ``None``
            makes the drift 0 by definition and is only valid when *taking* the
            reference.

    Returns:
        :class:`~models.owlv2.guard.Owlv2ProbeMetrics`.
    """
    import torch
    from PIL import Image

    was_training = model.training
    model.eval()
    entropies: list[float] = []
    stds: list[float] = []
    distinct_rates: list[float] = []
    permutation_deltas: list[float] = []
    n_boxes = 0
    terms = list(terms)
    reversed_terms = list(reversed(terms))
    # Undoing the reversal is an index permutation; written explicitly so the
    # comparison cannot silently become "compare the matrix with itself".
    undo = [reversed_terms.index(term) for term in terms]

    try:
        with torch.no_grad():
            for start in range(0, len(records), config.batch_size):
                chunk = records[start:start + config.batch_size]
                images = []
                sizes = []
                for record in chunk:
                    with Image.open(record.image_path) as raw:
                        images.append(raw.convert("RGB"))
                    sizes.append((record.height, record.width))
                inputs = processor(images=images, text=[terms] * len(images),
                                   return_tensors="pt").to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=config.use_bf16 and device.startswith("cuda")):
                    outputs = model(**inputs)
                scores = torch.sigmoid(outputs.logits.float())
                boxes = _scaled_boxes(outputs.pred_boxes.float(), sizes, device)
                for index in range(scores.shape[0]):
                    entropies.append(normalised_channel_entropy(scores[index]))
                    stds.append(float(scores[index].std()))
                    per_image = scores[index]
                    keep = min(config.max_detections, per_image.numel())
                    _, flat = torch.topk(per_image.reshape(-1), keep)
                    patch_index = torch.div(flat, per_image.shape[-1], rounding_mode="floor")
                    distinct_rates.append(distinct_box_rate(boxes[index][patch_index]))
                    n_boxes += keep

                if start < config.permutation_images:
                    permuted = processor(images=images,
                                         text=[reversed_terms] * len(images),
                                         return_tensors="pt").to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=config.use_bf16 and device.startswith("cuda")):
                        permuted_outputs = model(**permuted)
                    restored = torch.sigmoid(permuted_outputs.logits.float())[..., undo]
                    permutation_deltas.append(max_abs_drift(restored, scores))
    finally:
        model.train(was_training)

    if not entropies:
        raise RuntimeError("the health probe saw no images; refusing to report a "
                           "health metric from an empty measurement")

    current_text = text_query_embeddings(model, processor, terms, device)
    drift = (0.0 if reference_text_embeds is None
             else max_abs_drift(current_text, reference_text_embeds))

    return Owlv2ProbeMetrics(
        step=step,
        n_images=len(entropies),
        n_boxes=n_boxes,
        channel_entropy=float(np.mean(entropies)),
        score_std=float(np.mean(stds)),
        distinct_box_rate=float(np.mean(distinct_rates)),
        text_embedding_drift=float(drift),
        permutation_drift=float(max(permutation_deltas) if permutation_deltas else 0.0),
        loss_class_ema=loss_class_ema,
    )


def baseline_record(metrics: Owlv2ProbeMetrics,
                    thresholds: Owlv2Thresholds) -> dict:
    """Baseline + threshold + headroom, in the shape the risk register wants."""
    rows = {
        "channel_entropy": (metrics.channel_entropy, thresholds.channel_entropy_alarm,
                            thresholds.channel_entropy_abort, "up"),
        "score_std": (metrics.score_std, thresholds.score_std_alarm,
                      thresholds.score_std_abort, "down"),
        "distinct_box_rate": (metrics.distinct_box_rate,
                              thresholds.distinct_box_rate_alarm,
                              thresholds.distinct_box_rate_abort, "down"),
        "permutation_drift": (metrics.permutation_drift,
                              thresholds.permutation_drift_alarm,
                              thresholds.permutation_drift_abort, "up"),
        "text_embedding_drift": (metrics.text_embedding_drift,
                                 thresholds.text_embedding_drift_alarm,
                                 thresholds.text_embedding_drift_abort, "up"),
    }
    out: dict[str, dict] = {}
    for name, (value, alarm, abort, direction) in rows.items():
        if direction == "up":
            headroom = (alarm / value) if value else None
        else:
            headroom = (value / alarm) if alarm else None
        out[name] = {"baseline": value, "alarm": alarm, "abort": abort,
                     "collapse_direction": direction, "headroom_factor": headroom}
    return out
