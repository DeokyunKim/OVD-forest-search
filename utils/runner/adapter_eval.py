"""Load a V6 LoRA checkpoint into the **zero-shot** GDINO / OWLv2 detectors.

Spec: the frozen-hyperparameter specification §1 (LoRA), §2 (placement), §6
(evaluation). Companion of :mod:`models.omdet.detector`, which does the same
job for the cross-architecture arm; that module is not imported from here
because its injection is OmDet-specific, but its structure is followed
deliberately so the two arms behave the same way.

One implementation, two arms
----------------------------
Nothing below builds a detector. It takes the model a zero-shot detector has
*already* built and adds the adapter to it. The read-out, the protocol values
and the post-processing therefore cannot drift between the zero-shot and the
fine-tuned arm of a model, because there is only one copy of them (§6 requires
ZS and FT to be scored identically, and a second code path is the cheapest way
to break that silently).

Three properties this module enforces
-------------------------------------
1. **Never merged.** The adapter is applied at forward time as
   ``base(x) + scaling * B(A(x))`` and the base weights are left untouched.
   Merging a PEFT adapter on a non-``cuda:0`` device has a recorded history on
   this project of producing *run-to-run different* results for the same image
   and query (``CLAUDE.md``: 0 detections one run, 4 the next), invisible in
   logs. Not merging makes that failure structurally impossible, so
   :func:`assert_not_merged` and :func:`assert_adapter_applied_at_forward` are
   run on every load rather than trusted.
2. **The manifest is verified against the live model, not believed.**
   :func:`utils.train.checkpoint.verify_loaded_checkpoint` re-runs §2 compliance,
   compares per-group counts (a matching total can hide modules moving between
   groups) and reads the alpha/r fingerprint off ``module.scaling``. The
   placement it is given is built for the scope the *checkpoint* records, not a
   hardcoded one.
3. **§1 is re-asserted.** A manifest is a record, not an authority: a
   checkpoint claiming ``alpha=8`` would verify against itself perfectly. The
   recorded hyperparameters are therefore compared to the frozen §1 values as
   well. Alpha has already been silently halved once on this project.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Mapping

from utils.runner.logging_utils import get_logger
from utils.train.checkpoint import (
    CheckpointMismatchError,
    V6CheckpointManifest,
    load_adapter_weights,
    read_manifest,
    verify_loaded_checkpoint,
)
from models.shared.lora_spec_v6line import SpecViolationError, is_lora_module, verify_lora_scaling
from utils.protocol.train_caption import training_caption
from utils.train.training_schedule import LORA

logger = get_logger(__name__)

#: Substring every fine-tuned arm label must carry. The fragment file name is
#: built from that label, so without this a fine-tuned run under a zero-shot
#: label would overwrite the zero-shot fragment of the same condition — and the
#: overwritten file would look entirely well-formed.
FT_LABEL_TOKEN = "ft"

#: OWLv2 §2 placement, per projection: 24 ViT blocks x 6 projections = 144.
#: Measured on ``google/owlv2-large-patch14-ensemble`` (see
#: ``models/owlv2/train.py``), not derived from the config.
EXPECTED_OWLV2_GROUP_COUNTS: Mapping[str, int] = {
    "q_proj": 24, "k_proj": 24, "v_proj": 24,
    "out_proj": 24, "fc1": 24, "fc2": 24,
}

#: Relative tolerance of the "the adapter is applied at forward time" identity.
#: fp32 matmuls of different shapes do not associate bit-for-bit, so an exact
#: comparison would fail for a correct implementation.
_FORWARD_IDENTITY_RTOL = 1e-4


class FtLabelError(RuntimeError):
    """A fine-tuned run was asked to write under a label that hides that fact."""


def assert_ft_label(label: str, adapter_dir: str | None) -> None:
    """Refuse a fine-tuned run whose output label does not say ``ft``.

    Args:
        label: The label the fragment (and the detection dump) is named after.
        adapter_dir: ``None`` for a zero-shot run, which is never refused.

    Raises:
        FtLabelError: an adapter was given under a label without ``ft``.
    """
    if adapter_dir is None:
        return
    if FT_LABEL_TOKEN not in label.lower():
        raise FtLabelError(
            f"--v6-adapter {adapter_dir} was given but the output label is "
            f"{label!r}, which does not contain {FT_LABEL_TOKEN!r}. The fragment "
            f"and dump file names are built from that label, so this run would "
            f"land on top of the zero-shot result for the same condition and be "
            f"indistinguishable from it afterwards. Pass a label such as "
            f"'swinb-ft-seed1' (GDINO) or 'large-ft-seed1' (OWLv2)")


def assert_spec1_hyperparameters(manifest: V6CheckpointManifest) -> None:
    """Compare the checkpoint's recorded §1 values to the frozen ones.

    ``verify_loaded_checkpoint`` checks the live model *against the manifest*,
    which a checkpoint trained with the wrong alpha would pass: it is
    self-consistent. This is the second half — the manifest against the spec.

    Raises:
        SpecViolationError: r, alpha or dropout differ from §1.
    """
    expected = {"lora_r": LORA.r, "lora_alpha": LORA.lora_alpha,
                "lora_dropout": LORA.lora_dropout}
    observed = {"lora_r": manifest.lora_r, "lora_alpha": manifest.lora_alpha,
                "lora_dropout": manifest.lora_dropout}
    if observed != expected:
        raise SpecViolationError(
            f"checkpoint hyperparameters {observed} != spec §1 {expected}. The "
            f"retired preset used lora_alpha=8 (scaling 0.5) and this project "
            f"has already had alpha silently halved once; refusing to evaluate "
            f"a checkpoint the spec does not describe")
    if abs(manifest.scaling - LORA.lora_alpha / LORA.r) > 1e-6:
        raise SpecViolationError(
            f"manifest scaling {manifest.scaling} != alpha/r "
            f"{LORA.lora_alpha / LORA.r}")


def scope_from_manifest(manifest: V6CheckpointManifest) -> str:
    """The LoRA scope the checkpoint was trained with, from its own record.

    ``train.py --model gdino`` writes ``scope=<name>`` into the manifest's
    ``notes``. It is read back rather than assumed, because the placement built
    from it decides which modules §2 verification expects to find: verifying a
    ``spec2_attention`` checkpoint against a ``spec2`` placement would compare
    the live model to a rule it was never trained under.

    Raises:
        CheckpointMismatchError: the manifest records no scope.
    """
    match = re.search(r"\bscope=([A-Za-z0-9_]+)", manifest.notes or "")
    if match is None:
        raise CheckpointMismatchError(
            f"the manifest of this {manifest.model!r} checkpoint records no LoRA "
            f"scope (notes={manifest.notes!r}). The §2 placement to verify "
            f"against cannot be reconstructed, and guessing one would verify the "
            f"model against a rule it may never have been trained under")
    return match.group(1)


def per_group_from_manifest(manifest: V6CheckpointManifest) -> dict[str, int] | None:
    """Per-group module counts recorded at training time, or ``None``.

    Finer than ``manifest.injection_counts`` (which is per allowed *region*):
    the GDINO trainer records one count per §2 group, and 36 + 36 + 0 is also
    reachable as 24 + 24 + 24.
    """
    match = re.search(r"\bper_group=(\{[^}]*\})", manifest.notes or "")
    if match is None:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): int(value) for key, value in parsed.items()}


def assert_not_merged(model) -> int:
    """Refuse if any LoRA module has folded its delta into the base weights.

    Returns:
        Number of LoRA modules checked.

    Raises:
        SpecViolationError: a module reports itself merged, or none was found.
    """
    checked = 0
    merged: list[str] = []
    for name, module in model.named_modules():
        if not is_lora_module(module):
            continue
        checked += 1
        # loralib-style (the external Grounding DINO repo) exposes a bool;
        # PEFT exposes `merged` (property) plus `merged_adapters` (list).
        if bool(getattr(module, "merged", False)) or list(
                getattr(module, "merged_adapters", []) or []):
            merged.append(name)
    if checked == 0:
        raise SpecViolationError(
            "no LoRA module found after loading the adapter; the checkpoint "
            "would have no effect and the arm would silently be zero-shot")
    if merged:
        raise SpecViolationError(
            f"{len(merged)} LoRA module(s) have been MERGED into the base "
            f"weights, e.g. {merged[:5]}. Merging on this project has produced "
            f"run-to-run different results for the same image and query, with no "
            f"log evidence; the V6 evaluation path never merges")
    return checked


def assert_adapter_applied_at_forward(model) -> dict:
    """Prove on the live model that the delta is computed at forward time.

    Picks the first LoRA module, runs one random input through it and compares
    the output with ``base(x) + scaling * B(A(x))`` recomputed from the module's
    own tensors. Two things are asserted: the identity holds (so the adapter is
    applied explicitly, exactly as an unmerged adapter must be) and the delta is
    **non-zero** (so the loaded weights actually act — a checkpoint that failed
    to load would leave ``lora_B`` at its zero initialisation and change
    nothing, which is otherwise indistinguishable from a working zero-shot run).

    Returns:
        ``{"module", "max_abs_delta", "max_abs_error"}`` for the log.

    Raises:
        SpecViolationError: no LoRA module, a zero delta, or the identity fails.
    """
    import torch
    import torch.nn.functional as F

    target = next(((n, m) for n, m in model.named_modules() if is_lora_module(m)),
                  None)
    if target is None:
        raise SpecViolationError("no LoRA module to check the forward identity on")
    name, module = target

    with torch.no_grad():
        was_training = module.training
        module.eval()  # LoRA dropout must not perturb the identity
        try:
            if hasattr(module, "base_layer"):  # PEFT
                adapter = next(iter(module.lora_A))
                weight = module.base_layer.weight
                x = torch.randn(4, weight.shape[1], device=weight.device,
                                dtype=weight.dtype)
                base = module.base_layer(x)
                delta = module.lora_B[adapter](module.lora_A[adapter](x))
                scaling = float(module.scaling[adapter])
                reference = base + scaling * delta
            else:  # loralib-style: the external Grounding DINO repo's lora.Linear
                weight = module.weight
                x = torch.randn(4, weight.shape[1], device=weight.device,
                                dtype=weight.dtype)
                base = F.linear(x, weight, module.bias)
                delta = x @ module.lora_A.transpose(0, 1) @ module.lora_B.transpose(0, 1)
                scaling = float(module.scaling)
                reference = base + scaling * delta
            observed = module(x)
        finally:
            module.train(was_training)

        contribution = float((scaling * delta).abs().max())
        error = float((observed - reference).abs().max())
        tolerance = _FORWARD_IDENTITY_RTOL * max(float(base.abs().max()), 1.0)

    if contribution == 0.0:
        raise SpecViolationError(
            f"the LoRA delta of {name!r} is exactly zero, so the adapter changes "
            f"nothing. `lora_B` is zero-initialised, so this is what an adapter "
            f"that was never loaded looks like — and it would score as zero-shot "
            f"under a fine-tuned label")
    if error > tolerance:
        raise SpecViolationError(
            f"{name!r} does not compute base(x) + scaling * B(A(x)): max "
            f"deviation {error:.3e} > {tolerance:.3e}. The V6 evaluation path "
            f"requires the adapter to be applied at forward time and never "
            f"merged into the base weights")
    return {"module": name, "max_abs_delta": contribution, "max_abs_error": error}


def _provenance(manifest: V6CheckpointManifest, directory: Path,
                per_group: Mapping[str, int], n_loaded: int,
                observed_scalings: tuple[float, ...],
                forward_check: Mapping[str, Any]) -> dict:
    """The record a fine-tuned fragment carries, so it can never pass for ZS."""
    return {
        "adapter_dir": str(directory),
        "model": manifest.model,
        "seed": manifest.seed,
        "step": manifest.step,
        "total_steps": manifest.total_steps,
        "lora_r": manifest.lora_r,
        "lora_alpha": manifest.lora_alpha,
        "lora_dropout": manifest.lora_dropout,
        "scaling": manifest.scaling,
        "observed_scalings": list(observed_scalings),
        "n_lora_modules": manifest.n_lora_modules,
        "per_group_modules": dict(per_group),
        "injection_counts": dict(manifest.injection_counts),
        "trainable_params": manifest.trainable_params,
        "trainable_pct": manifest.trainable_pct,
        "training_caption": manifest.training_caption,
        "spec_version": manifest.spec_version,
        "tensors_loaded": n_loaded,
        "merged_into_base_weights": False,
        "forward_identity": dict(forward_check),
    }


def _log_load(manifest: V6CheckpointManifest, directory: Path,
              per_group: Mapping[str, int], n_loaded: int,
              observed_scalings: tuple[float, ...],
              forward_check: Mapping[str, Any]) -> None:
    """The acceptance evidence, in the log, at load time."""
    logger.info("V6 %s adapter loaded from %s", manifest.model, directory)
    logger.info("  LoRA modules (live)  : %d", manifest.n_lora_modules)
    logger.info("  per-group counts     : %s", dict(per_group))
    logger.info("  per-region counts    : %s", dict(manifest.injection_counts))
    logger.info("  scaling (live module): %s (alpha=%d / r=%d)",
                list(observed_scalings), manifest.lora_alpha, manifest.lora_r)
    logger.info("  training step        : %d/%d", manifest.step, manifest.total_steps)
    logger.info("  seed                 : %d", manifest.seed)
    logger.info("  training caption     : %r", manifest.training_caption)
    logger.info("  tensors loaded       : %d", n_loaded)
    logger.info("  merged into base     : False (delta applied at forward time; "
                "%s contributes max |delta| %.3e, identity error %.3e)",
                forward_check["module"], forward_check["max_abs_delta"],
                forward_check["max_abs_error"])


def ft_checkpoint_name(variant: str, provenance: Mapping[str, Any]) -> str:
    """Detector ``checkpoint_name`` for a fine-tuned arm.

    Built from the provenance record the load returned, so the seed and step in
    the name are the ones that were *verified against the live model* rather
    than a second reading of the file.

    A trailing ``zeroshot`` is stripped from the variant: the fine-tuned arm is
    built from the same base weights the zero-shot arm uses, so the variant name
    carries that word, and ``swinb_zeroshot-ft-seed1`` would read as a
    contradiction in every table it appears in.
    """
    base = variant
    for suffix in ("_zeroshot", "-zeroshot"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return f"{base}-ft-seed{provenance['seed']}-step{provenance['step']}"


# --- Grounding DINO ----------------------------------------------------------

def attach_gdino_adapter(model, adapter_dir: str,
                         expected_caption: str | None = None) -> dict:
    """Inject spec §2 LoRA into a built GDINO model, load a V6 checkpoint, verify.

    The injection repeats ``train.py --model gdino::build_lora_model``
    step for step — decompose the fused attentions, collect §2 targets for the
    checkpoint's recorded scope, replace each ``nn.Linear`` with the external
    repo's ``lora.Linear`` — because the parameter names in the checkpoint are
    the names that procedure produces. Anything else would either fail to load
    or, worse, load partially.

    ``lora.Linear.forward`` in that repo always computes
    ``F.linear(x, W, b) + scaling * B(A(x))`` (its merge branch is commented
    out upstream), so this path cannot merge even by accident; it is asserted
    anyway.

    Args:
        model: A model built exactly as the zero-shot arm builds it.
        adapter_dir: Directory holding ``v6_adapter.pt`` + ``v6_manifest.json``.
        expected_caption: Caption evaluation will send. Defaults to
            :func:`utils.protocol.train_caption.training_caption`; the load is refused
            if the checkpoint was trained on a different one.

    Returns:
        Provenance recorded verbatim into the fragment.
    """
    import torch

    # Carried in this repository rather than imported from the checkout: the
    # fork never committed its LoRA layer. See models/gdino/lora.
    from models.gdino.lora import Linear as LoRALinear, apply_custom_attention

    from models.gdino.lora import (
        build_placement,
        collect_targets,
        expected_group_counts,
        group_counts,
    )

    directory = Path(adapter_dir)
    manifest = read_manifest(directory)
    if manifest.model != "gdino":
        raise CheckpointMismatchError(
            f"{directory} holds a {manifest.model!r} checkpoint, not a GDINO one")
    assert_spec1_hyperparameters(manifest)
    scope = scope_from_manifest(manifest)

    device = next(model.parameters()).device
    _decompose_fused_attention(model, apply_custom_attention)

    targets = collect_targets(model, scope=scope)
    for target in targets:
        old = target.linear
        adapted = LoRALinear(
            in_features=old.in_features,
            out_features=old.out_features,
            r=manifest.lora_r,
            lora_alpha=manifest.lora_alpha,
            lora_dropout=manifest.lora_dropout,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            adapted.weight.copy_(old.weight)
            if old.bias is not None:
                adapted.bias.copy_(old.bias)
        setattr(target.parent, target.attr, adapted.to(old.weight.device))

    model.to(device)
    if hasattr(model, "device"):
        # Plain attribute, not updated by nn.Module.to(); preprocess_image reads
        # it (CLAUDE.md). The decomposition/injection above replaced modules, so
        # it is re-asserted here rather than assumed still correct.
        model.device = str(device)

    n_loaded = load_adapter_weights(model, directory, strict=True)
    placement = build_placement(scope)
    verify_loaded_checkpoint(
        model, manifest, placement,
        expected_caption=training_caption() if expected_caption is None
        else expected_caption)

    per_group = group_counts(collect_targets(model, scope=scope))
    _assert_per_group(per_group, expected_group_counts(scope), "the §2 table")
    recorded = per_group_from_manifest(manifest)
    if recorded is not None:
        _assert_per_group(per_group, recorded, "the manifest")

    assert_not_merged(model)
    model.eval()  # LoRA dropout is 0.05: a train-mode module would randomise eval
    forward_check = assert_adapter_applied_at_forward(model)
    scalings = verify_lora_scaling(model, expected_r=manifest.lora_r,
                                   expected_alpha=manifest.lora_alpha).observed_scalings

    _log_load(manifest, directory, per_group, n_loaded, scalings, forward_check)
    record = _provenance(manifest, directory, per_group, n_loaded, scalings,
                         forward_check)
    record["scope"] = scope
    return record


def _decompose_fused_attention(model, apply_custom_attention) -> int:
    """Replace fused ``nn.MultiheadAttention`` with its q/k/v/o decomposition.

    ``nn.MultiheadAttention`` holds one fused ``in_proj_weight`` and exposes
    **zero** ``nn.Linear`` children, so three of the nine §2 groups do not exist
    until this runs — 72 of 192 modules. The external repo's
    ``apply_custom_attention`` copies the chunked fused weights across, so the
    decomposition is weight-preserving; it is the same call the trainer makes,
    which is why the checkpoint's parameter names match afterwards.

    Idempotent: an already-decomposed layer is skipped, because the upstream
    helper reads ``in_proj_weight`` and would raise on a second pass.
    """
    n = 0
    groups = [(model.transformer.decoder.layers, ("self_attn", "ca_text")),
              (model.transformer.encoder.text_layers, ("self_attn",))]
    for layers, names in groups:
        for layer in layers:
            todo = [name for name in names
                    if not hasattr(getattr(layer, name), "linear_q")]
            if todo:
                apply_custom_attention(layer, todo)
                n += len(todo)
    logger.info("decomposed %d fused attention module(s) into linear_{q,k,v,o}", n)
    return n


def _assert_per_group(observed: Mapping[str, int], expected: Mapping[str, int],
                      source: str) -> None:
    if dict(observed) != dict(expected):
        raise CheckpointMismatchError(
            f"per-group LoRA counts on the live model {dict(observed)} != "
            f"{dict(expected)} from {source}. The same total is reachable from "
            f"different group combinations, so the per-group comparison is the "
            f"one that matters")


# --- OWLv2 -------------------------------------------------------------------

def attach_owlv2_adapter(model, adapter_dir: str,
                         expected_caption: str | None = None) -> dict:
    """Inject spec §2 LoRA into a built OWLv2 model, load a V6 checkpoint, verify.

    ``inject_adapter_in_model``, never ``get_peft_model``: the wrapper form adds
    a ``LoraModel`` root and clones every ``modules_to_save`` module, which
    inflates the parameter denominator and corrupts the trainable-% figure that
    goes into the §2 table. The training path
    (``models/owlv2/train.py::inject_lora``) uses the same call with the same
    target list, which is what makes the checkpoint's parameter names match.

    PEFT applies an unmerged adapter as ``base_layer(x) + scaling *
    B(A(x))``; ``merge_adapter``/``merge_and_unload`` are never called here.
    """
    from peft import LoraConfig, inject_adapter_in_model

    from models.shared.lora_spec_v6line import OWLV2_PLACEMENT, select_targets
    from models.owlv2.train import EXPECTED_LORA_MODULES

    directory = Path(adapter_dir)
    manifest = read_manifest(directory)
    if manifest.model != "owlv2":
        raise CheckpointMismatchError(
            f"{directory} holds a {manifest.model!r} checkpoint, not an OWLv2 one")
    assert_spec1_hyperparameters(manifest)

    targets = select_targets([n for n, _ in model.named_modules()], OWLV2_PLACEMENT)
    if len(targets) != EXPECTED_LORA_MODULES:
        raise SpecViolationError(
            f"§2 regex selected {len(targets)} modules, expected "
            f"{EXPECTED_LORA_MODULES} (24 ViT blocks x 6 projections) measured on "
            f"the real checkpoint")
    inject_adapter_in_model(
        LoraConfig(r=manifest.lora_r, lora_alpha=manifest.lora_alpha,
                   lora_dropout=manifest.lora_dropout, bias=LORA.bias,
                   target_modules=targets),
        model)

    n_loaded = load_adapter_weights(model, directory, strict=True)
    verify_loaded_checkpoint(
        model, manifest, OWLV2_PLACEMENT,
        expected_caption=training_caption() if expected_caption is None
        else expected_caption)

    per_group = owlv2_group_counts(model)
    _assert_per_group(per_group, EXPECTED_OWLV2_GROUP_COUNTS,
                      "the OWLv2 §2 table measured on the checkpoint")

    text_tower = [n for n, m in model.named_modules()
                  if is_lora_module(m) and ".text_model." in f".{n}."]
    if text_tower:
        raise SpecViolationError(
            f"{len(text_tower)} LoRA module(s) under the CLIP text tower, e.g. "
            f"{text_tower[:3]}. §2 freezes it, and a frozen text tower is what "
            f"makes the permutation-invariance regression meaningful")

    assert_not_merged(model)
    model.eval()
    forward_check = assert_adapter_applied_at_forward(model)
    scalings = verify_lora_scaling(model, expected_r=manifest.lora_r,
                                   expected_alpha=manifest.lora_alpha).observed_scalings

    _log_load(manifest, directory, per_group, n_loaded, scalings, forward_check)
    record = _provenance(manifest, directory, per_group, n_loaded, scalings,
                         forward_check)
    record["lora_modules_in_text_tower"] = 0
    return record


def owlv2_group_counts(model) -> dict[str, int]:
    """Live LoRA count per ViT projection (``q_proj`` … ``fc2``)."""
    counts: dict[str, int] = {name: 0 for name in EXPECTED_OWLV2_GROUP_COUNTS}
    for name, module in model.named_modules():
        if not is_lora_module(module):
            continue
        projection = name.rsplit(".", 1)[-1]
        counts[projection] = counts.get(projection, 0) + 1
    return counts
