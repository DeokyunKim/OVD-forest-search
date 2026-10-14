"""V6 checkpoint format — self-describing, and re-verified on load.

Why V6 owns this
----------------
``models/gdino/variants.py`` reaches into ``scripts.train_gdino_hinge`` for its
``lora_hinge`` variants. That script now lives in ``backup/scripts_20260808/``,
so every one of those variants raises ``ModuleNotFoundError`` on load. Reviving
the script would be the wrong repair: what was retired is the **recipe**
(alpha=8, 216 modules spanning frozen regions, task-loss hinge), and reading a
V6 checkpoint through that loader would resurrect a retired convention inside a
new experiment. V6 therefore owns its format end to end and never imports the
retired path.

The rule this format enforces
-----------------------------
From the infrastructure_incidents.md note: *a value reconstructed from config at
load time and never stored alongside the weights is indistinguishable from a
wrong one.* That is exactly how alpha was silently halved — ``scaling =
alpha / r`` is computed at build time from a config file that served fourteen
variants, and an alpha=8 checkpoint loaded through an alpha=16 config yields a
working model with every adapter doubled and no error anywhere.

So the manifest stores everything needed to know **what was trained**, and
:func:`verify_loaded_checkpoint` re-derives it from the live model and refuses
on disagreement:

* alpha, r and the resulting ``scaling`` — checked against ``module.scaling``
* per-group injection counts — not only the total, because 72 is also
  reachable as 36+36+0
* spec §2 compliance — re-run on the live model, because a mismatch between
  save time and load time silently evaluates a different model
* the training caption — so an FT arm cannot be scored against a caption it
  was not trained on
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

from utils.runner.logging_utils import get_logger
from models.shared.lora_spec_v6line import (
    LoraPlacement,
    SpecViolationError,
    assert_compliant,
    assert_scaling_compliant,
    verify_injection,
    verify_lora_scaling,
)

logger = get_logger(__name__)

MANIFEST_NAME = "v6_manifest.json"
ADAPTER_NAME = "v6_adapter.pt"
#: Bumped when the format changes incompatibly.
CHECKPOINT_FORMAT_VERSION = "v6-ckpt.1"


class CheckpointMismatchError(RuntimeError):
    """The live model disagrees with what the manifest says was trained."""


@dataclass(frozen=True)
class V6CheckpointManifest:
    """Everything needed to know what produced these weights."""

    format_version: str
    spec_version: str
    model: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    #: alpha / r. Stored because it is otherwise derived-not-stored.
    scaling: float
    n_lora_modules: int
    #: Per-group counts. A total alone cannot detect modules moving between
    #: groups while the sum is preserved.
    injection_counts: Mapping[str, int]
    modules_to_save: tuple[str, ...]
    step: int
    total_steps: int
    seed: int
    effective_batch: int
    training_caption: str
    trainable_params: int
    total_params: int
    trainable_pct: float
    notes: str = ""
    extra: Mapping = field(default_factory=dict)

    def as_record(self) -> dict:
        return asdict(self)


def build_manifest(
    *,
    model_name: str,
    report,
    scaling_report,
    step: int,
    total_steps: int,
    seed: int,
    effective_batch: int,
    training_caption: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    spec_version: str = "frozen-2026-08-08",
    notes: str = "",
) -> V6CheckpointManifest:
    """Assemble a manifest from the live model's own verification reports."""
    observed = (scaling_report.observed_scalings[0]
                if scaling_report.observed_scalings else lora_alpha / lora_r)
    return V6CheckpointManifest(
        format_version=CHECKPOINT_FORMAT_VERSION,
        spec_version=spec_version,
        model=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        scaling=float(observed),
        n_lora_modules=report.n_lora_modules,
        injection_counts=dict(report.per_region_counts),
        modules_to_save=tuple(report.modules_to_save_found),
        step=step,
        total_steps=total_steps,
        seed=seed,
        effective_batch=effective_batch,
        training_caption=training_caption,
        trainable_params=report.trainable_params,
        total_params=report.total_params,
        trainable_pct=report.trainable_pct,
        notes=notes,
    )


def save_v6_checkpoint(model, manifest: V6CheckpointManifest, directory: Path) -> Path:
    """Write adapter weights plus the manifest. Weights alone are not enough."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    trainable = {name: param.detach().cpu()
                 for name, param in model.named_parameters() if param.requires_grad}
    if not trainable:
        raise CheckpointMismatchError(
            "no trainable parameters to save; refusing to write an empty adapter")
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION,
                "state_dict": trainable}, directory / ADAPTER_NAME)
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest.as_record(), indent=2))
    logger.info("saved V6 checkpoint step=%d -> %s (%d tensors)",
                manifest.step, directory, len(trainable))
    return directory


def read_manifest(directory: Path) -> V6CheckpointManifest:
    """Load the manifest. Absence is an error, not a default."""
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise CheckpointMismatchError(
            f"{path} is missing. A V6 checkpoint without its manifest cannot be "
            f"verified, and an unverifiable checkpoint must not be evaluated")
    payload = json.loads(path.read_text())
    payload.pop("format_version", None)
    return V6CheckpointManifest(format_version=CHECKPOINT_FORMAT_VERSION, **payload)


def verify_loaded_checkpoint(model, manifest: V6CheckpointManifest,
                             placement: LoraPlacement,
                             expected_caption: str | None = None) -> None:
    """Re-verify §2 and the alpha fingerprint on the **live** loaded model.

    Raises:
        SpecViolationError: the loaded model violates spec §2.
        CheckpointMismatchError: the live model disagrees with the manifest.
    """
    report = verify_injection(model, placement)
    assert_compliant(report)                     # §2, re-checked after loading

    scaling_report = verify_lora_scaling(
        model, expected_r=manifest.lora_r, expected_alpha=manifest.lora_alpha)
    assert_scaling_compliant(scaling_report)

    # alpha fingerprint: the manifest's scaling must equal the live one.
    observed = scaling_report.observed_scalings
    if any(abs(value - manifest.scaling) > 1e-6 for value in observed):
        raise CheckpointMismatchError(
            f"alpha fingerprint mismatch: manifest says scaling={manifest.scaling} "
            f"(alpha={manifest.lora_alpha}, r={manifest.lora_r}) but the live "
            f"modules read {list(observed)}. This is the failure mode that "
            f"silently halved alpha once already")

    if report.n_lora_modules != manifest.n_lora_modules:
        raise CheckpointMismatchError(
            f"module count mismatch: manifest {manifest.n_lora_modules}, live "
            f"{report.n_lora_modules}")

    for group, count in manifest.injection_counts.items():
        live = report.per_region_counts.get(group, 0)
        if live != count:
            raise CheckpointMismatchError(
                f"per-group injection mismatch in {group!r}: manifest {count}, "
                f"live {live}. A matching total can still hide modules moving "
                f"between groups")

    if expected_caption is not None and expected_caption != manifest.training_caption:
        raise CheckpointMismatchError(
            f"caption mismatch: this checkpoint was trained with "
            f"{manifest.training_caption!r} but evaluation would send "
            f"{expected_caption!r}")

    logger.info("checkpoint verified: %d LoRA modules, scaling %.4f, step %d/%d, "
                "seed %d, caption %r", report.n_lora_modules, manifest.scaling,
                manifest.step, manifest.total_steps, manifest.seed,
                manifest.training_caption)


def load_adapter_weights(model, directory: Path, strict: bool = True) -> int:
    """Load adapter weights into an already-built model. Returns tensors loaded."""
    import torch

    path = Path(directory) / ADAPTER_NAME
    if not path.is_file():
        raise CheckpointMismatchError(f"{path} is missing")
    payload = torch.load(path, map_location="cpu")
    state = payload["state_dict"]
    own = dict(model.named_parameters())
    missing = [name for name in state if name not in own]
    if missing and strict:
        raise CheckpointMismatchError(
            f"{len(missing)} saved tensor(s) have no counterpart in the model, "
            f"e.g. {missing[:5]}. The architecture does not match the checkpoint")
    with torch.no_grad():
        for name, tensor in state.items():
            if name in own:
                own[name].copy_(tensor.to(own[name].device))
    return len(state)
