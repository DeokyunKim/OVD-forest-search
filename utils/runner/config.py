"""Typed configuration loading for the analysis pipeline.

Reads ``configs/default.yaml`` (or any compatible YAML file), resolves the
``${external_repo}`` interpolation token, and validates that referenced paths
exist so failures happen at config-load time rather than deep inside model
building or dataset iteration.
"""
from __future__ import annotations

import re
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os

import yaml

from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)

_INTERP_PATTERN = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass(frozen=True)
class VariantSpec:
    """A single GroundingDINO variant to load for analysis."""

    config: str
    checkpoint: str | None
    kind: str  # one of: base, full_state, zira_adapter, lora_*, ...
    # kind=lora_hinge only: optional post-hoc per-injection-group multiplier
    # on the LoRA `scaling` (selective alpha dial, e.g. alpha_text) applied
    # by scripts.train_gdino_hinge.apply_lora_group_scales before merge.
    # Keys must be injection-group names; 1.0 = untouched, 0.0 = revert the
    # group to the frozen base weights.
    lora_group_scales: dict[str, float] | None = None
    # kind=lora_hinge only: which scripts.train_gdino_hinge INJECTION_PRESETS
    # scope the checkpoint was trained with (None == "both_towers", the
    # original hinge arms). Must match training or the strict load fails.
    lora_injection_preset: str | None = None
    # Set true for a variant whose checkpoint is still training: skips the
    # startup existence check ONLY (load_variant still fails loudly if the
    # file is missing when the variant is actually used), so a config
    # carrying a not-yet-finished arm does not block every other run.
    pending_checkpoint: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    """A single dataset source description."""

    kind: str  # one of: coco, forest_belongings
    images: str | None = None
    ann: str | None = None
    root: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """Top-level run configuration shared by all extraction/comparison scripts."""

    external_repo: str
    base_checkpoint: str
    device: str
    caption: str
    variants: dict[str, VariantSpec] = field(default_factory=dict)
    datasets: dict[str, DatasetSpec] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)


def _interpolate(value: Any, context: dict[str, str]) -> Any:
    """Recursively substitute ``${key}`` tokens using ``context`` values."""
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            key = match.group(1)
            if key not in context:
                raise ValueError(f"Unresolved interpolation token '${{{key}}}' in value: {value!r}")
            return context[key]

        return _INTERP_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, context) for v in value]
    return value


def _require_keys(raw: dict[str, Any], keys: list[str], scope: str) -> None:
    missing = [k for k in keys if k not in raw]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} in '{scope}' section of config")


def _build_variant(name: str, raw: dict[str, Any]) -> VariantSpec:
    _require_keys(raw, ["config", "checkpoint", "kind"], f"variants.{name}")
    return VariantSpec(
        config=raw["config"],
        checkpoint=raw["checkpoint"],
        kind=raw["kind"],
        lora_group_scales=raw.get("lora_group_scales"),
        lora_injection_preset=raw.get("lora_injection_preset"),
        pending_checkpoint=bool(raw.get("pending_checkpoint", False)),
    )


def _build_dataset(name: str, raw: dict[str, Any]) -> DatasetSpec:
    _require_keys(raw, ["kind"], f"datasets.{name}")
    return DatasetSpec(
        kind=raw["kind"],
        images=raw.get("images"),
        ann=raw.get("ann"),
        root=raw.get("root"),
    )


def _validate_paths(config: RunConfig) -> None:
    """Fail fast if referenced filesystem paths do not exist."""
    if not Path(config.external_repo).is_dir():
        raise FileNotFoundError(f"external_repo directory not found: {config.external_repo}")

    for name, variant in config.variants.items():
        if not Path(variant.config).is_file():
            raise FileNotFoundError(f"variants.{name}.config not found: {variant.config}")
        if (variant.checkpoint is not None and not variant.pending_checkpoint
                and not Path(variant.checkpoint).is_file()):
            raise FileNotFoundError(f"variants.{name}.checkpoint not found: {variant.checkpoint}")

    for name, dataset in config.datasets.items():
        if dataset.kind == "coco":
            if dataset.images is None or dataset.ann is None:
                raise ValueError(f"datasets.{name}: kind=coco requires 'images' and 'ann'")
            if not Path(dataset.images).is_dir():
                raise FileNotFoundError(f"datasets.{name}.images not found: {dataset.images}")
            if not Path(dataset.ann).is_file():
                raise FileNotFoundError(f"datasets.{name}.ann not found: {dataset.ann}")
        elif dataset.kind == "forest_belongings":
            if dataset.root is None:
                raise ValueError(f"datasets.{name}: kind=forest_belongings requires 'root'")
            if not Path(dataset.root).is_dir():
                raise FileNotFoundError(f"datasets.{name}.root not found: {dataset.root}")
        else:
            raise ValueError(f"datasets.{name}: unknown kind '{dataset.kind}'")


def _resolve_relative(value: str, base_dir: Path) -> str:
    """Resolve a possibly-relative filesystem path against ``base_dir``.

    Absolute paths pass through unchanged, so existing configs that already
    use absolute paths (e.g. ``/home/.../ZiRaGroundingDINO``) keep working
    from any CWD exactly as before. Relative paths (e.g. a packaged config's
    ``../ZiRaGroundingDINO``) are resolved against the YAML file's own
    directory rather than the process CWD, so ``load_config`` no longer
    depends on where it is invoked from.
    """
    candidate = Path(value)
    return value if candidate.is_absolute() else str((base_dir / candidate).resolve())


def load_config(path: str | Path) -> RunConfig:
    """Load, interpolate, and validate a ``RunConfig`` from a YAML file.

    Args:
        path: Path to the YAML config file (e.g. ``configs/default.yaml``).

    Returns:
        A validated, immutable ``RunConfig``.

    Raises:
        FileNotFoundError: If the file or a referenced path does not exist.
        ValueError: If required keys are missing or malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping")

    _require_keys(raw, ["external_repo", "base_checkpoint", "device", "caption"], "root")

    base_dir = path.resolve().parent
    raw["external_repo"] = _resolve_relative(
        _interpolate(raw["external_repo"],
                     {k: v for k, v in os.environ.items() if k.startswith("UC_")}),
        base_dir)
    # `${UC_GDINO_REPO}` and friends come from the environment; `${checkpoint_root}`
    # from configs/paths.yaml. Both are resolved here so a config file can be
    # written once and read on any machine.
    from utils.runner.paths import paths as _paths
    context = {"external_repo": raw["external_repo"],
               "checkpoint_root": str(_paths().checkpoint_root),
               **{k: v for k, v in os.environ.items() if k.startswith("UC_")}}
    # Only the keys this config object is built from. `configs/models.yaml`
    # also holds a `models:` block, one section per architecture, and GLIP's
    # names `${UC_MMDET_ROOT}`. Interpolating the whole document coupled the
    # four models together: evaluating Grounding DINO, which has no use for
    # mmdet, failed with `Unresolved interpolation token '${UC_MMDET_ROOT}'`
    # because a section belonging to another model happened to sit in the same
    # file. `utils/runner/model_registry.py` already resolves that block one
    # model at a time, on purpose and for this reason; doing it here as well
    # undid it.
    #
    # Derived from the dataclass rather than listed, so a field added to
    # `RunConfig` is interpolated without anyone remembering to say so here.
    consumed = {f.name for f in dataclasses.fields(RunConfig)}
    raw = {**raw, **{k: _interpolate(v, context)
                     for k, v in raw.items() if k in consumed}}
    raw["base_checkpoint"] = _resolve_relative(raw["base_checkpoint"], base_dir)

    variants_raw = raw.get("variants", {})
    datasets_raw = raw.get("datasets", {})
    for variant in variants_raw.values():
        variant["config"] = _resolve_relative(variant["config"], base_dir)
        if variant.get("checkpoint") is not None:
            variant["checkpoint"] = _resolve_relative(variant["checkpoint"], base_dir)

    config = RunConfig(
        external_repo=raw["external_repo"],
        base_checkpoint=raw["base_checkpoint"],
        device=raw["device"],
        caption=raw["caption"],
        variants={name: _build_variant(name, v) for name, v in variants_raw.items()},
        datasets={name: _build_dataset(name, v) for name, v in datasets_raw.items()},
        layers=list(raw.get("layers", [])),
    )

    _validate_paths(config)
    logger.info(
        "Loaded config from %s: %d variant(s), %d dataset(s), %d layer(s)",
        path, len(config.variants), len(config.datasets), len(config.layers),
    )
    return config
