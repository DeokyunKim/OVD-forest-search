"""Per-model settings, read from `configs/models.yaml` rather than from code.

What a model is measured with -- its default variant, how many candidates its
detector emits, how a batch of 16 is split across accumulation steps, where its
upstream config lives -- describes the experiment, so it belongs in a file the
reader opens rather than in a dict they would have to go looking for.

Unknown keys are refused. A setting that is silently ignored is worse than one
that is missing: the file says the run used it and the run did not.

    from utils.runner.model_registry import model_settings

    settings = model_settings("owlv2")
    settings.max_output_detections   # 500
    settings.per_device_batch        # 4
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / 'pyproject.toml').is_file())
DEFAULT_PATH = REPO_ROOT / "configs" / "models.yaml"

MODELS: tuple[str, ...] = ("gdino", "owlv2", "glip", "omdet")

#: Every key a model block may carry. Adding a setting means adding it here,
#: which is the point: the list is what makes a typo an error.
KNOWN_KEYS = frozenset({
    "variant", "per_device_batch", "max_output_detections",
    "mmdet_root", "mmdet_config", "checkpoint",
})

_TOKEN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class ModelSettings:
    """One model's block, validated."""

    model: str
    variant: str
    per_device_batch: int
    #: Candidates the detector emits BEFORE the protocol's per-category cap.
    #: None where the architecture has no such knob -- absent, not zero.
    max_output_detections: int | None = None
    mmdet_root: str | None = None
    mmdet_config: str | None = None
    checkpoint: str | None = None

    @property
    def mmdet_config_path(self) -> str:
        """Absolute path to the upstream config, or a refusal naming the cause."""
        if self.mmdet_root is None or self.mmdet_config is None:
            raise ValueError(f"{self.model} has no mmdet config in configs/models.yaml")
        return f"{self.mmdet_root.rstrip('/')}/{self.mmdet_config}"


def _interpolate(value: Any) -> Any:
    """Substitute `${VAR}` from the environment, raising when it is unset.

    No empty-string fallback: `f"{root}/configs/..."` with an unset root
    composes into an absolute path at the filesystem root that reads like a
    path and resolves to nothing, and the failure then arrives as a missing
    file rather than as a missing setting.
    """
    if isinstance(value, str):
        def sub(match: re.Match) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if not resolved:
                raise SystemExit(
                    f"{name} is not set, and configs/models.yaml needs it for "
                    f"{value!r}. Set it to the checkout it names "
                    f"(see requirements/).")
            return resolved
        return _TOKEN.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


@lru_cache(maxsize=None)
def _blocks(path: str) -> Mapping[str, Mapping[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    blocks = raw.get("models")
    if not isinstance(blocks, dict):
        raise SystemExit(f"{path}: no `models:` mapping")
    missing = sorted(set(MODELS) - set(blocks))
    if missing:
        raise SystemExit(f"{path}: no block for {missing}; every model the "
                         f"benchmark reports needs one")
    for model, block in blocks.items():
        unknown = sorted(set(block) - KNOWN_KEYS)
        if unknown:
            raise SystemExit(
                f"{path}: model {model!r} has unknown key(s) {unknown}. "
                f"Known keys are {sorted(KNOWN_KEYS)}. A key that is ignored "
                f"reads as a setting that was applied.")
    return blocks


def model_settings(model: str, path: str | Path = DEFAULT_PATH) -> ModelSettings:
    """Settings for one model. Interpolation happens here, so an unset variable
    fails when the model is actually used rather than at import."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {list(MODELS)}")
    block = dict(_blocks(str(path))[model])
    return ModelSettings(model=model, **{k: _interpolate(v) for k, v in block.items()})


def default_variants(path: str | Path = DEFAULT_PATH) -> dict[str, str]:
    """`model -> default variant`, without interpolating anything else.

    Used where only the name is wanted; reading the full block would raise on a
    model whose upstream checkout is not installed on this machine.
    """
    return {model: block["variant"] for model, block in _blocks(str(path)).items()}


def per_device_batches(path: str | Path = DEFAULT_PATH) -> dict[str, int]:
    """`model -> per-device batch`. The accumulation factor is NOT here: it is
    derived from this and the effective batch, and two sources for one number
    is how they come to disagree."""
    return {model: int(block["per_device_batch"])
            for model, block in _blocks(str(path)).items()}
