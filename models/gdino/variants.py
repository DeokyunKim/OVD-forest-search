"""GroundingDINO variant loading.

Mirrors the load path used by ``groundingdino/util/inference.py::load_model``
and the checkpoint-restoration pattern in ``eval_coco_dithub.py`` (base
weights first, then adapter/full-state weights on top with
``strict=False``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from utils.runner.config import VariantSpec
from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)


def build_base_model(config_path: str, device: str) -> nn.Module:
    """Build an un-initialized (random-weight) GroundingDINO model from a config.

    Args:
        config_path: Path to a GroundingDINO SLConfig ``.py`` file.
        device: Target device string, e.g. ``cuda:0``.

    Returns:
        The constructed model, moved to ``device`` and set to eval mode.
    """
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig

    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    model = model.to(device)

    if hasattr(model, "device"):
        # Detectron2-style ("dt*"/ZiRa-family) GroundingDINO classes stash a
        # plain `self.device` attribute (default "cuda", see e.g.
        # groundingdino_dt.py:106) that they read directly inside
        # forward()/preprocess_image() (e.g. `x["image"].to(self.device)`,
        # groundingdino_dt.py:697). Unlike buffers/parameters, this plain
        # attribute is NOT updated by `nn.Module.to()` above, so it stays
        # "cuda" even when building on CPU -- verified: forward() otherwise
        # raises "RuntimeError: No CUDA GPUs are available" when
        # `device="cpu"`. The plain (non-"dt") GroundingDINO class has no
        # such attribute, so this is a no-op for it.
        model.device = device

    model.eval()
    return model


def _load_base_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load the pretrained GroundingDINO base weights into ``model`` in place."""
    from groundingdino.util.utils import clean_state_dict

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    logger.info(
        "Loaded base checkpoint %s: missing=%d unexpected=%d",
        checkpoint_path, len(missing), len(unexpected),
    )
    # Defensive re-placement: certain detectron2 'model_final.pth' files have
    # been observed to leave every parameter on CPU after load_state_dict
    # (root cause unresolved; sibling periodic checkpoints of the same
    # iteration load normally). Re-issuing .to() is a no-op in the normal
    # case and repairs the pathological one.
    device = next(model.parameters()).device
    cpu_leak = sum(1 for _, p in model.named_parameters() if p.device.type == "cpu")
    if cpu_leak and device.type == "cpu":
        logger.warning(
            "Checkpoint load left %d parameter tensors on CPU; model device "
            "repair is the caller's responsibility via load_variant.", cpu_leak,
        )


def _load_lora_plain(
    spec: VariantSpec, device: str, base_checkpoint: str | None
) -> nn.Module:
    """Load a plain-LoRA checkpoint trained by train_lora_forest_persons.py.

    The LoRA-wrapped architecture (custom-attention decomposition + lora.Linear
    wrappers) can only be reconstructed by that script's ``load_lora_model``,
    which lives at the ZiRa repo root (importable once repo_bridge has run).
    The checkpoint contains base + LoRA weights together, loaded strict=True.
    """
    from groundingdino.util.utils import clean_state_dict
    from train_lora_forest_persons import load_lora_model

    if base_checkpoint is None:
        raise ValueError("Variant kind 'lora_plain' requires 'base_checkpoint' for model construction")

    model = load_lora_model(spec.config, base_checkpoint)
    if spec.checkpoint is None:
        # Fresh LoRA wrapper (lora_B is zero-initialized) == exact base model
        # in the SAME architecture — the correct paired-drift baseline.
        logger.info("lora_plain with no checkpoint: returning zero-LoRA base model")
    else:
        # weights_only=False: locally produced research checkpoint (see note below).
        checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
        logger.info("Loaded plain-LoRA checkpoint %s (strict)", spec.checkpoint)
    model = model.to(device)
    if hasattr(model, "device"):
        model.device = device
    model.eval()
    return model


def _load_lora_bzil(
    spec: VariantSpec, device: str, base_checkpoint: str | None
) -> nn.Module:
    """Load a bottleneck LoRA+ZiL checkpoint (5 injection points).

    Architecture rebuilt via train_lora_bottleneck_zil_forest_persons.py's
    load_bottleneck_lora_model (importable once repo_bridge has run);
    checkpoint=None returns the zero-branch base-equivalent model.
    """
    from groundingdino.util.utils import clean_state_dict
    from train_lora_bottleneck_zil_forest_persons import load_bottleneck_lora_model

    if base_checkpoint is None:
        raise ValueError("Variant kind 'lora_bzil' requires 'base_checkpoint'")

    model = load_bottleneck_lora_model(spec.config, base_checkpoint)
    if spec.checkpoint is None:
        logger.info("lora_bzil with no checkpoint: zero-branch base model")
    else:
        checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=True)
        logger.info("Loaded bottleneck LoRA+ZiL checkpoint %s (strict)", spec.checkpoint)
    model = model.to(device)
    if hasattr(model, "device"):
        model.device = device
    model.eval()
    return model


class LoraConfigMismatchError(ValueError):
    """Raised when a variant's config disagrees with its checkpoint's own record."""


#: Backwards-compatible alias (the alpha case was the first instance found).
LoraAlphaMismatchError = LoraConfigMismatchError

#: Values that are DERIVED FROM CONFIG at load time and never stored in the
#: checkpoint tensors, mapped to their key in the run's ``injection_map.json``.
#: Each is silent when wrong -- the model loads, runs, and returns plausible
#: numbers -- which is exactly why each needs an explicit assert.
#:
#: ``lora_alpha``  : scaling = alpha/r; wrong value rescales every adapter.
#: ``lora_r``      : also feeds scaling; a wrong r additionally changes shapes,
#:                   so strict-load would usually catch it -- checked anyway
#:                   because "usually" is not "always".
#: ``injection_preset``: which modules carry LoRA. A mismatch normally trips
#:                   strict-load on missing/unexpected keys (loud), but the
#:                   preset is asserted here so the failure names the cause.
#:
#: NOT covered, and why: ``lora_dropout`` is identity in eval mode, so a wrong
#: value cannot change inference; ``lora_group_scales`` is a deliberate
#: post-hoc dial with no training-time counterpart to compare against.
_CONFIG_CHECKPOINT_KEYS: tuple[tuple[str, str], ...] = (
    ("lora_alpha", "lora_alpha"),
    ("lora_r", "lora_r"),
)


def _assert_alpha_matches_checkpoint(spec: VariantSpec) -> None:
    """Cross-check config-derived LoRA hyper-parameters against the run's record.

    LoRA applies ``scaling = lora_alpha / r``, rebuilt FROM THE CONFIG at load
    time and never stored in the checkpoint tensors. Loading an alpha=8
    checkpoint through an alpha=16 config therefore doubles every adapter
    contribution and loading the reverse halves it -- in both directions
    silently, with a plausible-looking model and no error anywhere.

    ``train_gdino_hinge.py`` (research tree) writes ``injection_map.json`` next to each
    checkpoint recording the alpha actually trained with, so the two can be
    compared. A run predating that file is skipped rather than blocked, but a
    disagreement is always fatal.
    """
    import json
    from pathlib import Path

    record = Path(spec.checkpoint).parent / "injection_map.json"
    if not record.exists():
        logger.warning("no injection_map.json beside %s: cannot verify lora_alpha",
                       spec.checkpoint)
        return
    try:
        summary = json.loads(record.read_text())["_summary"]
    except (KeyError, ValueError, TypeError):
        logger.warning("%s has no readable _summary; skipping config cross-check", record)
        return

    text = Path(spec.config).read_text()
    checked: dict[str, float] = {}
    for cfg_name, record_key in _CONFIG_CHECKPOINT_KEYS:
        declared = [ln for ln in text.splitlines() if ln.strip().startswith(cfg_name)]
        if not declared or record_key not in summary:
            logger.warning("cannot verify %s for %s; skipping", cfg_name, spec.checkpoint)
            continue
        config_value = float(declared[-1].split("=")[1].split("#")[0].strip())
        trained_value = float(summary[record_key])
        if abs(config_value - trained_value) > 1e-9:
            ratio = ""
            if cfg_name == "lora_alpha" and trained_value:
                ratio = (f" LoRA scaling is alpha/r and is rebuilt from config, so this "
                         f"silently scales every adapter by {config_value / trained_value:g}x.")
            raise LoraConfigMismatchError(
                f"{spec.checkpoint} was trained with {cfg_name}={trained_value:g} but "
                f"{spec.config} declares {cfg_name}={config_value:g}.{ratio} Point the "
                "variant at the matching config (groundingdino_swinb_dt_lora_a16.py for "
                "alpha=16 arms, groundingdino_swinb_dt_lora_hinge.py for legacy alpha=8)."
            )
        checked[cfg_name] = config_value

    trained_preset = summary.get("injection_preset")
    declared_preset = getattr(spec, "lora_injection_preset", None) or "both_towers"
    if trained_preset and declared_preset != trained_preset:
        raise LoraConfigMismatchError(
            f"{spec.checkpoint} was trained with injection_preset={trained_preset!r} but the "
            f"variant declares {declared_preset!r}; a different module set would be adapted."
        )
    logger.info("config/checkpoint agree: %s, injection_preset=%s",
                ", ".join(f"{k}={v:g}" for k, v in checked.items()), trained_preset)


def _load_lora_hinge(spec: VariantSpec, device: str) -> nn.Module:
    """Load a LoRA + task-loss-hinge checkpoint trained by train_gdino_hinge.py (research tree).

    The architecture (custom-attention decomposition + vanilla LoRA wrappers on
    the hinge run's injection groups) is rebuilt by that script's own
    ``load_hinge_eval_model``, which strict-loads the full checkpoint (base +
    LoRA weights together) — no separate base checkpoint is needed. Importing
    the training script pops ``HF_HOME`` (bert-base-uncased lives in
    ``~/.cache/huggingface``) exactly as every gdino eval path already
    requires.
    """
    from scripts.train_gdino_hinge import load_hinge_eval_model

    if spec.checkpoint is None:
        raise ValueError("Variant kind 'lora_hinge' requires a trained checkpoint")

    _assert_alpha_matches_checkpoint(spec)
    model = load_hinge_eval_model(
        spec.config,
        spec.checkpoint,
        lora_group_scales=getattr(spec, "lora_group_scales", None),
        injection_preset=getattr(spec, "lora_injection_preset", None),
    )
    model = model.to(device)
    if hasattr(model, "device"):
        model.device = device
    _assert_preprocessing(spec, model)
    model.eval()  # merges LoRA deltas into the base Linear weights
    return model


def _assert_preprocessing(spec: VariantSpec, model: nn.Module) -> None:
    """Verify resolution/normalisation against the run's manifest, if it has one.

    Absent manifest => warn and pass: every checkpoint predating the manifest
    has none, and blocking them would break reproduction of published numbers
    to fix a problem they do not have.
    """
    import json
    from pathlib import Path

    from utils.data.preprocessing_manifest import assert_preprocessing_matches

    recorded = None
    record = Path(spec.checkpoint).parent / "injection_map.json"
    if record.exists():
        try:
            recorded = json.loads(record.read_text())["_summary"].get("preprocessing")
        except (KeyError, ValueError, TypeError):
            recorded = None
    if recorded is None:
        sidecar = Path(spec.checkpoint).parent / "reconstructed_manifest.json"
        if sidecar.exists():
            try:
                recorded = json.loads(sidecar.read_text()).get("preprocessing")
            except (KeyError, ValueError, TypeError):
                recorded = None
    assert_preprocessing_matches(recorded, model, source=str(spec.checkpoint))


def load_variant(spec: VariantSpec, device: str, base_checkpoint: str | None = None) -> nn.Module:
    """Build and load a GroundingDINO variant according to its ``kind``.

    Supported kinds:
        - ``base``: load the pretrained checkpoint (``spec.checkpoint``)
          directly (strict=False).
        - ``full_state``: same loading path as ``base`` but semantically a
          fully fine-tuned (joint-trained) checkpoint.
        - ``zira_adapter``: build with the ZiRa (rep) config, load
          ``base_checkpoint`` (the pretrained GroundingDINO weights) first,
          then restore adapter-only weights from ``spec.checkpoint`` on top
          (following eval_coco_dithub.py:96-117).

    Args:
        spec: Variant specification (config path, checkpoint path, kind).
        device: Target device string, e.g. ``cuda:0``.
        base_checkpoint: Path to the pretrained GroundingDINO checkpoint.
            Required (and used) only when ``spec.kind == "zira_adapter"``,
            since the adapter checkpoint only contains the adapter/LoRA
            deltas, not the full backbone weights.

    Returns:
        The loaded model in eval mode on ``device``.

    Raises:
        ValueError: If ``spec.kind`` is unknown, or if a checkpoint-requiring
            kind has no checkpoint configured yet.
    """
    if spec.kind == "lora_plain":
        return _load_lora_plain(spec, device, base_checkpoint)
    if spec.kind == "lora_bzil":
        return _load_lora_bzil(spec, device, base_checkpoint)
    if spec.kind == "lora_hinge":
        return _load_lora_hinge(spec, device)

    if spec.kind not in {"base", "full_state", "zira_adapter"}:
        raise ValueError(f"Unknown variant kind: {spec.kind}")

    if spec.kind in {"base", "full_state"} and spec.checkpoint is None:
        raise ValueError(
            f"Variant kind '{spec.kind}' requires a checkpoint but none is configured — "
            "checkpoint not yet trained — configure in configs/default.yaml"
        )
    if spec.kind == "zira_adapter" and spec.checkpoint is None:
        raise ValueError(
            "Variant kind 'zira_adapter' requires a checkpoint but none is configured — "
            "checkpoint not yet trained — configure in configs/default.yaml"
        )
    if spec.kind == "zira_adapter" and base_checkpoint is None:
        raise ValueError(
            "Variant kind 'zira_adapter' requires 'base_checkpoint' "
            "(the pretrained GroundingDINO weights) to be provided."
        )

    model = build_base_model(spec.config, device)

    if spec.kind in {"base", "full_state"}:
        _load_base_checkpoint(model, spec.checkpoint)
    else:  # zira_adapter
        from groundingdino.util.utils import clean_state_dict

        _load_base_checkpoint(model, base_checkpoint)

        # weights_only=False: these are locally produced research checkpoints
        # whose 'model' dict may be wrapped in non-tensor training metadata;
        # never point this at a checkpoint from an untrusted source.
        adapter_state = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        state = adapter_state["model"] if "model" in adapter_state else adapter_state
        missing, unexpected = model.load_state_dict(clean_state_dict(state), strict=False)
        logger.info(
            "Restored ZiRa adapter weights from %s: %d keys, missing=%d unexpected=%d",
            spec.checkpoint, len(state), len(missing), len(unexpected),
        )

    # Unconditional re-placement (no-op when already on `device`): guards
    # against the pathological detectron2 'model_final.pth' load that leaves
    # parameters on CPU (see note in _load_base_checkpoint).
    model = model.to(device)
    if hasattr(model, "device"):
        model.device = device
    model.eval()
    return model
