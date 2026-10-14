"""Record and verify preprocessing constants that checkpoints do not store.

THE FAILURE CLASS
-----------------
A value reconstructed from config at load time and never stored beside the
weights is never compared against anything, so a wrong value is
indistinguishable from a right one: the model builds, runs, and returns
plausible numbers. On 2026-08-05 this cost a full forensic investigation when
``lora_alpha`` briefly changed under 14 variants sharing one config file (see
the infrastructure_incidents.md note).

``lora_alpha`` / ``lora_r`` / ``injection_preset`` are now asserted against the
training run's ``injection_map.json``. This module closes the remaining gap
named in that audit: **input resolution and normalisation constants**, which no
artefact recorded at all.

Both are silent when wrong:

* resolution -- a model trained at short edge 800 and evaluated at 640 simply
  detects worse; nothing errors.
* normalisation -- GroundingDINO's detectron2-style forward normalises inside
  ``preprocess_image`` using ``model.pixel_mean`` / ``model.pixel_std``. Wrong
  constants shift every activation, and the network still produces boxes.

Values are read OFF THE LIVE MODEL rather than parsed out of config files, so
the manifest records what the run actually used, not what a config was believed
to say.
"""
from __future__ import annotations

from typing import Any, Mapping

from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)

#: Evaluation-time resize, shared with ``models/gdino/loader.py``
#: so the manifest and the adapter can never drift apart.
EVAL_SHORT_EDGE = 800
EVAL_MAX_SIZE = 1333

#: Tolerance for float comparison of normalisation constants.
_TOL = 1e-6


class PreprocessingMismatchError(ValueError):
    """Raised when a checkpoint's recorded preprocessing differs from the loader's."""


def describe_model_preprocessing(model: Any, short_edge: int = EVAL_SHORT_EDGE,
                                 max_size: int = EVAL_MAX_SIZE) -> dict[str, Any]:
    """Snapshot the preprocessing a built model actually applies.

    ``pixel_mean`` / ``pixel_std`` are plain attributes on the detectron2-style
    GroundingDINO classes; a model that has neither records ``null`` rather
    than a guess, so a later comparison can tell "not applicable" from "equal".
    """
    def _as_list(value: Any) -> list[float] | None:
        if value is None:
            return None
        try:
            return [float(v) for v in value]
        except TypeError:
            return None

    return {
        "pixel_mean": _as_list(getattr(model, "pixel_mean", None)),
        "pixel_std": _as_list(getattr(model, "pixel_std", None)),
        "eval_short_edge": int(short_edge),
        "eval_max_size": int(max_size),
    }


def assert_preprocessing_matches(recorded: Mapping[str, Any] | None, model: Any,
                                 source: str = "checkpoint") -> None:
    """Compare a recorded manifest against the model being loaded.

    A MISSING manifest warns and passes: every checkpoint produced before this
    module existed has none, and refusing to load them would break reproduction
    of published numbers to fix a problem they do not have. A manifest that is
    PRESENT and disagrees is always fatal.
    """
    if not recorded:
        logger.warning("%s records no preprocessing manifest; cannot verify resolution / "
                       "normalisation (pre-dates the manifest, or was written by an older "
                       "trainer)", source)
        return

    current = describe_model_preprocessing(model)
    for key in ("eval_short_edge", "eval_max_size"):
        want, got = recorded.get(key), current.get(key)
        if want is not None and got is not None and int(want) != int(got):
            raise PreprocessingMismatchError(
                f"{source} was produced with {key}={want} but this loader uses {got}. "
                "Resolution mismatches degrade detection silently -- nothing errors."
            )
    for key in ("pixel_mean", "pixel_std"):
        want, got = recorded.get(key), current.get(key)
        if want is None or got is None:
            continue
        if len(want) != len(got) or any(abs(a - b) > _TOL for a, b in zip(want, got)):
            raise PreprocessingMismatchError(
                f"{source} was produced with {key}={want} but this model has {got}. "
                "Wrong normalisation shifts every activation and still returns boxes."
            )
    logger.info("preprocessing verified against %s: short_edge=%s max_size=%s",
                source, current["eval_short_edge"], current["eval_max_size"])
