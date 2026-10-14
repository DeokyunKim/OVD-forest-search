"""Serialisation for query-sensitivity results.

One rule: an unmeasured quantity is rendered as the string
:data:`~.dump_adapter.UNMEASURED`, never as ``0``. ``n_det = 0`` (the model was
silent) and ``recall = 0`` (the model spoke and was wrong) are different facts,
and so is "we did not measure this". A table that renders all three as ``0`` is
unreadable, and this project has already been burned by exactly that.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from utils.runner.logging_utils import get_logger
from utils.metrics.dump_adapter import UNMEASURED

logger = get_logger(__name__)


def serialise(value: Any) -> Any:
    """Recursively convert dataclasses/arrays to JSON, ``None`` -> ``_unmeasured_``."""
    if value is None:
        return UNMEASURED
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: serialise(getattr(value, f.name))
                for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): serialise(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return serialise(value.tolist())
    if isinstance(value, (list, tuple, set)):
        return [serialise(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_record(record: Mapping, path: Path) -> Path:
    """Write a result record atomically, so a killed run leaves no half table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(serialise(record), indent=2))
    tmp.rename(path)
    logger.info("wrote query-sensitivity record %s", path)
    return path
