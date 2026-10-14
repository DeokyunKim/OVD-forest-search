"""Bridge to the external ZiRaGroundingDINO codebase.

The GroundingDINO implementation (including the compiled
``MultiScaleDeformableAttention`` CUDA extension) lives in a separate,
already-built repository. Rather than vendoring it, we insert it at the
front of ``sys.path`` so ``import groundingdino`` resolves to that
checkout's package.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from utils.runner.logging_utils import get_logger

logger = get_logger(__name__)


def _prefer_local_hf_cache() -> None:
    """Skip HuggingFace Hub freshness-check HEAD requests when a local cache exists.

    The BERT text encoder (``bert-base-uncased``) is already cached under
    ``~/.cache/huggingface/hub`` on this machine, but ``transformers``
    otherwise still attempts a network HEAD request per call, which can hang
    for tens of seconds behind restrictive network policies before falling
    back to cache. Setting the offline env vars makes it use the cache
    directly. Only applied if not already set, so callers can override.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ensure_external_repo(path: str) -> None:
    """Idempotently prepend ``path`` to ``sys.path`` and verify importability.

    Args:
        path: Root directory of the external ZiRaGroundingDINO checkout
            (the directory containing the ``groundingdino`` package).

    Raises:
        RuntimeError: If ``path`` does not exist, or if ``import
            groundingdino`` fails after inserting it on ``sys.path``.
    """
    repo_path = Path(path)
    if not repo_path.is_dir():
        raise RuntimeError(
            f"External repo path does not exist: {path}. "
            "Both callers resolve it from UC_GDINO_REPO -- evaluation through "
            "`external_repo` in configs/models.yaml, training directly."
        )

    _prefer_local_hf_cache()

    resolved = str(repo_path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        logger.info("Inserted external repo at sys.path[0]: %s", resolved)

    try:
        import groundingdino  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to 'import groundingdino' after adding {resolved} to sys.path. "
            "Verify the external repo is intact and its compiled CUDA extension "
            "(groundingdino/_C*.so) was built in the target venv."
        ) from exc
