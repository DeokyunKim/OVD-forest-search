"""Idempotent logger factory shared across the analysis toolkit."""
from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped logger with a single stream handler.

    Safe to call repeatedly (e.g. once per module import) without
    accumulating duplicate handlers.

    Args:
        name: Usually ``__name__`` of the calling module.
        level: Logging level applied to both the logger and its handler.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        logger.addHandler(handler)

    logger.propagate = False
    return logger
