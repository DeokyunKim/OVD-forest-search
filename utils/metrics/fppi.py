"""False-positives-per-image (FPPI) on negative (person-free) image pools.

Every prediction returned for a query on an image with zero person ground
truth is, by construction, a false positive -- there is no matching step
here (unlike ``srcs.calibration.attribution``, which matches against real
person GT). FPPI at a given score threshold is simply the count of
predictions surviving that threshold, divided by the number of images the
pool was drawn from (including images with zero surviving predictions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FppiCurve:
    """FPPI evaluated at a sweep of score thresholds.

    Attributes:
        thresholds: The score thresholds the curve was evaluated at.
        fppi: ``fppi[i]`` is FPPI at ``thresholds[i]``, same length.
        n_images: Number of images the false-positive count was normalised by.
    """

    thresholds: np.ndarray
    fppi: np.ndarray
    n_images: int


def fppi_at(scores: np.ndarray, n_images: int, threshold: float) -> float:
    """FPPI at a single score threshold.

    Args:
        scores: All prediction scores pooled across every image in the tier
            (flat array; an empty array is valid and yields FPPI 0).
        n_images: Number of images in the pool (the FPPI denominator). Must
            include images with zero predictions -- callers must not filter
            them out before calling this function.
        threshold: Score cutoff; predictions with ``score >= threshold`` count.

    Returns:
        ``count(scores >= threshold) / n_images``.

    Raises:
        ValueError: If ``n_images <= 0``.
    """
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    scores = np.asarray(scores, dtype=float)
    return float(np.sum(scores >= threshold)) / n_images


def fppi_curve(scores: np.ndarray, n_images: int, thresholds: np.ndarray) -> FppiCurve:
    """FPPI swept across a range of thresholds.

    Args:
        scores: All prediction scores pooled across every image in the tier.
        n_images: Number of images in the pool (see ``fppi_at``).
        thresholds: 1-D array of score thresholds to evaluate at.

    Returns:
        An ``FppiCurve`` with one FPPI value per threshold.

    Raises:
        ValueError: If ``n_images <= 0``.
    """
    if n_images <= 0:
        raise ValueError(f"n_images must be positive, got {n_images}")
    scores = np.asarray(scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    # Vectorised over thresholds: for each threshold count scores >= it.
    fppi = np.array([np.sum(scores >= t) for t in thresholds], dtype=float) / n_images
    return FppiCurve(thresholds=thresholds, fppi=fppi, n_images=n_images)


def matched_threshold_table(
    scores_by_tier: dict[str, np.ndarray],
    n_images_by_tier: dict[str, int],
    thresholds: list[float],
) -> dict[str, dict[float, float]]:
    """FPPI per tier at a small fixed set of "matched" thresholds.

    Args:
        scores_by_tier: ``tier_name -> pooled prediction scores``.
        n_images_by_tier: ``tier_name -> image count`` (same key set as
            ``scores_by_tier``).
        thresholds: The thresholds to report (e.g. ``[0.25, 0.45, 0.65]``).

    Returns:
        ``tier_name -> {threshold -> fppi}``.

    Raises:
        ValueError: If the two input dicts have different key sets.
    """
    if set(scores_by_tier) != set(n_images_by_tier):
        raise ValueError("scores_by_tier and n_images_by_tier must share the same tier keys")
    return {
        tier: {t: fppi_at(scores_by_tier[tier], n_images_by_tier[tier], t) for t in thresholds}
        for tier in scores_by_tier
    }
