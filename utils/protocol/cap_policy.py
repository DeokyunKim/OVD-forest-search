"""How the detection cap is applied — the two policies, and which is standard.

Spec §6 says "max detections per image 100 (COCO standard)". That sentence is
ambiguous, and the ambiguity turned out to matter: it silently decides whether
the queried terms compete for a shared budget.

What the COCO standard actually is
----------------------------------
``pycocotools`` applies ``maxDets`` **per (image, category)**, not per image.
From the installed source (``pycocothe internal checks``):

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        ...
        dt = self._dts[imgId, catId]
        ...
        dt = [dt[i] for i in dtind[0:maxDet]]

``evaluate`` calls it once per ``(imgId, catId)`` pair, and ``accumulate``
indexes recall as ``recall[T, K, A, M]`` with ``K`` the category axis, each
category's curve built from its own ``dtScores[0:maxDet]``. So ``AR@100`` means
"average recall allowing up to 100 detections **of that category** per image".

Verified empirically, not only by reading: submitting 150 detections for each
of two categories on one image, with category 2's true box at global rank ~240
but rank 90 *within its category*, still recalls it (AR = 1.0). A global
per-image budget of 100 would have discarded it.

The two policies
----------------
``PER_CATEGORY`` (standard, default)
    Distractor detections are dropped first; the surviving scored-class
    detections are handed to COCOeval, which applies its own per-category
    ``maxDets``. Queried terms do not compete for a shared evaluation budget.

``GLOBAL_PER_IMAGE`` (deployment view, supplementary)
    The top 100 detections across *all* terms are kept, then distractors are
    dropped. Models this: an operator's screen shows 100 boxes and a distractor
    that outranks a person box really does displace it.

``GLOBAL_PER_IMAGE`` was used for the first census pass. It is not wrong as a
deployment measurement, but it is not the COCO standard and it makes the
queried terms zero-sum, which mechanically drives ``AR`` down whenever a
distractor captures budget (measured r = -0.96 between distractor budget share
and ``AR_person``). Because §4's stealing test is defined as "AR down", that
artefact would masquerade as the mechanism under test. See
the cap-policy note.

The model's own cap is untouched under either policy
----------------------------------------------------
GroundingDINO returns ``select_box_nums_for_evaluation`` (200) candidates over
the flattened (query x class) axis. That is a genuine architectural budget and
§6 says to keep model defaults, so it stays under both policies. A residual
model-level competition therefore remains and must be acknowledged; what
``PER_CATEGORY`` removes is the *additional* 100-wide bottleneck that the
harness was imposing on top of it.
"""
from __future__ import annotations

from typing import Mapping, Sequence

PER_CATEGORY = "per_category"
GLOBAL_PER_IMAGE = "global_per_image"

CAP_POLICIES: tuple[str, ...] = (PER_CATEGORY, GLOBAL_PER_IMAGE)

#: The standard, and the default for every primary result.
DEFAULT_CAP_POLICY = PER_CATEGORY

Detection = tuple[str, float, tuple[float, float, float, float]]


def apply_cap_policy(
    detections: Sequence[Detection],
    term_to_gt_class: Mapping[str, str | None],
    policy: str,
    max_detections: int,
) -> list[Detection]:
    """Apply the cap and drop distractors, in the order ``policy`` dictates.

    Args:
        detections: One image's ``(term, score, bbox_xyxy)`` triples.
        term_to_gt_class: Condition's term -> V6 class, ``None`` for distractors.
        policy: :data:`PER_CATEGORY` or :data:`GLOBAL_PER_IMAGE`.
        max_detections: The per-image cap, used only by ``GLOBAL_PER_IMAGE``.

    Returns:
        The scored-class detections to hand to COCOeval. Under
        ``PER_CATEGORY`` no harness-side cap is applied at all — COCOeval
        applies the standard per-category ``maxDets`` itself.
    """
    if policy not in CAP_POLICIES:
        raise ValueError(f"unknown cap policy {policy!r}; expected one of {CAP_POLICIES}")

    if policy == GLOBAL_PER_IMAGE:
        kept = sorted(detections, key=lambda d: d[1], reverse=True)[:max_detections]
    else:
        kept = detections
    return [d for d in kept if term_to_gt_class.get(d[0]) is not None]


def budget_is_shared(policy: str) -> bool:
    """Whether queried terms compete for one evaluation budget under ``policy``."""
    return policy == GLOBAL_PER_IMAGE
