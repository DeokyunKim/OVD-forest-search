"""The training caption — one source, shared with evaluation.

Spec §4.0 trains on the V6 vocabulary and §5 evaluates condition
``A4_full_v6``, whose caption is ``person . clothes . hat . shoes . gloves .
bag .``. If the trainer rendered its own caption, a drift as small as a
trailing space or a reordered term would make the FT arm's in-vocabulary
result incomparable to its own zero-shot baseline — and nothing would error.

So there is exactly one renderer: :func:`training_caption` returns the
``A4_full_v6`` condition's caption object, byte for byte. The class-name list
handed to a detectron2 mapper is derived from the same condition, in the same
order, so the caption the mapper builds and the caption the evaluator sends are
the same string.

the regression tests drives the real evaluation path and compares
bytes rather than asserting against a literal.
"""
from __future__ import annotations

from utils.protocol.conditions import get_condition

#: The condition whose vocabulary defines training. Training uses the full V6
#: vocabulary, which is exactly this condition's term list.
TRAINING_CONDITION_ID = "A4_full_v6"


def training_condition():
    """The condition object that defines the training vocabulary."""
    return get_condition(TRAINING_CONDITION_ID)


def training_caption() -> str:
    """The verbatim caption used for training. Identical to evaluation's."""
    return training_condition().caption


def training_class_names() -> list[str]:
    """Ordered class names for a detectron2 mapper's ``categories_names``.

    The mapper builds its caption by joining these, so order is part of the
    contract: permuting it would produce a different caption and a different
    experiment.
    """
    return list(training_condition().terms)


def caption_from_class_names(class_names: list[str]) -> str:
    """Reproduce the mapper's caption from a class-name list.

    Mirrors the external repo's ``DetrDatasetMapper`` convention
    (``" . ".join(names) + " ."``) so the two can be compared without running
    the mapper.
    """
    return " . ".join(class_names) + " ."
