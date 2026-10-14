"""The V6 vocabulary and the source-to-V6 label mappings (frozen).

Spec: the vocabulary specification §1.

    V6 = person . clothes . hat . shoes . gloves . bag

The order above is the spec's order and it is the order used for category
ids (``person``=1 .. ``bag``=6) and for every rendered caption. It differs
from the pre-2026-08-08 ``srcs.vocab.CLASS_NAMES`` order
(``person, clothes, hat, gloves, bag, shoes``); that module belongs to the
retired experiment system and is not imported here.

Three user decisions of 2026-08-08 are encoded in this module:

1. ForestPersons' 13 pose/visibility subclasses (``standing_100`` etc.) all
   map to the single class ``person``.
2. ForestBelongings singulars are unified to plurals: ``cloth``->``clothes``,
   ``glove``->``gloves``, ``shoe``->``shoes``. ``hat``/``bag``/``person``
   are unchanged.
3. ForestPersonsIR is excluded from both training and evaluation.

``etc`` handling
----------------
ForestBelongings' ``etc`` category is *not* in V6 and is carried through as
an ``iscrowd=1`` ignore region under category id 7, outside the queried
vocabulary. Rationale (see ``etc_patch.py`` in the research tree for the canonical
citation): the dataset's own category definition document defines ``etc`` as
"a belonging that does not fall into the five core types" -- an
out-of-vocabulary *belonging*, not background. Dropping the boxes would
teach "a visible belonging here is background", which is the wrong signal;
mapping them into any V6 class would fabricate labels. ``iscrowd=1`` under a
non-queried id is the dataset's existing convention for "annotated but
excluded from scoring", and pycocotools' per-category matching never scores
category 7 against any V6 class.
"""
from __future__ import annotations

from typing import Mapping

#: The frozen V6 vocabulary in spec §1 order. Category ids are 1-indexed in
#: this order.
V6_CLASSES: tuple[str, ...] = ("person", "clothes", "hat", "shoes", "gloves", "bag")

CATEGORY_ID: Mapping[str, int] = {name: idx + 1 for idx, name in enumerate(V6_CLASSES)}

CLASS_BY_ID: Mapping[int, str] = {idx: name for name, idx in CATEGORY_ID.items()}

#: The five belongings classes (V6 minus ``person``).
BELONGINGS_CLASSES: tuple[str, ...] = tuple(c for c in V6_CLASSES if c != "person")

PERSON_CLASS: str = "person"

#: Out-of-vocabulary ignore region. Never queried, never scored.
ETC_CATEGORY_NAME: str = "etc"
ETC_CATEGORY_ID: int = len(V6_CLASSES) + 1  # 7

#: ForestBelongings singular -> V6 plural (user decision 2, 2026-08-08).
FB_RENAME: Mapping[str, str] = {"cloth": "clothes", "glove": "gloves", "shoe": "shoes"}

#: ForestPersons pose/visibility subclass names, all collapsed to ``person``
#: (user decision 1, 2026-08-08). Listed explicitly so that a source-side
#: change (a new pose class) fails loudly instead of being silently absorbed.
FP_POSE_SUBCLASSES: tuple[str, ...] = (
    "sitting_100", "sitting_70", "sitting_40", "sitting_20",
    "lying_100", "lying_70", "lying_40", "lying_20",
    "standing_100", "standing_70", "standing_40", "standing_20",
    "unknown_0",
)

#: Names that must NOT appear in a V6 ``categories`` block. Used by the
#: build-time verification gate.
FORBIDDEN_CATEGORY_NAMES: frozenset[str] = frozenset(FB_RENAME) | frozenset(FP_POSE_SUBCLASSES)


def build_categories() -> list[dict]:
    """The V6 ``categories`` block: 6 queried classes + the ``etc`` ignore id."""
    return [{"id": CATEGORY_ID[name], "name": name} for name in V6_CLASSES] + [
        {"id": ETC_CATEGORY_ID, "name": ETC_CATEGORY_NAME}
    ]


def render_caption(terms: tuple[str, ...]) -> str:
    """Render a GroundingDINO-style caption from an ordered term list.

    The exact string returned here is what gets recorded as the verbatim
    query string in every result fragment (spec §7).
    """
    if not terms:
        raise ValueError("cannot render a caption from an empty term list")
    return " . ".join(terms) + " ."


V6_CAPTION: str = render_caption(V6_CLASSES)
