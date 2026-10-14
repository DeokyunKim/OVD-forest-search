"""The 32 query conditions — single definition site.

Spec: the vocabulary specification §2-§6.

===  =====  ==========================================================
axis  count  content
===  =====  ==========================================================
A       18  singletons 6 + person pairs 5 + leave-one-out 6 + full 1
B        9  ``person`` variants 6 + ``clothes`` variants 3
C        5  distractor sets
===  =====  ==========================================================

Nothing else in the codebase may construct a query string. Every evaluation
run takes its caption from :func:`all_conditions`, so the caption recorded in
a result fragment is by construction the caption that was evaluated.

In/out-of-vocabulary labelling follows spec §6 verbatim: axis A is
in-vocabulary (V6, its leave-one-out sets and its subsets are all inside the
FT training vocabulary); **all** of axis B and **all** of axis C are
out-of-vocabulary, including conditions such as C1 whose first six terms are
exactly V6 — the condition as a whole contains expressions the FT model never
saw.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from utils.protocol.coco80 import COCO80_CLASSES
from utils.protocol.vocabulary import BELONGINGS_CLASSES, PERSON_CLASS, V6_CLASSES, render_caption

IN_VOCABULARY = "in_vocabulary"
OUT_OF_VOCABULARY = "out_of_vocabulary"

#: Spec §3: ``person`` paraphrases, B1..B6, in spec order.
PERSON_VARIANTS: tuple[str, ...] = ("human", "people", "hiker", "missing person", "victim", "body")
#: Spec §3: auxiliary ``clothes`` paraphrases, B7..B9.
CLOTHES_VARIANTS: tuple[str, ...] = ("clothing", "garment", "jacket")

#: Spec §4 distractor sets.
C_NAT: tuple[str, ...] = ("tree", "rock", "bush", "stream", "fern", "moss")
C_CONF: tuple[str, ...] = ("tree stump", "fallen log", "boulder", "tree trunk", "shadow")
C_MAN: tuple[str, ...] = ("tent", "trail sign", "fence", "bench", "litter")


@dataclass(frozen=True)
class Condition:
    """One evaluated query condition.

    Attributes:
        condition_id: Stable identifier; the fragment filename stem.
        axis: ``"A"``, ``"B"`` or ``"C"``.
        subgroup: Spec subsection, e.g. ``"A2_person_pair"``.
        terms: The query terms in the order they are rendered. This order is
            part of the condition — permuting it is a different experiment.
        term_to_gt_class: term -> the V6 ground-truth class its detections are
            scored against, or ``None`` for a distractor whose predictions are
            discarded (no ground truth exists for it).
        vocabulary_status: :data:`IN_VOCABULARY` or :data:`OUT_OF_VOCABULARY`.
        description: One line for the report tables.
    """

    condition_id: str
    axis: str
    subgroup: str
    terms: tuple[str, ...]
    term_to_gt_class: Mapping[str, str | None]
    vocabulary_status: str
    description: str

    def __post_init__(self) -> None:
        if len(set(self.terms)) != len(self.terms):
            raise ValueError(f"{self.condition_id}: duplicate terms in {self.terms}")
        if set(self.term_to_gt_class) != set(self.terms):
            raise ValueError(f"{self.condition_id}: term_to_gt_class does not cover terms exactly")
        bad = {v for v in self.term_to_gt_class.values() if v is not None} - set(V6_CLASSES)
        if bad:
            raise ValueError(f"{self.condition_id}: maps to non-V6 classes {sorted(bad)}")
        if self.vocabulary_status not in (IN_VOCABULARY, OUT_OF_VOCABULARY):
            raise ValueError(f"{self.condition_id}: bad vocabulary_status {self.vocabulary_status!r}")

    @property
    def caption(self) -> str:
        """The verbatim query string handed to the model and recorded in §7."""
        return render_caption(self.terms)

    @property
    def scored_classes(self) -> tuple[str, ...]:
        """V6 classes with a queried term, in V6 order. Others are not scored."""
        mapped = {v for v in self.term_to_gt_class.values() if v is not None}
        return tuple(c for c in V6_CLASSES if c in mapped)

    @property
    def distractor_terms(self) -> tuple[str, ...]:
        """Queried terms with no ground truth; their detections are discarded."""
        return tuple(t for t in self.terms if self.term_to_gt_class[t] is None)


def _identity(terms: tuple[str, ...]) -> dict[str, str | None]:
    return {t: t for t in terms}


def _with_distractors(v6_terms: tuple[str, ...],
                      distractors: tuple[str, ...]) -> dict[str, str | None]:
    return {**_identity(v6_terms), **{d: None for d in distractors}}


def _axis_a() -> list[Condition]:
    conditions: list[Condition] = []

    # A1 singletons (6) -- the interference-free baseline, denominator of every
    # interference figure.
    for name in V6_CLASSES:
        conditions.append(Condition(
            condition_id=f"A1_singleton_{name.replace(' ', '_')}",
            axis="A", subgroup="A1_singleton", terms=(name,),
            term_to_gt_class=_identity((name,)),
            vocabulary_status=IN_VOCABULARY,
            description=f"singleton `{name}`",
        ))

    # A2 person pairs (5) -- first-order interference I(c).
    for name in BELONGINGS_CLASSES:
        terms = (PERSON_CLASS, name)
        conditions.append(Condition(
            condition_id=f"A2_pair_person_{name}",
            axis="A", subgroup="A2_person_pair", terms=terms,
            term_to_gt_class=_identity(terms),
            vocabulary_status=IN_VOCABULARY,
            description=f"person paired with `{name}`",
        ))

    # A3 leave-one-out (6) -- marginal contribution M(c) within V6.
    for name in V6_CLASSES:
        terms = tuple(c for c in V6_CLASSES if c != name)
        conditions.append(Condition(
            condition_id=f"A3_loo_{name}",
            axis="A", subgroup="A3_leave_one_out", terms=terms,
            term_to_gt_class=_identity(terms),
            vocabulary_status=IN_VOCABULARY,
            description=f"V6 without `{name}`",
        ))

    # A4 full V6 (1).
    conditions.append(Condition(
        condition_id="A4_full_v6",
        axis="A", subgroup="A4_full", terms=V6_CLASSES,
        term_to_gt_class=_identity(V6_CLASSES),
        vocabulary_status=IN_VOCABULARY,
        description="full V6",
    ))
    return conditions


def _substituted(target: str, variant: str) -> tuple[tuple[str, ...], dict[str, str | None]]:
    """Replace one V6 term by a paraphrase, holding every other term fixed."""
    terms = tuple(variant if c == target else c for c in V6_CLASSES)
    mapping: dict[str, str | None] = {c: c for c in V6_CLASSES if c != target}
    mapping[variant] = target
    return terms, mapping


def _axis_b() -> list[Condition]:
    conditions: list[Condition] = []
    for idx, variant in enumerate(PERSON_VARIANTS, start=1):
        terms, mapping = _substituted(PERSON_CLASS, variant)
        conditions.append(Condition(
            condition_id=f"B{idx}_person_as_{variant.replace(' ', '_')}",
            axis="B", subgroup="B_person_variant", terms=terms,
            term_to_gt_class=mapping,
            vocabulary_status=OUT_OF_VOCABULARY,
            description=f"V6 with `person` -> `{variant}`",
        ))
    for offset, variant in enumerate(CLOTHES_VARIANTS, start=len(PERSON_VARIANTS) + 1):
        terms, mapping = _substituted("clothes", variant)
        conditions.append(Condition(
            condition_id=f"B{offset}_clothes_as_{variant.replace(' ', '_')}",
            axis="B", subgroup="B_clothes_variant", terms=terms,
            term_to_gt_class=mapping,
            vocabulary_status=OUT_OF_VOCABULARY,
            description=f"V6 with `clothes` -> `{variant}`",
        ))
    return conditions


#: COCO80 minus the names already in V6. Only ``person`` collides. The spec
#: writes C5 as "V6 + COCO80"; emitting the collision verbatim would put two
#: identical text spans in one caption, which makes the span-to-class
#: assignment ambiguous for a phrase-grounding model and is not what the
#: condition is testing (extreme vocabulary size). The V6 occurrence is kept
#: and the COCO duplicate dropped, giving 6 + 79 = 85 terms.
COCO80_EXTRA: tuple[str, ...] = tuple(c for c in COCO80_CLASSES if c not in V6_CLASSES)


def _axis_c() -> list[Condition]:
    return [
        Condition(
            condition_id="C1_v6_plus_nat",
            axis="C", subgroup="C1_natural", terms=(*V6_CLASSES, *C_NAT),
            term_to_gt_class=_with_distractors(V6_CLASSES, C_NAT),
            vocabulary_status=OUT_OF_VOCABULARY,
            description="V6 + C-nat (natural, spatially disjoint from people)",
        ),
        Condition(
            condition_id="C2_v6_plus_conf",
            axis="C", subgroup="C2_confusable", terms=(*V6_CLASSES, *C_CONF),
            term_to_gt_class=_with_distractors(V6_CLASSES, C_CONF),
            vocabulary_status=OUT_OF_VOCABULARY,
            description="V6 + C-conf (person-shaped confusables) — absorption vs stealing",
        ),
        Condition(
            condition_id="C3_v6_plus_man",
            axis="C", subgroup="C3_manmade", terms=(*V6_CLASSES, *C_MAN),
            term_to_gt_class=_with_distractors(V6_CLASSES, C_MAN),
            vocabulary_status=OUT_OF_VOCABULARY,
            description="V6 + C-man (forest artefacts, not search targets)",
        ),
        Condition(
            condition_id="C4_person_plus_conf",
            axis="C", subgroup="C4_confusable_person_only",
            terms=(PERSON_CLASS, *C_CONF),
            term_to_gt_class=_with_distractors((PERSON_CLASS,), C_CONF),
            vocabulary_status=OUT_OF_VOCABULARY,
            description="person + C-conf (vocabulary size controlled)",
        ),
        Condition(
            condition_id="C5_v6_plus_coco80",
            axis="C", subgroup="C5_coco80", terms=(*V6_CLASSES, *COCO80_EXTRA),
            term_to_gt_class=_with_distractors(V6_CLASSES, COCO80_EXTRA),
            vocabulary_status=OUT_OF_VOCABULARY,
            description="V6 + COCO80 (extreme vocabulary size; `person` de-duplicated)",
        ),
    ]


def all_conditions() -> tuple[Condition, ...]:
    """The frozen 32 conditions, axis A then B then C."""
    return tuple([*_axis_a(), *_axis_b(), *_axis_c()])


#: Expected per-axis counts from spec §5. Asserted by the test suite.
EXPECTED_AXIS_COUNTS: Mapping[str, int] = {"A": 18, "B": 9, "C": 5}
EXPECTED_TOTAL_CONDITIONS: int = 32


#: The frozen condition whose term order the permutation control reverses.
PERMUTATION_SOURCE_ID: str = "A2_pair_person_clothes"


def all_control_conditions() -> tuple[Condition, ...]:
    """Controls: runnable, but deliberately NOT part of the frozen 32.

    A control measures the harness, not the model, so it does not belong in the
    spec §5 results table. Putting ``A2_pair_clothes_person`` into
    :func:`all_conditions` would make that table 33 rows with axis A at 19 and
    would quietly redefine every "all conditions" aggregate that has already
    been computed. It lives here instead: :func:`get_condition` resolves it, so
    the runner can run it, and no aggregate over the 32 sees it.

    ``A2_pair_clothes_person`` is ``A2_pair_person_clothes`` with the two terms
    swapped. It is the permutation noise floor: for a dual-encoder that scores
    each query channel independently (OWLv2) the per-channel logits must be
    exactly equal between the two orders, so any non-zero difference is a
    harness bug rather than a model property. For a fusion model (Grounding
    DINO), where the terms share one caption, a non-zero difference IS a model
    property and is the quantity of interest.
    """
    # Derived from the frozen condition rather than retyped, so the control can
    # never drift into being a permutation of something else.
    source = conditions_by_id()[PERMUTATION_SOURCE_ID]
    terms = tuple(reversed(source.terms))
    return (
        Condition(
            condition_id="A2_pair_clothes_person",
            axis="A", subgroup="A2_person_pair_permutation_control", terms=terms,
            term_to_gt_class=_identity(terms),
            vocabulary_status=IN_VOCABULARY,
            description="permutation control: A2_pair_person_clothes with the "
                        "two terms swapped (order noise floor)",
        ),
    )


def conditions_by_id() -> Mapping[str, Condition]:
    """Exact-key lookup. Never select a condition by prefix or first match."""
    return {c.condition_id: c for c in all_conditions()}


def control_conditions_by_id() -> Mapping[str, Condition]:
    return {c.condition_id: c for c in all_control_conditions()}


def get_condition(condition_id: str) -> Condition:
    """Exact-match lookup over the frozen 32 **and** the controls.

    Controls resolve here so the runner needs no special case, but they are
    absent from :func:`all_conditions` and therefore from every table built
    over it. An id collision between the two registries is impossible by
    construction and is asserted by the test suite.
    """
    table = conditions_by_id()
    if condition_id in table:
        return table[condition_id]
    return control_conditions_by_id()[condition_id]


def get_control_condition(condition_id: str) -> Condition:
    """Exact-match lookup restricted to the controls."""
    return control_conditions_by_id()[condition_id]
