"""Belongings-only 2-element conditions — the person-free competition axis.

Rationale and the pre-registered selection rule: the belongings-pair note.
Measured results: the v6_belongings_target_flip.md note.

Why this axis has to exist
--------------------------
Every 2-element condition in the frozen 32 contains ``person``
(``A2_pair_person_*``), and so does every 3-element condition in
:mod:`utils.protocol.conditions_triples` (``A5_triple_person_*``). Consequently there
is **no measurement anywhere in the campaign of two belongings competing with
each other**. When ``AP_clothes`` moves between ``clothes .`` and
``person . clothes . hat .`` the design cannot say whether ``person`` or ``hat``
did it, because the two never arrive separately.

These conditions supply the missing contrast::

    q0 = A1_singleton_clothes        `clothes .`
    q1 = A6_pair_clothes_hat         `clothes . hat .`      <- no person

Read against ``A2_pair_person_clothes`` (person arrives, no second belongings)
and ``A5_triple_person_clothes_hat`` (both arrive), the three isolate the two
terms and their interaction.

Why a separate module (do not move these into ``conditions.py``)
----------------------------------------------------------------
``utils/runner/runner_version.py`` lists ``conditions.py`` in ``CORE_MODULES``, so
editing it changes the ``runner_version`` fingerprint of all four models and
puts every fragment written so far into a different generation. This module is
**not** in ``CORE_MODULES`` — which is an explicit tuple, not a glob — so adding
it moves no fingerprint. :mod:`utils.protocol.conditions_triples` was created for
exactly this reason and this module follows it verbatim: it imports the
:class:`~utils.protocol.conditions.Condition` dataclass and the same caption renderer,
so a caption produced here obeys the same convention as the frozen 32 and a
fragment written from either is directly comparable. ``conditions.py`` still
returns exactly 32 from ``all_conditions``.

Selection rule (fixed before any run; see the doc)
--------------------------------------------------
1. **Every** 2-element subset of the five belongings, rendered in the frozen V6
   order. ``C(5, 2) = 10``. This is a complete enumeration, not a sample, so it
   cannot be cherry-picked after seeing results.
2. **Two order-only permutation controls**, chosen by a rule that reads only the
   ground-truth counts of ``datasets/v6_fp_fb/test.json`` — a fixed property of
   the data that no model measurement can influence: the pair with the largest
   ``n(X)·n(Y)`` and the pair with the smallest. With
   ``clothes 3228, bag 701, shoes 696, hat 600, gloves 498`` that is
   ``clothes+bag`` (2,262,828) and ``hat+gloves`` (298,800). See
   :data:`PERMUTATION_CONTROL_PAIRS`.

Two controls rather than one (the person axis had one,
``A2_pair_clothes_person``) because a single control cannot distinguish "the
order-noise floor is a property of the model" from "this particular pair happens
to be order-stable", and because bracketing by ground-truth size puts one
control where a single flipped object weighs least (clothes+bag) and one where
it weighs most (hat+gloves: 1/498 = 0.201 pp). Two controls rather than ten
because the floor was already measured to be a model-level constant on the
person axis (OWLv2 exactly 0, GDINO 1e-5, GLIP 1e-3, OmDet 1e-2) and eight more
would cost roughly another 105 GPU-hours to re-establish that.
"""
from __future__ import annotations

from itertools import combinations

from utils.protocol.conditions import IN_VOCABULARY, Condition
from utils.protocol.vocabulary import BELONGINGS_CLASSES, PERSON_CLASS, V6_CLASSES

#: Subgroup label. The axis stays ``"A"`` — this is the same interference axis
#: as A1/A2/A5, with ``person`` removed from the term list rather than a new
#: kind of question. ``A6`` because ``A5`` is the triples.
PAIR_SUBGROUP = "A6_belongings_pair"

#: C(5, 2) = 10. Asserted by the test suite so a vocabulary change is loud.
EXPECTED_PAIR_CONDITIONS: int = 10

#: The permutation controls, as unordered pairs in frozen V6 order. Selected by
#: the ground-truth-size rule in the module docstring, which is evaluated
#: against the census and never against a result.
PERMUTATION_CONTROL_PAIRS: tuple[tuple[str, str], ...] = (
    ("clothes", "bag"),    # largest  n(X)*n(Y) = 3228 * 701
    ("hat", "gloves"),     # smallest n(X)*n(Y) =  600 * 498
)

#: 10 forward + 2 reversed. Asserted by the test suite.
EXPECTED_TOTAL_CONDITIONS: int = EXPECTED_PAIR_CONDITIONS + len(PERMUTATION_CONTROL_PAIRS)


def _v6_order(names: tuple[str, ...]) -> tuple[str, ...]:
    """Sort ``names`` into the frozen V6 order (spec §1)."""
    return tuple(c for c in V6_CLASSES if c in names)


def _condition(terms: tuple[str, ...], description: str) -> Condition:
    if PERSON_CLASS in terms:
        raise ValueError(
            f"{terms} contains {PERSON_CLASS!r}; this axis exists precisely to "
            "hold a pair with person ABSENT, and a person-containing pair is "
            "already A2_pair_person_*")
    return Condition(
        condition_id=f"A6_pair_{terms[0]}_{terms[1]}",
        axis="A",
        subgroup=PAIR_SUBGROUP,
        terms=terms,
        term_to_gt_class={t: t for t in terms},
        vocabulary_status=IN_VOCABULARY,
        description=description,
    )


def belongings_pair_conditions() -> tuple[Condition, ...]:
    """The 10 person-free belongings pairs, rendered in frozen V6 order.

    Terms are ordered by V6 position, matching how ``A2_pair_*``, ``A3_loo_*``
    and ``A5_triple_*`` order theirs, so a term's position in the caption is
    never a confound between these conditions and the rest of the campaign.
    """
    return tuple(
        _condition(_v6_order((first, second)), f"`{first}` paired with `{second}`")
        for first, second in combinations(BELONGINGS_CLASSES, 2)
    )


def permutation_control_conditions() -> tuple[Condition, ...]:
    """The two order-only controls: the same term multiset, reversed.

    A control and its partner differ in **word order only**, which is what makes
    the contrast between them a noise floor rather than a vocabulary effect.
    ``analyze_v6_query_sensitivity.py::permuted_partner`` (research tree) pairs them by
    term multiset, so the naming here is presentational and the pairing is not.
    """
    out: list[Condition] = []
    for first, second in PERMUTATION_CONTROL_PAIRS:
        forward = _v6_order((first, second))
        reverse = (forward[1], forward[0])
        out.append(_condition(
            reverse,
            f"order-only control: `{reverse[0]}` before `{reverse[1]}` "
            f"(reverse of A6_pair_{forward[0]}_{forward[1]})"))
    return tuple(out)


def all_belongings_pair_conditions() -> tuple[Condition, ...]:
    """The 10 pairs followed by the 2 permutation controls, in a stable order."""
    return belongings_pair_conditions() + permutation_control_conditions()


def belongings_pair_conditions_by_id() -> dict[str, Condition]:
    """Exact-key lookup. Never select a condition by prefix or first match."""
    table: dict[str, Condition] = {}
    for condition in all_belongings_pair_conditions():
        if condition.condition_id in table:
            raise ValueError(f"duplicate condition id {condition.condition_id!r}")
        table[condition.condition_id] = condition
    return table


def get_belongings_pair_condition(condition_id: str) -> Condition:
    """Exact-match lookup; raises :class:`KeyError` on an unknown id."""
    return belongings_pair_conditions_by_id()[condition_id]


def permuted_control_of(condition: Condition) -> Condition:
    """The order-only control for ``condition``, or raise if it has none.

    Only the two pairs named in :data:`PERMUTATION_CONTROL_PAIRS` have one; the
    other eight report ``S_norm`` as ``_unmeasured_`` rather than borrowing a
    floor measured on a different term multiset.
    """
    if condition.subgroup != PAIR_SUBGROUP:
        raise ValueError(f"{condition.condition_id} is not a belongings-pair condition")
    reverse = (condition.terms[1], condition.terms[0])
    control_id = f"A6_pair_{reverse[0]}_{reverse[1]}"
    table = belongings_pair_conditions_by_id()
    if control_id not in table:
        raise KeyError(
            f"{condition.condition_id} has no order-only control; only "
            f"{PERMUTATION_CONTROL_PAIRS} were selected, by the ground-truth-size "
            "rule fixed before any run. Report S_norm as _unmeasured_.")
    return table[control_id]
