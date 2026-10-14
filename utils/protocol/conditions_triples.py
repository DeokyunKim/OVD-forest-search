"""Axis-A 3-element subsets — the spec's conditional extra, in its own module.

Spec: the vocabulary specification §2, rationale and pre-registered
selection rule in the query-axis note.

    불성립 -> 고차 상호작용 존재. 그 자체가 발견이며, 이때만 3원소
    부분집합 일부(10개)를 추가로 돌린다.

These 10 conditions are **not** part of the frozen 32. ``conditions.py`` still
returns exactly 32 from :func:`~utils.protocol.conditions.all_conditions`, and this
module never touches it — it only reuses its :class:`~utils.protocol.conditions.Condition`
dataclass and the same caption renderer, so the query strings obey the same
convention as the 32 and a fragment written from either is directly comparable.

Selection rule (fixed before any run, see the doc)
--------------------------------------------------
Every 3-element subset of V6 that contains ``person``. Since V6 is ``person``
plus five belongings, that is ``C(5, 2) = 10`` — exactly the count the spec
names. **There is no freedom in the selection**: it is a complete enumeration,
not a sample, so it cannot be cherry-picked after seeing results.

``person`` must be in every subset because ``I(c)``, ``M(c)`` and the
additivity identity are all defined on ``AP_person``; a subset without
``person`` has no value on that axis.

What they test
--------------
If first-order additivity held at order 3::

    I(c1, c2) = AP_person(singleton) - AP_person(person + c1 + c2)
              ?= I(c1) + I(c2)

The gap between the two sides is the **second-order interaction term**. A large
negative gap means the first added term already produces most of the effect and
further terms saturate — which is what GLIP's
``sum_c I(c) = -0.4431`` against a true total of ``-0.0815`` points to.
"""
from __future__ import annotations

from itertools import combinations

from utils.protocol.conditions import IN_VOCABULARY, Condition
from utils.protocol.vocabulary import BELONGINGS_CLASSES, PERSON_CLASS, V6_CLASSES

#: Subgroup label; the axis stays ``"A"`` so the extra conditions are visibly
#: part of the interference axis rather than a fourth axis.
TRIPLE_SUBGROUP = "A5_triple"

#: C(5, 2) = 10. Asserted by the test suite so a vocabulary change is loud.
EXPECTED_TRIPLE_CONDITIONS: int = 10


def _v6_order(names: tuple[str, ...]) -> tuple[str, ...]:
    """Sort ``names`` into the frozen V6 order (spec §1)."""
    return tuple(c for c in V6_CLASSES if c in names)


def triple_conditions() -> tuple[Condition, ...]:
    """The 10 person-containing 3-element subsets, in a stable order.

    Terms are rendered in V6 order, matching how ``A2_pair_*`` and ``A3_loo_*``
    order theirs, so a term's position in the caption is never a confound
    between these conditions and the frozen 32.
    """
    out: list[Condition] = []
    for first, second in combinations(BELONGINGS_CLASSES, 2):
        terms = _v6_order((PERSON_CLASS, first, second))
        out.append(Condition(
            condition_id=f"{TRIPLE_SUBGROUP}_person_{first}_{second}",
            axis="A",
            subgroup=TRIPLE_SUBGROUP,
            terms=terms,
            term_to_gt_class={t: t for t in terms},
            vocabulary_status=IN_VOCABULARY,
            description=f"person with `{first}` and `{second}`",
        ))
    return tuple(out)


def triple_conditions_by_id() -> dict[str, Condition]:
    """Exact-key lookup. Never select a condition by prefix or first match."""
    return {c.condition_id: c for c in triple_conditions()}


def get_triple_condition(condition_id: str) -> Condition:
    """Exact-match lookup; raises :class:`KeyError` on an unknown id."""
    return triple_conditions_by_id()[condition_id]


def belongings_pair(condition: Condition) -> tuple[str, str]:
    """The two belongings in a triple, in V6 order.

    Used by the analysis to pair ``I(c1, c2)`` against ``I(c1) + I(c2)``.
    """
    if condition.subgroup != TRIPLE_SUBGROUP:
        raise ValueError(f"{condition.condition_id} is not a triple condition")
    pair = tuple(t for t in condition.terms if t != PERSON_CLASS)
    if len(pair) != 2:
        raise ValueError(f"{condition.condition_id} does not hold exactly two belongings")
    return pair  # type: ignore[return-value]
