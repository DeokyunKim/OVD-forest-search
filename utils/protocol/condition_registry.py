"""Resolve a condition id against every registry that defines one.

The campaign's conditions live in three modules, and which one owns an id is
not visible from the id:

===========================  =====================================
``utils.protocol.conditions``       the frozen 32 (A1–A4, B, C)
``utils.protocol.conditions_triples``            A5 3-element subsets
``utils.protocol.conditions_belongings_pairs``   A6 belongings pairs
=====================================================================

The split exists because ``conditions.py`` is inside the ``runner_version``
fingerprint (``runner_version.CORE_MODULES``), so adding a condition there
would change the fingerprint of every model and orphan the fragments already
on disk. Each later axis therefore got its own module — correct, but it means
any code that resolves an id through **one** registry silently fails on the
other two.

That has now happened twice: the campaign allowlist parser rejected every A5
id, and ``analyze_v6_query_sensitivity.py::permuted_partner`` (research tree) raised
``KeyError: 'A6_pair_bag_clothes'`` for 64 of 320 runs — losing exactly the
permutation controls, which are the noise floor the other results are read
against. Two occurrences of one bug is a missing abstraction, so the lookup
lives here once instead of being inlined a third time.

Not in the fingerprint: ``CORE_MODULES`` is an explicit tuple of file names,
not a glob, and this file is not in it.
"""
from __future__ import annotations

from typing import Callable, Iterable

__all__ = ["ALL_LOOKUPS", "resolve_condition", "known_condition_ids"]


def _lookups() -> tuple[Callable[[str], object], ...]:
    """The registry accessors, imported lazily.

    Lazily because the A5/A6 modules import the ``Condition`` dataclass from
    ``conditions``; importing them at module scope would make this module's
    import order matter to callers that only need the frozen 32.
    """
    from utils.protocol.conditions import get_condition
    from utils.protocol.conditions_belongings_pairs import get_belongings_pair_condition
    from utils.protocol.conditions_triples import get_triple_condition

    return (get_condition, get_triple_condition, get_belongings_pair_condition)


#: Exposed so a caller can report *which* registries were searched.
ALL_LOOKUPS = _lookups


def resolve_condition(condition_id: str):
    """The :class:`~utils.protocol.conditions.Condition` for ``condition_id``.

    Raises:
        KeyError: no registry defines it — and the message names every
            registry that was tried, because "unknown condition" without that
            list sends the reader to the wrong module.
    """
    for lookup in _lookups():
        try:
            return lookup(condition_id)
        except (KeyError, ValueError):
            continue
    raise KeyError(
        f"{condition_id!r} is defined by none of utils.protocol.conditions, "
        f"utils.protocol.conditions_triples, utils.protocol.conditions_belongings_pairs"
    )


def known_condition_ids() -> frozenset[str]:
    """Every id any registry defines — for validating a list before running it.

    ``conditions_triples`` exposes no bulk accessor, only
    :func:`~utils.protocol.conditions_triples.triple_conditions`, so it is named
    explicitly rather than discovered: a silent ``except`` here would drop a
    whole axis from the "known" set and turn a typo into a missing condition.
    """
    from utils.protocol.conditions import all_conditions, all_control_conditions
    from utils.protocol.conditions_belongings_pairs import all_belongings_pair_conditions
    from utils.runner import conditions_triples

    triples = getattr(conditions_triples, "triple_conditions", None) \
        or getattr(conditions_triples, "all_triple_conditions", None)
    if triples is None:  # pragma: no cover — the module always defines one
        raise AttributeError(
            "utils.protocol.conditions_triples exposes no bulk accessor; "
            "known_condition_ids() would silently omit the A5 axis")

    ids: set[str] = set()
    for source in (all_conditions, all_control_conditions,
                   triples, all_belongings_pair_conditions):
        ids.update(c.condition_id for c in source())
    return frozenset(ids)
