"""Prompt-invariance metrics — the SINGLE implementation (schema revision 3).

FILE NAME: this module is still called ``rev2_metrics.py`` although it now
implements schema **revision 3** (the 12-rung ladder). The name is deliberately
frozen: it is imported by both repos and pinned by a twin byte-identity test,
and renaming it on every schema revision would break that synchronisation for
no benefit. The schema revision lives in the data (``schema_revision``), not in
the filename.

``outputs/unified_bench/prompt_invariance_SCHEMA.md`` revision 3 defines four
prediction-stability metrics (``pi_f1``, ``count_cv``, ``loc_jitter``,
``empty_fp_spread``) and requires that families A, B and C emit fragments that
"concatenate into one table without translation". That guarantee is only worth
anything if the numbers are produced by the SAME code: two independent
implementations of "greedy match at IoU>=0.5" or "both-empty contributes 1.0"
would differ subtly, and the difference would be read as a difference BETWEEN
FAMILIES -- the exact confound the schema exists to prevent. This module is
therefore the only place these metrics are computed; every family-specific
script is a thin wrapper that supplies dumps and naming.

This file is maintained as a VERBATIM TWIN at
``the RT-OWL research tree`` (the two repos do
not import across each other; the twin convention already exists for
``testset_metrics.py`` and ``vocab.py``). the regression tests pins
their byte-identity, so an edit to one that is not mirrored fails CI.

Dump contract (identical for every family)
------------------------------------------
``dumps[image_id][prompt_id]`` is an ``(n, 6)`` float array whose columns are
``[x1, y1, x2, y2, score, class_index]``, with ``class_index`` indexing into
``vocab``. Revision 1 dumped only the person channel; revision 2 scores all
six classes, so every class must be present in the dump.

Two conventions inherited from the family-B implementation, applied to every
family so the fragments stay comparable:

1. **Which variants enter a class's comparison.** A prompt that never asked for
   ``gloves`` cannot detect gloves, so pairing it against one that did would
   measure the vocabulary's presence rather than the model's stability. Each
   class is compared only over the variants whose query list CONTAINS it;
   ``n_variants``/``n_pairs`` is reported in every cell, and a class appearing
   in fewer than two variants is ``null`` (the schema's rule that null marks an
   undefined quantity, never 0.0).
2. **Degeneracy is measured, never assumed.** ``duplicate_score_rate`` counts
   the share of detections whose same-box group ties on score across classes.
   An arm whose phrase scores have collapsed scores perfectly on all four
   metrics while being useless, so ``degenerate_phrase_scores`` is derived from
   this measurement and must never be hard-coded.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

#: Box-matching IoU between two prompt variants (schema "Scoring conventions").
IOU_MATCH = 0.5
#: Primary score floor plus the two deployment operating points.
THRESHOLDS: tuple[float, ...] = (0.05, 0.25, 0.45)
#: Images sampled when estimating the duplicate-score rate (a degeneracy probe,
#: not a reported metric, so a cap is acceptable where the census is not).
DUP_RATE_MAX_IMAGES = 3000

#: Source-set labels carried by each pool record.
SOURCE_FP, SOURCE_FB, SOURCE_FPIR = "fp_test", "fb_test", "fpir_test"
#: Source sets whose sensor modality is infrared. Pooling one of these with a
#: visible-light set confounds domain shift with class confusion.
IR_SOURCE_SETS = frozenset({SOURCE_FPIR})
#: Size of the withdrawn three-way (FP+FPIR+FB) pool. Any aggregation landing
#: on exactly this count has silently merged everything -- the failure this
#: project has already been burned by -- so it is rejected by image count as
#: well as by source composition.
FORBIDDEN_THREE_WAY_N_IMAGES = 27171


#: The revision-3 ladder: 12 rungs. Rungs 1-5 are UNCHANGED from revision 2
#: (captions and ids), so revision-2 dumps stay valid and only the seven new
#: rungs need inferring. Every class now appears in at least four variants, so
#: the "compare a class only across variants containing it" rule leaves every
#: class with >= 6 pairs -- under revision 2, gloves/bag/shoes appeared in one
#: variant each and could not be measured at all.
LADDER_V3: dict[str, tuple[str, ...]] = {
    "p1_person": ("person",),
    "p2_plus_clothes": ("person", "clothes"),
    "p3_plus_hat": ("person", "clothes", "hat"),
    "p4_full6": ("person", "clothes", "hat", "gloves", "bag", "shoes"),
    "p5_reversed": ("clothes", "person"),
    "s_clothes": ("clothes",),
    "s_hat": ("hat",),
    "s_gloves": ("gloves",),
    "s_bag": ("bag",),
    "s_shoes": ("shoes",),
    "p6_belongings5": ("clothes", "hat", "gloves", "bag", "shoes"),
    "p7_full6_reversed": ("shoes", "bag", "gloves", "hat", "clothes", "person"),
}
#: Revision 4 adds twelve rungs that ISOLATE ONE INDEPENDENT VARIABLE EACH.
#: Revision 3's twelve gave only three clean contrasts (p2/p5, p4/p7, p4/p6);
#: every other comparison moved vocabulary SIZE and IDENTITY together, so
#: "a bigger vocabulary hurts" was strictly "changing size and identity at once
#: hurts". Ids and captions of the first twelve are UNCHANGED, so every dump
#: already collected stays valid and only the new twelve need inferring.
#:
#: Block B -- nested size chain. With p1/p2/p3/p4 this makes sizes 1..6 a
#: STRICT INCLUSION chain, so size is the only thing that moves along it.
LADDER_BLOCK_B: dict[str, tuple[str, ...]] = {
    "n4": ("person", "clothes", "hat", "gloves"),
    "n5": ("person", "clothes", "hat", "gloves", "bag"),
}
#: Block C -- size held at 2, only the PARTNER changes. With p2 this gives five
#: points at identical size. Any variation in person AP across them is a pure
#: identity effect, which revision 3 could not separate from size at all.
LADDER_BLOCK_C: dict[str, tuple[str, ...]] = {
    "c_hat": ("person", "hat"),
    "c_gloves": ("person", "gloves"),
    "c_bag": ("person", "bag"),
    "c_shoes": ("person", "shoes"),
}
#: Block D -- all 3! permutations of ONE fixed vocabulary {person, clothes,
#: bag}. 15 unordered pairs in which ORDER is the only difference. Revision 3's
#: two order contrasts were both full reversals, a special case of the
#: permutation space; this covers it exhaustively at size 3.
#: An AP50 spread across these six is a PURE ORDER EFFECT: family B encodes
#: queries independently and must score exactly 0, family A encodes one joint
#: caption through BERT (which applies positional encoding across the whole
#: string) and need not.
LADDER_BLOCK_D: dict[str, tuple[str, ...]] = {
    "d_pcb": ("person", "clothes", "bag"),
    "d_pbc": ("person", "bag", "clothes"),
    "d_cpb": ("clothes", "person", "bag"),
    "d_cbp": ("clothes", "bag", "person"),
    "d_bpc": ("bag", "person", "clothes"),
    "d_bcp": ("bag", "clothes", "person"),
}
#: Block E -- the REVERSE size chain, which removes the ORDER CONFOUND left in
#: Block B. Block B grows `person . clothes . hat . gloves . bag . shoes .` by
#: appending one word at a time, so `person` is always the FIRST word and every
#: addition lands LAST. What that chain calls a "vocabulary-size effect" is
#: therefore size entangled with position. Block E stacks the SAME six words in
#: exactly the reverse priority order -- `shoes` heads every caption and
#: `person` is the last word added -- so the size series 1..6 is identical and
#: only the positions are mirrored. Comparing the two curves at matched size is
#: what quantifies the confound.
#:
#: BOTH ENDPOINTS ARE SHARED with the existing ladder and are NOT re-measured:
#: r1 == `s_shoes` (("shoes",)) and r6 == `p7_full6_reversed`
#: (("shoes","bag","gloves","hat","clothes","person")), the exact reversal of
#: Block B's `p4_full6`. Only the four interior rungs are new.
LADDER_BLOCK_E: dict[str, tuple[str, ...]] = {
    "r2": ("shoes", "bag"),
    "r3": ("shoes", "bag", "gloves"),
    "r4": ("shoes", "bag", "gloves", "hat"),
    "r5": ("shoes", "bag", "gloves", "hat", "clothes"),
}
#: The twelve added by revision 4.
RUNGS_NEW_V4: tuple[str, ...] = (
    tuple(LADDER_BLOCK_B) + tuple(LADDER_BLOCK_C) + tuple(LADDER_BLOCK_D))
#: The four added by revision 5 -- the only ones needing new inference.
RUNGS_NEW_V5: tuple[str, ...] = tuple(LADDER_BLOCK_E)

#: Rungs already produced under revision 2; kept so a v2 dump can be reused.
RUNGS_REV2: tuple[str, ...] = ("p1_person", "p2_plus_clothes", "p3_plus_hat",
                               "p4_full6", "p5_reversed")
#: The seven rungs added by revision 3 -- the only ones needing new inference.
RUNGS_NEW_V3: tuple[str, ...] = tuple(r for r in LADDER_V3 if r not in RUNGS_REV2)
#: Order-only contrasts: same vocabulary, permuted position, at two lengths.
ORDER_PAIRS_V3: dict[str, tuple[str, str]] = {
    "p2_vs_p5": ("p2_plus_clothes", "p5_reversed"),
    "p4_vs_p7": ("p4_full6", "p7_full6_reversed"),
}

#: Every unordered pair of Block D: C(6,2) = 15 order-only contrasts at size 3.
ORDER_PAIRS_BLOCK_D: dict[str, tuple[str, str]] = {
    f"{a}_vs_{b}": (a, b)
    for i, a in enumerate(LADDER_BLOCK_D)
    for b in list(LADDER_BLOCK_D)[i + 1:]
}
#: All order contrasts reported by a revision-4 fragment: 2 + 15 = 17.
ORDER_PAIRS_V4: dict[str, tuple[str, str]] = {**ORDER_PAIRS_V3, **ORDER_PAIRS_BLOCK_D}

#: The size chain (strict inclusion) and the fixed-size identity set, named so
#: the block analyses cannot drift from the ladder definition.
SIZE_CHAIN_V4: tuple[str, ...] = ("p1_person", "p2_plus_clothes", "p3_plus_hat",
                                  "n4", "n5", "p4_full6")
IDENTITY_SET_V4: tuple[str, ...] = ("p2_plus_clothes", "c_hat", "c_gloves",
                                    "c_bag", "c_shoes")
#: The full revision-4 ladder: the original twelve (ids and captions frozen)
#: plus the twelve isolating blocks. 24 rungs.
LADDER_V4: dict[str, tuple[str, ...]] = {
    **LADDER_V3, **LADDER_BLOCK_B, **LADDER_BLOCK_C, **LADDER_BLOCK_D}

#: The REVERSE size chain, mirror image of :data:`SIZE_CHAIN_V4`. Its first and
#: last rungs are existing ladder members, so the chain costs four new rungs.
#: Sizes run 1..6 exactly as the forward chain does.
REVERSE_CHAIN_V5: tuple[str, ...] = ("s_shoes", "r2", "r3", "r4", "r5",
                                     "p7_full6_reversed")
#: Revision 5 = revision 4 + Block E. Every revision-4 rung id and caption is
#: unchanged, so a v4 census plus a four-rung top-up reconstructs a v5 arm.
LADDER_V5: dict[str, tuple[str, ...]] = {**LADDER_V4, **LADDER_BLOCK_E}
#: Order-only contrasts are unchanged by revision 5: Block E moves size, not
#: order, within its own chain. Aliased rather than redefined so the two
#: revisions cannot drift.
ORDER_PAIRS_V5: dict[str, tuple[str, str]] = dict(ORDER_PAIRS_V4)

#: Minimum variants a class must appear in for its pairwise metrics to exist.
MIN_VARIANTS_PER_CLASS = 4


class InvalidEvaluationUnitError(ValueError):
    """Raised when an aggregation unit violates the evaluation design."""


class IncompleteLadderError(ValueError):
    """Raised when a fragment is built from fewer than the full 12 rungs.

    Pairwise metrics are averaged over the pairs a class participates in, so a
    partial ladder changes the pair SET and the resulting numbers are not
    comparable with a full-ladder arm. Ranked at the same severity as the
    census gate for exactly that reason.
    """


def validate_ladder(present: Sequence[str], required: Mapping[str, tuple[str, ...]] = LADDER_V3
                    ) -> None:
    """Fail unless every required rung is present in the dumps."""
    missing = [r for r in required if r not in set(present)]
    unexpected = [r for r in present if r not in required]
    if missing:
        raise IncompleteLadderError(
            f"ladder incomplete: missing {missing} (have {sorted(present)}). Pairwise metrics "
            "computed on a partial ladder use a different pair set and cannot be compared "
            "across arms; finish inferring the remaining rungs first."
        )
    if unexpected:
        raise IncompleteLadderError(
            f"ladder contains unknown rungs {unexpected}; refusing to mix ladders."
        )


@dataclass(frozen=True)
class EvalUnit:
    """One aggregation unit: a named pooling of pool source sets.

    The evaluation design is fixed (schema rev 2, "Evaluation sets"):
    ``fp_fb_test`` (ForestPersons + ForestBelongings, 14,980) is the PRIMARY
    reporting unit -- the merge is deliberate, because it is what exposes
    person<->belongings confusion -- and ``fpir_test`` (12,191) is scored
    separately because it is a different sensor. Per-source splits are
    diagnostics only, never a headline.
    """

    unit_id: str
    source_sets: tuple[str, ...]
    primary: bool = False


#: Top-level reporting units. These, and only these, appear in ``sets``.
DEFAULT_EVAL_UNITS: tuple[EvalUnit, ...] = (
    EvalUnit("fp_fb_test", (SOURCE_FP, SOURCE_FB), primary=True),
    EvalUnit("fpir_test", (SOURCE_FPIR,)),
)
#: Reported under ``diagnostics``; may never be quoted as a headline number.
DEFAULT_DIAGNOSTIC_UNITS: tuple[EvalUnit, ...] = (
    EvalUnit(SOURCE_FP, (SOURCE_FP,)),
    EvalUnit(SOURCE_FB, (SOURCE_FB,)),
)

#: Full-census image count of each reporting unit. Sampling is prohibited, so a
#: unit that does not hit its count exactly is an incomplete run, not a smaller
#: experiment. A stratified 4,000-image sample once nearly reversed a
#: conclusion on this project, which is why this is a hard gate.
CENSUS_N_IMAGES: dict[str, int] = {"fp_fb_test": 14980, "fpir_test": 12191}


class IncompleteCensusError(ValueError):
    """Raised when a reporting unit does not cover its full census."""


def validate_eval_unit(unit: EvalUnit, n_images: int) -> None:
    """Reject any unit that pools across sensor modalities or the whole census.

    Enforced in code rather than left to convention: aggregating the full
    27,171-image pool as one block is a silent, plausible-looking mistake that
    produces numbers nobody can tell are wrong by inspection.
    """
    sources = set(unit.source_sets)
    if not sources:
        raise InvalidEvaluationUnitError(f"{unit.unit_id!r}: no source sets")
    ir = sources & IR_SOURCE_SETS
    if ir and sources - IR_SOURCE_SETS:
        raise InvalidEvaluationUnitError(
            f"{unit.unit_id!r} pools IR {sorted(ir)} with visible-light "
            f"{sorted(sources - IR_SOURCE_SETS)}; this confounds domain shift with "
            "person<->belongings class confusion and is not a valid unit"
        )
    if n_images == FORBIDDEN_THREE_WAY_N_IMAGES:
        raise InvalidEvaluationUnitError(
            f"{unit.unit_id!r} aggregated {n_images} images, the size of the withdrawn "
            "three-way FP+FPIR+FB pool; report fp_fb_test (14,980) and fpir_test (12,191) "
            "separately instead"
        )


def validate_census(unit: EvalUnit, n_images: int, strict: bool = True) -> None:
    """Fail unless ``unit`` covers its full census (``strict=False`` to screen)."""
    expected = CENSUS_N_IMAGES.get(unit.unit_id)
    if expected is None or not strict:
        return
    if n_images != expected:
        raise IncompleteCensusError(
            f"{unit.unit_id!r}: {n_images} images dumped, census requires exactly {expected}. "
            "Sampling is prohibited for reported numbers; finish the inference run "
            "(or pass strict=False for a screening pass that must never be published)."
        )


@dataclass(frozen=True)
class FragmentSpec:
    """Naming/provenance for one fragment; metrics are identical across specs."""

    arm: str
    family: str
    model: str
    vocab: tuple[str, ...]
    prompts: Mapping[str, Sequence[str]]
    #: Order-only contrasts, keyed by name. Revision 3 has TWO: ``p2_vs_p5``
    #: at two words and ``p4_vs_p7`` at six, so an order effect can be seen at
    #: two different caption lengths.
    order_pairs: Mapping[str, tuple[str, str]]
    full_prompt_id: str
    units: tuple[EvalUnit, ...] = DEFAULT_EVAL_UNITS
    diagnostic_units: tuple[EvalUnit, ...] = DEFAULT_DIAGNOSTIC_UNITS
    extra: Mapping[str, Any] = field(default_factory=dict)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two arrays of ``xyxy`` boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    ar_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ar_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(ar_a[:, None] + ar_b[None, :] - inter, 1e-9)


def greedy_match(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Greedy by descending score of ``a``, IoU >= :data:`IOU_MATCH`, one-to-one."""
    if len(a) == 0 or len(b) == 0:
        return []
    ious = iou_matrix(a[:, :4], b[:, :4])
    taken: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for ai in np.argsort(-a[:, 4]):
        row = ious[ai].copy()
        for bj in taken:
            row[bj] = -1.0
        bj = int(np.argmax(row))
        if row[bj] >= IOU_MATCH:
            taken.add(bj)
            pairs.append((int(ai), bj))
    return pairs


def class_detections(dets: np.ndarray, class_index: int, threshold: float) -> np.ndarray:
    """Rows of ``dets`` for one class at or above one score threshold."""
    if not len(dets):
        return dets.reshape(-1, 6)
    return dets[(dets[:, 5] == class_index) & (dets[:, 4] >= threshold)].reshape(-1, 6)


def variants_for_class(prompts: Mapping[str, Sequence[str]], cls: str) -> list[str]:
    """Prompt ids whose query list contains ``cls`` (see convention 1)."""
    return [pid for pid, words in prompts.items() if cls in words]


def stability_cell(ids: Sequence[int], dumps: Mapping[int, Mapping[str, np.ndarray]],
                   cls: str, threshold: float, spec: FragmentSpec) -> dict[str, Any]:
    """``pi_f1`` / ``count_cv`` / ``loc_jitter`` for one (set, class, threshold)."""
    ci = spec.vocab.index(cls)
    variants = variants_for_class(spec.prompts, cls)
    pairs = list(itertools.combinations(variants, 2))
    if len(variants) < 2:
        return {
            "pi_f1": {"value": None, "n_images": 0, "n_pairs_both_empty": 0},
            "count_cv": {"value": None, "n_images": 0, "n_images_all_empty": 0},
            "loc_jitter": {"value": None, "n_matched_pairs": 0},
            "n_variants": len(variants), "n_pairs": 0,
            "note": "class appears in fewer than two prompt variants; undefined",
        }

    f1_per_image: list[float] = []
    both_empty = 0
    cvs: list[float] = []
    all_empty = 0
    iou_sum = 0.0
    iou_n = 0
    for i in ids:
        per_variant = {p: class_detections(dumps[i][p], ci, threshold) for p in variants}
        f1s: list[float] = []
        for p, q in pairs:
            a, b = per_variant[p], per_variant[q]
            if len(a) == 0 and len(b) == 0:
                # Both prompts agreed there is nothing. Counted so this
                # convention -- what makes a silent model look invariant --
                # can be audited from the fragment.
                f1s.append(1.0)
                both_empty += 1
                continue
            if len(a) == 0 or len(b) == 0:
                f1s.append(0.0)
                continue
            matched = greedy_match(a, b)
            f1s.append(2 * len(matched) / (len(a) + len(b)))
            for ai, bj in matched:
                iou_sum += float(iou_matrix(a[ai:ai + 1, :4], b[bj:bj + 1, :4])[0, 0])
                iou_n += 1
        f1_per_image.append(float(np.mean(f1s)))
        counts = np.array([len(per_variant[p]) for p in variants], dtype=float)
        if counts.mean() == 0:
            all_empty += 1  # excluded from count_cv; a 0/0 CV is undefined
        else:
            cvs.append(float(counts.std() / counts.mean()))
    return {
        "pi_f1": {"value": float(np.mean(f1_per_image)) if f1_per_image else None,
                  "n_images": len(f1_per_image), "n_pairs_both_empty": both_empty},
        "count_cv": {"value": float(np.mean(cvs)) if cvs else None,
                     "n_images": len(cvs), "n_images_all_empty": all_empty},
        # micro over matched pairs, NOT a per-image macro: a one-box image must
        # not weigh the same as a thirty-box image.
        "loc_jitter": {"value": (iou_sum / iou_n) if iou_n else None, "n_matched_pairs": iou_n},
        "n_variants": len(variants), "n_pairs": len(pairs),
    }


def macro_over_classes(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Macro of each metric over the classes where it is defined."""
    out: dict[str, Any] = {}
    for metric in ("pi_f1", "count_cv", "loc_jitter"):
        vals = [c[metric]["value"] for c in cells.values() if c[metric]["value"] is not None]
        out[metric] = {"value": float(np.mean(vals)) if vals else None, "n_classes": len(vals)}
    return out


def empty_fp_block(ids_by_class: Mapping[str, Sequence[int]],
                   dumps: Mapping[int, Mapping[str, np.ndarray]], cls: str,
                   threshold: float, spec: FragmentSpec) -> dict[str, Any]:
    """False-positive dispersion across variants on frames with no GT of ``cls``."""
    ci = spec.vocab.index(cls)
    ids = ids_by_class[cls]
    variants = variants_for_class(spec.prompts, cls)
    if not ids or not variants:
        return {"fppi": {"spread": None, "per_variant": {}},
                "frame_rate": {"spread": None, "per_variant": {}},
                "argmax_variant": None, "argmin_variant": None, "n_images": len(ids)}
    fppi: dict[str, float] = {}
    frame_rate: dict[str, float] = {}
    for p in variants:
        n_fp = 0
        n_frames = 0
        for i in ids:
            d = class_detections(dumps[i][p], ci, threshold)
            n_fp += len(d)
            n_frames += 1 if len(d) else 0
        fppi[p] = n_fp / len(ids)
        frame_rate[p] = n_frames / len(ids)
    hi = max(fppi, key=lambda p: fppi[p])
    lo = min(fppi, key=lambda p: fppi[p])
    return {
        "fppi": {"spread": fppi[hi] - fppi[lo], "per_variant": fppi},
        "frame_rate": {"spread": max(frame_rate.values()) - min(frame_rate.values()),
                       "per_variant": frame_rate},
        "argmax_variant": hi, "argmin_variant": lo, "n_images": len(ids),
    }


def duplicate_score_rate(ids: Sequence[int], dumps: Mapping[int, Mapping[str, np.ndarray]],
                         spec: FragmentSpec, max_images: int = DUP_RATE_MAX_IMAGES) -> float:
    """Share of detections whose same-box group ties on score across classes.

    This is the degeneracy probe behind ``degenerate_phrase_scores``. A model
    whose phrase logits have collapsed emits the SAME score for several class
    names on the same box, which is invisible in the stability metrics (it in
    fact maximises them) but obvious here.
    """
    n = 0
    dup = 0
    for i in ids[:max_images]:
        d = dumps[i][spec.full_prompt_id]
        if not len(d):
            continue
        groups: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
        for row in d:
            groups[tuple(np.round(row[:4], 2))].append((int(row[5]), float(row[4])))
        for entries in groups.values():
            by_score: dict[float, set[int]] = defaultdict(set)
            for c, s in entries:
                by_score[round(s, 4)].add(c)
            degenerate = {s for s, classes in by_score.items() if len(classes) >= 2}
            for _c, s in entries:
                n += 1
                if round(s, 4) in degenerate:
                    dup += 1
    return (dup / n) if n else 0.0


def _one_order_contrast(ids: Sequence[int], dumps: Mapping[int, Mapping[str, np.ndarray]],
                        spec: FragmentSpec, pair: tuple[str, str]) -> dict[str, Any]:
    """One order-only contrast for every class; family B must be bit-identical."""
    out: dict[str, Any] = {}
    for cls in spec.vocab:
        ci = spec.vocab.index(cls)
        if not all(cls in spec.prompts[p] for p in pair):
            out[cls] = {"pi_f1": None, "loc_jitter": None, "identical": None}
            continue
        f1s: list[float] = []
        iou_sum = 0.0
        iou_n = 0
        identical = True
        for i in ids:
            a = class_detections(dumps[i][pair[0]], ci, THRESHOLDS[0])
            b = class_detections(dumps[i][pair[1]], ci, THRESHOLDS[0])
            if len(a) == 0 and len(b) == 0:
                f1s.append(1.0)
                continue
            if len(a) != len(b):
                identical = False
            matched = greedy_match(a, b) if len(a) and len(b) else []
            f1s.append(2 * len(matched) / (len(a) + len(b)) if (len(a) + len(b)) else 1.0)
            for ai, bj in matched:
                v = float(iou_matrix(a[ai:ai + 1, :4], b[bj:bj + 1, :4])[0, 0])
                iou_sum += v
                iou_n += 1
                if v < 1.0 - 1e-9:
                    identical = False
        out[cls] = {"pi_f1": float(np.mean(f1s)) if f1s else None,
                    "loc_jitter": (iou_sum / iou_n) if iou_n else None,
                    "identical": identical}
    return out


def order_contrast(ids: Sequence[int], dumps: Mapping[int, Mapping[str, np.ndarray]],
                   spec: FragmentSpec) -> dict[str, Any]:
    """All order-only contrasts, nested ``{class: {pair_name: metrics}}``."""
    per_pair = {name: _one_order_contrast(ids, dumps, spec, pair)
                for name, pair in spec.order_pairs.items()}
    return {cls: {name: per_pair[name][cls] for name in spec.order_pairs}
            for cls in spec.vocab}


class DumpMergeError(ValueError):
    """Raised when dumps from different runs cannot be safely merged."""


def merge_rung_dumps(parts: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Merge per-run dumps into ``{image_id: {rung: dets}}``, checking consistency.

    Each element of ``parts`` is one parsed JSONL row::

        {"image_id": int, "arm": str, "pool": str, "prompts": {rung: dets}}

    A silent merge is this project's single biggest risk: rungs inferred in
    different runs look interchangeable but may come from a different pool, a
    different arm, or a re-run that already wrote the same rung. All four are
    rejected:

    * mixed ``pool`` paths -- the image sets would differ;
    * mixed ``arm`` -- two models' detections would be averaged together;
    * a repeated ``(image_id, rung)`` -- silently keeping one of two values;
    * (checked by the caller) images present for some rungs but not others.
    """
    pools = {str(p.get("pool")) for p in parts if p.get("pool") is not None}
    if len(pools) > 1:
        raise DumpMergeError(f"dumps span multiple pools {sorted(pools)}; image sets differ")
    arms = {str(p.get("arm")) for p in parts if p.get("arm") is not None}
    if len(arms) > 1:
        raise DumpMergeError(f"dumps span multiple arms {sorted(arms)}")

    merged: dict[int, dict[str, Any]] = defaultdict(dict)
    for part in parts:
        image_id = int(part["image_id"])
        for rung, dets in part["prompts"].items():
            if rung in merged[image_id]:
                raise DumpMergeError(
                    f"duplicate (image_id={image_id}, rung={rung!r}) across dump files; "
                    "one value would be silently discarded. Remove the redundant run."
                )
            merged[image_id][rung] = dets
    return dict(merged)


def assert_uniform_rungs(dumps: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Return the rung set, failing if images disagree about which rungs exist."""
    if not dumps:
        raise DumpMergeError("no dumps to analyse")
    reference: set[str] | None = None
    for image_id, per_rung in dumps.items():
        rungs = set(per_rung)
        if reference is None:
            reference = rungs
        elif rungs != reference:
            raise DumpMergeError(
                f"image {image_id} has rungs {sorted(rungs)} but others have "
                f"{sorted(reference)}; a partial re-run left the ladder ragged"
            )
    return sorted(reference or ())


def _unit_ids(unit: EvalUnit, ids_by_source: Mapping[str, Sequence[int]]) -> list[int]:
    return sorted(i for s in unit.source_sets for i in ids_by_source.get(s, ()))


def _empty_frame_ids(ids: Sequence[int], by_id: Mapping[int, Mapping[str, Any]],
                     vocab: Sequence[str]) -> dict[str, list[int]]:
    """Per-class empty frames WITHIN one unit.

    Computed from ``ids`` -- the unit's own images -- never from a pool-wide
    precomputed count. The shared pool's ``empty_frames_per_class`` is over all
    27,171 images, so reusing it would silently score fp_fb_test's
    false-positive dispersion against FPIR frames the unit never contained.
    """
    return {cls: [i for i in ids
                  if not any(g["cls"] == cls and not g.get("iscrowd") for g in by_id[i]["gt"])]
            for cls in vocab}


# ---------------------------------------------------------------------------
# Per-rung detection accuracy (schema revision 3.1)
# ---------------------------------------------------------------------------
# The stability metrics answer "does the output MOVE when the prompt changes".
# They cannot answer "which prompt DETECTS BETTER", which is the study's actual
# question, so revision 3.1 adds COCO accuracy per rung alongside them.
#
# The calculator is the benchmark's own ``utils.metrics.cocoeval_metrics
# .cocoeval_block`` -- imported, never reimplemented. A second COCO
# implementation would drift from the tables these numbers must be read against,
# and the drift would surface as an apparent family difference. The import is
# lazy so this module still loads for stability-only work; when accuracy IS
# requested and the workshop repo is not importable, it fails loudly rather than
# silently substituting a different calculator.


def _cocoeval_block():
    try:
        from utils.metrics.cocoeval_metrics import cocoeval_block
    except ImportError as exc:  # pragma: no cover - environment wiring
        raise ImportError(
            "accuracy requires utils.metrics.cocoeval_metrics (the benchmark's own COCO "
            "calculator). Install this package. Reimplementing "
            "COCOeval here would let family A and family B diverge silently."
        ) from exc
    return cocoeval_block


def accuracy_block(ids: Sequence[int], dumps: Mapping[int, Mapping[str, np.ndarray]],
                   by_id: Mapping[int, Mapping[str, Any]], spec: FragmentSpec,
                   threshold: float = 0.05) -> dict[str, Any]:
    """``{class: {rung: {ap, ap50, ap75, ar, n_gt} | None}}`` plus ``macro6``.

    ``None`` -- never ``0.0`` -- for a (class, rung) whose query list omits the
    class: the model was not asked, so there is no performance to report, and
    0.0 would read as "asked and found nothing".

    ``macro6`` is macro over the classes PRESENT IN THAT RUNG, so the number of
    contributing classes varies by rung; the count is reported per rung so a
    table can say so.
    """
    cocoeval_block = _cocoeval_block()
    gt_by_class: dict[str, dict[int, np.ndarray]] = {}
    ignore_by_class: dict[str, dict[int, np.ndarray]] = {}
    for cls in spec.vocab:
        gt_by_class[cls] = {
            i: np.array([g["bbox_xyxy"] for g in by_id[i]["gt"]
                         if g["cls"] == cls and not g.get("iscrowd")],
                        dtype=float).reshape(-1, 4) for i in ids
        }
        ignore_by_class[cls] = {
            i: np.array([g["bbox_xyxy"] for g in by_id[i]["gt"] if g.get("iscrowd")],
                        dtype=float).reshape(-1, 4) for i in ids
        }

    out: dict[str, Any] = {cls: {} for cls in spec.vocab}
    macro: dict[str, Any] = {}
    for rung, words in spec.prompts.items():
        present = [c for c in spec.vocab if c in words]
        for cls in spec.vocab:
            if cls not in present:
                out[cls][rung] = None          # not asked -- not a zero
                continue
            ci = spec.vocab.index(cls)
            preds = {}
            for i in ids:
                d = class_detections(dumps[i][rung], ci, threshold)
                preds[i] = (d[:, :4], d[:, 4])
            block = cocoeval_block({cls: preds}, {cls: gt_by_class[cls]},
                                   {cls: ignore_by_class[cls]}, list(ids))
            cell = dict(block["per_class"][cls])
            cell["n_gt"] = int(sum(len(v) for v in gt_by_class[cls].values()))
            out[cls][rung] = cell
        vals = [out[c][rung] for c in present if out[c][rung] is not None]
        macro[rung] = {
            key: (float(np.mean([v[key] for v in vals if v.get(key) is not None]))
                  if any(v.get(key) is not None for v in vals) else None)
            for key in ("ap", "ap50", "ap75", "ar")
        }
        macro[rung]["n_classes"] = len(present)
        macro[rung]["classes"] = present
    out["macro6"] = macro
    return out


def accuracy_spread_block(accuracy: Mapping[str, Any], spec: FragmentSpec) -> dict[str, Any]:
    """Per class, min/max/spread of accuracy over the rungs CONTAINING it."""
    spread: dict[str, Any] = {}
    for cls in spec.vocab:
        per_rung = {r: v for r, v in accuracy[cls].items() if v is not None}
        entry: dict[str, Any] = {"n_rungs": len(per_rung)}
        for key in ("ap50", "ap"):
            pairs = [(r, v[key]) for r, v in per_rung.items() if v.get(key) is not None]
            if not pairs:
                entry[key] = {"min": None, "max": None, "spread": None,
                              "argmin_rung": None, "argmax_rung": None}
                continue
            lo = min(pairs, key=lambda kv: kv[1])
            hi = max(pairs, key=lambda kv: kv[1])
            entry[key] = {"min": float(lo[1]), "max": float(hi[1]),
                          "spread": float(hi[1] - lo[1]),
                          "argmin_rung": lo[0], "argmax_rung": hi[0]}
        spread[cls] = entry
    return spread


def _unit_block(unit: EvalUnit, ids: Sequence[int],  # noqa: C901
                dumps: Mapping[int, Mapping[str, np.ndarray]],
                by_id: Mapping[int, Mapping[str, Any]],
                spec: FragmentSpec) -> dict[str, Any]:
    """Stability / absolute / order-contrast / empty-FP for ONE unit."""
    stability: dict[str, Any] = {}
    absolute: dict[str, Any] = {}
    for cls in spec.vocab:
        stability[cls] = {f"{t}": stability_cell(ids, dumps, cls, t, spec) for t in THRESHOLDS}
        ci = spec.vocab.index(cls)
        absolute[cls] = {}
        for p in spec.prompts:
            per_thr = {f"{t}": sum(len(class_detections(dumps[i][p], ci, t)) for i in ids)
                       for t in THRESHOLDS}
            scores = (np.concatenate([class_detections(dumps[i][p], ci, THRESHOLDS[0])[:, 4]
                                      for i in ids]) if ids else np.zeros(0))
            absolute[cls][p] = {
                "n_dets": per_thr,
                "dets_per_image": {k: v / max(len(ids), 1) for k, v in per_thr.items()},
                "mean_score": float(scores.mean()) if len(scores) else 0.0,
            }
    stability["macro6"] = {
        f"{t}": macro_over_classes({c: stability[c][f"{t}"] for c in spec.vocab})
        for t in THRESHOLDS
    }
    empty_ids = _empty_frame_ids(ids, by_id, spec.vocab)
    empty_fp = {cls: {f"{t}": empty_fp_block(empty_ids, dumps, cls, t, spec) for t in THRESHOLDS}
                for cls in spec.vocab}
    return {
        "n_images": len(ids),
        "stability": stability,
        "_order_contrast_for_blocks": None,
        "absolute": absolute,
        "order_contrast": order_contrast(ids, dumps, spec),
        "empty_fp_spread": empty_fp,
        "empty_frames": {c: len(v) for c, v in empty_ids.items()},
    }


def build_fragment(spec: FragmentSpec, dumps: Mapping[int, Mapping[str, np.ndarray]],
                   pool: Sequence[Mapping[str, Any]], strict_census: bool = True,
                   pool_path: str | None = None,
                   with_accuracy: bool = True) -> dict[str, Any]:
    """Assemble one schema-revision-2 fragment. Family-agnostic by construction.

    Aggregation units are explicit and validated: ``fp_fb_test`` (primary) and
    ``fpir_test`` (separate). Nothing is ever pooled across the two.
    """
    # The ladder gate sits beside the census gate: a partial ladder changes the
    # pair set, so its pairwise numbers are not comparable across arms.
    validate_ladder(assert_uniform_rungs(dumps), {k: v for k, v in spec.prompts.items()})

    by_id = {r["pool_id"]: r for r in pool}
    ids_by_source: dict[str, list[int]] = defaultdict(list)
    for i in dumps:
        ids_by_source[by_id[i]["set"]].append(i)
    for v in ids_by_source.values():
        v.sort()

    dsr = duplicate_score_rate(ids_by_source.get(SOURCE_FB, []), dumps, spec)
    prompt_text = {k: ", ".join(v) for k, v in spec.prompts.items()}
    frag: dict[str, Any] = {
        "schema_revision": (5 if set(RUNGS_NEW_V5) <= set(spec.prompts)
                            else 4 if set(RUNGS_NEW_V4) <= set(spec.prompts) else 3),
        "arm": spec.arm,
        "family": spec.family,
        "model": spec.model,
        "census": "full" if strict_census else "PARTIAL-SCREENING-NOT-PUBLISHABLE",
        "pool": pool_path,
        "vocabulary": list(spec.vocab),
        "score_floor": THRESHOLDS[0],
        "deployment_thresholds": list(THRESHOLDS[1:]),
        "degenerate_phrase_scores": bool(dsr > 0.5),
        "duplicate_score_rate": dsr,
        **dict(spec.extra),
        "class_variant_rule": ("each class is compared only over prompt variants whose query "
                               "list contains it; classes in <2 variants are null"),
        "aggregation_rule": ("fp_fb_test = fp_test + fb_test pooled (PRIMARY, 14,980); "
                             "fpir_test scored separately (12,191); never pooled together; "
                             "empty_frames recomputed inside each unit"),
        "prompts": {k: {"text": prompt_text[k], "classes": list(v)}
                    for k, v in spec.prompts.items()},
        "sets": {}, "accuracy": {}, "accuracy_spread": {}, "stability": {},
        "empty_fp_spread": {}, "absolute": {}, "order_contrast": {}, "diagnostics": {},
    }

    empty_frames_by_unit: dict[str, dict[str, int]] = {}
    for unit in spec.units:
        ids = _unit_ids(unit, ids_by_source)
        validate_eval_unit(unit, len(ids))
        validate_census(unit, len(ids), strict=strict_census)
        block = _unit_block(unit, ids, dumps, by_id, spec)
        frag["sets"][unit.unit_id] = {"n_images": block["n_images"],
                                      "source_sets": list(unit.source_sets),
                                      "primary": unit.primary}
        frag["stability"][unit.unit_id] = block["stability"]
        frag["absolute"][unit.unit_id] = block["absolute"]
        frag["order_contrast"][unit.unit_id] = block["order_contrast"]
        frag["empty_fp_spread"][unit.unit_id] = block["empty_fp_spread"]
        empty_frames_by_unit[unit.unit_id] = block["empty_frames"]
        if with_accuracy:
            acc = accuracy_block(ids, dumps, by_id, spec)
            frag["accuracy"][unit.unit_id] = acc
            frag["accuracy_spread"][unit.unit_id] = accuracy_spread_block(acc, spec)
            # Revision-4 isolating blocks: size / identity / order, each holding
            # everything else fixed. Only meaningful once the 24-rung ladder is
            # complete, which validate_ladder has already enforced above.
            if set(RUNGS_NEW_V4) <= set(spec.prompts):
                frag.setdefault("block_analyses", {})[unit.unit_id] = block_analyses(
                    acc, block["order_contrast"], spec)
    frag["sets"]["empty_frames"] = {"n_images": empty_frames_by_unit}

    # Per-source breakdowns: valid ONLY as a diagnostic row under a headline.
    for unit in spec.diagnostic_units:
        ids = _unit_ids(unit, ids_by_source)
        validate_eval_unit(unit, len(ids))
        block = _unit_block(unit, ids, dumps, by_id, spec)
        frag["diagnostics"][unit.unit_id] = {
            "n_images": block["n_images"],
            "headline_eligible": False,
            "stability": block["stability"],
            "empty_fp_spread": block["empty_fp_spread"],
        }
    return frag


# NOTE: per-rung detection accuracy is computed by ``accuracy_block`` /
# ``accuracy_spread_block`` above, which delegate to the benchmark's own
# ``utils.metrics.cocoeval_metrics.cocoeval_block``. An earlier revision of
# there is exactly one pycocotools calculator in this file. Two would make the
# numbers that reach a fragment depend on which one a keyword argument selected
# rather than on the data, which is the precise failure the single-calculator
# rule exists to prevent -- so a duplicate is removed rather than kept "for
# reference".


# ── Revision-4 block analyses: one independent variable each ─────────────────
#
# Revision 3 could not separate vocabulary SIZE from vocabulary IDENTITY: every
# step up its ladder changed both at once. These three summaries each hold
# everything else fixed, so the axis named is the only thing moving.
#
# Shared by families A and B, like every other metric here -- a per-family
# implementation would make the two blocks' numbers incomparable for reasons
# unrelated to the models.


def _ap_by_rung(accuracy_for_class: Mapping[str, Any], rungs: Sequence[str],
                metric: str = "ap50") -> dict[str, float]:
    """``{rung: value}`` over rungs where the class was actually asked."""
    out: dict[str, float] = {}
    for rung in rungs:
        entry = accuracy_for_class.get(rung)
        if entry and entry.get(metric) is not None:
            out[rung] = float(entry[metric])
    return out


def _spread_block(values: Mapping[str, float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "max": None, "spread": None,
                "argmin_rung": None, "argmax_rung": None, "n_rungs": 0}
    lo = min(values, key=lambda r: values[r])
    hi = max(values, key=lambda r: values[r])
    return {"min": values[lo], "max": values[hi], "spread": values[hi] - values[lo],
            "argmin_rung": lo, "argmax_rung": hi, "n_rungs": len(values),
            "per_rung": dict(values)}


def size_axis_block(accuracy: Mapping[str, Any], cls: str, spec: FragmentSpec,
                    metric: str = "ap50") -> dict[str, Any]:
    """Block B: AP along the STRICT-INCLUSION size chain 1->6.

    Every rung's vocabulary is a superset of the previous one, so a monotone
    trend here is attributable to size alone. Under revision 3 the same
    comparison also changed which words were present.
    """
    chain = [r for r in SIZE_CHAIN_V4 if cls in spec.prompts.get(r, ())]
    values = _ap_by_rung(accuracy.get(cls, {}), chain, metric)
    block = _spread_block(values)
    block["axis"] = "vocabulary_size"
    block["chain"] = chain
    block["sizes"] = {r: len(spec.prompts[r]) for r in chain}
    return block


def identity_axis_block(accuracy: Mapping[str, Any], cls: str, spec: FragmentSpec,
                        metric: str = "ap50") -> dict[str, Any]:
    """Block C: AP across size-2 prompts that differ ONLY in the partner word.

    A non-zero spread here is a pure identity effect: the vocabulary is the
    same size everywhere, so nothing that revision 3 called a "size effect"
    can explain it.
    """
    rungs = [r for r in IDENTITY_SET_V4 if cls in spec.prompts.get(r, ())]
    values = _ap_by_rung(accuracy.get(cls, {}), rungs, metric)
    block = _spread_block(values)
    block["axis"] = "vocabulary_identity"
    block["fixed_size"] = 2
    block["partners"] = {r: [w for w in spec.prompts[r] if w != cls] for r in rungs}
    return block


def order_axis_block(accuracy: Mapping[str, Any], cls: str, spec: FragmentSpec,
                     order_contrast_for_class: Mapping[str, Any] | None = None,
                     metric: str = "ap50") -> dict[str, Any]:
    """Block D: AP across all 3! permutations of ONE fixed vocabulary.

    The vocabulary is identical in all six rungs, so any spread is a PURE
    ORDER EFFECT. Family B encodes each query independently and must score
    exactly 0; family A encodes one joint caption whose tokens carry positional
    encoding, so a non-zero value there is the cleanest available evidence of
    the structural difference between the two families.

    ``order_contrast_for_class`` (the 15 pairwise entries) is summarised
    alongside, because a spread of 0 with pairwise ``identical=False`` means
    the boxes moved without changing AP -- a different finding from true
    invariance.
    """
    rungs = [r for r in LADDER_BLOCK_D if cls in spec.prompts.get(r, ())]
    values = _ap_by_rung(accuracy.get(cls, {}), rungs, metric)
    block = _spread_block(values)
    block["axis"] = "word_order"
    block["fixed_vocabulary"] = sorted({w for r in rungs for w in spec.prompts[r]})
    block["n_permutations"] = len(rungs)

    if order_contrast_for_class:
        pairs = {k: v for k, v in order_contrast_for_class.items() if k in ORDER_PAIRS_BLOCK_D}
        f1 = [v["pi_f1"] for v in pairs.values() if v.get("pi_f1") is not None]
        jit = [v["loc_jitter"] for v in pairs.values() if v.get("loc_jitter") is not None]
        ident = [v.get("identical") for v in pairs.values() if v.get("identical") is not None]
        block["pairwise_summary"] = {
            "n_pairs": len(pairs),
            "pi_f1": {"mean": float(np.mean(f1)) if f1 else None,
                      "min": float(np.min(f1)) if f1 else None,
                      "max": float(np.max(f1)) if f1 else None},
            "loc_jitter": {"mean": float(np.mean(jit)) if jit else None,
                           "min": float(np.min(jit)) if jit else None,
                           "max": float(np.max(jit)) if jit else None},
            "n_identical": int(sum(1 for i in ident if i)),
            "all_identical": bool(ident) and all(ident),
        }
    return block


def block_analyses(accuracy: Mapping[str, Any], order_contrast: Mapping[str, Any],
                   spec: FragmentSpec) -> dict[str, Any]:
    """All three isolating-block summaries, per class."""
    return {
        cls: {
            "size_axis": size_axis_block(accuracy, cls, spec),
            "identity_axis": identity_axis_block(accuracy, cls, spec),
            "order_axis": order_axis_block(accuracy, cls, spec,
                                           order_contrast.get(cls)),
        }
        for cls in spec.vocab
    }
