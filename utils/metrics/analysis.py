"""Axis-A interference analysis: I(c), M(c) and the additivity test.

Spec: the vocabulary specification §2.

    I(c) = AP_person(singleton) - AP_person(person + c)      first-order interference
    M(c) = AP_person(V6 \\ c)   - AP_person(V6)               marginal contribution

Additivity test — the justification for running 18 conditions instead of 63::

    sum_c I(c)  ?=  AP_person(singleton) - AP_person(V6)

Holds  -> interference is first-order; the remaining subsets are predictable.
Fails -> higher-order interaction exists, which is itself a finding, and only
then are the extra 3-element subsets worth running.

Every value here is read from fragment files. Nothing is transcribed from a
conversation or from a pre-2026-08-08 artefact.

The covariate regression the coordinator substituted for the spec's overlap
ranking lives in :func:`interference_covariates`: on this data
``person``-belongings spatial overlap is ~0 for every class with no variation,
so interference is regressed on **box area fraction** and **instances per
image** instead. See the v6_gt_statistics.md note.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from utils.runner.logging_utils import get_logger
from utils.protocol.vocabulary import BELONGINGS_CLASSES, PERSON_CLASS

logger = get_logger(__name__)

SINGLETON_PERSON = "A1_singleton_person"
FULL_V6 = "A4_full_v6"


def fragment_path(directory: Path, model: str, variant: str, condition_id: str) -> Path:
    return directory / f"{model}__{variant}__{condition_id}.json"


class MixedCapPolicyError(ValueError):
    """Fragments from two different cap policies were about to be aggregated."""


class MixedEnvironmentError(ValueError):
    """Fragments produced under different interpreters/library versions.

    ``runner_version`` hashes our source, not the environment it ran in. On
    2026-08-10 two OWLv2 runs with identical fingerprints came from
    transformers 4.57.3 and 5.14.1, whose OWLv2 image processors pad the
    letterbox differently (0.5 grey vs 0 black). ~44% of a 1080p canvas is
    pad, so those were different experiments and nothing in the fragment said
    so — it had to be reconstructed from shell scripts afterwards.

    A ``transformers`` MAJOR version split is always rejected. Fragments with
    no ``env`` key at all predate this recording; they are reported as
    ``pre-recording``, never conflated with a measured mismatch.
    """


class MixedDumpFormatError(ValueError):
    """Fragments written by different detection-dump formats.

    ``dump_format`` is inside the ``runner_version`` hash, so a mixed set
    already fails the version check. This exists anyway because the hash gives
    an opaque "these differ" while this names the axis. Debugging time differs
    by an order of magnitude, and the hash cannot say *which* thing changed.
    """


class StaleCosineError(ValueError):
    """A fragment's stamped axis-B cosine disagrees with the canonical table.

    Fourth guard of the same family as :class:`MixedCapPolicyError`,
    ``MixedPromptVersionError`` and ``MixedCheckpointError``: it refuses a
    silent mixture rather than averaging two incompatible measurements.
    """


def load_fragments(directory: Path, model: str, variant: str,
                   require_cap_policy: str | None = None) -> dict[str, dict]:
    """All fragments for one (model, variant), keyed by exact condition id.

    Guards against mixing cap policies. Pass 1 ran under ``global_per_image``
    and pass 2 under ``per_category``; the two are not comparable, and a table
    silently built from both would be wrong in a way no reader could detect.
    Fragments predating the field are treated as ``global_per_image``, which is
    what they in fact used.

    Raises:
        MixedCapPolicyError: if the directory holds more than one policy, or a
            policy other than ``require_cap_policy``.
    """
    out: dict[str, dict] = {}
    prefix = f"{model}__{variant}__"
    policies: set[str] = set()
    dump_formats: set[str] = set()
    for path in sorted(directory.glob(f"{prefix}*.json")):
        fragment = json.loads(path.read_text())
        policies.add(fragment.get("cap_policy", "global_per_image"))
        rv = fragment.get("runner_version") or {}
        if isinstance(rv, dict) and "dump_format" in rv:
            dump_formats.add(rv["dump_format"] or "none")
        out[fragment["condition_id"]] = fragment
    envs: dict[str, set] = {}
    n_pre_recording = 0
    for fragment in out.values():
        env = fragment.get("env")
        if not env:
            n_pre_recording += 1
            continue
        for key in ("transformers", "torch", "venv_path"):
            envs.setdefault(key, set()).add(env.get(key))
    tf = envs.get("transformers", set())
    if len({str(v).split(".")[0] for v in tf if v}) > 1:
        raise MixedEnvironmentError(
            f"{directory} mixes transformers MAJOR versions {sorted(tf)}. The "
            f"OWLv2 image processor pads differently across the 4->5 boundary "
            f"(0.5 grey vs 0 black, ~44% of a 1080p canvas), so these are "
            f"different experiments, not different code versions.")
    split = {k: sorted(str(x) for x in v) for k, v in envs.items() if len(v) > 1}
    if split:
        raise MixedEnvironmentError(
            f"{directory} mixes execution environments {split}. runner_version "
            f"hashes source, not the environment, so these fragments are "
            f"indistinguishable by fingerprint alone.")
    if n_pre_recording and envs:
        raise MixedEnvironmentError(
            f"{directory} mixes {n_pre_recording} fragment(s) from before env "
            f"recording with {len(out) - n_pre_recording} that record it. "
            f"'pre-recording' is not 'same environment' -- the older ones must "
            f"be attributed from run records before they are pooled.")

    if len(dump_formats) > 1:
        raise MixedDumpFormatError(
            f"{directory} holds fragments written with different detection-dump "
            f"formats {sorted(dump_formats)}. They are not comparable: the dump "
            f"format is part of the runner_version hash, so these came from "
            f"different code. Keep each format in its own directory.")
    if len(policies) > 1:
        raise MixedCapPolicyError(
            f"{directory} holds fragments from several cap policies {sorted(policies)}; "
            f"they are not comparable. Keep each pass in its own directory.")
    if require_cap_policy and policies and policies != {require_cap_policy}:
        raise MixedCapPolicyError(
            f"expected cap_policy={require_cap_policy!r} but found {sorted(policies)}")
    return out


def metric(fragment: Mapping, class_name: str, key: str = "AP") -> float | None:
    """One per-class metric, or ``None`` if unmeasured/absent.

    Exact key lookup — a class that was not queried is absent, and absence must
    not silently become 0.0.
    """
    per_class = fragment["per_class"]
    if class_name not in per_class:
        return None
    return per_class[class_name].get(key)


@dataclass(frozen=True)
class InterferenceResult:
    """Axis-A interference for one (model, variant)."""

    model: str
    variant: str
    metric_key: str
    ap_person_singleton: float | None
    ap_person_full_v6: float | None
    #: class -> I(c); ``None`` where a fragment is missing.
    interference: Mapping[str, float | None]
    #: class -> M(c); ``None`` where a fragment is missing.
    marginal: Mapping[str, float | None]
    sum_interference: float | None
    total_drop: float | None
    additivity_residual: float | None
    additivity_holds: bool | None
    missing_conditions: tuple[str, ...]


def compute_interference(fragments: Mapping[str, Mapping], model: str, variant: str,
                         metric_key: str = "AP",
                         tolerance: float = 0.02) -> InterferenceResult:
    """Compute I(c), M(c) and the additivity residual from loaded fragments.

    Args:
        tolerance: Absolute AP tolerance for calling additivity satisfied.
            Reported alongside the residual so a reader can apply their own.
    """
    missing: list[str] = []

    def _ap(condition_id: str) -> float | None:
        fragment = fragments.get(condition_id)
        if fragment is None:
            missing.append(condition_id)
            return None
        return metric(fragment, PERSON_CLASS, metric_key)

    singleton = _ap(SINGLETON_PERSON)
    full = _ap(FULL_V6)

    interference: dict[str, float | None] = {}
    marginal: dict[str, float | None] = {}
    for name in BELONGINGS_CLASSES:
        pair = _ap(f"A2_pair_person_{name}")
        interference[name] = (singleton - pair) if (singleton is not None and pair is not None) else None
        loo = _ap(f"A3_loo_{name}")
        marginal[name] = (loo - full) if (loo is not None and full is not None) else None

    values = [v for v in interference.values() if v is not None]
    complete = len(values) == len(BELONGINGS_CLASSES)
    sum_i = sum(values) if complete else None
    total_drop = (singleton - full) if (singleton is not None and full is not None) else None
    residual = (sum_i - total_drop) if (sum_i is not None and total_drop is not None) else None

    return InterferenceResult(
        model=model, variant=variant, metric_key=metric_key,
        ap_person_singleton=singleton, ap_person_full_v6=full,
        interference=interference, marginal=marginal,
        sum_interference=sum_i, total_drop=total_drop,
        additivity_residual=residual,
        additivity_holds=(abs(residual) <= tolerance) if residual is not None else None,
        missing_conditions=tuple(sorted(set(missing))),
    )


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson r, or ``None`` when it is undefined (n<3 or zero variance)."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx * syy) ** 0.5


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """Least-squares ``(slope, intercept)``, or ``None`` if undefined."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


@dataclass(frozen=True)
class AxisBPoint:
    """One paraphrase condition: ΔAP against its measured embedding cosine.

    Spec §3 forbids reporting the B axis as a synonym / non-synonym split. The
    independent variable is the cosine measured with **this model's** text
    encoder, so the same term can sit at different x positions for GDINO and
    OWLv2 — which is the point.
    """

    condition_id: str
    replaced_class: str
    variant_term: str
    cosine: float | None
    ap: float | None
    ar: float | None
    baseline_ap: float | None
    baseline_ar: float | None
    delta_ap: float | None
    delta_ar: float | None


#: Canonical axis-B cosines. A fragment's stamped copy is a record of what that
#: run did; this file is what the analysis plots. See the v6_text_cosines.md note.
CANONICAL_COSINES_PATH = (Path(__file__).resolve().parents[2]
                          / "outputs" / "v6" / "text_cosines.json")

#: Fragments written before weights provenance was recorded (2026-08-09).
UNRECORDED_WEIGHTS = "unrecorded (pre-2026-08-09)"

#: Tolerance for "the stamped cosine is the canonical one". Float32 round-off
#: across transformers builds is ~1e-7; anything above this is a real change.
COSINE_TOLERANCE = 1e-6


def load_canonical_cosines(path: Path | None = None) -> dict[tuple[str, str, str], float]:
    """``(model, reference_term, term) -> cosine_mean_subtoken``, or ``{}``."""
    from utils.metrics.text_similarity import cosine_lookup, load_cosines

    target = path or CANONICAL_COSINES_PATH
    if not target.is_file():
        return {}
    return dict(cosine_lookup(load_cosines(target)))


def _resolve_cosine(model: str, replaced_class: str, variant_term: str,
                    stamped: float | None, canonical: Mapping[tuple[str, str, str], float],
                    condition_id: str, strict: bool,
                    weights_source: str) -> float | None:
    """Return the canonical cosine, refusing a silent disagreement.

    The fragment is **not** rewritten. A fragment records what its run did; a
    value edited in afterwards would make it a document rather than a
    measurement. The canonical table is joined in here at analysis time
    instead.
    """
    key = (model, replaced_class, variant_term)
    truth = canonical.get(key)
    if truth is None:
        return stamped
    if stamped is None:
        return truth
    if abs(stamped - truth) <= COSINE_TOLERANCE:
        return truth
    detail = (
        f"{condition_id}: fragment stamps cos{key} = {stamped!r} but the canonical "
        f"table says {truth!r} (|diff| = {abs(stamped - truth):.6g}). "
        f"Canonical source: {CANONICAL_COSINES_PATH}. "
        f"Fragment weights_source = {weights_source!r}. "
        "Fragments are execution records and are never re-stamped; the analysis "
        "joins the canonical table instead. If the stamped value is the "
        "superseded vanilla-BERT one, it is expected to differ -- rerun with "
        "strict=False to join the canonical value and keep going."
    )
    if strict:
        raise StaleCosineError(detail)
    logger.warning("stale cosine joined from canonical table -- %s", detail)
    return truth


def axis_b_points(fragments: Mapping[str, Mapping],
                  metric_key: str = "AP",
                  canonical_cosines: Mapping[tuple[str, str, str], float] | None = None,
                  strict_cosines: bool = True) -> list[AxisBPoint]:
    """ΔAP and ΔAR for every axis-B condition, against the V6 baseline.

    The baseline is ``A4_full_v6`` because each B condition is V6 with exactly
    one term swapped — the minimal contrast pair of spec §3.

    The x axis comes from :data:`CANONICAL_COSINES_PATH`, not from the
    fragment. GDINO's pass-2 fragments carry cosines computed with **vanilla**
    ``bert-base-uncased`` rather than GDINO's own BERT (corrected 2026-08-09);
    plotting those would put every GDINO point at the wrong x.

    Args:
        canonical_cosines: Override the table, for tests.
        strict_cosines: Raise :class:`StaleCosineError` when a fragment's
            stamped cosine disagrees with the canonical one. Set ``False`` to
            join the canonical value with a warning instead — which is what the
            GDINO pass-2 fragments need, since their stamped values are the
            superseded ones.
    """
    canonical = (dict(canonical_cosines) if canonical_cosines is not None
                 else load_canonical_cosines())
    baseline = fragments.get(FULL_V6)
    points: list[AxisBPoint] = []
    for condition_id, fragment in sorted(fragments.items()):
        if fragment.get("axis") != "B":
            continue
        cosines = fragment.get("text_embedding_cosines") or {}
        terms = fragment.get("terms", [])
        model = fragment.get("model", "")
        weights_source = fragment.get("text_embedding_weights_source", UNRECORDED_WEIGHTS)
        # The replaced term is the one that is not its own gt class. Fragments
        # that stamp no cosines at all (GLIP, OmDet) still yield a point: the
        # variant term is recovered from the condition definition.
        replaced_class, variant_term = None, None
        for term in terms:
            if term in cosines:
                variant_term = term
        subgroup = fragment.get("subgroup", "")
        replaced_class = "clothes" if subgroup == "B_clothes_variant" else PERSON_CLASS
        if variant_term is None:
            variant_term = next((t for t in terms if t not in fragment.get(
                "scored_classes", [])), None)
        if variant_term is None:
            continue
        ap = metric(fragment, replaced_class, metric_key)
        ar = metric(fragment, replaced_class, "AR")
        base_ap = metric(baseline, replaced_class, metric_key) if baseline else None
        base_ar = metric(baseline, replaced_class, "AR") if baseline else None
        points.append(AxisBPoint(
            condition_id=condition_id,
            replaced_class=replaced_class,
            variant_term=variant_term,
            cosine=_resolve_cosine(model, replaced_class, variant_term,
                                   cosines.get(variant_term), canonical,
                                   condition_id, strict_cosines, weights_source),
            ap=ap, ar=ar, baseline_ap=base_ap, baseline_ar=base_ar,
            delta_ap=(ap - base_ap) if (ap is not None and base_ap is not None) else None,
            delta_ar=(ar - base_ar) if (ar is not None and base_ar is not None) else None,
        ))
    return points


#: Spec §4 mechanism table. Absorption: the distractor takes boxes that would
#: have been person FALSE positives -> AP up, AR held. Stealing: it takes boxes
#: that would have been person TRUE positives -> AR down.
ABSORPTION = "absorption"
STEALING = "stealing"
MIXED = "mixed"
NO_EFFECT = "no_effect"


@dataclass(frozen=True)
class AxisCMechanism:
    """Absorption vs stealing for one distractor condition (spec §4)."""

    condition_id: str
    delta_ap_person: float | None
    delta_ar_person: float | None
    baseline_empty_frame_fp_per_image: float | None
    empty_frame_fp_per_image: float | None
    delta_empty_frame_fp: float | None
    verdict: str


def axis_c_mechanism(fragments: Mapping[str, Mapping],
                     ar_tolerance: float = 0.005,
                     ap_tolerance: float = 0.005) -> list[AxisCMechanism]:
    """Classify each C condition as absorption, stealing, mixed or no effect.

    C4 (``person`` + C-conf) is compared against the ``person`` singleton, not
    against V6 — it holds vocabulary size at one scored class, so the singleton
    is its matched control. C1/C2/C3/C5 are compared against V6.
    """
    verdicts: list[AxisCMechanism] = []
    for condition_id, fragment in sorted(fragments.items()):
        if fragment.get("axis") != "C":
            continue
        control_id = (SINGLETON_PERSON if condition_id.startswith("C4_") else FULL_V6)
        control = fragments.get(control_id)
        ap = metric(fragment, PERSON_CLASS)
        ar = metric(fragment, PERSON_CLASS, "AR")
        base_ap = metric(control, PERSON_CLASS) if control else None
        base_ar = metric(control, PERSON_CLASS, "AR") if control else None
        d_ap = (ap - base_ap) if (ap is not None and base_ap is not None) else None
        d_ar = (ar - base_ar) if (ar is not None and base_ar is not None) else None

        empty = fragment["empty_frame_false_positives_per_image"]["per_image"]
        base_empty = (control["empty_frame_false_positives_per_image"]["per_image"]
                      if control else None)
        d_empty = (empty - base_empty) if (empty is not None and base_empty is not None) else None

        if d_ap is None or d_ar is None:
            verdict = "unmeasured"
        elif d_ar < -ar_tolerance:
            verdict = STEALING if d_ap <= ap_tolerance else MIXED
        elif d_ap > ap_tolerance:
            verdict = ABSORPTION
        else:
            verdict = NO_EFFECT
        verdicts.append(AxisCMechanism(
            condition_id=condition_id, delta_ap_person=d_ap, delta_ar_person=d_ar,
            baseline_empty_frame_fp_per_image=base_empty,
            empty_frame_fp_per_image=empty, delta_empty_frame_fp=d_empty,
            verdict=verdict,
        ))
    return verdicts


@dataclass(frozen=True)
class CovariateRegression:
    """Interference regressed on a ground-truth covariate."""

    covariate: str
    classes: tuple[str, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    pearson_r: float | None
    slope: float | None
    intercept: float | None
    note: str = ""


def interference_covariates(
    result: InterferenceResult,
    gt_statistics: Mapping[str, Mapping],
) -> list[CovariateRegression]:
    """Regress I(c) on the covariates that actually vary on this data.

    Spec §1 asks for interference vs. person-overlap ranking. On the V6 data
    that correlation is **not computable**: mean IoU with ``person`` is ~0 for
    every belongings class because the two source datasets are disjoint at the
    image level. The substitutes are box area fraction and instances per image,
    both of which vary by 3-4x across classes.
    """
    covariates = {
        "mean_box_area_fraction": "mean box area as a fraction of the image",
        "mean_instances_per_image": "mean instances per image over the whole split",
        "mean_iou_with_person": "mean IoU with person (RETAINED FOR THE RECORD: ~0 "
                                "for every class, no variation, not interpretable)",
    }
    out: list[CovariateRegression] = []
    for key, note in covariates.items():
        pairs = [(name, result.interference[name], gt_statistics[name][key])
                 for name in BELONGINGS_CLASSES
                 if result.interference.get(name) is not None and name in gt_statistics]
        if len(pairs) < 3:
            out.append(CovariateRegression(covariate=key, classes=(), x=(), y=(),
                                           pearson_r=None, slope=None, intercept=None,
                                           note=f"{note} — too few measured classes"))
            continue
        names = tuple(p[0] for p in pairs)
        ys = tuple(float(p[1]) for p in pairs)
        xs = tuple(float(p[2]) for p in pairs)
        fit = linear_fit(xs, ys)
        out.append(CovariateRegression(
            covariate=key, classes=names, x=xs, y=ys,
            pearson_r=pearson(xs, ys),
            slope=fit[0] if fit else None,
            intercept=fit[1] if fit else None,
            note=note,
        ))
    return out
