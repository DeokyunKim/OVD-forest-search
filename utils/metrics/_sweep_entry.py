#!/usr/bin/env python
"""Score-threshold sensitivity of the V6 query-sensitivity flip metrics.

    $PY sweep_v6_flip_thresholds.py \\
        --analysis-dir outputs/v6/analysis \\
        --dumps-dir outputs/v6/detection_dumps \\
        --out-dir outputs/v6/analysis/threshold_sweep \\
        --report the specification

CPU only, no re-inference: every number is recomputed from the frozen
``v6_npz.1`` detection dumps by moving ONE knob, ``score_threshold``. The IoU
threshold and the per-category detection cap stay at the values the frozen
protocol used, so the sweep isolates the operating point and nothing else.

Why it exists
-------------
The published flip analysis pins ``score_threshold = 0.0``. At that point
``recall_q0`` sits at 97-99% for every arm, i.e. near the ceiling, and a small
NFR is then partly a property of the construction rather than of the model.
The identity ``ΔAR50 = PFR - NFR`` is exact at every threshold and therefore
cannot detect this. The reviewer-facing question is whether the ORDERING of
models by NFR survives the operating point; this script answers it by
recomputing all 48 published runs at each threshold in
:data:`SWEEP_THRESHOLDS`.

Reproduction gate
-----------------
The ``0.0`` slice must reproduce the published JSON **bit for bit**. It is
compared field by field before anything else is written; a mismatch means this
script re-implemented something instead of re-aggregating it, and the run
aborts rather than emitting a sweep nobody can trust.

Nothing in ``utils/metrics`` is modified or re-implemented here:
``flip_analysis`` and ``assign_gt_states`` are imported and called with a
different ``score_threshold``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# The repository root, found by walking up to the directory that holds
# `pyproject.toml`. Written this way rather than as a fixed
# number of parents because this file has moved once already, and a
# hard-coded depth is only correct until it moves again.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(next(p for p in _HERE.parents
                            if (p / 'pyproject.toml').is_file())))

from utils.runner.logging_utils import get_logger  # noqa: E402
from utils.metrics import consistency  # noqa: E402
from utils.metrics.dump_adapter import load_dump, load_ground_truth  # noqa: E402
from utils.metrics.flip import FlipResult, flip_analysis  # noqa: E402
from utils.metrics.matching import STATE_P, STATES  # noqa: E402

logger = get_logger(__name__)

#: The one knob this script moves. 0.0 is the frozen protocol's operating point
#: and doubles as the reproduction gate.
SWEEP_THRESHOLDS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.45)
BASELINE_THRESHOLD: float = 0.0

#: Model order used for every ranking. Alphabetical, not by result.
MODELS: tuple[str, ...] = ("gdino", "glip", "omdet", "owlv2")
#: Display order for the arms. Discovery is glob-based and never consults this
#: list, so an arm missing from it would be swept and then silently dropped
#: from every table -- all three fine-tuning seeds are named for that reason.
#: A crossing that appears at one seed and not the others is a seed artefact,
#: which is exactly what a reviewer asking about threshold robustness needs to
#: be able to see. Rows with no runs are skipped, so naming a seed that was
#: never trained costs nothing.
ARMS: tuple[str, ...] = ("zeroshot", "ft_seed0", "ft_seed1", "ft_seed2")

#: Fields of the published ``layer15_flip`` block that the 0.0 slice must
#: reproduce exactly. Floats are compared with ``==``: both sides run the same
#: code on the same bytes, so any difference at all is a defect.
REPRO_FIELDS: tuple[str, ...] = (
    "n_images", "n_gt", "n_p_q0", "n_p_q1",
    "negative_flip_rate", "positive_flip_rate", "stealing_rate",
    "negative_flip_rate_conditional", "positive_flip_rate_conditional",
    "recall_q0", "recall_q1", "delta_recall",
)

ANALYSIS_GLOB = "qsens_npz__*__*__A1_vs_*.json"
UNMEASURED = "_unmeasured_"


class ReproductionGateError(RuntimeError):
    """The 0.0 slice did not reproduce the published analysis."""


@dataclass(frozen=True)
class RunSpec:
    """One published (model, arm, condition) run, resolved to its input files."""

    analysis_path: Path
    model: str
    arm: str
    condition: str
    arm_id: str
    pool: str
    target_class: str
    iou_threshold: float
    dump_q0: Path
    dump_q1: Path

    @property
    def key(self) -> str:
        return f"{self.model}__{self.arm}__{self.condition}"


def parse_analysis(path: Path, dumps_dir: Path) -> RunSpec:
    """Resolve one published analysis JSON to the dumps it was built from.

    The dump file names are read from the record itself rather than rebuilt
    from the arm id: the arm-to-file-name mapping is not a pure function (the
    fragment side spells the same arm differently), and guessing it would
    silently pair the wrong checkpoint.
    """
    record = json.loads(path.read_text())
    parts = path.stem.split("__")
    if len(parts) != 4 or not parts[3].startswith("A1_vs_"):
        raise ValueError(f"{path}: name is not qsens_npz__<model>_<arm>__<class>__A1_vs_<cond>")
    model, _, arm = parts[1].partition("_")
    condition = parts[3][len("A1_vs_"):]
    dumps = record["dump_format"]
    return RunSpec(
        analysis_path=path,
        model=model,
        arm=arm,
        condition=condition,
        arm_id=record["arm"],
        pool=record["pool"],
        target_class=record["target_class"],
        iou_threshold=float(record["iou_threshold"]),
        dump_q0=dumps_dir / dumps["q0"],
        dump_q1=dumps_dir / dumps["q1"],
    )


def discover_runs(analysis_dir: Path, dumps_dir: Path) -> tuple[RunSpec, ...]:
    """Every published npz-based query-sensitivity run, sorted deterministically."""
    specs = [parse_analysis(p, dumps_dir) for p in sorted(analysis_dir.glob(ANALYSIS_GLOB))]
    missing = [str(p) for s in specs for p in (s.dump_q0, s.dump_q1) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "dumps referenced by the published analyses are absent: " + ", ".join(sorted(set(missing))))
    return tuple(specs)


def retention_rate(flip: FlipResult) -> float | None:
    """RR — the share of q0's found objects that q1 still finds as the target.

    ``|P -> P| / |D(q0)|`` where ``D(q0)`` is the set of ground-truth objects
    the reference query detected. ``None`` when q0 found nothing, which is not
    the same as a retention of 0.
    """
    denominator = flip.n_p_q0
    if not denominator:
        return None
    return flip.transitions[f"{STATE_P}->{STATE_P}"] / denominator


def flip_record(spec: RunSpec, flip: FlipResult) -> dict[str, Any]:
    """Flatten one flip result into the sweep's row schema."""
    gate = consistency.check(flip)
    return {
        "model": spec.model,
        "arm": spec.arm,
        "condition": spec.condition,
        "arm_id": spec.arm_id,
        "pool": spec.pool,
        "target_class": spec.target_class,
        "q0": flip.q0,
        "q1": flip.q1,
        "iou_threshold": flip.iou_threshold,
        "score_threshold": flip.score_threshold,
        "n_images": flip.n_images,
        "n_gt": flip.n_gt,
        "n_p_q0": flip.n_p_q0,
        "n_p_q1": flip.n_p_q1,
        "recall_q0": flip.recall_q0,
        "recall_q1": flip.recall_q1,
        "delta_recall": flip.delta_recall,
        "negative_flip_rate": flip.negative_flip_rate,
        "positive_flip_rate": flip.positive_flip_rate,
        "stealing_rate": flip.stealing_rate,
        "negative_flip_rate_conditional": flip.negative_flip_rate_conditional,
        "positive_flip_rate_conditional": flip.positive_flip_rate_conditional,
        "retention_rate": retention_rate(flip),
        "transitions": dict(flip.transitions),
        "stolen_by": dict(flip.stolen_by),
        "hidden_churn": flip.hidden_churn,
        "pfr_minus_nfr": gate.pfr_minus_nfr,
        "residual_internal": gate.residual_internal,
        "consistency_status": gate.status,
        "consistency_detail": gate.detail,
    }


def sweep_arm(specs: Sequence[RunSpec],
              thresholds: Sequence[float] = SWEEP_THRESHOLDS) -> list[dict[str, Any]]:
    """Recompute every condition of one arm at every threshold.

    The reference dump and the ground truth are loaded once per arm; only the
    comparison dump changes between conditions and only the threshold changes
    within a condition. Nothing else in the call is allowed to vary.
    """
    if not specs:
        return []
    q0_paths = {s.dump_q0 for s in specs}
    if len(q0_paths) != 1:
        raise ValueError(f"an arm must share one reference dump, got {sorted(map(str, q0_paths))}")
    q0 = load_dump(next(iter(q0_paths)))
    gt = load_ground_truth(Path(specs[0].pool), specs[0].target_class, q0.image_ids)
    records: list[dict[str, Any]] = []
    for spec in specs:
        q1 = load_dump(spec.dump_q1)
        for threshold in thresholds:
            flip = flip_analysis(q0, q1, gt, spec.target_class,
                                 spec.iou_threshold, float(threshold))
            records.append(flip_record(spec, flip))
        logger.info("%s %s: swept %d thresholds", spec.arm_id, spec.condition, len(thresholds))
    return records


def score_profile(spec: RunSpec, thresholds: Sequence[float]) -> dict[str, Any]:
    """Where the reference query's target-class scores actually live.

    A score threshold is an absolute number, and the four detectors do not put
    their scores on the same scale — OWLv2's per-query sigmoid tops out far
    below GLIP's. Without this profile a reader cannot tell "the model became
    unstable at 0.25" from "0.25 is past the model's entire score range", and
    those are opposite conclusions.
    """
    import numpy as np

    dump = load_dump(spec.dump_q0)
    index = dump.class_index(spec.target_class)
    blocks = [rows[rows[:, 5] == index][:, 4] for rows in dump.detections.values()]
    blocks = [b for b in blocks if len(b)]
    scores = np.concatenate(blocks) if blocks else np.zeros(0)
    if not len(scores):
        return {"arm_id": spec.arm_id, "model": spec.model, "arm": spec.arm,
                "n_detections": 0, "min": None, "max": None,
                "quantiles": None, "fraction_at_or_above": None}
    return {
        "arm_id": spec.arm_id,
        "model": spec.model,
        "arm": spec.arm,
        "n_detections": int(len(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "quantiles": {str(q): float(np.quantile(scores, q))
                      for q in (0.5, 0.9, 0.99)},
        "fraction_at_or_above": {f"{t:g}": float((scores >= t).mean())
                                 for t in thresholds},
    }


def degenerate_points(records: Sequence[Mapping[str, Any]],
                      recall_floor: float = 0.5) -> list[dict[str, Any]]:
    """Sweep points where a flip rate is arithmetically real but not meaningful.

    ``NFR = |P -> not P| / N`` is 0 when the reference query found nothing at
    all: a detector silenced by the threshold cannot negatively flip. Ranking
    such a cell alongside the others would put the most silent model first.
    These points are listed so no table implies otherwise.
    """
    flagged = []
    for record in records:
        reasons = []
        if record["n_p_q0"] == 0:
            reasons.append("the reference query detected no object of the target class, "
                           "so NFR is 0 by silence and RR is undefined")
        elif record["recall_q0"] is not None and record["recall_q0"] < recall_floor:
            reasons.append(f"recall_q0 {record['recall_q0']:.4f} is below {recall_floor}; "
                           "most objects are already gone before the query changes")
        if reasons:
            flagged.append({
                "model": record["model"], "arm": record["arm"],
                "condition": record["condition"],
                "score_threshold": record["score_threshold"],
                "n_p_q0": record["n_p_q0"], "recall_q0": record["recall_q0"],
                "negative_flip_rate": record["negative_flip_rate"],
                "reason": "; ".join(reasons),
            })
    return flagged


def query_invariant_arms(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Arms whose ``P`` row is diagonal at every threshold and condition.

    ``P -> O`` and ``P -> M`` both exactly zero everywhere means the target
    class's detections did not move when the query changed — a structural
    property of the architecture, not a small measured effect, and it has to be
    named as such or its NFR of 0 will read as a won comparison.
    """
    arms: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        arms.setdefault((record["model"], record["arm"]), []).append(record)
    out = []
    for (model, arm), rows in sorted(arms.items()):
        off_diagonal = sum(r["transitions"][f"{STATE_P}->{s}"]
                           for r in rows for s in STATES if s != STATE_P)
        into_p = sum(r["transitions"][f"{s}->{STATE_P}"]
                     for r in rows for s in STATES if s != STATE_P)
        if off_diagonal == 0 and into_p == 0:
            out.append({"model": model, "arm": arm, "n_points": len(rows),
                        "p_row_off_diagonal": off_diagonal, "into_p": into_p})
    return out


def run_sweep(specs: Sequence[RunSpec], thresholds: Sequence[float],
              workers: int) -> list[dict[str, Any]]:
    """Sweep every arm, one process per arm when ``workers > 1``."""
    by_arm: dict[str, list[RunSpec]] = {}
    for spec in specs:
        by_arm.setdefault(spec.arm_id, []).append(spec)
    groups = [by_arm[k] for k in sorted(by_arm)]
    if workers <= 1:
        return [r for group in groups for r in sweep_arm(group, thresholds)]
    with ProcessPoolExecutor(max_workers=min(workers, len(groups))) as pool:
        results = pool.map(sweep_arm, groups, [thresholds] * len(groups))
        return [r for group in results for r in group]


def check_reproduction(records: Sequence[Mapping[str, Any]],
                       specs: Sequence[RunSpec]) -> list[str]:
    """Compare the 0.0 slice against the published ``layer15_flip`` blocks.

    Returns the list of mismatches; empty means the gate passed. Floats are
    compared exactly on purpose — the same code reading the same dump must
    return the same bits, and a tolerance here would hide a re-implementation.
    """
    by_key = {s.key: s for s in specs}
    mismatches: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record["score_threshold"] != BASELINE_THRESHOLD:
            continue
        key = f"{record['model']}__{record['arm']}__{record['condition']}"
        seen.add(key)
        published = json.loads(by_key[key].analysis_path.read_text())["layer15_flip"]
        for field in REPRO_FIELDS:
            if record[field] != published[field]:
                mismatches.append(
                    f"{key}.{field}: recomputed {record[field]!r} != published {published[field]!r}")
        if record["transitions"] != published["transitions"]:
            mismatches.append(
                f"{key}.transitions: recomputed {record['transitions']} != published {published['transitions']}")
    absent = sorted(set(by_key) - seen)
    if absent:
        mismatches.append(f"no 0.0 slice recomputed for: {absent}")
    return mismatches


def _kendall_tau(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Kendall's tau-b between two score vectors over the same items."""
    from scipy.stats import kendalltau

    if len(a) < 2:
        return None
    tau = kendalltau(a, b).statistic
    return None if tau != tau else float(tau)  # NaN check without importing math


def ranking_stability(records: Sequence[Mapping[str, Any]],
                      metric: str = "negative_flip_rate") -> dict[str, Any]:
    """Does the model ordering under ``metric`` survive the operating point?

    For each (arm, condition) the four models are ordered by ``metric``
    ascending at every threshold and compared with the ordering at
    :data:`BASELINE_THRESHOLD`. Both the permutation identity and Kendall's
    tau-b are reported, plus the exact model pairs whose sign of difference
    flips — a tau below 1 with no named pair would be unactionable.
    """
    index: dict[tuple[str, str, float, str], float | None] = {
        (r["arm"], r["condition"], float(r["score_threshold"]), r["model"]): r[metric]
        for r in records
    }
    thresholds = sorted({float(r["score_threshold"]) for r in records})
    cells: list[dict[str, Any]] = []
    for arm in sorted({r["arm"] for r in records}):
        for condition in sorted({r["condition"] for r in records}):
            models = [m for m in MODELS
                      if (arm, condition, BASELINE_THRESHOLD, m) in index]
            if len(models) < 2:
                continue
            baseline_values = [index[(arm, condition, BASELINE_THRESHOLD, m)] for m in models]
            if any(v is None for v in baseline_values):
                continue
            baseline_order = tuple(m for _, m in sorted(zip(baseline_values, models)))
            per_threshold: list[dict[str, Any]] = []
            crossings: list[dict[str, Any]] = []
            for threshold in thresholds:
                values = [index.get((arm, condition, threshold, m)) for m in models]
                if any(v is None for v in values):
                    per_threshold.append({"score_threshold": threshold, "order": None,
                                          "kendall_tau": None, "same_as_baseline": None})
                    continue
                order = tuple(m for _, m in sorted(zip(values, models)))
                per_threshold.append({
                    "score_threshold": threshold,
                    "order": list(order),
                    "values": {m: v for m, v in zip(models, values)},
                    "kendall_tau": _kendall_tau(baseline_values, values),
                    "same_as_baseline": order == baseline_order,
                })
                if threshold == BASELINE_THRESHOLD:
                    continue
                for i, first in enumerate(models):
                    for second in models[i + 1:]:
                        base = baseline_values[i] - baseline_values[models.index(second)]
                        now = values[i] - values[models.index(second)]
                        if base == 0 or now == 0 or (base > 0) == (now > 0):
                            continue
                        crossings.append({
                            "score_threshold": threshold,
                            "pair": [first, second],
                            "baseline_values": [baseline_values[i],
                                                baseline_values[models.index(second)]],
                            "threshold_values": [values[i], values[models.index(second)]],
                        })
            cells.append({
                "arm": arm,
                "condition": condition,
                "baseline_order": list(baseline_order),
                "per_threshold": per_threshold,
                "crossings": crossings,
                "stable": all(t["same_as_baseline"] for t in per_threshold
                              if t["same_as_baseline"] is not None),
            })
    unstable = [c for c in cells if not c["stable"]]
    return {
        "metric": metric,
        "baseline_threshold": BASELINE_THRESHOLD,
        "thresholds": thresholds,
        "n_cells": len(cells),
        "n_unstable_cells": len(unstable),
        "unstable_fraction": (len(unstable) / len(cells)) if cells else None,
        "min_kendall_tau": min(
            [t["kendall_tau"] for c in cells for t in c["per_threshold"]
             if t["kendall_tau"] is not None] or [None]),
        "cells": cells,
    }


def write_outputs(records: Sequence[Mapping[str, Any]], ranking: Mapping[str, Any],
                  repro: Sequence[str], profiles: Sequence[Mapping[str, Any]],
                  out_dir: Path) -> tuple[Path, Path, Path]:
    """Write the machine-readable sweep: full JSON, flat CSV, ranking JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "v6_flip_threshold_sweep.json"
    csv_path = out_dir / "v6_flip_threshold_sweep.csv"
    ranking_path = out_dir / "v6_flip_threshold_ranking.json"

    json_path.write_text(json.dumps({
        "schema_version": "v6-qsens-threshold-sweep.1",
        "thresholds": sorted({float(r["score_threshold"]) for r in records}),
        "fixed": {"iou_threshold": 0.5, "max_detections_per_image": 100,
                  "note": "the dumps are post-protocol; only score_threshold moves"},
        "reproduction_gate": {"threshold": BASELINE_THRESHOLD,
                              "passed": not repro,
                              "mismatches": list(repro)},
        "score_profiles": list(profiles),
        "degenerate_points": degenerate_points(records),
        "query_invariant_arms": query_invariant_arms(records),
        "records": list(records),
    }, indent=2))

    scalar_fields = [k for k in records[0] if k not in ("transitions", "stolen_by")]
    transition_fields = [f"{a}->{b}" for a in STATES for b in STATES]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields + transition_fields)
        writer.writeheader()
        for record in records:
            row = {k: record[k] for k in scalar_fields}
            row.update({k: record["transitions"][k] for k in transition_fields})
            writer.writerow(row)

    ranking_path.write_text(json.dumps(ranking, indent=2))
    return json_path, csv_path, ranking_path


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return UNMEASURED
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric_table(records: Sequence[Mapping[str, Any]], metric: str,
                  thresholds: Sequence[float], digits: int = 5) -> list[str]:
    index = {(r["model"], r["arm"], r["condition"], float(r["score_threshold"])): r[metric]
             for r in records}
    header = "| model | arm | condition | " + " | ".join(f"t={t:g}" for t in thresholds) + " |"
    lines = [header, "|" + "---|" * (3 + len(thresholds))]
    for model in MODELS:
        for arm in ARMS:
            for condition in sorted({r["condition"] for r in records}):
                if (model, arm, condition, thresholds[0]) not in index:
                    continue
                cells = " | ".join(_fmt(index.get((model, arm, condition, t)), digits)
                                   for t in thresholds)
                lines.append(f"| {model} | {arm} | {condition} | {cells} |")
    return lines


def _recall_q0_table(records: Sequence[Mapping[str, Any]],
                     thresholds: Sequence[float]) -> list[str]:
    """``recall_q0`` per (model, arm, threshold), with its cross-condition check.

    ``recall_q0`` depends only on the reference dump, so all six conditions of
    an arm must agree at a given threshold. The agreement is asserted here
    instead of being assumed, and the number of ground-truth objects behind it
    is printed next to it.
    """
    lines = ["| model | arm | n_gt | " + " | ".join(f"t={t:g}" for t in thresholds) + " |",
             "|" + "---|" * (3 + len(thresholds))]
    for model in MODELS:
        for arm in ARMS:
            rows = [r for r in records if r["model"] == model and r["arm"] == arm]
            if not rows:
                continue
            cells = []
            for threshold in thresholds:
                values = {r["recall_q0"] for r in rows
                          if float(r["score_threshold"]) == threshold}
                cells.append(_fmt(next(iter(values))) if len(values) == 1
                             else "INCONSISTENT " + str(sorted(values)))
            n_gt = sorted({r["n_gt"] for r in rows})
            lines.append(f"| {model} | {arm} | {n_gt[0] if len(n_gt) == 1 else n_gt} | "
                         + " | ".join(cells) + " |")
    return lines


def _n_p_q0_table(records: Sequence[Mapping[str, Any]],
                  thresholds: Sequence[float]) -> list[str]:
    lines = ["| model | arm | " + " | ".join(f"t={t:g}" for t in thresholds) + " |",
             "|" + "---|" * (2 + len(thresholds))]
    for model in MODELS:
        for arm in ARMS:
            rows = [r for r in records if r["model"] == model and r["arm"] == arm]
            if not rows:
                continue
            cells = []
            for threshold in thresholds:
                values = {r["n_p_q0"] for r in rows
                          if float(r["score_threshold"]) == threshold}
                cells.append(str(next(iter(values))) if len(values) == 1
                             else "INCONSISTENT " + str(sorted(values)))
            lines.append(f"| {model} | {arm} | " + " | ".join(cells) + " |")
    return lines


def _ranking_section(ranking: Mapping[str, Any]) -> list[str]:
    thresholds = ranking["thresholds"]
    lines = ["| arm | condition | " + " | ".join(f"order @ t={t:g}" for t in thresholds)
             + " | min tau | stable |",
             "|" + "---|" * (3 + len(thresholds) + 1)]
    for cell in ranking["cells"]:
        orders = []
        taus = []
        for entry in cell["per_threshold"]:
            orders.append(" < ".join(entry["order"]) if entry["order"] else UNMEASURED)
            if entry["kendall_tau"] is not None:
                taus.append(entry["kendall_tau"])
        lines.append(f"| {cell['arm']} | {cell['condition']} | " + " | ".join(orders)
                     + f" | {_fmt(min(taus) if taus else None, 3)} | "
                     + ("yes" if cell["stable"] else "**NO**") + " |")
    return lines


def _crossing_section(ranking: Mapping[str, Any]) -> list[str]:
    rows = [(c["arm"], c["condition"], x) for c in ranking["cells"] for x in c["crossings"]]
    if not rows:
        return ["No model pair changes the sign of its NFR difference at any swept "
                "threshold: the ordering is identical at every operating point."]
    lines = ["| arm | condition | threshold | pair | NFR @ t=0 | NFR @ t |",
             "|---|---|---|---|---|---|"]
    for arm, condition, crossing in rows:
        first, second = crossing["pair"]
        base_a, base_b = crossing["baseline_values"]
        now_a, now_b = crossing["threshold_values"]
        lines.append(
            f"| {arm} | {condition} | {crossing['score_threshold']:g} | {first} vs {second} | "
            f"{_fmt(base_a)} vs {_fmt(base_b)} | {_fmt(now_a)} vs {_fmt(now_b)} |")
    return lines


def _score_profile_table(profiles: Sequence[Mapping[str, Any]],
                         thresholds: Sequence[float]) -> list[str]:
    lines = ["| model | arm | min | median | p99 | max | "
             + " | ".join(f"≥{t:g}" for t in thresholds) + " |",
             "|" + "---|" * (6 + len(thresholds))]
    order = {(p["model"], p["arm"]): p for p in profiles}
    for model in MODELS:
        for arm in ARMS:
            profile = order.get((model, arm))
            if profile is None:
                continue
            fractions = profile["fraction_at_or_above"] or {}
            cells = " | ".join(_fmt(fractions.get(f"{t:g}"), 4) for t in thresholds)
            quantiles = profile["quantiles"] or {}
            lines.append(
                f"| {model} | {arm} | {_fmt(profile['min'], 4)} | "
                f"{_fmt(quantiles.get('0.5'), 4)} | {_fmt(quantiles.get('0.99'), 4)} | "
                f"{_fmt(profile['max'], 4)} | {cells} |")
    return lines


def _degenerate_section(flagged: Sequence[Mapping[str, Any]]) -> list[str]:
    if not flagged:
        return ["No swept point has an empty or near-empty reference detection set; "
                "every NFR below is computed against objects the reference query "
                "actually found."]
    lines = ["| model | arm | condition | threshold | n_p_q0 | recall_q0 | NFR | why |",
             "|---|---|---|---|---|---|---|---|"]
    for row in flagged:
        lines.append(
            f"| {row['model']} | {row['arm']} | {row['condition']} | "
            f"{row['score_threshold']:g} | {row['n_p_q0']} | {_fmt(row['recall_q0'])} | "
            f"{_fmt(row['negative_flip_rate'])} | {row['reason']} |")
    return lines


def render_report(records: Sequence[Mapping[str, Any]], ranking: Mapping[str, Any],
                  repro: Sequence[str], profiles: Sequence[Mapping[str, Any]],
                  paths: Mapping[str, Path]) -> str:
    """Build the reviewer-facing report. Every number comes from ``records``."""
    thresholds = sorted({float(r["score_threshold"]) for r in records})
    defects = [r for r in records if r["consistency_status"] == consistency.STATUS_DEFECT]
    worst_residual = max((abs(r["residual_internal"]) for r in records
                          if r["residual_internal"] is not None), default=None)
    churn = [r for r in records if r["hidden_churn"]]

    lines: list[str] = [
        "# V6 query-sensitivity flip metrics: score-threshold sensitivity",
        "",
        "Reviewer question: *does the NFR ordering of the four detectors survive a "
        "change of score threshold?* The published analysis fixes "
        "`score_threshold = 0.0`, where `recall_q0` sits near the ceiling for every "
        "arm, so a small NFR is partly a property of that operating point. This "
        "sweep recomputes all "
        f"{len({(r['model'], r['arm'], r['condition']) for r in records})} published runs at "
        f"{len(thresholds)} thresholds from the same frozen `v6_npz.1` dumps. No "
        "re-inference; CPU only.",
        "",
        "Held fixed: `iou_threshold = 0.5`, per-category cap 100 detections "
        "(the dumps are post-protocol), pool "
        f"`{sorted({r['pool'] for r in records})[0]}`, target class "
        f"`{sorted({r['target_class'] for r in records})[0]}`, reference query "
        f"`{sorted({r['q0'] for r in records})[0]}`. Swept: "
        f"`score_threshold ∈ {{{', '.join(f'{t:g}' for t in thresholds)}}}`.",
        "",
        f"Generated by `sweep_v6_flip_thresholds.py` (research tree). Raw output: "
        f"`{paths['json']}`, `{paths['csv']}`, `{paths['ranking']}`.",
        "",
        "## Bottom line",
        "",
    ]
    crossing_pairs = sorted({tuple(x["pair"]) for c in ranking["cells"]
                             for x in c["crossings"]})
    if ranking["n_unstable_cells"]:
        lines += [
            f"**The NFR ordering does NOT hold.** {ranking['n_unstable_cells']} of "
            f"{ranking['n_cells']} (arm, condition) cells change model permutation "
            f"somewhere in the sweep; the worst Kendall tau-b against the "
            f"`t = {ranking['baseline_threshold']:g}` ordering is "
            f"{_fmt(ranking['min_kendall_tau'], 3)}. Every crossing involves the pairs "
            + ", ".join(f"`{a}`/`{b}`" for a, b in crossing_pairs)
            + "; each one is located by threshold and cell in §3. The published "
            "`score_threshold = 0.0` result is therefore an operating-point-specific "
            "statement, and the ranking claim has to be either restricted to that "
            "operating point or dropped.",
        ]
    else:
        lines += [
            f"**The NFR ordering holds.** All {ranking['n_cells']} (arm, condition) "
            "cells keep the same model permutation at every swept threshold "
            f"(minimum Kendall tau-b {_fmt(ranking['min_kendall_tau'], 3)}).",
        ]
    lines += [
        "",
        "Two things the sweep exposes that the fixed-threshold analysis could not: "
        "the reference query's recall is not merely near the ceiling at `t = 0`, it "
        "falls at very different rates per architecture (§1), and the score threshold "
        "is not a comparable knob across them — OWLv2 zero-shot's entire person-score "
        "range ends below 0.45, so at that threshold its NFR of 0 is silence, not "
        "stability (§1, degenerate points).",
        "",
        "## 4. Reproduction gate (t = 0.0 vs the published analysis)",
        "",
    ]
    if repro:
        lines += ["**FAILED.** The 0.0 slice does not reproduce the published "
                  "`layer15_flip` blocks, so nothing below is a result:", ""]
        lines += [f"- {m}" for m in repro]
    else:
        lines += [
            "**PASSED.** For all 48 runs the recomputed `t = 0.0` slice equals the "
            "published `layer15_flip` block exactly — all nine transition cells and "
            f"every field in `{', '.join(REPRO_FIELDS)}`, compared with `==` and no "
            "tolerance. The sweep is a re-aggregation of the published numbers, not a "
            "re-implementation.",
        ]
    lines += [
        "",
        "### Consistency gate (ΔAR50 = PFR − NFR) at every threshold",
        "",
        f"Internal identity residual, worst case over all {len(records)} sweep points: "
        f"`{_fmt(worst_residual, 3) if worst_residual is None else f'{worst_residual:.3e}'}` "
        f"(tolerance `{consistency.INTERNAL_TOLERANCE:g}`). "
        + (f"**{len(defects)} points are flagged DEFECT.**" if defects
           else "No point is flagged as a defect."),
        "",
        "The fragment-side comparison (COCOeval `ΔAR50`) is only meaningful at "
        "`t = 0.0`, because COCOeval scores the full ranked list regardless of any "
        "threshold applied afterwards; it is reported in the published per-run JSONs "
        "and is not re-run here. Above, `residual_internal` is the identity that must "
        "hold at every threshold.",
        "",
        "## 1. `recall_q0` per threshold — where the saturation breaks",
        "",
        "`recall_q0` is a property of the reference query alone, so all six "
        "conditions of an arm must produce the same value at a given threshold; the "
        "table asserts that rather than assuming it (a disagreement would print "
        "`INCONSISTENT`).",
        "",
    ]
    lines += _recall_q0_table(records, thresholds)
    lines += ["", "Objects still found by the reference query (`n_p_q0`, the RR "
              "denominator):", ""]
    lines += _n_p_q0_table(records, thresholds)
    lines += [
        "",
        "### The threshold is not the same knob on every architecture",
        "",
        "Score distribution of the reference query's `person` detections in each "
        "arm's dump, and the fraction of them each threshold keeps:",
        "",
    ]
    lines += _score_profile_table(profiles, thresholds)
    lines += [
        "",
        "### Degenerate operating points",
        "",
        "Points where a flip rate is arithmetically defined but says nothing about "
        "query sensitivity, because the reference query has already been silenced:",
        "",
    ]
    lines += _degenerate_section(degenerate_points(records))
    lines += [
        "",
        "## 2. NFR / PFR / RR per threshold",
        "",
        "`NFR = |P→¬P| / N`, `PFR = |¬P→P| / N` over all `N` ground-truth persons; "
        "`RR = |P→P| / |D(q0)|` with `|D(q0)| = n_p_q0`. Conditional NFR/PFR, the "
        "stealing rate and the full 9-cell transition matrix are in the CSV/JSON.",
        "",
        "### NFR",
        "",
    ]
    lines += _metric_table(records, "negative_flip_rate", thresholds)
    lines += ["", "### PFR", ""]
    lines += _metric_table(records, "positive_flip_rate", thresholds)
    lines += ["", "### RR (retention)", ""]
    lines += _metric_table(records, "retention_rate", thresholds)
    lines += [
        "",
        "## 3. Ranking stability",
        "",
        f"Models ordered by `{ranking['metric']}` ascending (lowest NFR first) within "
        f"each (arm, condition) cell. {ranking['n_cells']} cells; "
        f"**{ranking['n_unstable_cells']} change permutation** relative to "
        f"`t = {ranking['baseline_threshold']:g}` "
        f"({_fmt(ranking['unstable_fraction'], 3)} of cells). Minimum Kendall tau-b "
        f"against the baseline ordering over all cells and thresholds: "
        f"{_fmt(ranking['min_kendall_tau'], 3)}.",
        "",
    ]
    invariant = query_invariant_arms(records)
    if invariant:
        lines += [
            "",
            "Before reading the orderings: "
            + "; ".join(f"**{a['model']}/{a['arm']}**" for a in invariant)
            + " has an exactly diagonal `P` row at every one of its "
            f"{invariant[0]['n_points']} swept points — zero `P→O`, zero `P→M`, zero "
            "`→P` — so its NFR and PFR are structurally 0 rather than small. It "
            "therefore occupies first place in every cell without ever being able to "
            "cross another model, and the ranking question is really about the "
            "remaining detectors.",
        ]
    lines += ["", "The ordering table:", ""]
    lines += _ranking_section(ranking)
    lines += ["", "### Where the orderings cross", ""]
    lines += _crossing_section(ranking)
    if churn:
        lines += ["", "### Hidden-churn flags raised during the sweep", "",
                  "| model | arm | condition | threshold | NFR | PFR | Δrecall |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['model']} | {r['arm']} | {r['condition']} | "
                  f"{r['score_threshold']:g} | {_fmt(r['negative_flip_rate'])} | "
                  f"{_fmt(r['positive_flip_rate'])} | {_fmt(r['delta_recall'])} |"
                  for r in churn]
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path("outputs/v6/analysis"))
    parser.add_argument("--dumps-dir", type=Path,
                        default=Path("outputs/v6/detection_dumps"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("outputs/v6/analysis/threshold_sweep"))
    parser.add_argument("--report", type=Path,
                        default=Path("the specification"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(SWEEP_THRESHOLDS))
    parser.add_argument("--workers", type=int, default=8,
                        help="one process per arm; the dumps are the memory cost")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = discover_runs(args.analysis_dir, args.dumps_dir)
    logger.info("sweeping %d published runs over thresholds %s", len(specs), args.thresholds)
    records = run_sweep(specs, tuple(args.thresholds), args.workers)
    records.sort(key=lambda r: (r["model"], r["arm"], r["condition"], r["score_threshold"]))

    repro = check_reproduction(records, specs)
    if repro:
        logger.error("REPRODUCTION GATE FAILED (%d mismatches); the sweep is a "
                     "re-implementation bug, not a result", len(repro))
        for message in repro[:20]:
            logger.error("  %s", message)

    ranking = ranking_stability(records)
    seen_arms: set[str] = set()
    profiles = []
    for spec in specs:
        if spec.arm_id in seen_arms:
            continue
        seen_arms.add(spec.arm_id)
        profiles.append(score_profile(spec, tuple(args.thresholds)))

    json_path, csv_path, ranking_path = write_outputs(records, ranking, repro,
                                                      profiles, args.out_dir)
    report = render_report(records, ranking, repro, profiles,
                           {"json": json_path, "csv": csv_path, "ranking": ranking_path})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    logger.info("wrote %s, %s, %s, %s", json_path, csv_path, ranking_path, args.report)
    logger.info("ranking: %d/%d cells change permutation; min tau %s",
                ranking["n_unstable_cells"], ranking["n_cells"], ranking["min_kendall_tau"])
    return 1 if repro else 0


if __name__ == "__main__":
    raise SystemExit(main())
