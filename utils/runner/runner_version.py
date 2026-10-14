"""A fingerprint of the code that decides a fragment's numbers.

Why a *derived* version and not a declared one
---------------------------------------------
A hand-maintained ``RUNNER_VERSION = "1.2"`` is a claim in a file. It goes stale
the first time someone edits the loop and forgets to bump it, and a stale
version string is worse than none: it asserts that two fragments are comparable
when they are not.

So the version here is a **content hash of the source files that can change the
numbers**. It cannot go stale, because changing any of those files changes it.

What this is for
----------------
``utils/runner/runner.py`` has a prefetch that overlaps CPU preprocessing with GPU
compute. Python reads a module at import time, so processes launched before the
change keep the old loop and processes launched after get the new one — and
both write into the same fragment directory. Whether a table mixed the two must
be a **query**, not a reconstruction from launch timestamps.

The hash answers "was this the same code?". It cannot answer "which path
through that code?", because ``--no-prefetch`` selects the serial loop from the
very same source file. So the record also carries a human-readable ``path``
label (:data:`RUNNER_PATH_SERIAL` / :data:`RUNNER_PATH_PREFETCH`) and the
prefetch settings that were in force.

The same hole existed, and mattered more, for the detection dump. Hashing
``detection_dump.py`` says the code *could* dump; it does not say that this run
did, nor in which format, because ``--no-detection-dump`` and the format switch
select different behaviour from identical bytes. The fragment
``omdet__swin-tiny-ft-seed1__A1_singleton_hat`` is the live example: it carries
a fingerprint from the dump era and no dump at all, and nothing in that
fingerprint separates "dumping was off" from "the dump was lost". So the record
now carries ``dump_format`` explicitly -- the format string a reader would have
to implement, or ``None`` for "this run wrote no dump". A fragment can no longer
imply a dump it does not have, or a format it is not in.

The prefetch is a scheduling change and is expected to leave every number bit
identical. This fingerprint is what lets that expectation be *checked* rather
than assumed: if the numbers agree across a fingerprint change, that is
evidence; if they disagree, the fingerprint says exactly which pair to compare.

Scope
-----
Only files whose contents can alter a detection, a score or a metric. Files that
merely orchestrate (the chain shell script, argument parsing) are excluded — a
change there moves *which* conditions run, not what they produce.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Mapping

# The repository root. Scope entries are repo-relative
# (`utils/runner/runner.py`, `models/gdino/detector.py`) so one root serves both.
SRC_ROOT = next(q for q in Path(__file__).resolve().parents
                if (q / 'pyproject.toml').is_file())

#: Modules that determine a fragment's numbers. The detector module is added
#: per model by :func:`runner_fingerprint`, because a GLIP change cannot affect
#: an OmDet fragment and pretending otherwise would invalidate fragments for no
#: reason.
#: Where the scope starts. Everything these modules import, transitively, is in
#: it -- the list is not maintained by hand.
#:
#: It used to be. A hand-kept list of ten modules missed `owlv2_guard.py`,
#: `adapter_eval.py` and `checkpoint.py`, all of which run inside an OWLv2
#: evaluation. A hand-kept list can declare "same code" while an unlisted module
#: differs; a closure cannot, because the thing it hashes is the thing that runs.
ENTRY_MODULES: tuple[str, ...] = (
    "utils/runner/runner.py",           # the evaluation loop
    "utils/protocol/cap_policy.py",     # which detections are submitted for scoring
    "utils/metrics/coco_eval.py",       # how AP/AR are computed
    "utils/protocol/eval_protocol.py",  # the frozen protocol values
    "utils/runner/fragment.py",         # what is recorded and validated
    "utils/protocol/conditions.py",     # the queried terms
    "utils/protocol/vocabulary.py",     # the caption rendering
)

DETECTOR_MODULES: Mapping[str, str] = {
    "gdino": "models/gdino/detector.py",
    "owlv2": "models/owlv2/detector.py",
    "glip": "models/glip/detector.py",
    "omdet": "models/omdet/detector.py",
}


def _package_imports(path: Path) -> set[str]:
    """First-party modules imported by this file, as repo-relative paths."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
    out = set()
    for name in names:
        # The two first-party top-level packages. A third would have to be
        # added here, and the closure would silently stop following it until
        # someone noticed -- so the list is short and stated, not inferred.
        if not name.startswith(("utils.", "models.")):
            continue
        rel = name.replace(".", "/")
        for candidate in (f"{rel}.py", f"{rel}/__init__.py"):
            if (SRC_ROOT / candidate).is_file():
                out.add(candidate)
    return out


def scope_for(model: str | None) -> list[str]:
    """Every package file reachable from the entry modules, sorted.

    Import-time reachability, not runtime: a module imported inside a function
    is still in scope, because editing it can still change the numbers. The
    inverse error -- hashing something that cannot affect the result -- costs a
    spurious version change, which is cheap next to a fingerprint that claims
    two different runs were the same code.
    """
    seen: set[str] = set()
    queue = list(ENTRY_MODULES)
    if model is not None:
        detector = DETECTOR_MODULES.get(model)
        if detector is None:
            raise KeyError(f"no detector module registered for model {model!r}")
        queue.append(detector)
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        path = SRC_ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is in the fingerprint scope but does not exist; refusing "
                f"to fingerprint a subset of the code that decides the numbers")
        seen.add(rel)
        # `core.detector.build_detector` imports all four detectors, each inside
        # the branch that needs it. Following every branch would put GLIP in an
        # OmDet fragment's scope, so a GLIP edit would invalidate OmDet results
        # that it cannot have touched. Only the requested model's package is
        # followed; everything outside `models/` is followed unconditionally.
        for candidate in _package_imports(path) - seen:
            if candidate.startswith("models/") and model is not None:
                if not candidate.startswith(f"models/{model}/") \
                        and not candidate.startswith("models/shared/"):
                    continue
            queue.append(candidate)
    return sorted(seen)


#: The loop that existed before the prefetch, and the one ``--no-prefetch``
#: still selects: load -> preprocess -> forward, strictly serially.
RUNNER_PATH_SERIAL: str = "v6-runner.1-serial"

#: CPU stages staged ahead on worker threads, GPU stage on the main thread,
#: results consumed in census order.
RUNNER_PATH_PREFETCH: str = "v6-runner.2-prefetch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runner_fingerprint(model: str | None = None,
                       dump_format: str | None = None) -> dict:
    """``{"version": <12 hex>, "files": {name: <12 hex>}}`` for the live source.

    ``dump_format`` is folded INTO the hash, not merely recorded beside it.
    Recording it alongside was the original design and it failed the same way
    the ``A1_singleton_hat`` fragment did: a value that sits next to the hash
    is a value nobody compares. With it inside, a campaign that mixes
    ``v6_npz.1`` with ``rev2_jsonl``, or dump-on with dump-off, produces
    different versions and is rejected instead of silently pooled.

    Args:
        model: Adds that model's detector module. ``None`` fingerprints only
            the model-independent core.
        dump_format: The dump format this run writes, or ``None`` for no dump.

    Raises:
        FileNotFoundError: a module in the scope list is missing — better to
            fail than to fingerprint a subset and call it the whole.
    """
    names = scope_for(model)

    digests: dict[str, str] = {}
    combined = hashlib.sha256()
    for name in names:
        digest = _sha256(SRC_ROOT / name)
        digests[name] = digest[:12]
        combined.update(name.encode())
        combined.update(digest.encode())
    combined.update(b"dump_format=")
    combined.update((dump_format or "none").encode())
    return {"version": combined.hexdigest()[:12], "files": digests}


def runner_version_record(
    model: str | None,
    path: str,
    prefetch_workers: int | None = None,
    prefetch_depth: int | None = None,
    detector_two_phase: bool = False,
    dump_format: str | None = None,
) -> dict:
    """The ``runner_version`` block a fragment carries.

    A superset of :func:`runner_fingerprint` (``version``/``files`` keep their
    meaning and position, so existing readers are unaffected) plus which loop
    path ran. ``detector_two_phase`` records whether the detector's own
    preprocessing was staged as well or only its image loading was.

    Args:
        dump_format: The detection dump format this run wrote
            (``utils.runner.detection_dump_binary.DUMP_FORMAT_NPZ`` or
            ``DUMP_FORMAT_JSONL``), or ``None`` if it wrote no dump. Always
            present in the record, so its absence identifies a fragment from
            before this field existed rather than a run that chose not to dump.

    An unregistered ``model`` degrades to the model-independent core rather
    than raising: refusing to record any version because the detector module is
    unknown would leave the fragment with no provenance at all, which is worse.
    """
    key = model if model in DETECTOR_MODULES else None
    record = dict(runner_fingerprint(key, dump_format=dump_format))
    if key is None and model is not None:
        record["detector_module"] = None
    record["path"] = path
    record["prefetch_workers"] = prefetch_workers
    record["prefetch_depth"] = prefetch_depth
    record["detector_two_phase"] = detector_two_phase
    record["dump_format"] = dump_format
    # Lossless-guard telemetry. Present even when zero, so a reader can tell
    # "checked, none found" from "never checked". Populated by the runner after
    # the dump is committed; see the specification.
    record.setdefault("dump_inexact_rows", 0)
    record.setdefault("dump_max_rel_err", 0.0)
    return record


def describe(fingerprint: Mapping) -> str:
    path = fingerprint.get("path")
    suffix = f", path {path}" if path else ""
    if "dump_format" in fingerprint:
        suffix += f", dump {fingerprint['dump_format'] or 'off'}"
    return (f"runner_version={fingerprint['version']} "
            f"({len(fingerprint['files'])} source files hashed{suffix})")
