"""One environment record, and something that compares it.

On 2026-08-10 two OWLv2 runs with identical ``runner_version`` fingerprints came
from different ``transformers`` releases whose image processors pad the OWLv2
letterbox differently — 4.57.3 fills 0.5 grey, 5.14.1 fills 0 black — and ~44%
of a 1080p canvas is pad. Those were different experiments and nothing in the
fragment said so; it had to be reconstructed from shell scripts afterwards.

Both runners were then taught to stamp their environment, but independently,
and they disagreed on every name:

===================  ==========================  ==========================
                     ``run_v6_condition.py``     ``run_v6_condition_xarch.py``
===================  ==========================  ==========================
fragment key         ``env``                     ``environment``
venv                 ``venv_path`` (full path)   ``venv`` (basename)
transformers         ``transformers``            ``transformers_version``
torch                ``torch``                   ``torch_version``
===================  ==========================  ==========================

So a reader looking for ``environment`` saw ``null`` on GDINO and OWLv2 — the
two models the padding incident actually bit — and no comparison across the
corpus was possible at all. Recording beside the hash was never the flaw;
having nothing that *compares* the record was, and that is what this module
is for.

:func:`environment_record` is the single writer. :func:`read_environment`
normalises both historical shapes so existing fragments stay readable — they
are not rewritten, because a fragment records what its run actually saw and
editing that would be falsification. :func:`compare_environments` is the piece
that was missing: it reports, per model, whether every fragment agrees.

Not in the ``runner_version`` fingerprint scope (see
``utils/runner/runner_version.py``), so changing anything here cannot move a
fingerprint or invalidate a result.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: The fragment key. ``env`` is the older GDINO/OWLv2 spelling and is still
#: read, never written.
FRAGMENT_KEY = "environment"
LEGACY_FRAGMENT_KEY = "env"

#: Libraries whose version can change a result without changing any of our
#: code. ``transformers`` is here because of the padding incident above;
#: ``torch``/``torchvision`` because kernel selection moves numerics; the mm*
#: and ``peft`` entries because they own GLIP's and OmDet's inference paths.
TRACKED_LIBRARIES: tuple[str, ...] = (
    "transformers", "torch", "torchvision", "mmdet", "mmcv", "peft",
)

#: Old field name -> canonical field name, for both historical shapes.
_ALIASES: Mapping[str, str] = {
    "venv_path": "venv_path",
    "venv": "venv_name",
    "python_executable": "python_executable",
    "transformers_version": "transformers",
    "torch_version": "torch",
    "torchvision_version": "torchvision",
    "mmdet_version": "mmdet",
    "mmcv_version": "mmcv",
    "peft_version": "peft",
}


def environment_record() -> dict:
    """The interpreter and library versions this process actually has.

    A library that is not installed records ``None`` rather than being
    omitted: absent and unknown are different, and a comparison that silently
    skips a missing key cannot tell them apart.
    """
    record: dict[str, object] = {
        "python_executable": sys.executable,
        "venv_path": sys.prefix,
        "venv_name": Path(sys.prefix).name,
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
    }
    for name in TRACKED_LIBRARIES:
        try:
            record[name] = getattr(__import__(name), "__version__", None)
        except Exception:
            record[name] = None
    try:
        import torch

        record["cuda"] = torch.version.cuda
        record["device_name"] = (torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else None)
    except Exception:
        record.setdefault("cuda", None)
        record.setdefault("device_name", None)
    return record


def read_environment(fragment: Mapping) -> dict | None:
    """The environment of ``fragment``, whichever shape it was written in.

    Returns ``None`` only when the fragment predates environment stamping
    altogether — which is itself worth reporting, and is distinct from a
    fragment that recorded ``None`` for one library.
    """
    raw = fragment.get(FRAGMENT_KEY) or fragment.get(LEGACY_FRAGMENT_KEY)
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, object] = {}
    for key, value in raw.items():
        canonical = _ALIASES.get(key, key)
        out[canonical] = value
    if "venv_name" not in out and isinstance(out.get("venv_path"), str):
        out["venv_name"] = Path(out["venv_path"]).name
    return out


@dataclass(frozen=True)
class Divergence:
    """One field on which a model's fragments disagree."""

    model: str
    field: str
    #: value -> the fragments that recorded it, so a disagreement can be traced
    #: back to specific runs rather than only counted.
    values: Mapping[str, tuple[str, ...]]

    def __str__(self) -> str:
        parts = ", ".join(f"{v!r} x{len(f)}" for v, f in sorted(self.values.items()))
        return f"{self.model}.{self.field}: {parts}"


def compare_environments(
    fragments: Iterable[tuple[str, Mapping]],
    fields: Sequence[str] = ("venv_name", "python_version", *TRACKED_LIBRARIES),
) -> tuple[list[Divergence], list[str]]:
    """Per model, the fields on which its fragments do not all agree.

    Args:
        fragments: ``(name, fragment)`` pairs; ``name`` identifies the fragment
            in the report, so a path is the useful thing to pass.
        fields: canonical field names to compare.

    Returns:
        ``(divergences, unstamped)`` — the fields that disagree, and the names
        of fragments carrying no environment record at all.

    Comparison is per model, not global: the four models deliberately run in
    four different venvs, so a cross-model difference is the design rather
    than a defect. A difference *within* one model is the defect.
    """
    by_model: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    unstamped: list[str] = []
    for name, fragment in fragments:
        env = read_environment(fragment)
        if env is None:
            unstamped.append(name)
            continue
        by_model[str(fragment.get("model", "?"))].append((name, env))

    divergences: list[Divergence] = []
    for model, entries in sorted(by_model.items()):
        for field in fields:
            seen: dict[str, list[str]] = defaultdict(list)
            for name, env in entries:
                seen[repr(env.get(field))].append(name)
            if len(seen) > 1:
                divergences.append(Divergence(
                    model=model, field=field,
                    values={k: tuple(v) for k, v in seen.items()}))
    return divergences, sorted(unstamped)
