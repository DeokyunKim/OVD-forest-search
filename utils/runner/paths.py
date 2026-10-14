"""The one place that knows where anything lives.

Every other module asks this for a path and never builds one from a literal.
That rule is what lets a reader run this on their own machine without editing
files.

Resolution order, first hit wins:

1. an environment variable -- ``UC_DATA_ROOT``, ``UC_OUTPUT_ROOT``,
   ``UC_CHECKPOINT_ROOT``, or ``UC_CONFIG`` to point at a different YAML;
2. ``configs/paths.yaml``;
3. the repo-relative defaults in that file.

Relative values resolve against the repository root, so a fresh clone works
with no configuration at all once ``link_datasets.sh`` has run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml

#: Repository root: the directory holding ``pyproject.toml``.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / 'pyproject.toml').is_file())
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"


def _resolve(value: str | Path, base: Path) -> Path:
    """Absolute values are honoured as given; relative ones hang off ``base``."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path)


@dataclass(frozen=True)
class Paths:
    """Resolved locations. Frozen: a run must not move its own outputs."""

    config_file: Path
    data_root: Path
    output_root: Path
    checkpoint_root: Path
    _datasets: Mapping[str, str]
    _outputs: Mapping[str, str]

    def dataset(self, name: str) -> Path:
        """Directory of a named corpus (``forest_persons``, ``benchmark``, ...).

        Unknown names raise rather than returning a plausible-looking path: a
        typo that silently yields ``data/forest_person`` would surface much
        later as an empty evaluation, which is far harder to diagnose.
        """
        try:
            return _resolve(self._datasets[name], self.data_root)
        except KeyError:
            raise KeyError(
                f"unknown dataset {name!r}; {self.config_file} defines "
                f"{sorted(self._datasets)}") from None

    def output(self, name: str) -> Path:
        """Directory for a class of artefact (``fragments``, ``figures``, ...)."""
        try:
            return _resolve(self._outputs[name], self.output_root)
        except KeyError:
            raise KeyError(
                f"unknown output {name!r}; {self.config_file} defines "
                f"{sorted(self._outputs)}") from None

    def require(self, name: str) -> Path:
        """A dataset that must already exist, or a message saying how to get it.

        Used by entry points before they load a model, so a missing corpus costs
        a second rather than the minutes it takes to put weights on a GPU.
        """
        path = self.dataset(name)
        # The symlink case is checked FIRST. `Path.exists()` follows links, so a
        # dangling one is simply "not there" to it, and the generic message
        # below would tell the reader to create a link they already created.
        # The two failures need different fixes: make the link, or repair its
        # target.
        if path.is_symlink() and not path.resolve().exists():
            raise FileNotFoundError(
                f"dataset {name!r} at {path} is a dangling symlink: it points at "
                f"{os.readlink(path)}, which does not exist. Repoint it, or "
                f"re-run link_datasets.sh. Reported here because a "
                f"dangling link does not crash a run -- it yields no images, "
                f"scores zero and says nothing.")
        if not path.exists():
            raise FileNotFoundError(
                f"dataset {name!r} is not at {path}. Either run "
                f"link_datasets.sh to symlink it, or set UC_DATA_ROOT / "
                f"edit {self.config_file}. See DATA.md for how to obtain it.")
        return path


@lru_cache(maxsize=None)
def paths(config: Path | None = None) -> Paths:
    """Load and cache the resolved paths."""
    config_file = Path(os.environ.get("UC_CONFIG", config or DEFAULT_CONFIG))
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}

    def root(key: str, env: str) -> Path:
        value = os.environ.get(env, raw.get(key, key.replace("_root", "s")))
        return _resolve(value, REPO_ROOT)

    return Paths(
        config_file=config_file,
        data_root=root("data_root", "UC_DATA_ROOT"),
        output_root=root("output_root", "UC_OUTPUT_ROOT"),
        checkpoint_root=root("checkpoint_root", "UC_CHECKPOINT_ROOT"),
        _datasets=dict(raw.get("datasets", {})),
        _outputs=dict(raw.get("outputs", {})),
    )
