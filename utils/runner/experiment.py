"""One YAML describes one experiment, and both entry points read it.

`train.py --config x.yaml` fine-tunes; `eval.py --config x.yaml` measures what
that produced. The same file drives both so the pair cannot drift: the adapter
`eval` reads is the one `train` was told to write, rather than a path retyped on
a second command line months later.

Every field a run depends on is here and nothing is inferred from a file name.
Names are labels people choose; a checkpoint that says `seed1` in its path
because someone copied a directory is not evidence of anything.

    model: gdino          # gdino | owlv2 | glip | omdet
    seed: 1
    train:
      steps: 5667
      lora: {rank: 16, alpha: 32}
    eval:
      conditions: [A1_singleton_person, A4_full_v6]
      per_dataset: true   # report ForestPersons and ForestBelongings apart
    output_dir: checkpoints/gdino/seed1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

MODELS: tuple[str, ...] = ("gdino", "owlv2", "glip", "omdet")

#: LoRA settings the published results used. Named here rather than left to each
#: trainer's default so a config that omits them still describes the run, and a
#: config that sets them is checked against something.
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_STEPS = 5667


@dataclass(frozen=True)
class LoraSpec:
    """Adapter geometry. `alpha / rank` is the scaling the merge applies."""

    rank: int = DEFAULT_RANK
    alpha: int = DEFAULT_ALPHA

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"lora.rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"lora.alpha must be positive, got {self.alpha}")


@dataclass(frozen=True)
class TrainSpec:
    steps: int = DEFAULT_STEPS
    lora: LoraSpec = field(default_factory=LoraSpec)
    num_workers: int = 4


@dataclass(frozen=True)
class EvalSpec:
    """What to measure, and how finely to report it."""

    #: Query combinations. Empty means the whole A axis -- a term alone, that
    #: term beside `person`, and the full six-word caption.
    conditions: tuple[str, ...] = ()
    #: Report ForestPersons and ForestBelongings separately as well as together.
    #: The two corpora answer different questions -- can it still find people,
    #: can it find what they dropped -- and a pooled number hides one behind the
    #: other, weighted by nothing more meaningful than how many frames each has.
    per_dataset: bool = True
    #: Raw detections are needed to score the two corpora apart without running
    #: inference twice, so per_dataset implies dumping.
    dump: bool = True
    #: Fragment label, when the default will not do. The label names the file
    #: and is recorded as the checkpoint, so comparing against results produced
    #: elsewhere means writing the label that side used -- the four models do
    #: not share one convention (`swinb_ft_seed1` against `large-ft-seed1`).
    #: It must contain `ft`, or a fine-tuned run would overwrite a zero-shot one.
    label: str | None = None
    #: Leave a condition alone when its fragment already exists. Off by
    #: default: a run that measures nothing and reports success is how a check
    #: passes without checking. Turn it on to continue a long sweep that died
    #: partway -- and because it is written here, the file records that the run
    #: may have reused earlier results rather than producing them.
    resume: bool = False
    #: `detection` runs the frozen protocol and writes a fragment. `flip`
    #: re-reads two dumps this model already produced and reports the
    #: query-sensitivity rates. Both are "what is measured", so both are
    #: described here rather than selected by a flag at the call site.
    mode: str = "detection"
    #: flip only: the query the rates are measured against, and the class scored.
    reference: str = "A1_singleton_person"
    target: str = "person"
    #: `finetuned` measures the adapter `output_dir` holds; `zeroshot` measures
    #: the base weights and reads no adapter at all. The paper reports both, and
    #: the presence of an adapter is the ONLY difference between them -- same
    #: protocol, same read-out, same post-processing -- so which arm is being
    #: measured belongs in the file rather than in whether someone remembered a
    #: flag.
    arm: str = "finetuned"


@dataclass(frozen=True)
class Experiment:
    model: str
    seed: int
    output_dir: Path
    train: TrainSpec
    eval: EvalSpec
    config_file: Path

    @property
    def label(self) -> str:
        """Fragment label for the fine-tuned arm.

        It must contain `ft`: the fragment and the detection dump are named
        after it, so a fine-tuned run under a zero-shot label would overwrite
        the zero-shot result and be indistinguishable from it afterwards.
        """
        return self.eval.label or f"ft-seed{self.seed}"

    def fragment_label(self, variant: str) -> str:
        """The name the fragment and the dump are filed under.

        An explicit `eval.label` is used whole, because the point of setting it
        is to match a name produced elsewhere and the four models do not share
        one convention. Otherwise the variant is prefixed, which is what makes
        the default label read as `large-ft-seed1`.
        """
        if self.eval.label:
            return self.eval.label
        # Zero-shot carries the variant unchanged: there is no seed to record,
        # and a label saying `ft` would be a fine-tuned filename over a
        # zero-shot measurement.
        return variant if self.eval.arm == "zeroshot" else f"{variant}-{self.label}"


def _require(raw: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in raw:
        raise ValueError(f"{path}: missing required key {key!r}")
    return raw[key]


def load_experiment(path: str | Path) -> Experiment:
    """Read and validate a config. Unknown keys raise rather than being ignored.

    Silently ignoring an unknown key is how `lora_rank: 32` sits in a file for a
    month while every run uses 16, and the config is then cited as evidence of
    what was run.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse to a mapping")

    known = {"model", "seed", "output_dir", "train", "eval"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}; expected {sorted(known)}")

    model = _require(raw, "model", path)
    if model not in MODELS:
        raise ValueError(f"{path}: model {model!r} is not one of {list(MODELS)}")
    seed = _require(raw, "seed", path)
    if not isinstance(seed, int):
        raise ValueError(f"{path}: seed must be an integer, got {seed!r}")

    train_raw = dict(raw.get("train") or {})
    lora_raw = dict(train_raw.pop("lora", None) or {})
    unknown = sorted(set(train_raw) - {"steps", "num_workers"})
    if unknown:
        raise ValueError(f"{path}: unknown train key(s) {unknown}")
    unknown = sorted(set(lora_raw) - {"rank", "alpha"})
    if unknown:
        raise ValueError(f"{path}: unknown train.lora key(s) {unknown}")

    eval_raw = dict(raw.get("eval") or {})
    unknown = sorted(set(eval_raw) - {"conditions", "per_dataset", "dump", "label",
                                  "resume", "mode", "reference", "target", "arm"})
    if unknown:
        raise ValueError(f"{path}: unknown eval key(s) {unknown}")
    per_dataset = bool(eval_raw.get("per_dataset", True))

    # `checkpoints/<model>/seed<n>/`, so one directory per architecture rather
    # than a flat pile where four models' runs interleave alphabetically. The
    # base weights already sit at `checkpoints/<model>/`, so trained adapters
    # joining them keeps everything about a model in one place.
    output_dir = Path(raw.get("output_dir")
                      or f"checkpoints/{model}/seed{seed}")
    if not output_dir.is_absolute():
        output_dir = path.resolve().parents[2] / output_dir

    mode = str(eval_raw.get("mode", "detection"))
    if mode not in ("detection", "flip"):
        raise ValueError(f"{path}: eval.mode {mode!r} is not 'detection' or 'flip'")

    arm = str(eval_raw.get("arm", "finetuned"))
    if arm not in ("finetuned", "zeroshot"):
        raise ValueError(f"{path}: eval.arm {arm!r} is not 'finetuned' or 'zeroshot'")

    label = eval_raw.get("label")
    if arm == "zeroshot" and label is not None and "ft" in str(label):
        raise ValueError(
            f"{path}: eval.arm is 'zeroshot' but eval.label {label!r} contains "
            f"'ft'. The label names the fragment and the dump; a zero-shot "
            f"result filed under a fine-tuned name is indistinguishable from "
            f"one afterwards.")
    if arm == "finetuned" and label is not None and "ft" not in str(label):
        raise ValueError(
            f"{path}: eval.label {label!r} does not contain 'ft'. The fragment "
            f"and the dump are named after it, so a fine-tuned run under a "
            f"zero-shot label would overwrite the zero-shot result and be "
            f"indistinguishable from it afterwards.")

    return Experiment(
        model=model,
        seed=seed,
        output_dir=output_dir,
        train=TrainSpec(steps=int(train_raw.get("steps", DEFAULT_STEPS)),
                        num_workers=int(train_raw.get("num_workers", 4)),
                        lora=LoraSpec(rank=int(lora_raw.get("rank", DEFAULT_RANK)),
                                      alpha=int(lora_raw.get("alpha", DEFAULT_ALPHA)))),
        eval=EvalSpec(conditions=tuple(eval_raw.get("conditions") or ()),
                      label=label,
                      resume=bool(eval_raw.get("resume", False)),
                      mode=str(eval_raw.get("mode", "detection")),
                      reference=str(eval_raw.get("reference", "A1_singleton_person")),
                      target=str(eval_raw.get("target", "person")),
                      arm=arm,
                      per_dataset=per_dataset,
                      # per_dataset cannot be honoured without raw detections,
                      # so asking for one and refusing the other is a
                      # contradiction, not a preference.
                      dump=bool(eval_raw.get("dump", True)) or per_dataset),
        config_file=path.resolve(),
    )
