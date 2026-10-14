# Four environments, not one

The four detectors cannot share an interpreter. GLIP needs `mmdet` 3.3 and its
pinned `mmcv`; OmDet-Turbo and OWLv2 need `transformers` versions that are years
apart; GroundingDINO needs a compiled CUDA op (`MultiScaleDeformableAttention`)
built against its own torch. Installing all four together resolves to a set
where at least one silently misbehaves.

So each model gets its own virtual environment, and `utils/` — the
protocol, the COCO evaluation, the fragment writer — is installed into every one
of them. That is what keeps the four arms comparable: they run different
inference stacks against a single definition of the measurement.

    python3 -m venv ~/venvs/uc-gdino
    ~/venvs/uc-gdino/bin/pip install -r requirements/gdino.txt -e .

Repeat per model. Every fragment records the versions it ran under, so a result
can always be traced to the stack that produced it.

Each file is a full `pip freeze` of the interpreter the published numbers were
produced on — 82 to 123 pins — not a hand-maintained shortlist. `torch` carries a
`+cu124` or `+cu118` local version that PyPI does not serve, so each file opens
with the `--extra-index-url` that does; dropping that line is the usual way this
fails.

Three dependencies are named in the headers rather than pinned, because pip
cannot resolve them in the same run that installs torch:

| package | why | needed by |
|---|---|---|
| `groundingdino`, `detectron2` | a deformable-attention CUDA op compiles against the torch already installed | Grounding DINO |
| `MultiScaleDeformableAttention` | built by GroundingDINO's own setup | Grounding DINO |
| `mmdet` | GLIP's model configs are data files read from the checkout | GLIP |

Install those from a checkout after the requirements file has landed, and point
`UC_GDINO_REPO` / `UC_MMDET_ROOT` at them.
