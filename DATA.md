# Data

Three corpora. None is redistributed here: `data/` holds symlinks you create,
and `.gitignore` keeps its contents out of the repository.

| corpus | role | availability |
|---|---|---|
| **ForestPersons** | people under canopy; the `person` class | [on the Hub](https://huggingface.co/datasets/etri/ForestPersons) |
| **ForestBelongings** | dropped belongings; `clothes` `hat` `shoes` `gloves` `bag` | [on the Hub](https://huggingface.co/datasets/etri/ForestBelongings) |
| **COCO 2017** | zero-shot retention only; never trained on | [cocodataset.org](https://cocodataset.org) |

## ForestPersons

[`etri/ForestPersons`](https://huggingface.co/datasets/etri/ForestPersons)
— 44.3 GB, **CC BY-NC-SA 4.0** (non-commercial research use).

```bash
pip install huggingface_hub && hf auth login --add-to-git-credential

git lfs install
git clone https://huggingface.co/datasets/etri/ForestPersons
```

Access is automatic but not anonymous: the repository is gated, so accept its
terms once while logged in. `--add-to-git-credential` is what lets `git` use the
token — `hf auth login` on its own authenticates the Python client and not git,
and the clone then fails with 401 rather than falling back to a partial fetch.

A clone keeps a `.git` copy of every LFS object, so budget about twice the size.
`hf download etri/ForestPersons --repo-type dataset --local-dir <dir>` fetches
the working tree only, if disk is tight.

COCO-format: `annotations/{train,val,test}.json` plus `images/`. The category
field of the standard `val.json` carries pose subclasses (`sitting_100` and
similar); the six-class benchmark uses the person-only variant, and
`prepare.py` selects it.

An infrared companion corpus, `etri/ForestPersonsIR`, is published alongside it.
This benchmark does not use it — the six-class splits are RGB only.

## ForestBelongings

[`etri/ForestBelongings`](https://huggingface.co/datasets/etri/ForestBelongings)
— 31,897 images over 250 sequences, 10.2 GB, **CC BY-NC-SA 4.0** (non-commercial
research use).
Gated the same way as ForestPersons: accept its terms once while logged in.

```bash
git lfs install
git clone https://huggingface.co/datasets/etri/ForestBelongings
```

Layout: `images/<sequence>/<frame>.jpg` plus COCO
`annotations/{train,val,test,total}.json`. Categories are `person` `clothes`
`hat` `gloves` `bag` `shoes` `etc` with ids 1–7 — the same names the benchmark
queries, so nothing is renamed on the way in except the ForestPersons pose
subclasses.

Two conventions `prepare.py` depends on:

* `iscrowd=1` marks *annotated but not scored* — every `etc` box, and boxes
  under `sqrt(area) < 16px` at the usual detector input scale. `etc` is carried
  into the benchmark as an ignore region rather than dropped or remapped.
* Frames with no scored annotation are kept, including 2,161 deliberate
  object-free background sequences. Removing them changes the negative half of
  the benchmark.

Twenty frames are face-anonymised, one of which is in the test split; the effect
on the published numbers is −5.6e-05 macro AP, measured rather than assumed.

## COCO 2017

Only `val2017` and its annotations are needed; the retention measurement never
trains. Standard layout: `annotations/instances_val2017.json`, `val2017/`.

## Linking them

```bash
./link_datasets.sh \
    --forest-persons     /path/to/forest_persons \
    --forest-belongings  /path/to/ForestBelongings \
    --coco               /path/to/COCO
```

The script verifies each link and exits non-zero if one resolves somewhere
without an `annotations/` directory. That check exists because `ln` will happily
create a symlink to nothing, and an evaluation over a dangling link does not
crash — it scores zero and says nothing.

Any subset may be linked. `UC_DATA_ROOT`, or editing `configs/paths.yaml`, does
the same job if you would rather not use symlinks.

## Building the benchmark

```bash
python prepare.py
```

Merges the two forest corpora into the six-class splits every evaluation reads.
Run once. The image paths inside the produced JSON are absolute, so rebuild it
if you move the corpora.

## Licences

Each corpus carries its own terms, and they are not this repository's to grant.
COCO in particular restricts commercial use of some annotations. Check before
deploying anything trained here.
