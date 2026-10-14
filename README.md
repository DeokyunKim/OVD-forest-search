# Query-Composition Sensitivity of Open-Vocabulary Detectors in Under-Canopy Search and Rescue

Deokyun Kim\*, Giyoung Lee\*, Yookyung Kim, Myungseok Ki, Jihun Cha

<sub>\* Equal contribution</sub>

*NeurIPS 2026 Workshop on Vision-Language Models for Real-World Deployment
([VLM4RWD](https://vlm4rwd.github.io/))*

Dataset: [`etri/ForestBelongings`](https://huggingface.co/datasets/etri/ForestBelongings) on the Hugging Face Hub

Open-vocabulary detection of missing persons and belongings under forest canopy.
Benchmark, training and evaluation code for the paper.

Four detectors — [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[OWLv2](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit),
[GLIP](https://github.com/microsoft/GLIP),
[OmDet-Turbo](https://github.com/om-ai-lab/OmDet) — fine-tuned with LoRA on a
six-class benchmark, measured under one frozen protocol. `train.py` fine-tunes,
`eval.py` measures; `--config` is the only option either takes.

## Quick start

### 1. Setup Environment

One per model; they cannot share one. Grounding DINO shown here.

```bash
python3 -m venv ~/venvs/uc-gdino
~/venvs/uc-gdino/bin/pip install -r requirements/gdino.txt -e .
```

### 2. Download Datasets

54 GB. Both datasets are gated: accept their terms on each dataset page, then
give git the credential.

```bash
pip install huggingface_hub && hf auth login --add-to-git-credential

git lfs install
git clone https://huggingface.co/datasets/etri/ForestPersons
git clone https://huggingface.co/datasets/etri/ForestBelongings
```

### 3. Link Datasets

Symlinks; nothing is copied.

```bash
./link_datasets.sh \
    --forest-persons     /path/to/forest_persons \
    --forest-belongings  /path/to/ForestBelongings \
    --coco               /path/to/COCO
```

### 4. Download Pretrained Weights

The two that are files. OWLv2 and OmDet-Turbo download themselves — see
[Weights](#weights).

```bash
mkdir -p checkpoints/gdino checkpoints/glip

hf download ShilongLiu/GroundingDINO groundingdino_swinb_cogcoor.pth \
    --local-dir checkpoints/gdino
curl -L -o checkpoints/glip/glip_tiny_c_mmdet-2fc427dd.pth \
    https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth
```

### 5. Build Benchmark

Once.

```bash
python prepare.py
```

### 6. Fine-tune and Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/experiments/gdino_seed1.yaml
CUDA_VISIBLE_DEVICES=0 python eval.py  --config configs/experiments/gdino_seed1.yaml
```

Required: `TMPDIR` on a volume with room (DataLoader shared memory; a full volume
fails with `OSError 28` hours in), `UC_GDINO_REPO` for Grounding DINO,
`UC_MMDET_ROOT` for GLIP.

Pick the card with `CUDA_VISIBLE_DEVICES`; there is no device option. Adapted
Grounding DINO variants fail on a non-zero ordinal, and merging an OmDet-Turbo
PEFT adapter outside `cuda:0` has produced run-to-run different results on the
same input with nothing in the log.

## Commands

```bash
# fine-tune, then measure
CUDA_VISIBLE_DEVICES=0 ~/venvs/uc-gdino/bin/python train.py --config configs/experiments/gdino_seed1.yaml
CUDA_VISIBLE_DEVICES=0 ~/venvs/uc-gdino/bin/python eval.py  --config configs/experiments/gdino_seed1.yaml

# zero-shot
CUDA_VISIBLE_DEVICES=0 ~/venvs/uc-owlv2/bin/python eval.py --config configs/experiments/owlv2_zeroshot.yaml

# every cell
for config in configs/experiments/*_seed[012].yaml; do
    model=$(awk '/^model:/ {print $2}' "$config")
    CUDA_VISIBLE_DEVICES=0 "$HOME/venvs/uc-$model/bin/python" train.py --config "$config"
done

# cells that did not finish
grep -L "step 5667/5667" outputs/logs/train_*.log
```

`eval.py` refuses if the adapter directory is absent. A process can exit 0
having measured nothing, so judge completion from the log.

## Environments

The four cannot share an interpreter: GLIP needs mmdet 3.3 with its pinned mmcv,
OmDet-Turbo and OWLv2 need `transformers` versions years apart, Grounding DINO
needs a CUDA op compiled against its own torch. `utils/` installs into all four
— different inference stacks, one definition of the measurement.

| model | requirements | stack |
|---|---|---|
| Grounding DINO | `requirements/gdino.txt` | torch 2.4.0+cu124, transformers 4.43.3 |
| OWLv2 | `requirements/owlv2.txt` | torch 2.6.0+cu124, transformers 5.14.1 |
| GLIP | `requirements/glip.txt` | torch 2.4.1+cu118, mmdet 3.3.0, mmcv 2.2.0 |
| OmDet-Turbo | `requirements/omdet.txt` | torch 2.7.1+cu118, transformers 4.57.3 |

## Directories

```
prepare.py     build the benchmark from the two corpora
train.py       fine-tune with LoRA
eval.py        measure

models/        gdino, owlv2, glip, omdet -- architecture and adapter injection
utils/
  protocol/    what is measured: conditions, cap policy, vocabulary, thresholds
  runner/      how it is measured: the loop, dumps, fragments, paths
  metrics/     flip rates, per-corpus scoring
  data/        benchmark construction
  train/       the training loop and its schedule
configs/       paths.yaml, models.yaml, experiments/, reproduce/
```

## Outputs

```
checkpoints/<model>/seed<n>/ckpt_final_step5667/    the adapter
outputs/fragments/<model>__<label>__<condition>.json
outputs/analysis/per_dataset__<model>__<label>.json
outputs/logs/<stage>_<config name>.log
```

A **fragment** holds per-class AP/AP50/AP75/AR, the query, the post-processing
settings, the environment, and a fingerprint of the source files that decide the
numbers. Logs are named after the config, not the seed: a zero-shot and a
fine-tuned arm share a seed.

`per_dataset: true` also scores each corpus separately, from one detection dump
rather than a second inference pass. Classes with no ground truth in a corpus
come back unmeasured, not 0.0.

## Runtime

GPU-hours on a single 48 GB card, read from the logs and fragments rather than
estimated. Cells are independent, so divide by the cards you can run in parallel.

| model | training | evaluation, per condition |
|---|--:|--:|
| OmDet-Turbo | 0.4 h | 16 min |
| Grounding DINO | 4.6 h | 31 min |
| GLIP | 7.1 h | 43 min |
| OWLv2 | 8.1 h | 90 min |

The twelve training cells are about 25 GPU-hours. Evaluation is the larger
number: one seed of one model across the 15-condition A axis is 4-23 GPU-hours.

## Weights

Two are files this repository reads from `checkpoints/`; two are pulled by
`transformers` on first use and need nothing here.

| model | weights | where |
|---|---|---|
| Grounding DINO | `groundingdino_swinb_cogcoor.pth` — 938 MB, Swin-B | [`ShilongLiu/GroundingDINO`](https://huggingface.co/ShilongLiu/GroundingDINO) → `checkpoints/gdino/` |
| GLIP | `glip_tiny_c_mmdet-2fc427dd.pth` — 928 MB | [mmdetection model zoo](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth) → `checkpoints/glip/` |
| OWLv2 | [`google/owlv2-large-patch14-ensemble`](https://huggingface.co/google/owlv2-large-patch14-ensemble) | downloaded on first use |
| OmDet-Turbo | [`omlab/omdet-turbo-swin-tiny-hf`](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) | downloaded on first use |

The commands for the two files are quick-start step 4, **Download Pretrained
Weights**. Fine-tuning Grounding DINO additionally needs the Swin-T release
weights, `groundingdino_swint_ogc.pth` — 694 MB — at the **root of the checkout**
`UC_GDINO_REPO` points at, which is where `configs/models.yaml` looks:

```bash
curl -L -o "$UC_GDINO_REPO/groundingdino_swint_ogc.pth" \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

The file names are what the configs expect; renaming one means editing
`configs/models.yaml`, and a symlink works as well as a copy. Each set of
weights carries its own licence — check it before deploying anything trained on
them.

## Data

| corpus | where | licence |
|---|---|---|
| **ForestBelongings** | [`etri/ForestBelongings`](https://huggingface.co/datasets/etri/ForestBelongings) — 31,897 images, 10.2 GB, gated | CC BY-NC-SA 4.0 |
| **ForestPersons** | [`etri/ForestPersons`](https://huggingface.co/datasets/etri/ForestPersons) — 44.3 GB, gated | CC BY-NC-SA 4.0 |
| **COCO 2017** | [cocodataset.org](https://cocodataset.org); `val2017` only | see source |

Cloning is quick-start step 2. A clone keeps a `.git` copy of every LFS object,
so budget about twice the sizes above; `hf download <repo> --repo-type dataset
--local-dir <dir>` fetches the working tree only, if disk is tight.

`prepare.py` rebuilds the published splits from these two corpora: 90,670 train
/ 22,729 val / 14,980 test. COCO is used only for zero-shot retention and is
never trained on. Nothing is copied into the repository; `link_datasets.sh`
symlinks. See [DATA.md](DATA.md).

**The code is Apache-2.0; both corpora are CC BY-NC-SA 4.0.** Non-commercial
terms apply to anything trained or evaluated on them.

## Citation

```bibtex
@inproceedings{kim2026querycomposition,
  title     = {Query-Composition Sensitivity of Open-Vocabulary Detectors in Under-Canopy Search and Rescue},
  author    = {Kim, Deokyun and Lee, Giyoung and Kim, Yookyung and Ki, Myungseok and Cha, Jihun},
  booktitle = {NeurIPS 2026 Workshop on Vision-Language Models for Real-World Deployment (VLM4RWD)},
  year      = {2026}
}
```

## Acknowledgements

We especially thank the authors of
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[GLIP](https://github.com/microsoft/GLIP),
[OWLv2](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
and [OmDet-Turbo](https://github.com/om-ai-lab/OmDet) for releasing their models
and code openly.

## Licence

Apache-2.0, following the models this builds on — Grounding DINO, mmdetection,
transformers and PEFT are Apache-2.0, GLIP's original release is MIT. See
[NOTICE](NOTICE) for attributions. Model weights carry their own licences and are
downloaded, not redistributed here.
