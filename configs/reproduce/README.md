# Query-sensitivity reproduction

Sixteen configs: four models by four arms (zero-shot, and fine-tuned at seeds
0/1/2). Each names the same 21 conditions, which is every query the sensitivity
tables are built from.

```bash
python eval.py --config configs/reproduce/owlv2_seed0.yaml
```

One config is one card, in sequence — 21 full-census passes. Budget by model:
Grounding DINO ~34 min per condition, OWLv2 ~82, GLIP ~44, OmDet-Turbo ~14.
The whole set is about 219 GPU-hours; on one card that is nine days, so run the
arms you need rather than all sixteen.

The fine-tuned configs read `checkpoints/<model>/seed<N>`, which is where
`train.py --config configs/experiments/<model>_seed<N>.yaml` writes. Train first,
or point `output_dir` at an adapter you already have.

`per_dataset: false` here: these tables are scored over the pooled test set, not
per corpus. The detection dump is still written, because the query-turnover
metrics are computed from dumps rather than from a second inference pass.
