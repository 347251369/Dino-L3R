# Dino-L3R

The final abdominal multi-organ segmentation framework has two stages:

1. **GSP-DINO**: global multi-organ segmentation with DINOv3.
2. **OCRE**: single-pass organ-conditioned physical-space refinement.

## Structure

```text
configs/                    dataset-specific experiment configuration
datasets/                   dataset metadata, splits, and shared loaders
docs/                       locked baselines and protocol notes
evaluation/                 Dice, ASSD, HD95, and transition audits
models/gsp_dino/            GSP model, training, and inference
models/ocre/                OCRE model, training, and checkpoint loading
pipeline/                   two-stage orchestration and volume refinement
results/                    server artifact layout and ownership
scripts/                    optional analysis utilities
utils/                      shared configuration, I/O, and geometry helpers
run_pipeline.py             single GSP -> OCRE entry point
```

Each dataset has one unversioned split manifest and one configuration. Formal
artifacts use `results/<dataset>/dino_l3r`, with checkpoints, predictions, and
test reports owned directly by `gsp_dino/` or `ocre/`.

## Two-stage commands

```bash
python run_pipeline.py train --config configs/btcv_dino_l3r.toml
python run_pipeline.py train --config configs/flare22_dino_l3r.toml
python run_pipeline.py reproduce --config configs/flare22_dino_l3r.toml --overwrite
```

Both dataset configurations now point to selected epoch-24 OCRE checkpoints.
Locked metrics and artifact hashes are recorded under `results/`.
