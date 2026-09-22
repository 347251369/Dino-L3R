# Dino-L3R final code audit

Date: 2026-08-24

## Final naming and scope

- Framework: **Dino-L3R**
- Stage 1: **GSP-DINO**
- Stage 2: **OCRE**
- Public OCRE package: `models.ocre`
- Public OCRE model class: `OCRE`
- FLARE2022 final OCRE checkpoint: selected epoch 24
- BTCV final OCRE checkpoint: selected epoch 24
- No third-stage model or versioned experiment is part of the final project

## Frozen assets

### FLARE2022

- Split: 45 train / 5 fixed development-test cases
- GSP-DINO checkpoint SHA256: `27e74a73a6f13b0dd1caf2e826aaeade07e2154cb5c0810ea45338d074bf3445`
- OCRE checkpoint SHA256: `e8d58eb01eecae721abe7d190eaf3c07fff877cf126d6406845fcea947a94ad1`
- Mean Dice: 94.0294%
- Mean ASSD: 0.5867 mm
- Mean HD95: 2.7116 mm

### BTCV

- Split: 24 train / 6 fixed cases
- GSP-DINO checkpoint SHA256: `71e4db77579ae968b8f3fbb65eca94079dcb6d2042682e117556ccdfe28c4fa3`
- OCRE checkpoint SHA256: `5f646d2f8dd0c532f714b91dd6f446619c9582d040d690ab6b4f6ebf71d8ccae`
- Mean Dice: 83.8709%
- Mean ASSD: 1.8132 mm
- Mean HD95: 9.3785 mm

## Verification

- Both OCRE checkpoints completed 24 epochs and load strictly
- FLARE2022 has exactly 45 train and 5 test GSP-DINO predictions
- BTCV has exactly 24 train and 6 test GSP-DINO predictions
- Final OCRE test predictions contain exactly 5 FLARE2022 and 6 BTCV cases
- Stage manifests and SHA256 inventories are stored with each formal result
- Active code contains only GSP-DINO and OCRE

## Organization

- Server active code: `/pd/heyang/dinov3-main`
- Server formal results: `/pd/heyang/dinov3-main/results/<dataset>/dino_l3r`
- Local active code and lightweight result records: `Dino-L3R`
- The training server remains authoritative for checkpoints and volumetric predictions

Obsolete experiment backups, aliases, caches, intermediate epoch checkpoints,
partial predictions, and Python bytecode were removed after the final assets were
locked. Formal GSP-DINO and OCRE checkpoints, required predictions, reports,
manifests, and SHA256 inventories were retained.
