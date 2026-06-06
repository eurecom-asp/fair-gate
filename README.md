# FAIR-GATE-ASV

Official research-code layout for **FAIR-GATE**, a gender-fair automatic speaker verification framework based on ECAPA-TDNN.

## Features

- ECAPA-TDNN speaker verification backbone
- Complementary identity/sensitive gating
- Capacity and saturation regularization for gate control
- Gender supervision on the sensitive branch
- Adversarial gender removal from the identity branch
- Optional decorrelation and Risk Extrapolation regularization
- VoxCeleb1-O/E/H evaluation
- Gender-disaggregated FMR/FNMR metrics
- GARBE fairness metric at a fixed global operating point
- Optional AS-Norm evaluation

## Layout

```text
FAIR-GATE-ASV/
├── fairgate/                  # Core model, losses, dataloader, metrics
├── configs/                   # Example configurations
├── scripts/
│   ├── training/              # Training scripts
│   ├── evaluation/            # Evaluation scripts
│   └── analysis/              # Result aggregation scripts
├── recipes/
│   ├── stage0_model_selection/# Stage-0 search and selection
│   ├── stage1_ablation/       # Core and extended ablations
│   └── evaluation/            # Batch evaluation recipes
├── examples/                  # Example file formats
├── docs/                      # Documentation
└── results/                   # Small example results only
```

## Quick start

```bash
pip install -r requirements.txt
```

Train FAIR-GATE r=0.05:

```bash
bash scripts/training/train_fairgate_r005.sh
```

Evaluate without AS-Norm:

```bash
bash scripts/evaluation/evaluate_vox1_no_asnorm.sh /path/to/model.pt
```

Evaluate with AS-Norm:

```bash
bash scripts/evaluation/evaluate_vox1_asnorm.sh /path/to/model.pt
```

Aggregate metric logs:

```bash
python scripts/analysis/aggregate_metric_lines.py --log_dir logs --out_csv results/metrics.csv
```

See `docs/ablation_naming.md` and `docs/data_format.md`.

## Public API and naming

This release preserves legacy ECAPA-TDNN names for backward compatibility and provides clearer public aliases for new code. See `docs/public_api.md`.
