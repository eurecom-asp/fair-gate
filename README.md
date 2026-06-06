# Fair-Gate

Official implementation of **Fair-Gate: Fairness-Aware Interpretable Risk Gating for Sex-Fair Voice Biometrics**.

Fair-Gate is a fairness-aware automatic speaker verification (ASV) framework built on an ECAPA-TDNN backbone. It introduces a complementary gating mechanism that separates speaker-discriminative identity information from sex-related variation, and uses branch-specific training objectives to reduce sex-linked error disparities under a shared operating threshold.

Paper: [Fair-Gate: Fairness-Aware Interpretable Risk Gating for Sex-Fair Voice Biometrics](https://arxiv.org/abs/2603.11360)  
Pretrained checkpoint: [Cody9517/fairgate-ecapa-voxceleb](https://huggingface.co/Cody9517/fairgate-ecapa-voxceleb)

---

## Overview

This repository provides code for:

- ECAPA-TDNN based speaker verification;
- Fair-Gate complementary identity/sensitive feature routing;
- capacity and saturation regularization for gate control;
- sex supervision on the sensitive branch;
- adversarial sex removal from the identity branch;
- optional decorrelation and Risk Extrapolation regularization;
- VoxCeleb1-O/E/H evaluation;
- sex-disaggregated FMR/FNMR analysis;
- GARBE fairness evaluation at a fixed global operating point.

The released checkpoint corresponds to the final **r=0.05** Fair-Gate configuration used for the main evaluation.

---

## Repository structure

```text
fair-gate/
├── train.py                         # Main training/evaluation entry
├── fairgate/                        # Core model, losses, dataloader, metrics
├── configs/                         # Configuration and selected hyperparameters
├── scripts/
│   ├── training/                    # Training scripts
│   ├── evaluation/                  # Evaluation scripts
│   └── analysis/                    # Log/result aggregation utilities
├── recipes/
│   ├── stage0_model_selection/      # Stage-0 model selection
│   ├── stage1_ablation/             # Core and extended ablations
│   └── evaluation/                  # Batch evaluation recipes
├── docs/                            # Documentation
├── examples/                        # Example input file formats
└── results/                         # Small reproducibility outputs
```

---

## Installation

```bash
git clone https://github.com/eurecom-asp/fair-gate.git
cd fair-gate

pip install -r requirements.txt
```

---

## Download the pretrained checkpoint

The pretrained Fair-Gate checkpoint is hosted on Hugging Face:

```bash
mkdir -p checkpoints

huggingface-cli download Cody9517/fairgate-ecapa-voxceleb \
  fairgate_ecapa_voxceleb_r005_epoch30.pt \
  --local-dir checkpoints/
```

The checkpoint was selected from the `r=0.05` configuration:

```text
--css_target_ratio 0.05
--lambda_css_cap 0.005
--lambda_css_sat 0.001
--lambda_gender_s 0.05
--lambda_gender_adv 0.002
--grl_warmup_epochs 8
--adv_warmup_epochs 8
```

---

## Data preparation

Prepare the following files before evaluation:

1. VoxCeleb1 waveform root;
2. VoxCeleb1-O/E/H trial lists;
3. speaker-level sex annotation file for subgroup evaluation.

Example trial-list and metadata formats are provided in:

```text
examples/
docs/data_format.md
```

Expected trial format:

```text
<label> <enroll_wav> <test_wav>
```

where `label=1` denotes a target trial and `label=0` denotes a non-target trial.

---

## Evaluation without AS-Norm

```bash
EVAL_PATH=/path/to/VoxCeleb1 \
GENDER_MAP=/path/to/gender_map_eval.json \
VOX1_O=/path/to/vox1-O-abs.txt \
VOX1_E=/path/to/vox1-E-abs.txt \
VOX1_H=/path/to/vox1-H-abs.txt \
SAVE_PATH=exps/eval_no_asnorm \
N_CLASS=5994 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/evaluation/evaluate_vox1_no_asnorm.sh \
  checkpoints/fairgate_ecapa_voxceleb_r005_epoch30.pt
```

The evaluation reports:

- EER;
- minDCF;
- FMR/FNMR at a fixed global FMR operating point;
- sex-disaggregated FMR/FNMR;
- GARBE fairness metrics.

---

## Training the final Fair-Gate configuration

```bash
TRAIN_LIST=/path/to/voxceleb2_train_list.txt \
TRAIN_PATH=/path/to/VoxCeleb2 \
GENDER_MAP=/path/to/gender_map_train.json \
PRETRAINED_ECAPA=/path/to/pretrain.model \
SAVE_PATH=exps/fairgate_r005 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/training/train_fairgate_r005.sh
```

The final configuration is also stored in:

```text
configs/fairgate_r005.yaml
configs/fairgate_r005_best_args.txt
```

---

## Ablation recipes

The main ablation scripts are provided under:

```text
recipes/stage1_ablation/
```

Key components:

| Name | Argument | Meaning |
|---|---|---|
| Cap | `--lambda_css_cap` | capacity regularization for complementary gating |
| Sat | `--lambda_css_sat` | gate saturation regularization |
| Gs | `--lambda_gender_s` | sex supervision on the sensitive branch |
| Adv | `--lambda_gender_adv` | adversarial sex removal from the identity branch |
| Dec | `--lambda_decor` | decorrelation regularization between identity and sensitive branches |
| REx | `--lambda_rex` | Risk Extrapolation penalty across sex groups |
| r | `--css_target_ratio` | target ratio for the sensitive branch |

See `docs/ablation_naming.md` for the full mapping between experiment IDs and components.

---

## Citation

If you use this code or checkpoint, please cite:

```bibtex
@article{qu2026fairgate,
  title   = {Fair-Gate: Fairness-Aware Interpretable Risk Gating for Sex-Fair Voice Biometrics},
  author  = {Qu, Yangyang and Todisco, Massimiliano and Galdi, Chiara and Evans, Nicholas},
  journal = {arXiv preprint arXiv:2603.11360},
  year    = {2026}
}
```

---

## Notes

- The checkpoint is hosted on Hugging Face and is not stored in this GitHub repository.
- AS-Norm analysis is not enabled in the default release entry point.
- This release keeps several legacy ECAPA-TDNN names for compatibility. See `docs/public_api.md` for cleaner public aliases.
- The model and code are released for research use. Fairness behavior may vary across datasets, demographic distributions, acoustic conditions, and operating thresholds.
