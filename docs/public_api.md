# Public API

This repository keeps several legacy ECAPA-TDNN names for backward compatibility. For readability, the following aliases are recommended for new code.

## Fairness and verification metrics

| Recommended name | Legacy name | Meaning |
|---|---|---|
| `select_threshold_at_fmr` | `pick_tau_at_fmr` | Select a global threshold at a target false match rate |
| `compute_error_rates_at_threshold` | `err_rates` | Compute FMR and FNMR at a fixed threshold |
| `compute_garbe_at_threshold` | `garbe` | Compute GARBE at a fixed threshold |
| `compute_group_eer` | `group_eer` | Compute group-wise EER |

## Legacy ECAPA utilities

| Recommended name | Legacy name | Meaning |
|---|---|---|
| `tune_threshold_from_score` | `tuneThresholdfromScore` | Tune a score threshold from scores and labels |
| `compute_error_rates` | `ComputeErrorRates` | Compute error-rate curves |
| `compute_min_dcf` | `ComputeMinDcf` | Compute minimum detection cost function |

## Losses

| Recommended name | Legacy name | Meaning |
|---|---|---|
| `AAMSoftmax` | `AAMsoftmax` | Additive angular margin softmax |

## Data

| Recommended name | Legacy name | Meaning |
|---|---|---|
| `VoxCelebTrainDataset` | `train_loader` | VoxCeleb training dataset |

## FAIR-GATE components

| Name | Argument | Meaning |
|---|---|---|
| Cap | `--lambda_css_cap` | Capacity regularization for complementary gating |
| Sat | `--lambda_css_sat` | Gate saturation regularization |
| Gs | `--lambda_gender_s` | Gender supervision on the sensitive branch |
| Adv | `--lambda_gender_adv` | Adversarial gender removal from the identity branch |
| Dec | `--lambda_decor` | Decorrelation regularization between identity and sensitive branches |
| REx | `--lambda_rex` | Risk Extrapolation penalty across gender groups |
| r | `--css_target_ratio` | Target ratio for the sensitive branch |
