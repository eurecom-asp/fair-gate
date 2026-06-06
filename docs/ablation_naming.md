# Ablation Naming

## Component names

| Short name | Meaning | Typical argument |
|---|---|---|
| Cap | Capacity regularization for complementary gating | `--lambda_css_cap` |
| Sat | Gate saturation regularization | `--lambda_css_sat` |
| Gs | Gender supervision on the sensitive branch | `--lambda_gender_s` |
| Adv | Adversarial gender removal from identity branch | `--lambda_gender_adv` |
| Dec | Decorrelation between identity and sensitive branches | `--lambda_decor` |
| REx | Risk Extrapolation regularization | `--lambda_rex` |
| r | CSS target ratio | `--css_target_ratio` |

## Stage-1 core ablations

| ID | Publication name | Meaning |
|---|---|---|
| S1 | Baseline | Fair branch disabled |
| S2 | Cap only | Only capacity regularization |
| S3 | Cap + Sat | Capacity + saturation regularization |
| S4 | Cap + Gs | Capacity + gender supervision |
| S5 | Cap + Adv | Capacity + adversarial gender removal |
| S6 | Main | Cap + Sat + Gs + Adv |
| S7 | Main + Dec | Main plus branch decorrelation |
| S8 | Main + REx | Main plus Risk Extrapolation |
| S9 | Main + Dec + REx | Main plus Dec and REx |

## Extended ablations

| ID | Publication name | Meaning |
|---|---|---|
| S10 | Main - Cap | Main without capacity regularization |
| S11 | Main - Sat | Main without saturation regularization |
| S12 | Main - Gs | Main without gender supervision |
| S13 | Main - Adv | Main without adversarial gender removal |
| S14 | Sat only | Saturation-only setting |
| S15 | Gs only | Gender-supervision-only setting |
| S16 | Adv only | Adversarial-only setting |
| S17 | Main, r=0.05 | Main with CSS target ratio 0.05 |
| S18 | Main, r=0.10 | Main with CSS target ratio 0.10 |
| S19 | Main, r=0.20 | Main with CSS target ratio 0.20 |
| B2 | GRL-only baseline | Adversarial baseline |
| B3 | REx-only baseline | Risk Extrapolation baseline |
