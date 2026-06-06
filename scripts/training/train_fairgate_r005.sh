#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_LIST:=/path/to/voxceleb2_train_list.txt}"
: "${TRAIN_PATH:=/path/to/VoxCeleb2}"
: "${EVAL_LIST:=/path/to/vox1-O-abs.txt}"
: "${EVAL_PATH:=/path/to/VoxCeleb1}"
: "${SAVE_PATH:=exps/fairgate_r005}"
: "${CUDA_VISIBLE_DEVICES:=0}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --standalone --nproc_per_node=1 train.py \
  --train_list "${TRAIN_LIST}" \
  --train_path "${TRAIN_PATH}" \
  --eval_list "${EVAL_LIST}" \
  --eval_path "${EVAL_PATH}" \
  --save_path "${SAVE_PATH}" \
  --lambda_css_cap 1.0 \
  --lambda_css_sat 1.0 \
  --lambda_gender_s 1.0 \
  --lambda_gender_adv 1.0 \
  --css_target_ratio 0.05
