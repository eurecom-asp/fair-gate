#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?Usage: bash scripts/evaluation/evaluate_vox1_no_asnorm.sh /path/to/model.pt}

: "${EVAL_PATH:=/path/to/VoxCeleb1}"
: "${GENDER_MAP:=/path/to/gender_map.json}"
: "${VOX1_O:=/path/to/vox1-O-abs.txt}"
: "${VOX1_E:=/path/to/vox1-E-abs.txt}"
: "${VOX1_H:=/path/to/vox1-H-abs.txt}"
: "${N_CLASS:=5994}"
: "${SAVE_PATH:=exps/eval_no_asnorm}"
: "${CUDA_VISIBLE_DEVICES:=0}"

mkdir -p "${SAVE_PATH}"

EVAL_LIST="${VOX1_O},${VOX1_E},${VOX1_H}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --standalone --nproc_per_node=1 train.py \
  --eval \
  --resume "${MODEL}" \
  --eval_list "${EVAL_LIST}" \
  --eval_path "${EVAL_PATH}" \
  --eval_gender_map "${GENDER_MAP}" \
  --fixed_fmr 0.01 \
  --n_class "${N_CLASS}" \
  --save_path "${SAVE_PATH}"
