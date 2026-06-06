#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
SHARD_TAG="${SHARD_TAG:-$(hostname -s)}"
SCRIPT="${SCRIPT:-trainECAPAModel_fair_ddp.py}"
PY="${PY:-python}"

ROOT="${ROOT:-exps/_two_stage/G1}"
STAGE0_ROOT="${ROOT}/stage0_tune"
mkdir -p "${STAGE0_ROOT}"

# 可选：只跑指定 name（逗号分隔）
RUN_NAMES="${RUN_NAMES:-}"

# ====== 数据/模型路径（按你的工程）======
TRAIN_LIST="${TRAIN_LIST:-/medias/speech/projects/quy/dataset/voxceleb2_train_list.txt}"
TRAIN_PATH="${TRAIN_PATH:-/medias/speech/data/VoxCeleb2}"
GENDER_MAP="${GENDER_MAP:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/gender_map_train.json}"
PRETRAIN="${PRETRAIN:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-ori/ECAPA-TDNN/exps/pretrain.model}"
N_CLASS="${N_CLASS:-5994}"

EVAL_LIST="${EVAL_LIST:-/medias/speech/projects/quy/dataset/VoxCeleb1_test/vox1-E-abs.txt,/medias/speech/projects/quy/dataset/VoxCeleb1_test/vox1-H-abs.txt,/medias/speech/projects/quy/dataset/VoxCeleb1_test/vox1-O-abs.txt}"
EVAL_PATH="${EVAL_PATH:-/medias/speech/data/VoxCeleb1}"
EVAL_GMAP="${EVAL_GMAP:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/gender_map.json}"

STAGE0_MAX_EPOCH="${STAGE0_MAX_EPOCH:-25}"
SAVE_EVERY="${SAVE_EVERY:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"       # 单卡时就是全局 batch
NUM_FRAMES="${NUM_FRAMES:-200}"
LR="${LR:-3e-4}"
ENCODER_LR="${ENCODER_LR:-0}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-5}"
LR_DECAY="${LR_DECAY:-0.995}"
USE_AMP="${USE_AMP:-1}"

BASE_ARGS_COMMON=(
  --train_list "${TRAIN_LIST}"
  --train_path "${TRAIN_PATH}"
  --gender_map "${GENDER_MAP}"
  --n_class "${N_CLASS}"
  --pretrained_ecapa "${PRETRAIN}"
  --pretrained_load_classifier 1
  --freeze_encoder_epochs "${FREEZE_ENCODER_EPOCHS}"
  --lr "${LR}"
  --encoder_lr "${ENCODER_LR}"
  --lr_decay "${LR_DECAY}"
  --batch_size "${BATCH_SIZE}"
  --num_frames "${NUM_FRAMES}"
  --save_every "${SAVE_EVERY}"
  --max_epoch "${STAGE0_MAX_EPOCH}"
)
if [[ "${USE_AMP}" == "1" ]]; then BASE_ARGS_COMMON+=(--amp); fi

EVAL_ARGS=(
  --eval
  --n_class "${N_CLASS}"
  --eval_list "${EVAL_LIST}"
  --eval_path "${EVAL_PATH}"
  --eval_gender_map "${EVAL_GMAP}"
  --fixed_fmr 0.01
  --eval_batch_size 1
)

GRID_TSV="${STAGE0_ROOT}/grid_stage0_9runs.tsv"
cat > "${GRID_TSV}" <<'TSV'
#name	css_target_ratio	lambda_css_cap	lambda_css_sat	lambda_gender_s	lambda_gender_adv	grl_warmup_epochs	adv_warmup_epochs
T01_r1_cap005_sat001_adv002	0.01	0.005	0.001	0.05	0.002	8	8
T02_r1_cap005_sat002_adv002	0.01	0.005	0.002	0.05	0.002	8	8
T03_r1_cap005_sat003_adv002	0.01	0.005	0.003	0.05	0.002	8	8
T04_r1_cap020_sat002_adv002	0.01	0.020	0.002	0.05	0.002	8	8
T05_r3_cap005_sat002_adv002	0.03	0.005	0.002	0.05	0.002	8	8
T06_r5_cap005_sat002_adv002	0.05	0.005	0.002	0.05	0.002	8	8
T07_r1_cap005_sat002_adv005	0.01	0.005	0.002	0.05	0.005	8	8
T08_r3_cap005_sat002_adv005	0.03	0.005	0.002	0.05	0.005	8	8
T09_r5_cap005_sat002_adv005	0.05	0.005	0.002	0.05	0.005	8	8
TSV

STAGE0_SUMMARY="${STAGE0_ROOT}/results_stage0_shard.${SHARD_TAG}.tsv"
echo -e "name\tckpt\texit_code\tgpu\targs" > "${STAGE0_SUMMARY}"

# 读取 runs
mapfile -t ALL_LINES < <(grep -v '^\s*#' "${GRID_TSV}" | sed '/^\s*$/d')

# RUN_NAMES 过滤
if [[ -n "${RUN_NAMES}" ]]; then
  IFS=',' read -r -a WANT <<< "${RUN_NAMES}"
  declare -A WANTSET=()
  for x in "${WANT[@]}"; do WANTSET["$x"]=1; done
  RUN_LINES=()
  for line in "${ALL_LINES[@]}"; do
    name="$(echo "${line}" | cut -f1)"
    [[ -n "${WANTSET[$name]:-}" ]] && RUN_LINES+=("${line}")
  done
else
  RUN_LINES=("${ALL_LINES[@]}")
fi

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPUS="${#GPU_ARR[@]}"

# 分配到每个 GPU 的队列（串行）
declare -A QUEUE=()
for idx in "${!RUN_LINES[@]}"; do
  gpu="${GPU_ARR[$((idx % NGPUS))]}"
  QUEUE["$gpu"]+="${RUN_LINES[$idx]}"$'\n'
done

run_line_on_gpu() {
  local gpu="$1"
  local line="$2"

  IFS=$'\t' read -r name r cap sat gs adv grl advw <<< "${line}"
  local save_path="${STAGE0_ROOT}/${name}"
  local model_save_path="${save_path}/models"
  mkdir -p "${model_save_path}"

  local extra_args=(
    --css_target_ratio "${r}"
    --lambda_css_cap "${cap}"
    --lambda_css_sat "${sat}"
    --lambda_gender_s "${gs}"
    --lambda_gender_adv "${adv}"
    --grl_warmup_epochs "${grl}"
    --adv_warmup_epochs "${advw}"
  )
  local args_str; args_str="$(printf "%q " "${extra_args[@]}")"

  echo "[GPU ${gpu}] START ${name}"

  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
  ${PY} -u "${SCRIPT}" \
    --save_path "${save_path}" \
    --model_save_path "${model_save_path}" \
    "${BASE_ARGS_COMMON[@]}" \
    "${extra_args[@]}" \
    > "${save_path}/train.out" 2>&1
  rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo -e "${name}\tNA\t${rc}\t${gpu}\t${args_str}" >> "${STAGE0_SUMMARY}"
    echo "[GPU ${gpu}] FAIL ${name} rc=${rc}"
    return 0
  fi

  ckpt="$(ls -1 "${model_save_path}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo -e "${name}\tNA\t3\t${gpu}\t${args_str}" >> "${STAGE0_SUMMARY}"
    echo "[GPU ${gpu}] FAIL ${name} no ckpt"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" \
  ${PY} -u "${SCRIPT}" \
    --save_path "${save_path}" \
    --model_save_path "${model_save_path}" \
    --resume "${ckpt}" \
    "${EVAL_ARGS[@]}" \
    > "${save_path}/eval_last.out" 2>&1 || true

  echo -e "${name}\t$(basename "${ckpt}")\t0\t${gpu}\t${args_str}" >> "${STAGE0_SUMMARY}"
  echo "[GPU ${gpu}] DONE ${name}"
}

worker_gpu() {
  local gpu="$1"
  local q="${QUEUE[$gpu]:-}"
  [[ -n "${q}" ]] || return 0
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    run_line_on_gpu "${gpu}" "${line}"
  done <<< "${q}"
}

echo "============================================================"
echo "[INFO] ROOT=${ROOT}"
echo "[INFO] STAGE0_ROOT=${STAGE0_ROOT}"
echo "[INFO] SHARD_TAG=${SHARD_TAG}"
echo "[INFO] GPUS=${GPUS} (one job per GPU, serial queue)"
echo "============================================================"

pids=()
for gpu in "${GPU_ARR[@]}"; do
  worker_gpu "${gpu}" &
  pids+=($!)
done

for p in "${pids[@]}"; do
  wait "${p}" || true
done

echo
echo "[DONE] Stage-0 shard finished -> ${STAGE0_SUMMARY}"
column -t -s $'\t' "${STAGE0_SUMMARY}" | sed 's/\r$//'
