#!/usr/bin/env bash
set -euo pipefail

############################################
# 用法：
#   GPUS=0,1,2,3 ROOT=exps/_two_stage/G1 \
#   RUN_NAMES="B2_GRL_ONLY,B3_REX_ONLY" \
#   bash run_stage1_fair_baselines_1gpu_each.sh
#
# 环境变量（可覆盖）：
#   GPUS=0,1,2,3
#   RUN_NAMES=... (逗号分隔；不填则全跑=两组)
#   ROOT=exps/_two_stage/G1
#   STAGE1_MAX_EPOCH=40  SAVE_EVERY=10  EVAL_EVERY=10
#   BATCH_SIZE=512  LR=3e-4  ...
#   SCORE_ALPHA=20
############################################

GPUS="${GPUS:-0,1,2,3}"
RUN_NAMES="${RUN_NAMES:-}"
SHARD_TAG="${SHARD_TAG:-$(hostname -s)}"

PY="${PY:-python}"

# 训练：回到你原始 fair ddp 版本（不需要 plus_baselines）
SCRIPT="${SCRIPT:-trainECAPAModel_fair_ddp.py}"

ROOT="${ROOT:-exps/_two_stage/G1}"
STAGE1_ROOT="${ROOT}/stage1_fair_baselines"
mkdir -p "${STAGE1_ROOT}"

############################################
# 数据/模型路径（按你工程实际）
############################################
TRAIN_LIST="${TRAIN_LIST:-/medias/speech/projects/quy/dataset/voxceleb2_train_list.txt}"
TRAIN_PATH="${TRAIN_PATH:-/medias/speech/data/VoxCeleb2}"

# 训练 gender map：你当前确实存在的是 gender_map.json（先用它，避免 train.json 不存在）
GENDER_MAP="${GENDER_MAP:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/gender_map.json}"

PRETRAIN="${PRETRAIN:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-ori/ECAPA-TDNN/exps/pretrain.model}"
N_CLASS="${N_CLASS:-5994}"

EVAL_LIST="${EVAL_LIST:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-H-abs.txt,/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-O-abs.txt,/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-E-abs.txt}"
EVAL_PATH="${EVAL_PATH:-/medias/speech/data/VoxCeleb1}"
EVAL_GMAP="${EVAL_GMAP:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/gender_map.json}"

############################################
# 基础 sanity check（避免一跑就挂）
############################################
[[ -f "${TRAIN_LIST}" ]] || { echo "[FATAL] TRAIN_LIST not found: ${TRAIN_LIST}" >&2; exit 2; }
[[ -d "${TRAIN_PATH}" ]] || { echo "[FATAL] TRAIN_PATH not found: ${TRAIN_PATH}" >&2; exit 2; }
[[ -f "${GENDER_MAP}" ]] || { echo "[FATAL] GENDER_MAP not found: ${GENDER_MAP}" >&2; exit 2; }
[[ -f "${PRETRAIN}" ]] || { echo "[FATAL] PRETRAIN not found: ${PRETRAIN}" >&2; exit 2; }

############################################
# 训练超参
############################################
STAGE1_MAX_EPOCH="${STAGE1_MAX_EPOCH:-40}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-10}"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_FRAMES="${NUM_FRAMES:-200}"

LR="${LR:-3e-4}"
ENCODER_LR="${ENCODER_LR:-0}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-5}"
LR_DECAY="${LR_DECAY:-0.995}"
USE_AMP="${USE_AMP:-1}"

SCORE_ALPHA="${SCORE_ALPHA:-20}"

############################################
# 公共参数（训练/评测）
############################################
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
  --max_epoch "${STAGE1_MAX_EPOCH}"
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

############################################
# 解析 eval 输出 -> pooled_eer pooled_mindcf mean_garbe score
############################################
PARSE_PY="${STAGE1_ROOT}/parse_eval_and_score.py"
cat > "${PARSE_PY}" <<'PY'
#!/usr/bin/env python3
import re, sys

p = sys.argv[1]
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
txt = open(p, "r", errors="ignore").read()

def grab(pat):
    m = re.findall(pat, txt, flags=re.M)
    return m[-1] if m else None

pooled_eer = grab(r'^\[EVAL\]\s+POOLED:.*EER=([0-9.]+)')
pooled_mindcf = grab(r'^\[EVAL\]\s+POOLED:.*minDCF=([0-9.]+)')

def g(pat):
    s = grab(pat)
    return float(s) if s is not None else None

H = g(r'^\[EVAL\]\s+sub-vox1-H-abs\.txt:.*GARBE_fmr=([0-9.]+)')
O = g(r'^\[EVAL\]\s+sub-vox1-O-abs\.txt:.*GARBE_fmr=([0-9.]+)')
E = g(r'^\[EVAL\]\s+sub-vox1-E-abs\.txt:.*GARBE_fmr=([0-9.]+)')

vals = [x for x in [H, O, E] if x is not None]
mean_garbe = sum(vals) / len(vals) if vals else None

if pooled_eer is None:
    score = 1e9
else:
    score = float(pooled_eer) + alpha * (float(mean_garbe) if mean_garbe is not None else 0.0)

print(f"{pooled_eer or 'NA'}\t{pooled_mindcf or 'NA'}\t{mean_garbe if mean_garbe is not None else 'NA'}\t{score:.6f}")
PY
chmod +x "${PARSE_PY}"

############################################
# Stage-1: 2 个 fairness baselines（只保留 B2/B3）
############################################
declare -a ALL_NAMES=(
  "B2_GRL_ONLY"
  "B3_REX_ONLY"
)

# 注意：这里不再使用 --gender_balance_sampler / --drop_unknown_gender
# 如果你的 trainECAPAModel_fair_ddp.py 里对应参数名不同（如 grl_*），需要按实际脚本调整。
declare -a ALL_EXTRA=(
  # B2：GRL-only
  "--lambda_gender_adv 1.0 --lambda_gender_s 0.0 \
   --lambda_css_cap 0.0 --lambda_css_sat 0.0 \
   --lambda_decor 0.0 --lambda_rex 0.0 --lambda_fair_fmr 0.0 \
   --grl_lambda 1.0 --grl_warmup_epochs 5 --adv_warmup_epochs 5"

  # B3：REx-only
  "--lambda_rex 0.05 \
   --lambda_gender_adv 0.0 --lambda_gender_s 0.0 \
   --lambda_css_cap 0.0 --lambda_css_sat 0.0 \
   --lambda_decor 0.0 --lambda_fair_fmr 0.0"
)

############################################
# 选择本机要跑的子集 RUN_NAMES
############################################
want_name() {
  local n="$1"
  if [[ -z "${RUN_NAMES}" ]]; then
    return 0
  fi
  IFS=',' read -r -a arr <<< "${RUN_NAMES}"
  for x in "${arr[@]}"; do
    [[ "${x}" == "${n}" ]] && return 0
  done
  return 1
}

RUN_TASK_NAMES=()
RUN_TASK_EXTRA=()
for i in "${!ALL_NAMES[@]}"; do
  n="${ALL_NAMES[$i]}"
  if want_name "${n}"; then
    RUN_TASK_NAMES+=("${n}")
    RUN_TASK_EXTRA+=("${ALL_EXTRA[$i]}")
  fi
done

if [[ "${#RUN_TASK_NAMES[@]}" -eq 0 ]]; then
  echo "[FATAL] RUN_NAMES matched nothing. RUN_NAMES=${RUN_NAMES}" >&2
  exit 3
fi

############################################
# summary（每台机一份）
############################################
STAGE1_SUMMARY="${STAGE1_ROOT}/results_fair_baselines_shard.${SHARD_TAG}.tsv"
echo -e "name\tbest_epoch\tbest_ckpt\tpooled_eer\tpooled_mindcf\tmean_garbe_fmr\tscore\texit_code\tgpu\targs" > "${STAGE1_SUMMARY}"

############################################
# GPU 列表
############################################
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPUS="${#GPU_ARR[@]}"
[[ "${NGPUS}" -ge 1 ]] || { echo "[FATAL] no GPUs parsed from GPUS='${GPUS}'" >&2; exit 4; }

############################################
# 单个任务：单卡 train + eval_all_ckpts + 选 best
############################################
run_one() {
  local gpu="$1"; shift
  local name="$1"; shift
  local extra_str="$1"; shift

  local save_path="${STAGE1_ROOT}/${name}"
  local model_save_path="${save_path}/models"
  mkdir -p "${model_save_path}"

  # shellcheck disable=SC2206
  local extra_args=(${extra_str})

  local args_str
  args_str="$(printf "%q " "${BASE_ARGS_COMMON[@]}" "${extra_args[@]}")"

  echo "[GPU ${gpu}] START ${name}"

  # train
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
    ${PY} -u "${SCRIPT}" \
      --save_path "${save_path}" \
      --model_save_path "${model_save_path}" \
      "${BASE_ARGS_COMMON[@]}" \
      "${extra_args[@]}" \
      > "${save_path}/train.out" 2>&1
  local rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo -e "${name}\tNA\tNA\tNA\tNA\tNA\t1000000000\t${rc}\t${gpu}\t${args_str}" > "${save_path}/result.tsv"
    echo "[GPU ${gpu}] FAIL ${name} rc=${rc} (see ${save_path}/train.out)"
    return 0
  fi

  # eval all ckpts
  local out_dir="${save_path}/eval_all_ckpts"
  mkdir -p "${out_dir}"
  local tsv="${out_dir}/eval_all_ckpts.tsv"
  echo -e "epoch\tckpt\tpooled_eer\tpooled_mindcf\tmean_garbe_fmr\tscore" > "${tsv}"

  mapfile -t ckpts < <(ls -1 "${model_save_path}"/model_*.pt 2>/dev/null | sort -V)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "[GPU ${gpu}] FAIL ${name}: no ckpts found in ${model_save_path}"
    echo -e "${name}\tNA\tNA\tNA\tNA\tNA\t1000000000\t3\t${gpu}\t${args_str}" > "${save_path}/result.tsv"
    return 0
  fi

  local best_epoch="NA"
  local best_ckpt="NA"
  local best_metrics=$'NA\tNA\tNA\t1000000000'
  local best_score="1000000000"

  for c in "${ckpts[@]}"; do
    local bn epoch
    bn="$(basename "${c}")"
    epoch="$(echo "${bn}" | sed -n 's/^model_\([0-9]\+\)\.pt$/\1/p')"
    [[ -n "${epoch}" ]] || continue

    if [[ "${EVAL_EVERY}" -gt 1 ]]; then
      if (( 10#${epoch} % ${EVAL_EVERY} != 0 )); then
        continue
      fi
    fi

    echo "============================================================"
    echo "[GPU ${gpu}][EVAL] ${name} epoch=${epoch} ckpt=${bn}"
    echo "============================================================"

    local out="${out_dir}/eval_${epoch}.out"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      ${PY} -u "${SCRIPT}" \
        --save_path "${save_path}" \
        --model_save_path "${model_save_path}" \
        --resume "${c}" \
        "${EVAL_ARGS[@]}" \
        > "${out}" 2>&1 || true

    local metrics
    metrics="$(${PARSE_PY} "${out}" "${SCORE_ALPHA}" 2>/dev/null)" || metrics=$'NA\tNA\tNA\t1000000000'

    echo "[GPU ${gpu}][EVAL] ${name} epoch=${epoch} -> ${metrics}"
    echo -e "${epoch}\t${bn}\t${metrics}" >> "${tsv}"

    local score
    score="$(echo "${metrics}" | awk -F'\t' '{print $4}')"

    if [[ -n "${score}" ]]; then
      if ${PY} -c 'import sys; sys.exit(0 if float(sys.argv[2]) < float(sys.argv[1]) else 1)' "${best_score}" "${score}"; then
        best_score="${score}"
        best_epoch="${epoch}"
        best_ckpt="${bn}"
        best_metrics="${metrics}"
      fi
    fi
  done

  echo -e "${name}\t${best_epoch}\t${best_ckpt}\t${best_metrics}\t0\t${gpu}\t${args_str}" > "${save_path}/result.tsv"
  echo "[GPU ${gpu}] DONE ${name} best_epoch=${best_epoch} best_ckpt=${best_ckpt} best_score=${best_score}"
}

############################################
# 调度器：保证每张 GPU 同时只跑一个任务
############################################
declare -a gpu_busy_pids=()
for ((i=0;i<NGPUS;i++)); do gpu_busy_pids+=(""); done

acquire_gpu_idx() {
  while true; do
    for ((gi=0;gi<NGPUS;gi++)); do
      local pid="${gpu_busy_pids[$gi]}"
      if [[ -z "${pid}" ]]; then
        echo "${gi}"
        return 0
      fi
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        gpu_busy_pids[$gi]=""
        echo "${gi}"
        return 0
      fi
    done
    sleep 2
  done
}

############################################
# 主循环：发任务
############################################
echo "============================================================"
echo "[FAIR BASELINES] SHARD_TAG=${SHARD_TAG}"
echo "[FAIR BASELINES] ROOT=${ROOT}"
echo "[FAIR BASELINES] STAGE1_ROOT=${STAGE1_ROOT}"
echo "[FAIR BASELINES] GPUS=${GPUS} (NGPUS=${NGPUS})"
echo "[FAIR BASELINES] WILL_RUN=${#RUN_TASK_NAMES[@]} runs"
echo "[FAIR BASELINES] RUN_NAMES=${RUN_NAMES:-<ALL>}"
echo "[FAIR BASELINES] STAGE1_MAX_EPOCH=${STAGE1_MAX_EPOCH} SAVE_EVERY=${SAVE_EVERY} EVAL_EVERY=${EVAL_EVERY}"
echo "[FAIR BASELINES] SCORE_ALPHA=${SCORE_ALPHA}"
echo "============================================================"
echo

for idx in "${!RUN_TASK_NAMES[@]}"; do
  name="${RUN_TASK_NAMES[$idx]}"
  extra="${RUN_TASK_EXTRA[$idx]}"

  gi="$(acquire_gpu_idx)"
  gpu="${GPU_ARR[$gi]}"

  run_one "${gpu}" "${name}" "${extra}" &
  gpu_busy_pids[$gi]=$!
done

for ((gi=0;gi<NGPUS;gi++)); do
  pid="${gpu_busy_pids[$gi]}"
  if [[ -n "${pid}" ]]; then
    wait "${pid}" 2>/dev/null || true
  fi
done

for name in "${RUN_TASK_NAMES[@]}"; do
  f="${STAGE1_ROOT}/${name}/result.tsv"
  if [[ -f "${f}" ]]; then
    cat "${f}" >> "${STAGE1_SUMMARY}"
  else
    echo -e "${name}\tNA\tNA\tNA\tNA\tNA\t1000000000\t99\tNA\tmissing_result" >> "${STAGE1_SUMMARY}"
  fi
done

echo
echo "[DONE] Fair baselines shard finished -> ${STAGE1_SUMMARY}"
column -t -s $'\t' "${STAGE1_SUMMARY}" | sed 's/\r$//'

############################################
# Stage-1 完成后：自动评测（no-AS-norm & AS-norm）并保存 Excel
############################################

# 让路径不依赖“你从哪里运行脚本”
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 评测用脚本（输出 [METRIC] 行）
EVAL_SCRIPT="${EVAL_SCRIPT:-trainECAPAModel_fair_ddp_asnorm.py}"

# eval sweep + merge 工具：默认用“脚本同目录”的文件
EVAL_SWEEP_SH="${EVAL_SWEEP_SH:-${SCRIPT_DIR}/eval_sweep_ckpts_allmetrics.sh}"
MERGE_PY="${MERGE_PY:-${SCRIPT_DIR}/merge_shards_split_protocols.py}"

# 强制检查，避免跑半天才发现没评测
[[ -f "${EVAL_SWEEP_SH}" ]] || { echo "[FATAL] EVAL_SWEEP_SH not found: ${EVAL_SWEEP_SH}" >&2; exit 2; }
[[ -f "${MERGE_PY}" ]]      || { echo "[FATAL] MERGE_PY not found: ${MERGE_PY}" >&2; exit 2; }

COHORT_LIST="${COHORT_LIST:-exps/_cohort/cohort_10000.txt}"
ASN_TOPK="${ASN_TOPK:-200}"
ASN_BS="${ASN_BS:-64}"

run_eval_condition() {
  local tag_prefix="$1"
  local asnorm_args="$2"
  local out_xlsx="$3"
  local out_merged_tsv="$4"

  echo
  echo "============================================================"
  echo "[EVAL] tag=${tag_prefix}  EXP_ROOTS=${STAGE1_ROOT}"
  echo "============================================================"

  local nshards="${NGPUS}"

  for ((i=0; i<nshards; i++)); do
    local gpu="${GPU_ARR[$i]}"
    (
      set -euo pipefail
      GPU="${gpu}" NUM_SHARDS="${nshards}" SHARD_IDX="${i}" \
      TAG="${tag_prefix}_s${i}.shard${i}" \
      EXP_ROOTS="${STAGE1_ROOT}" \
      EVAL_LIST="${EVAL_LIST}" EVAL_PATH="${EVAL_PATH}" GENDER_MAP="${EVAL_GMAP}" \
      N_CLASS="${N_CLASS}" \
      SCRIPT="${EVAL_SCRIPT}" \
      WRITE_ALL_METRICS=1 \
      ASNORM_ARGS="${asnorm_args}" \
      bash "${EVAL_SWEEP_SH}" \
        > "${STAGE1_ROOT}/logs_eval_${tag_prefix}_s${i}.log" 2>&1
    ) &
  done
  wait

  echo "[EVAL] all shards done for ${tag_prefix}"

  local shards_list=""
  for ((i=0; i<nshards; i++)); do
    local f="${STAGE1_ROOT}/_eval_summary_${tag_prefix}_s${i}.shard${i}.tsv"
    if [[ -z "${shards_list}" ]]; then
      shards_list="${f}"
    else
      shards_list="${shards_list},${f}"
    fi
  done

  ${PY} "${MERGE_PY}" \
    --shards "${shards_list}" \
    --out_merged_tsv "${out_merged_tsv}" \
    --out_xlsx "${out_xlsx}" \
    --include_pooled

  echo "[EVAL] wrote ${out_xlsx}"
}

# no-AS-norm
run_eval_condition \
  "noasn" \
  "" \
  "${STAGE1_ROOT}/results_noasn_merged_split.xlsx" \
  "${STAGE1_ROOT}/_eval_summary_noasn_merged.tsv"

# AS-norm
if [[ ! -f "${COHORT_LIST}" ]]; then
  echo "[WARN] COHORT_LIST not found: ${COHORT_LIST}"
  echo "[WARN] Skip AS-norm evaluation."
else
  ASNORM_ARGS="--use_asnorm --cohort_list ${COHORT_LIST} --asnorm_topk ${ASN_TOPK} --asnorm_batch_size ${ASN_BS}"
  run_eval_condition \
    "asn" \
    "${ASNORM_ARGS}" \
    "${STAGE1_ROOT}/results_asn_merged_split.xlsx" \
    "${STAGE1_ROOT}/_eval_summary_asn_merged.tsv"
fi

echo
echo "[DONE] Evaluation finished."
echo "  - ${STAGE1_ROOT}/results_noasn_merged_split.xlsx"
echo "  - ${STAGE1_ROOT}/results_asn_merged_split.xlsx"

