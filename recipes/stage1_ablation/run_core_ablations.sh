#!/usr/bin/env bash
set -euo pipefail

############################################
# 用法（两台机器分别跑不同 RUN_NAMES）：
#  serverA:
#    GPUS=0,1,2,3 RUN_NAMES="S1_BASELINE,S2_G_Cap_only,S3_G_Cap_Sat,S4_G_Cap_Gs" ROOT=exps/_two_stage/G1 ./run_stage1_shard_1gpu_each.sh
#  serverB:
#    GPUS=0,1,2,3 RUN_NAMES="S5_G_Cap_Adv,S6_MAIN(best),S7_MAIN_plus_Decor,S8_MAIN_plus_REx,S9_MAIN_plus_Decor_REx" ROOT=exps/_two_stage/G1 ./run_stage1_shard_1gpu_each.sh
#
# 环境变量（可覆盖）：
#   GPUS=0,1,2,3
#   RUN_NAMES=... (逗号分隔；不填则全跑)
#   ROOT=exps/_two_stage/G1
#   BEST_ARGS_FILE=.../best_args.txt
#   STAGE1_MAX_EPOCH=80  SAVE_EVERY=10  EVAL_EVERY=10
#   BATCH_SIZE=512  LR=3e-4  ...
#   SCORE_ALPHA=20  (score = pooled_eer + SCORE_ALPHA * mean_garbe_fmr)
############################################

GPUS="${GPUS:-0,1,2,3}"
RUN_NAMES="${RUN_NAMES:-}"
SHARD_TAG="${SHARD_TAG:-$(hostname -s)}"

PY="${PY:-python}"
SCRIPT="${SCRIPT:-trainECAPAModel_fair_ddp.py}"

ROOT="${ROOT:-exps/_two_stage/G1}"
STAGE1_ROOT="${ROOT}/stage1_ablation"
mkdir -p "${STAGE1_ROOT}"

############################################
# 数据/模型路径（按你工程实际）
############################################
TRAIN_LIST="${TRAIN_LIST:-/medias/speech/projects/quy/dataset/voxceleb2_train_list.txt}"
TRAIN_PATH="${TRAIN_PATH:-/medias/speech/data/VoxCeleb2}"
GENDER_MAP="${GENDER_MAP:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/gender_map_train.json}"

PRETRAIN="${PRETRAIN:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-ori/ECAPA-TDNN/exps/pretrain.model}"
N_CLASS="${N_CLASS:-5994}"

EVAL_LIST="${EVAL_LIST:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-H-abs.txt,/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-O-abs.txt,/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-E-abs.txt}"
EVAL_PATH="${EVAL_PATH:-/medias/speech/data/VoxCeleb1}"
EVAL_GMAP="${EVAL_GMAP:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/gender_map.json}"

############################################
# 训练超参（Stage-1 正式设置）
############################################
STAGE1_MAX_EPOCH="${STAGE1_MAX_EPOCH:-80}"
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
# best_args.txt（Stage-0 选出来的）
############################################
BEST_ARGS_FILE="${BEST_ARGS_FILE:-${ROOT}/stage0_tune/best_args.txt}"
[[ -f "${BEST_ARGS_FILE}" ]] || { echo "[FATAL] BEST_ARGS_FILE not found: ${BEST_ARGS_FILE}" >&2; exit 2; }
BEST_ARGS_STR="$(tr -d '\r' < "${BEST_ARGS_FILE}" | sed 's/^[ \t]*//;s/[ \t]*$//')"

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
# Stage-1 9 组（名字 + args）
############################################
declare -a ALL_NAMES=(
  "S1_BASELINE"
  "S2_G_Cap_only"
  "S3_G_Cap_Sat"
  "S4_G_Cap_Gs"
  "S5_G_Cap_Adv"
  "S6_MAIN(best)"
  "S7_MAIN_plus_Decor"
  "S8_MAIN_plus_REx"
  "S9_MAIN_plus_Decor_REx"
)

declare -a ALL_EXTRA=(
  "--baseline"
  "$(echo "${BEST_ARGS_STR}" | sed -E 's/--lambda_css_sat [^ ]+/--lambda_css_sat 0.0/g; s/--lambda_gender_s [^ ]+/--lambda_gender_s 0.0/g; s/--lambda_gender_adv [^ ]+/--lambda_gender_adv 0.0/g')"
  "$(echo "${BEST_ARGS_STR}" | sed -E 's/--lambda_gender_s [^ ]+/--lambda_gender_s 0.0/g; s/--lambda_gender_adv [^ ]+/--lambda_gender_adv 0.0/g')"
  "$(echo "${BEST_ARGS_STR}" | sed -E 's/--lambda_css_sat [^ ]+/--lambda_css_sat 0.0/g; s/--lambda_gender_adv [^ ]+/--lambda_gender_adv 0.0/g')"
  "$(echo "${BEST_ARGS_STR}" | sed -E 's/--lambda_css_sat [^ ]+/--lambda_css_sat 0.0/g; s/--lambda_gender_s [^ ]+/--lambda_gender_s 0.0/g')"
  "${BEST_ARGS_STR}"
  "${BEST_ARGS_STR} --lambda_decor 0.01"
  "${BEST_ARGS_STR} --lambda_rex 0.05"
  "${BEST_ARGS_STR} --lambda_decor 0.01 --lambda_rex 0.05"
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
STAGE1_SUMMARY="${STAGE1_ROOT}/results_stage1_shard.${SHARD_TAG}.tsv"
echo -e "name\tbest_epoch\tbest_ckpt\tpooled_eer\tpooled_mindcf\tmean_garbe_fmr\tscore\texit_code\tgpu\targs" > "${STAGE1_SUMMARY}"

############################################
# GPU 列表
############################################
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPUS="${#GPU_ARR[@]}"

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
    echo "[GPU ${gpu}] FAIL ${name} rc=${rc}"
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

    # 这里用 python -c 做浮点比较，完全避免 here-doc
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
echo "[STAGE1 SHARD] SHARD_TAG=${SHARD_TAG}"
echo "[STAGE1 SHARD] ROOT=${ROOT}"
echo "[STAGE1 SHARD] STAGE1_ROOT=${STAGE1_ROOT}"
echo "[STAGE1 SHARD] GPUS=${GPUS} (NGPUS=${NGPUS})"
echo "[STAGE1 SHARD] WILL_RUN=${#RUN_TASK_NAMES[@]} runs"
echo "[STAGE1 SHARD] RUN_NAMES=${RUN_NAMES:-<ALL>}"
echo "[STAGE1 SHARD] STAGE1_MAX_EPOCH=${STAGE1_MAX_EPOCH} SAVE_EVERY=${SAVE_EVERY} EVAL_EVERY=${EVAL_EVERY}"
echo "[STAGE1 SHARD] SCORE_ALPHA=${SCORE_ALPHA}"
echo "[STAGE1 SHARD] BEST_ARGS=${BEST_ARGS_STR}"
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
echo "[DONE] Stage-1 shard finished -> ${STAGE1_SUMMARY}"
column -t -s $'\t' "${STAGE1_SUMMARY}" | sed 's/\r$//'
echo
echo "[HINT] After both servers finish, merge shards:"
echo "  out=${STAGE1_ROOT}/results_stage1_merged.tsv"
echo "  head -n 1 ${STAGE1_ROOT}/results_stage1_shard.*.tsv > \$out"
echo "  for f in ${STAGE1_ROOT}/results_stage1_shard.*.tsv; do tail -n +2 \"\$f\" >> \$out; done"
echo "  column -t -s \$'\\t' \$out | sed 's/\\r\$//'"
