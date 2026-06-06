#!/usr/bin/env bash
set -euo pipefail

############################################
# Stage-1 EXTRA ablation ONLY (S10..S19)
#
# 两台服务器 8 卡跑法：
#  serverA:
#    GPUS=0,1,2,3 NUM_SHARDS=2 SHARD_IDX=0 ROOT=exps/_two_stage/G1 \
#      bash run_stage1_more_ablation_shard.sh
#  serverB:
#    GPUS=0,1,2,3 NUM_SHARDS=2 SHARD_IDX=1 ROOT=exps/_two_stage/G1 \
#      bash run_stage1_more_ablation_shard.sh
#
# 可选：
#   STAGE1_ROOT=exps/_two_stage/G1/stage1_ablation_more
#   VERBOSE=1   (eval 时 tee 到屏幕 + 文件)
#   TAIL_ON_FAIL=200
############################################

GPUS="${GPUS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SHARD_IDX="${SHARD_IDX:-0}"
SHARD_TAG="${SHARD_TAG:-$(hostname -s)}"

PY="${PY:-python}"
SCRIPT="${SCRIPT:-trainECAPAModel_fair_ddp.py}"

ROOT="${ROOT:-exps/_two_stage/G1}"

# 关键：后10个单独目录，避免和 S1..S9 冲突
STAGE1_ROOT="${STAGE1_ROOT:-${ROOT}/stage1_ablation_more}"
mkdir -p "${STAGE1_ROOT}"

VERBOSE="${VERBOSE:-0}"
TAIL_ON_FAIL="${TAIL_ON_FAIL:-200}"

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
# 训练超参（Stage-1）
############################################
STAGE1_MAX_EPOCH="${STAGE1_MAX_EPOCH:-80}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-10}"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_FRAMES="${NUM_FRAMES:-200}"

LR="${LR:-3e-4}"
ENCODER_LR="${ENCODER_LR:-1e-5}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-10}"

LR_DECAY="${LR_DECAY:-0.995}"
USE_AMP="${USE_AMP:-1}"

SCORE_ALPHA="${SCORE_ALPHA:-20}"

############################################
# best_args.txt（Stage-0 选出来的）
############################################
BEST_ARGS_FILE="${BEST_ARGS_FILE:-${ROOT}/stage0_tune/best_args.txt}"
[[ -f "${BEST_ARGS_FILE}" ]] || { echo "[FATAL] BEST_ARGS_FILE not found: ${BEST_ARGS_FILE}" >&2; exit 2; }
BEST_ARGS_STR="$(tr '\n' ' ' < "${BEST_ARGS_FILE}" | sed 's/\r//g' | xargs)"

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
# args override helper（稳，不靠脆弱 sed）
############################################
ARGS_TOOL="${STAGE1_ROOT}/_args_override.py"
cat > "${ARGS_TOOL}" <<'PY'
#!/usr/bin/env python3
import shlex, sys
base = sys.argv[1]
overrides = sys.argv[2:]
toks = shlex.split(base)
pos = {}
i=0
while i < len(toks):
    if toks[i].startswith("--"):
        flag = toks[i]
        if i+1 < len(toks) and (not toks[i+1].startswith("--")):
            pos[flag]=i; i+=2
        else:
            pos[flag]=i; i+=1
    else:
        i+=1

def set_kv(k,v):
    flag="--"+k
    if flag in pos:
        j=pos[flag]
        if j+1 >= len(toks) or toks[j+1].startswith("--"):
            toks.insert(j+1, v)
        else:
            toks[j+1]=v
    else:
        toks.extend([flag,v])

for ov in overrides:
    if "=" not in ov:
        flag = ov if ov.startswith("--") else ("--"+ov)
        if flag not in toks: toks.append(flag)
        continue
    k,v = ov.split("=",1)
    k = k[2:] if k.startswith("--") else k
    set_kv(k,v)

print(" ".join(shlex.quote(x) for x in toks))
PY
chmod +x "${ARGS_TOOL}"

############################################
# eval parser
############################################
PARSE_PY="${STAGE1_ROOT}/parse_eval_and_score.py"
cat > "${PARSE_PY}" <<'PY'
#!/usr/bin/env python3
import re, sys
p = sys.argv[1]
alpha = float(sys.argv[2]) if len(sys.argv)>2 else 20.0
txt = open(p,"r",errors="ignore").read()

def last(pat):
    m = re.findall(pat, txt, flags=re.M)
    return m[-1] if m else None

pooled_eer = last(r'^\[EVAL\]\s+POOLED:.*EER=([0-9.]+)')
pooled_mindcf = last(r'^\[EVAL\]\s+POOLED:.*minDCF=([0-9.]+)')

def g(pat):
    s = last(pat)
    return float(s) if s is not None else None

H = g(r'^\[EVAL\]\s+sub-vox1-H-abs\.txt:.*GARBE_fmr=([0-9.]+)')
O = g(r'^\[EVAL\]\s+sub-vox1-O-abs\.txt:.*GARBE_fmr=([0-9.]+)')
E = g(r'^\[EVAL\]\s+sub-vox1-E-abs\.txt:.*GARBE_fmr=([0-9.]+)')
vals = [x for x in (H,O,E) if x is not None]
mean_garbe = sum(vals)/len(vals) if vals else None

if pooled_eer is None:
    print("NA\tNA\tNA\t1000000000"); sys.exit(0)

eer=float(pooled_eer)
mg=float(mean_garbe) if mean_garbe is not None else 0.0
score=eer + alpha*mg
print(f"{eer:.4f}\t{(pooled_mindcf or 'NA')}\t{(mean_garbe if mean_garbe is not None else 'NA')}\t{score:.6f}")
PY
chmod +x "${PARSE_PY}"

############################################
# 只定义“后 10 个”
############################################
declare -a ALL_NAMES=(
  "S10_MAIN_minus_cap"
  "S11_MAIN_minus_sat"
  "S12_MAIN_minus_gs"
  "S13_MAIN_minus_adv"
  "S14_Sat_only"
  "S15_Gs_only"
  "S16_Adv_only"
  "S17_MAIN_r005"
  "S18_MAIN_r010"
  "S19_MAIN_r020"
)

extra_args_for_run() {
  local run_name="$1"
  case "${run_name}" in
    "S10_MAIN_minus_cap") "${ARGS_TOOL}" "${BEST_ARGS_STR}" "lambda_css_cap=0.0" ;;
    "S11_MAIN_minus_sat") "${ARGS_TOOL}" "${BEST_ARGS_STR}" "lambda_css_sat=0.0" ;;
    "S12_MAIN_minus_gs")  "${ARGS_TOOL}" "${BEST_ARGS_STR}" "lambda_gender_s=0.0" ;;
    "S13_MAIN_minus_adv") "${ARGS_TOOL}" "${BEST_ARGS_STR}" "lambda_gender_adv=0.0" ;;
    "S14_Sat_only") "${ARGS_TOOL}" "${BEST_ARGS_STR}" "lambda_css_cap=0.0" "lambda_gender_s=0.0" "lambda_gender_adv=0.0" ;;
    "S15_Gs_only")  "${ARGS_TOOL}" "${BEST_ARGS_STR}" "lambda_css_cap=0.0" "lambda_css_sat=0.0" "lambda_gender_adv=0.0" ;;
    "S16_Adv_only") "${ARGS_TOOL}" "${BEST_ARGS_STR}" "lambda_css_cap=0.0" "lambda_css_sat=0.0" "lambda_gender_s=0.0" ;;
    "S17_MAIN_r005") "${ARGS_TOOL}" "${BEST_ARGS_STR}" "css_target_ratio=0.05" ;;
    "S18_MAIN_r010") "${ARGS_TOOL}" "${BEST_ARGS_STR}" "css_target_ratio=0.10" ;;
    "S19_MAIN_r020") "${ARGS_TOOL}" "${BEST_ARGS_STR}" "css_target_ratio=0.20" ;;
    *) echo "${BEST_ARGS_STR}" ;;
  esac
}

want_idx() { local idx="$1"; [[ $((idx % NUM_SHARDS)) -eq "${SHARD_IDX}" ]]; }

RUN_TASK_NAMES=()
for i in "${!ALL_NAMES[@]}"; do
  want_idx "${i}" && RUN_TASK_NAMES+=("${ALL_NAMES[$i]}")
done

STAGE1_SUMMARY="${STAGE1_ROOT}/results_stage1_moreablation_shard.${SHARD_TAG}.S${SHARD_IDX}of${NUM_SHARDS}.tsv"
echo -e "name\tbest_epoch\tbest_ckpt\tpooled_eer\tpooled_mindcf\tmean_garbe_fmr\tscore\texit_code\tgpu\targs" > "${STAGE1_SUMMARY}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPUS="${#GPU_ARR[@]}"

list_ckpts() { (ls -1 "$1"/model_*.pt 2>/dev/null; ls -1 "$1"/model_*.model 2>/dev/null) | sort -V || true; }
extract_epoch() { echo "$1" | sed -n 's/^model_\([0-9]\+\)\.\(pt\|model\)$/\1/p'; }

run_one() {
  local gpu="$1"; shift
  local name="$1"; shift

  local save_path="${STAGE1_ROOT}/${name}"
  local model_save_path="${save_path}/models"
  mkdir -p "${model_save_path}"

  local extra_str; extra_str="$(extra_args_for_run "${name}")"
  # shellcheck disable=SC2206
  local extra_args=(${extra_str})
  local args_str; args_str="$(printf "%q " "${BASE_ARGS_COMMON[@]}" "${extra_args[@]}")"

  echo "[GPU ${gpu}] START ${name}"
  echo "${args_str}" > "${save_path}/args.txt"

  # ---- TRAIN
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
    echo "[GPU ${gpu}] FAIL ${name} rc=${rc}"
    tail -n "${TAIL_ON_FAIL}" "${save_path}/train.out" || true
    echo -e "${name}\tNA\tNA\tNA\tNA\tNA\t1000000000\t${rc}\t${gpu}\t${args_str}" > "${save_path}/result.tsv"
    return 0
  fi

  # ---- EVAL ALL CKPTS
  local out_dir="${save_path}/eval_all_ckpts"
  mkdir -p "${out_dir}"
  local tsv="${out_dir}/eval_all_ckpts.tsv"
  echo -e "epoch\tckpt\tpooled_eer\tpooled_mindcf\tmean_garbe_fmr\tscore\trc" > "${tsv}"

  mapfile -t ckpts < <(list_ckpts "${model_save_path}")
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "[GPU ${gpu}] FAIL ${name}: no ckpts"
    echo -e "${name}\tNA\tNA\tNA\tNA\tNA\t1000000000\t3\t${gpu}\t${args_str}" > "${save_path}/result.tsv"
    return 0
  fi

  local best_epoch="NA" best_ckpt="NA" best_metrics=$'NA\tNA\tNA\t1000000000' best_score="1000000000"

  for c in "${ckpts[@]}"; do
    local bn epoch
    bn="$(basename "${c}")"
    epoch="$(extract_epoch "${bn}")"
    [[ -n "${epoch}" ]] || continue

    if [[ "${EVAL_EVERY}" -gt 1 ]] && (( 10#${epoch} % ${EVAL_EVERY} != 0 )); then
      continue
    fi

    local out="${out_dir}/eval_${epoch}.out"
    echo "============================================================"
    echo "[GPU ${gpu}][EVAL] ${name} epoch=${epoch} ckpt=${bn}"
    echo "============================================================"

    set +e
    if [[ "${VERBOSE}" == "1" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" \
        ${PY} -u "${SCRIPT}" \
          --save_path "${save_path}" \
          --model_save_path "${model_save_path}" \
          --resume "${c}" \
          "${EVAL_ARGS[@]}" \
          "${extra_args[@]}" \
          2>&1 | tee "${out}"
      local eval_rc=${PIPESTATUS[0]}
    else
      CUDA_VISIBLE_DEVICES="${gpu}" \
        ${PY} -u "${SCRIPT}" \
          --save_path "${save_path}" \
          --model_save_path "${model_save_path}" \
          --resume "${c}" \
          "${EVAL_ARGS[@]}" \
          "${extra_args[@]}" \
          > "${out}" 2>&1
      local eval_rc=$?
    fi
    set -e

    if [[ "${eval_rc}" -ne 0 ]]; then
      echo "[GPU ${gpu}][EVAL][ERROR] ${name} epoch=${epoch} rc=${eval_rc} log=${out}"
      tail -n "${TAIL_ON_FAIL}" "${out}" || true
    fi

    local metrics
    metrics="$(${PARSE_PY} "${out}" "${SCORE_ALPHA}" 2>/dev/null || echo -e "NA\tNA\tNA\t1000000000")"
    echo "[GPU ${gpu}][EVAL] ${name} epoch=${epoch} -> ${metrics} (rc=${eval_rc})"
    echo -e "${epoch}\t${bn}\t${metrics}\t${eval_rc}" >> "${tsv}"

    local score
    score="$(echo "${metrics}" | awk -F'\t' '{print $4}')"
    if [[ -n "${score}" ]]; then
      if ${PY} -c 'import sys; sys.exit(0 if float(sys.argv[2]) < float(sys.argv[1]) else 1)' "${best_score}" "${score}"; then
        best_score="${score}"; best_epoch="${epoch}"; best_ckpt="${bn}"; best_metrics="${metrics}"
      fi
    fi
  done

  echo -e "${name}\t${best_epoch}\t${best_ckpt}\t${best_metrics}\t0\t${gpu}\t${args_str}" > "${save_path}/result.tsv"
  echo "[GPU ${gpu}] DONE ${name} best_epoch=${best_epoch} best_ckpt=${best_ckpt} best_score=${best_score}"
}

declare -a gpu_busy_pids=()
for ((i=0;i<NGPUS;i++)); do gpu_busy_pids+=(""); done

acquire_gpu_idx() {
  while true; do
    for ((gi=0;gi<NGPUS;gi++)); do
      local pid="${gpu_busy_pids[$gi]}"
      if [[ -z "${pid}" ]]; then echo "${gi}"; return 0; fi
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        gpu_busy_pids[$gi]=""
        echo "${gi}"; return 0
      fi
    done
    sleep 2
  done
}

echo "============================================================"
echo "[MORE ABLATION] ROOT=${ROOT}"
echo "[MORE ABLATION] STAGE1_ROOT=${STAGE1_ROOT}"
echo "[MORE ABLATION] GPUS=${GPUS} (NGPUS=${NGPUS})"
echo "[MORE ABLATION] NUM_SHARDS=${NUM_SHARDS} SHARD_IDX=${SHARD_IDX}"
echo "[MORE ABLATION] WILL_RUN=${#RUN_TASK_NAMES[@]} runs:"
printf "  - %s\n" "${RUN_TASK_NAMES[@]}"
echo "[MORE ABLATION] SAVE_EVERY=${SAVE_EVERY} EVAL_EVERY=${EVAL_EVERY} MAX_EPOCH=${STAGE1_MAX_EPOCH}"
echo "============================================================"
echo

for name in "${RUN_TASK_NAMES[@]}"; do
  gi="$(acquire_gpu_idx)"
  gpu="${GPU_ARR[$gi]}"
  run_one "${gpu}" "${name}" &
  gpu_busy_pids[$gi]=$!
done

for ((gi=0;gi<NGPUS;gi++)); do
  pid="${gpu_busy_pids[$gi]}"
  [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
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
echo "[DONE] shard summary -> ${STAGE1_SUMMARY}"
column -t -s $'\t' "${STAGE1_SUMMARY}" | sed 's/\r$//'
