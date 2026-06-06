#!/usr/bin/env bash
set -euo pipefail

# =========================
# 目的：
#   在“本机 4 张卡”上跑 Stage-0 的一个 shard（指定的若干组）
#   每张卡同一时间只跑 1 个实验；如果该 GPU 被分到多个 run，会顺序跑
#
# 用法示例：
#   # ServerA
#   GPUS=0,1,2,3 RUN_NAMES="T01_r1_cap005_sat001_adv002,T02_r1_cap005_sat002_adv002,T03_r1_cap005_sat003_adv002,T04_r1_cap020_sat002_adv002,T09_r5_cap005_sat002_adv005" ROOT=exps/_two_stage/G1_serverA ./run_stage0_shard_1gpu_each.sh
#
#   # ServerB
#   GPUS=0,1,2,3 RUN_NAMES="T05_r3_cap005_sat002_adv002,T06_r5_cap005_sat002_adv002,T07_r1_cap005_sat002_adv005,T08_r3_cap005_sat002_adv005" ROOT=exps/_two_stage/G1_serverB ./run_stage0_shard_1gpu_each.sh
# =========================

GPUS="${GPUS:-0,1,2,3}"
SHARD_TAG="${SHARD_TAG:-$(hostname -s)}"
SCRIPT="${SCRIPT:-trainECAPAModel_fair_ddp.py}"
PY="${PY:-python}"

ROOT="${ROOT:-exps/_two_stage}"
STAGE0_ROOT="${ROOT}/stage0_tune"
mkdir -p "${STAGE0_ROOT}"

# 数据/模型路径（按你的工程实际修改/覆盖）
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
BATCH_SIZE="${BATCH_SIZE:-256}"      # per-GPU（单卡就是全局）
NUM_FRAMES="${NUM_FRAMES:-200}"
LR="${LR:-3e-4}"
ENCODER_LR="${ENCODER_LR:-0}"
FREEZE_ENCODER_EPOCHS="${FREEZE_ENCODER_EPOCHS:-5}"
LR_DECAY="${LR_DECAY:-0.995}"
USE_AMP="${USE_AMP:-1}"

# 必须给：RUN_NAMES=逗号分隔的 name 列表
RUN_NAMES="${RUN_NAMES:-}"
if [[ -z "${RUN_NAMES}" ]]; then
  echo "[FATAL] RUN_NAMES is empty. Provide comma-separated names." >&2
  exit 2
fi

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

# Stage0 9 组 grid（保持与你当前一致）
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

# 解析 eval 输出（同你 two-stage 的口径）
PARSE_PY="${STAGE0_ROOT}/parse_eval_and_score.py"
cat > "${PARSE_PY}" <<'PY'
#!/usr/bin/env python3
import re, sys
out_path = sys.argv[1]
txt = open(out_path, "r", errors="ignore").read()

def grab(pat):
    m = re.findall(pat, txt, flags=re.M)
    return m[-1] if m else None

pooled_eer = grab(r'^\[EVAL\]\s+POOLED:.*EER=([0-9.]+)')
pooled_mindcf = grab(r'^\[EVAL\]\s+POOLED:.*minDCF=([0-9.]+)')

def g(p):
    s = grab(p)
    return float(s) if s is not None else None

H = g(r'^\[EVAL\]\s+.*H.*GARBE_fmr=([0-9.]+)')
O = g(r'^\[EVAL\]\s+.*O.*GARBE_fmr=([0-9.]+)')
E = g(r'^\[EVAL\]\s+.*E.*GARBE_fmr=([0-9.]+)')

vals = [x for x in [H,O,E] if x is not None]
mean_garbe = sum(vals)/len(vals) if vals else None

if pooled_eer is None:
    score = 1e9
else:
    score = float(pooled_eer) + 20.0 * float(mean_garbe or 0.0)

print(f"{pooled_eer or 'NA'}\t{pooled_mindcf or 'NA'}\t{mean_garbe if mean_garbe is not None else 'NA'}\t{score:.6f}")
PY
chmod +x "${PARSE_PY}"

STAGE0_SUMMARY="${STAGE0_ROOT}/results_stage0_shard.${SHARD_TAG}.tsv"
echo -e "name\tckpt\tpooled_eer\tpooled_mindcf\tmean_garbe_fmr\tscore\texit_code\tgpu\targs" > "${STAGE0_SUMMARY}"

# 只取 RUN_NAMES 指定的行
# 转为正则：^(A|B|C)$
RUN_RE="^($(echo "${RUN_NAMES}" | tr ',' '|' ))$"
mapfile -t RUN_LINES < <(grep -v '^\s*#' "${GRID_TSV}" | awk -F'\t' -v re="${RUN_RE}" '$1 ~ re {print $0}')

if [[ "${#RUN_LINES[@]}" -eq 0 ]]; then
  echo "[FATAL] No runs matched RUN_NAMES. RUN_NAMES=${RUN_NAMES}" >&2
  exit 3
fi

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NGPUS="${#GPU_ARR[@]}"

run_one() {
  local gpu="$1"; shift
  local line="$1"; shift

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
  local args_str
  args_str="$(printf "%q " "${extra_args[@]}")"

  echo "[GPU ${gpu}] START ${name}"

  # 单卡训练
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
    echo -e "${name}\tNA\tNA\tNA\tNA\t1000000000\t${rc}\t${gpu}\t${args_str}" >> "${STAGE0_SUMMARY}"
    echo "[GPU ${gpu}] FAIL ${name} rc=${rc}"
    return 0
  fi

  # eval last ckpt
  ckpt="$(ls -1 "${model_save_path}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo -e "${name}\tNA\tNA\tNA\tNA\t1000000000\t3\t${gpu}\t${args_str}" >> "${STAGE0_SUMMARY}"
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

  metrics="$(${PARSE_PY} "${save_path}/eval_last.out" || echo $'NA\tNA\tNA\t1000000000')"
  echo -e "${name}\t$(basename "${ckpt}")\t${metrics}\t0\t${gpu}\t${args_str}" >> "${STAGE0_SUMMARY}"
  echo "[GPU ${gpu}] DONE ${name}"
}

# 每张 GPU 一个 worker：顺序跑它被分配到的任务（避免同卡并发）
declare -a WORKER_PIDS=()
for gi in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$gi]}"
  (
    for idx in "${!RUN_LINES[@]}"; do
      if (( idx % NGPUS == gi )); then
        run_one "${gpu}" "${RUN_LINES[$idx]}"
      fi
    done
  ) &
  WORKER_PIDS+=($!)
done

for p in "${WORKER_PIDS[@]}"; do
  wait "${p}" || true
done

echo
echo "[INFO] SHARD_TAG=${SHARD_TAG}"
echo "[DONE] Stage-0 shard finished -> ${STAGE0_SUMMARY}"
column -t -s $'\t' "${STAGE0_SUMMARY}" | sed 's/\r$//'
