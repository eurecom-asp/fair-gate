#!/usr/bin/env bash
set -euo pipefail

############################################
# Run concurrently:
#   - no-ASnorm on 4 GPUs (0,1,2,3) with 4 shards
#   - ASnorm on 3 GPUs (4,5,6) with 3 shards
#
# Requires:
#   - eval_sweep_ckpts.sh  (supports NUM_SHARDS/SHARD_IDX/TAG/ASNORM_ARGS)
#   - parse_eval_to_excel.py
#   - trainECAPAModel_fair_ddp.py (supports --use_asnorm ... if ASNORM enabled)
############################################

EXP_ROOTS="${EXP_ROOTS:-exps/_two_stage/G1/stage1_ablation}"
EVAL_LIST="${EVAL_LIST:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-H-abs.txt,/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-O-abs.txt,/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-E-abs.txt}"
EVAL_PATH="${EVAL_PATH:-/medias/speech/data/VoxCeleb1}"
GENDER_MAP="${GENDER_MAP:-/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/gender_map.json}"

PY="${PY:-python}"
SCRIPT="${SCRIPT:-trainECAPAModel_fair_ddp_asnorm.py}"

# GPU allocation (edit if needed)
NOASN_GPUS="${NOASN_GPUS:-0,1,2,3}"
ASN_GPUS="${ASN_GPUS:-4,5,6}"

# AS-norm parameters (COHORT_LIST required for AS-norm)
COHORT_LIST="${COHORT_LIST:-}"
ASN_TOPK="${ASN_TOPK:-200}"
ASN_BS="${ASN_BS:-256}"

# Output root = first EXP_ROOT
OUT_ROOT="${OUT_ROOT:-${EXP_ROOTS%%,*}}"
mkdir -p "${OUT_ROOT}"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

# split CSV -> array
csv_to_arr() { IFS=',' read -r -a _a <<< "$1"; echo "${_a[@]}"; }

NOASN_GPU_ARR=($(csv_to_arr "$NOASN_GPUS"))
ASN_GPU_ARR=($(csv_to_arr "$ASN_GPUS"))

if [[ ${#NOASN_GPU_ARR[@]} -ne 4 ]]; then
  echo "[FATAL] NOASN_GPUS must have 4 GPUs, got: ${#NOASN_GPU_ARR[@]} ($NOASN_GPUS)" >&2
  exit 1
fi
if [[ ${#ASN_GPU_ARR[@]} -ne 3 ]]; then
  echo "[FATAL] ASN_GPUS must have 3 GPUs, got: ${#ASN_GPU_ARR[@]} ($ASN_GPUS)" >&2
  exit 1
fi

if [[ -z "${COHORT_LIST}" ]]; then
  echo "[FATAL] COHORT_LIST is not set. AS-norm group cannot run." >&2
  exit 1
fi
if [[ ! -f "${COHORT_LIST}" ]]; then
  echo "[FATAL] COHORT_LIST file not found: ${COHORT_LIST}" >&2
  exit 1
fi

ASNORM_ARGS="--use_asnorm --cohort_list ${COHORT_LIST} --asnorm_topk ${ASN_TOPK} --asnorm_batch_size ${ASN_BS}"

run_one() {
  local tag="$1" shard="$2" nshard="$3" gpu="$4" asnargs="$5" logfile="$6"
  echo "[$(ts)] [START] ${tag} shard=${shard}/${nshard} gpu=${gpu}"
  GPU="${gpu}" NUM_SHARDS="${nshard}" SHARD_IDX="${shard}" TAG="${tag}" \
  PY="${PY}" SCRIPT="${SCRIPT}" \
  EXP_ROOTS="${EXP_ROOTS}" EVAL_LIST="${EVAL_LIST}" EVAL_PATH="${EVAL_PATH}" GENDER_MAP="${GENDER_MAP}" \
  ASNORM_ARGS="${asnargs}" \
  bash eval_sweep_ckpts.sh > "${logfile}" 2>&1
  echo "[$(ts)] [DONE ] ${tag} shard=${shard}/${nshard} gpu=${gpu}"
}

merge_tsv() {
  local out="$1"; shift
  local parts=("$@")
  local header_src=""
  for p in "${parts[@]}"; do
    [[ -f "$p" ]] && { header_src="$p"; break; }
  done
  [[ -n "$header_src" ]] || { echo "[FATAL] No TSV parts found to merge into $out" >&2; return 1; }
  head -n 1 "$header_src" > "$out"
  for p in "${parts[@]}"; do
    if [[ -f "$p" ]]; then tail -n +2 "$p" >> "$out"; else echo "[WARN] missing $p" >&2; fi
  done
}

to_xlsx() {
  local tsv="$1" xlsx="$2"
  ${PY} parse_eval_to_excel.py --tsv "$tsv" --xlsx "$xlsx"
}

echo "[$(ts)] OUT_ROOT=${OUT_ROOT}"
echo "[$(ts)] EXP_ROOTS=${EXP_ROOTS}"
echo "[$(ts)] SCRIPT=${SCRIPT}"
echo "[$(ts)] noasn GPUs=${NOASN_GPUS} | asn GPUs=${ASN_GPUS}"
echo "[$(ts)] COHORT_LIST=${COHORT_LIST}"
echo "[$(ts)] Starting noasn(4) + asn(3) in parallel..."

# ---- noasn group (4 shards)
noasn_parts=()
noasn_pids=()
for i in 0 1 2 3; do
  tag="noasn_s${i}"
  gpu="${NOASN_GPU_ARR[$i]}"
  log="${OUT_ROOT}/logs_${tag}.log"
  noasn_parts+=("${OUT_ROOT}/_eval_summary_${tag}.tsv")
  ( run_one "${tag}" "${i}" 4 "${gpu}" "" "${log}" ) &
  noasn_pids+=($!)
done

# ---- asn group (3 shards)
asn_parts=()
asn_pids=()
for i in 0 1 2; do
  tag="asn_s${i}"
  gpu="${ASN_GPU_ARR[$i]}"
  log="${OUT_ROOT}/logs_${tag}.log"
  asn_parts+=("${OUT_ROOT}/_eval_summary_${tag}.tsv")
  ( run_one "${tag}" "${i}" 3 "${gpu}" "${ASNORM_ARGS}" "${log}" ) &
  asn_pids+=($!)
done

# wait all
for pid in "${noasn_pids[@]}"; do wait "$pid"; done
for pid in "${asn_pids[@]}"; do wait "$pid"; done

# merge + excel
MERGED_NOASN="${OUT_ROOT}/_eval_summary_noasn.tsv"
MERGED_ASN="${OUT_ROOT}/_eval_summary_asn.tsv"

merge_tsv "${MERGED_NOASN}" "${noasn_parts[@]}"
merge_tsv "${MERGED_ASN}" "${asn_parts[@]}"

to_xlsx "${MERGED_NOASN}" "${OUT_ROOT}/results_noasn.xlsx"
to_xlsx "${MERGED_ASN}" "${OUT_ROOT}/results_asn.xlsx"

echo
echo "[$(ts)] ALL DONE."
echo "  - ${MERGED_NOASN}"
echo "  - ${OUT_ROOT}/results_noasn.xlsx"
echo "  - ${MERGED_ASN}"
echo "  - ${OUT_ROOT}/results_asn.xlsx"
BASH

