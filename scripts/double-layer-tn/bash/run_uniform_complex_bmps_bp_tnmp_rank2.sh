#!/usr/bin/env bash
# Run the retained uniform-complex double-layer benchmark with BP, TNMP rank-2,
# and boundary-MPS. Each method can be toggled independently.
#
# Usage:
#   scripts/run_uniform_complex_bmps_bp_tnmp_rank2.sh [extra args forwarded to Julia]
#
# Environment overrides:
#   JL_L                      Grid size (default: 8)
#   JL_CHI                    PEPS bond dimension (default: 16)
#   JL_SEED                   RNG seed (default: 7)
#   JL_UNIFORM_LO             Lower bound for real/imag draws (default: -0.5)
#   JL_UNIFORM_HI             Upper bound for real/imag draws (default: 0.5)
#   JL_BP_MAX_ITER            BP iteration cap (default: 2000)
#   JL_BP_TOL                 BP tolerance (default: 1e-10)
#   JL_TNMP_REGION_L          TNMP rank-2 neighborhood side length (default: 3)
#   JL_TNMP_MAX_ITER          TNMP rank-2 iteration cap (default: 500)
#   JL_TNMP_TOL               TNMP rank-2 tolerance (default: 1e-8)
#   JL_TNMP_COMPLEXITY_OUTPUT Optional TNMP complexity dump path (default: empty)
#   JL_BMPS_CHI_MIN           boundary-MPS chi sweep start (default: 1)
#   JL_BMPS_CHI_MAX           boundary-MPS chi sweep cap (default: 256)
#   JL_BMPS_EPSILON           boundary-MPS marginal tolerance (default: 1e-4)
#   JL_BMPS_PARTITION_BY      boundary-MPS partitioning (default: row)
#   JL_PROGRESS_INTERVAL      Progress print interval; 0 disables per-step logs (default: 10)
#   JL_RUN_BP                 Set to 0 to skip BP (default: 1)
#   JL_RUN_TNMP_RANK2         Set to 0 to skip TNMP rank-2 (default: 1)
#   JL_RUN_BMPS               Set to 0 to skip BMPS (default: 1)
#   JULIA_NUM_THREADS         Julia thread count (default: nproc)
#   JULIA                     Julia executable (default: julia)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/../results"
JULIA_BIN="${JULIA:-julia}"
TNMP_PROJECT="${ROOT}"

env_or() {
  local key="$1" default="$2"
  if [[ -n "${!key:-}" ]]; then
    printf '%s' "${!key}"
  else
    printf '%s' "$default"
  fi
}

env_bool() {
  local key="$1" default="$2"
  local raw="${!key:-}"
  if [[ -z "${raw}" ]]; then
    printf '%s' "$default"
    return
  fi
  case "${raw,,}" in
    1|true|yes|on) printf '%s' "1" ;;
    0|false|no|off) printf '%s' "0" ;;
    *)
      echo "Invalid boolean for ${key}: ${raw}" >&2
      exit 1
      ;;
  esac
}

L="$(env_or JL_L "8")"
CHI="$(env_or JL_CHI "16")"
SEED="$(env_or JL_SEED "7")"
UNIFORM_LO="$(env_or JL_UNIFORM_LO "-0.5")"
UNIFORM_HI="$(env_or JL_UNIFORM_HI "0.5")"
BP_MAX_ITER="$(env_or JL_BP_MAX_ITER "2000")"
BP_TOL="$(env_or JL_BP_TOL "1e-10")"
TNMP_REGION_L="$(env_or JL_TNMP_REGION_L "3")"
TNMP_MAX_ITER="$(env_or JL_TNMP_MAX_ITER "500")"
TNMP_TOL="$(env_or JL_TNMP_TOL "1e-8")"
TNMP_COMPLEXITY_OUTPUT="$(env_or JL_TNMP_COMPLEXITY_OUTPUT "")"
BMPS_CHI_MIN="$(env_or JL_BMPS_CHI_MIN "1")"
BMPS_CHI_MAX="$(env_or JL_BMPS_CHI_MAX "256")"
BMPS_EPSILON="$(env_or JL_BMPS_EPSILON "1e-4")"
BMPS_PARTITION_BY="$(env_or JL_BMPS_PARTITION_BY "row")"
PROGRESS_INTERVAL="$(env_or JL_PROGRESS_INTERVAL "10")"
RUN_BP="$(env_bool JL_RUN_BP "1")"
RUN_TNMP_RANK2="$(env_bool JL_RUN_TNMP_RANK2 "1")"
RUN_BMPS="$(env_bool JL_RUN_BMPS "1")"

if [[ -z "${JULIA_NUM_THREADS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    export JULIA_NUM_THREADS="$(nproc)"
  else
    export JULIA_NUM_THREADS="1"
  fi
fi

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  "--L" "${L}"
  "--chi" "${CHI}"
  "--seed" "${SEED}"
  "--uniform-lo" "${UNIFORM_LO}"
  "--uniform-hi" "${UNIFORM_HI}"
  "--progress-interval" "${PROGRESS_INTERVAL}"
  "--tnmp-nthreads" "${JULIA_NUM_THREADS}"
)

if [[ -n "${TNMP_COMPLEXITY_OUTPUT}" ]]; then
  COMMON_ARGS+=("--tnmp-complexity-output" "${TNMP_COMPLEXITY_OUTPUT}")
fi

BP_ARGS=(
  "--bp-max-iter" "${BP_MAX_ITER}"
  "--bp-tol" "${BP_TOL}"
)

TNMP_ARGS=(
  "--tnmp-region-L" "${TNMP_REGION_L}"
  "--tnmp-max-iter" "${TNMP_MAX_ITER}"
  "--tnmp-tol" "${TNMP_TOL}"
)

BMPS_ARGS=(
  "--bmps-chi-min" "${BMPS_CHI_MIN}"
  "--bmps-chi-max" "${BMPS_CHI_MAX}"
  "--bmps-epsilon" "${BMPS_EPSILON}"
  "--bmps-partition-by" "${BMPS_PARTITION_BY}"
)

progress_log() {
  echo "[progress $(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

run_method() {
  local label="$1"
  local project="$2"
  local script="$3"
  local out="$4"
  shift 4

  local start_ts end_ts elapsed
  progress_log "START ${label} -> ${out}"
  echo "==> ${label}" >&2
  start_ts=$(date +%s)

  "${JULIA_BIN}" --project="${project}" \
    "${script}" \
    --output "${out}" \
    "${COMMON_ARGS[@]}" \
    "$@"

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  progress_log "DONE  ${label} elapsed=${elapsed}s"
}

EXTRA_ARGS=("$@")
PREFIX="L${L}_uniform_complex_lo${UNIFORM_LO}_hi${UNIFORM_HI}"
BP_OUT="${OUT_DIR}/${PREFIX}_chi${CHI}_seed${SEED}_bp.jls"
TNMP_OUT="${OUT_DIR}/${PREFIX}_r${TNMP_REGION_L}_chi${CHI}_seed${SEED}_tnmp_rank2.jls"
BMPS_OUT="${OUT_DIR}/${PREFIX}_chi${CHI}_seed${SEED}_boundarymps.jls"

BATCH_START=$(date +%s)
progress_log "batch start uniform_complex L=${L} chi=${CHI} seed=${SEED} lo=${UNIFORM_LO} hi=${UNIFORM_HI} JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
progress_log "TNMP_test project: ${TNMP_PROJECT}"

if [[ "${RUN_BP}" == "1" ]]; then
  run_method "bp" "${TNMP_PROJECT}" "${SCRIPT_DIR}/../src/run_bp_marginal.jl" "${BP_OUT}" \
    "${BP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
else
  progress_log "SKIP bp (JL_RUN_BP=0)"
fi

if [[ "${RUN_TNMP_RANK2}" == "1" ]]; then
  run_method "tnmp_rank2" "${TNMP_PROJECT}" "${SCRIPT_DIR}/../src/run_tnmp_rank2_marginal.jl" "${TNMP_OUT}" \
    "${TNMP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
else
  progress_log "SKIP tnmp_rank2 (JL_RUN_TNMP_RANK2=0)"
fi

if [[ "${RUN_BMPS}" == "1" ]]; then
  run_method "bmps" "${TNMP_PROJECT}" "${SCRIPT_DIR}/../src/run_boundarymps_marginal.jl" "${BMPS_OUT}" \
    "${BMPS_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
else
  progress_log "SKIP bmps (JL_RUN_BMPS=0)"
fi

BATCH_END=$(date +%s)
BATCH_ELAPSED=$((BATCH_END - BATCH_START))
progress_log "batch finished elapsed=${BATCH_ELAPSED}s"
echo "All uniform-complex BP / TNMP rank-2 / BMPS runs finished. Results in ${OUT_DIR}" >&2
