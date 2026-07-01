#!/usr/bin/env bash
# Sequentially run the 6 TreeSA cavity+neighborhood contraction-complexity
# sweeps requested for the L=10 lattice. Each job runs in its OWN detached
# `screen` session named cx_jobN, with 96 Julia threads. The controller waits
# for job N to finish before launching job N+1 (only one heavy job at a time).
#
# Jobs:
#   1) double-layer TN, 3x3 neighborhood, chi = 1,2,4,8,16,32,64,128,256,512
#   2) double-layer TN, 5x5 neighborhood, chi = 1,2,4,8,16,32,64,128,256,512
#   3) single-layer Potts, 3x3 neighborhood, q = 2,4,8,16,32,64,128,256,512
#   4) single-layer Potts, 5x5 neighborhood, q = 2,4,8,16,32,64,128,256,512
#   5) single-layer Potts, 7x7 neighborhood, q = 2,4,8,16,32,64,128
#   6) single-layer Potts, 9x9 neighborhood, q = 2,4,8,16,32,64,128
#
# Usage:
#   scripts/run_complexity_sweeps_sequential.sh
# (intended to itself be launched inside a detached screen, e.g. cx_controller)
set -euo pipefail

ROOT="/ssd/users/wangyijia/GTNMP/TNMP_test"
DL_SCRIPT="${ROOT}/contraction_process_tn/tnmp_rank2_complexity_sweep.jl"
POTTS_DIR="${ROOT}/scripts/potts/src"
POTTS_SCRIPT="complexity_sweep.jl"
SC_TARGET="${SC_TARGET:-32}"
OUTDIR="${ROOT}/contraction_process_tn/complexity_batch_20260630_sct${SC_TARGET}"
JULIA_BIN="${JULIA:-julia}"
NTHREADS="${NTHREADS:-96}"

mkdir -p "${OUTDIR}"
CTRL_LOG="${OUTDIR}/controller.log"

clog() { echo "[controller $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${CTRL_LOG}"; }

# Build the inner command string for one job.
# Double-layer path reads the TreeSA sc_target from TNMP_TREESA_SC_TARGET.
dl_cmd() {
  local region_L="$1" chi_list="$2" out="$3"
  printf 'cd %q && TNMP_TREESA_SC_TARGET=%s JULIA_NUM_THREADS=%s %q --project=%q %q --L 10 --region-L %s --chi-list %s --out %q' \
    "${ROOT}" "${SC_TARGET}" "${NTHREADS}" "${JULIA_BIN}" "${ROOT}" "${DL_SCRIPT}" "${region_L}" "${chi_list}" "${out}"
}

# Potts path takes sc_target via the --treesa-sc-target CLI flag.
potts_cmd() {
  local region_L="$1" q_list="$2" out="$3"
  printf 'cd %q && JULIA_NUM_THREADS=%s %q --project=%q %q --L 10 --region-L %s --q-list %s --treesa-sc-target %s --out %q' \
    "${POTTS_DIR}" "${NTHREADS}" "${JULIA_BIN}" "${ROOT}/env_gmp" "${POTTS_SCRIPT}" "${region_L}" "${q_list}" "${SC_TARGET}" "${out}"
}

# Launch one job in its own screen and block until that screen exits.
run_job() {
  local name="$1" desc="$2" inner="$3" log="$4"
  clog "START ${name}: ${desc}"
  clog "  log -> ${log}"
  # Run the inner command, capturing all output and the exit code to the log.
  screen -dmS "${name}" bash -lc "{ ${inner}; } > ${log} 2>&1; echo \"EXIT_CODE=\$?\" >> ${log}"
  # Wait for the screen session to disappear (job finished).
  local waited=0
  while screen -ls 2>/dev/null | grep -Eq "[0-9]+\.${name}[[:space:]]"; do
    sleep 15
    waited=$((waited + 15))
    if (( waited % 300 == 0 )); then
      clog "  ${name} still running (${waited}s elapsed)"
    fi
  done
  local rc
  rc="$(grep -oE 'EXIT_CODE=[0-9]+' "${log}" | tail -n1 | cut -d= -f2 || true)"
  clog "DONE  ${name} exit_code=${rc:-unknown}"
}

CHI_LIST="1,2,4,8,16,32,64,128,256,512"
Q_LIST_FULL="2,4,8,16,32,64,128,256,512"
Q_LIST_BIG="2,4,8,16,32,64,128"

BATCH_START=$(date +%s)
clog "batch start: 6 sequential jobs, ${NTHREADS} threads each, sc_target=${SC_TARGET}, outdir=${OUTDIR}"

run_job "cx_job1" "double-layer 3x3 chi=${CHI_LIST}" \
  "$(dl_cmd 3 "${CHI_LIST}" "${OUTDIR}/dl_L10_r3.md")" "${OUTDIR}/job1_dl_r3.log"

run_job "cx_job2" "double-layer 5x5 chi=${CHI_LIST}" \
  "$(dl_cmd 5 "${CHI_LIST}" "${OUTDIR}/dl_L10_r5.md")" "${OUTDIR}/job2_dl_r5.log"

run_job "cx_job3" "potts 3x3 q=${Q_LIST_FULL}" \
  "$(potts_cmd 3 "${Q_LIST_FULL}" "${OUTDIR}/potts_L10_r3.md")" "${OUTDIR}/job3_potts_r3.log"

run_job "cx_job4" "potts 5x5 q=${Q_LIST_FULL}" \
  "$(potts_cmd 5 "${Q_LIST_FULL}" "${OUTDIR}/potts_L10_r5.md")" "${OUTDIR}/job4_potts_r5.log"

run_job "cx_job5" "potts 7x7 q=${Q_LIST_BIG}" \
  "$(potts_cmd 7 "${Q_LIST_BIG}" "${OUTDIR}/potts_L10_r7.md")" "${OUTDIR}/job5_potts_r7.log"

run_job "cx_job6" "potts 9x9 q=${Q_LIST_BIG}" \
  "$(potts_cmd 9 "${Q_LIST_BIG}" "${OUTDIR}/potts_L10_r9.md")" "${OUTDIR}/job6_potts_r9.log"

BATCH_END=$(date +%s)
clog "batch finished elapsed=$((BATCH_END - BATCH_START))s. Results in ${OUTDIR}"
