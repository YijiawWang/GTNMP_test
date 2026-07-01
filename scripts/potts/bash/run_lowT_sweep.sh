#!/usr/bin/env bash
# Increase coupling K (lower temperature) and run TNMP + BP until BP fails to
# converge within max_iter. TNMP always runs for comparison.
#
# Usage:
#   ./run_lowT_sweep.sh [L] [q] [K_start] [K_step] [K_max] [bp_max_iter]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TNMPTEST="$(cd "$DIR/../../.." && pwd)"
SRC="$(cd "$DIR/../src" && pwd)"
RES="$DIR/../results"
mkdir -p "$RES"

L="${1:-10}"
Q="${2:-3}"
K_START="${3:-1.2}"
K_STEP="${4:-0.2}"
K_MAX="${5:-5.0}"
BP_MAX="${6:-500}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$RES/lowT_sweep_${TS}.log"
SUMMARY="$RES/lowT_sweep_${TS}_summary.csv"

echo "Low-T sweep: L=$L q=$Q K=$K_START:$K_STEP:$K_MAX bp_max_iter=$BP_MAX"
echo "started: $(date)"
echo "log: $LOG"

{
echo "K,tnmp_converged,tnmp_iters,tnmp_finalerr,bp_converged,bp_iters,bp_final_diff"

K="$K_START"
while awk -v k="$K" -v mx="$K_MAX" 'BEGIN{exit !(k <= mx + 1e-12)}'; do
    echo
    echo "##################################################################"
    echo "############ K = $K  (lower T => larger K) ############"
    echo "##################################################################"

    ct="$RES/lowT_L${L}_q${Q}_K${K}_tnmp.txt"
    cb="$RES/lowT_L${L}_q${Q}_K${K}_bp.txt"

    echo "---- TNMP ----"
    julia --project="$TNMPTEST/env_gmp" "$SRC/run_tnmp.jl" \
        --L "$L" --q "$Q" --coupling "$K" --verbose 0 --out "$ct"

    echo "---- BP ----"
    julia --project="$TNMPTEST" "$SRC/run_bp.jl" \
        --L "$L" --q "$Q" --coupling "$K" --bp-max-iter "$BP_MAX" --out "$cb"

    tnmp_conv=$(grep '^converged=' "$ct" | cut -d= -f2)
    tnmp_iters=$(grep '^iters=' "$ct" | cut -d= -f2)
    tnmp_err=$(grep '^finalerr=' "$ct" | cut -d= -f2)
    bp_conv=$(grep '^converged=' "$cb" | cut -d= -f2)
    bp_iters=$(grep '^iters=' "$cb" | cut -d= -f2)
    bp_diff=$(grep '^final_diff=' "$cb" | cut -d= -f2)

    echo "summary: K=$K  TNMP conv=$tnmp_conv iters=$tnmp_iters err=$tnmp_err  " \
         "BP conv=$bp_conv iters=$bp_iters diff=$bp_diff"
    echo "$K,$tnmp_conv,$tnmp_iters,$tnmp_err,$bp_conv,$bp_iters,$bp_diff"

    if [[ "$bp_conv" == "false" ]]; then
        echo
        echo ">>> BP failed to converge at K=$K (final_diff=$bp_diff after $bp_iters iters). Stopping."
        break
    fi

    K=$(awk -v k="$K" -v s="$K_STEP" 'BEGIN{printf "%.6f", k+s}')
done

echo
echo "finished: $(date)"
} | tee "$LOG" | tee >(grep '^[0-9]' > "$SUMMARY")

echo "summary csv -> $SUMMARY"
