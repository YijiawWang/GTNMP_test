#!/usr/bin/env bash
# Fully-frustrated q-state Potts: increase coupling K (lower temperature) and run
# CATN (reference) + 3x3 TNMP + single-site BP, all sharing the same signed bond
# matrices B_e, until BP fails to converge within bp_max_iter. TNMP/CATN keep
# running so we can see who stays right once BP breaks.
#
# Usage:
#   ./run_frustrated_sweep.sh [L] [q] [K_start] [K_step] [K_max] [bp_max_iter] [field]
# field defaults to the built-in symmetry-breaking field (REQUIRED to expose BP
# non-convergence: with an exactly symmetric field BP trivially sits at the
# uniform fixed point and never breaks). Pass an explicit field to override.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TNMPTEST="$(cd "$DIR/../../.." && pwd)"
SRC="$(cd "$DIR/../src" && pwd)"
RES="$DIR/../results"
mkdir -p "$RES"

L="${1:-10}"
Q="${2:-3}"
K_START="${3:-1.0}"
K_STEP="${4:-0.2}"
K_MAX="${5:-4.0}"
BP_MAX="${6:-1000}"
FIELD="${7:-}"   # empty -> use the built-in symmetry-breaking default field

CATN_DMAX="${CATN_DMAX:-128}"

# Build the optional --field argument (omitted when FIELD is empty).
FIELD_ARG=()
[[ -n "$FIELD" ]] && FIELD_ARG=(--field "$FIELD")

echo "Frustrated sweep: L=$L q=$Q K=$K_START:$K_STEP:$K_MAX bp_max_iter=$BP_MAX field=[${FIELD:-default}]"
echo "started: $(date)"

K="$K_START"
while awk -v k="$K" -v mx="$K_MAX" 'BEGIN{exit !(k <= mx + 1e-12)}'; do
    echo
    echo "##################################################################"
    echo "############ frustrated K = $K ############"
    echo "##################################################################"

    cc="$RES/frus_L${L}_q${Q}_K${K}_catn.txt"
    ct="$RES/frus_L${L}_q${Q}_K${K}_tnmp.txt"
    cb="$RES/frus_L${L}_q${Q}_K${K}_bp.txt"

    echo "---- CATN (reference) ----"
    julia --project="$TNMPTEST/env_catn" "$SRC/run_catn.jl" \
        --L "$L" --q "$Q" --coupling "$K" --couplings frustrated "${FIELD_ARG[@]}" \
        --catn-dmax "$CATN_DMAX" --catn-chi "$CATN_DMAX" --out "$cc" || echo "(CATN failed at K=$K)"

    echo "---- 3x3 TNMP ----"
    julia --project="$TNMPTEST/env_gmp" "$SRC/run_tnmp.jl" \
        --L "$L" --q "$Q" --coupling "$K" --couplings frustrated "${FIELD_ARG[@]}" \
        --verbose 0 --out "$ct"

    echo "---- single-site BP ----"
    julia --project="$TNMPTEST" "$SRC/run_bp.jl" \
        --L "$L" --q "$Q" --coupling "$K" --couplings frustrated "${FIELD_ARG[@]}" \
        --bp-max-iter "$BP_MAX" --out "$cb"

    echo "---- comparison (K=$K) ----"
    julia "$SRC/compare.jl" "$cc" "$ct" "$cb" || true

    bp_conv=$(grep '^converged=' "$cb" | cut -d= -f2)
    tnmp_conv=$(grep '^converged=' "$ct" | cut -d= -f2)
    echo "summary: K=$K  TNMP_conv=$tnmp_conv  BP_conv=$bp_conv"

    if [[ "$bp_conv" == "false" ]]; then
        echo
        echo ">>> BP failed to converge at K=$K. Stopping sweep."
        break
    fi

    K=$(awk -v k="$K" -v s="$K_STEP" 'BEGIN{printf "%.6f", k+s}')
done

echo
echo "finished: $(date)"
