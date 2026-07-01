#!/usr/bin/env bash
# Coupling sweep of the q-state Potts model on an L x L lattice, comparing the
# center single-site marginal from CATN (high-bond-dim reference), single-layer
# 3x3 TNMP, and single-layer BP. Each method runs in its own Julia environment.
#
# Usage:
#   ./run_sweep.sh [L] [q] [Dmax] ["c1 c2 c3 ..."]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TNMPTEST="$(cd "$DIR/../../.." && pwd)"
SRC="$(cd "$DIR/../src" && pwd)"
RES="$DIR/../results"
mkdir -p "$RES"

L="${1:-10}"
Q="${2:-3}"
DMAX="${3:-128}"
COUPLINGS="${4:-0.2 0.4 0.6 0.8 1.0}"

echo "Potts coupling sweep: L=$L q=$Q CATN_Dmax=$DMAX couplings=[$COUPLINGS]"
echo "started: $(date)"

for K in $COUPLINGS; do
    echo
    echo "##################################################################"
    echo "############ coupling K = $K ############"
    echo "##################################################################"

    cc="$RES/sweep_L${L}_q${Q}_K${K}_catn.txt"
    ct="$RES/sweep_L${L}_q${Q}_K${K}_tnmp.txt"
    cb="$RES/sweep_L${L}_q${Q}_K${K}_bp.txt"

    echo "---- [1/3] CATN ----"
    julia --project="$TNMPTEST/env_catn" "$SRC/run_catn.jl" \
        --L "$L" --q "$Q" --coupling "$K" --catn-dmax "$DMAX" --catn-chi "$DMAX" --out "$cc"

    echo "---- [2/3] single-layer TNMP (3x3) ----"
    julia --project="$TNMPTEST/env_gmp" "$SRC/run_tnmp.jl" \
        --L "$L" --q "$Q" --coupling "$K" --out "$ct"

    echo "---- [3/3] single-layer BP (TNQS) ----"
    julia --project="$TNMPTEST" "$SRC/run_bp.jl" \
        --L "$L" --q "$Q" --coupling "$K" --out "$cb"

    echo "---- comparison (K=$K) ----"
    julia "$SRC/compare.jl" "$cc" "$ct" "$cb"
done

echo
echo "finished: $(date)"
