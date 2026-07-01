#!/usr/bin/env bash
# Run the q-state Potts model with all three methods (each in its own Julia
# environment, since CATN pins OMEinsum 0.9.x while GenericMessagePassing pins
# 0.8.5 -> they cannot share one environment) and compare the center marginal.
#
# Usage:
#   ./run_all.sh [--L 10 --q 3 --coupling 0.3 ...]
# All extra args are forwarded verbatim to the three method scripts, so the model
# is guaranteed identical across methods.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TNMPTEST="$(cd "$DIR/../../.." && pwd)"
SRC="$(cd "$DIR/../src" && pwd)"
RES="$DIR/../results"
mkdir -p "$RES"

ARGS=("$@")

echo "########## [1/3] CATN ##########"
julia --project="$TNMPTEST/env_catn" "$SRC/run_catn.jl" "${ARGS[@]}" --out "$RES/potts_catn.txt"

echo "########## [2/3] single-layer TNMP (3x3) ##########"
julia --project="$TNMPTEST/env_gmp" "$SRC/run_tnmp.jl" "${ARGS[@]}" --out "$RES/potts_tnmp.txt"

echo "########## [3/3] single-layer BP (TNQS) ##########"
julia --project="$TNMPTEST" "$SRC/run_bp.jl" "${ARGS[@]}" --out "$RES/potts_bp.txt"

echo "########## comparison ##########"
julia "$SRC/compare.jl" "$RES/potts_catn.txt" "$RES/potts_tnmp.txt" "$RES/potts_bp.txt"
