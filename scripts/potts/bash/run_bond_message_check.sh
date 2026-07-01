#!/usr/bin/env bash
# Run TNMP + BP on frustrated Potts and compare messages on bond (3,3)-(4,3).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TNMPTEST="$(cd "$DIR/../../.." && pwd)"
SRC="$(cd "$DIR/../src" && pwd)"
RES="$DIR/../results"
mkdir -p "$RES"

L="${1:-10}"
Q="${2:-3}"
K="${3:-1.0}"
BX="${4:-3}"; BY="${5:-3}"
AX="${6:-4}"; AY="${7:-3}"

COMMON=(--L "$L" --q "$Q" --coupling "$K" --couplings frustrated
        --bx "$BX" --by "$BY" --ax "$AX" --ay "$AY")

TNMP_OUT="$RES/bond_L${L}_q${Q}_K${K}_tnmp.txt"
BP_OUT="$RES/bond_L${L}_q${Q}_K${K}_bp.txt"

echo "Bond message check: L=$L q=$Q K=$K bond=($BX,$BY)-($AX,$AY)"
echo "started: $(date)"

julia --project="$TNMPTEST/env_gmp" "$SRC/dump_bond_tnmp.jl" "${COMMON[@]}" --out "$TNMP_OUT"
julia --project="$TNMPTEST" "$SRC/dump_bond_bp.jl" "${COMMON[@]}" --out "$BP_OUT"
julia "$SRC/compare_bond_messages.jl" "$TNMP_OUT" "$BP_OUT"

echo "finished: $(date)"
