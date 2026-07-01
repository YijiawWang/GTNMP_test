#!/usr/bin/env bash
# Bootstrap Julia + package environments for TNMP_test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JULIAUP_BIN="${HOME}/.juliaup/bin"
export PATH="${JULIAUP_BIN}:${PATH}"

if ! command -v juliaup >/dev/null 2>&1; then
  echo "Installing Julia via juliaup..."
  curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.11.5
fi

export PATH="${JULIAUP_BIN}:${PATH}"
juliaup add 1.11.5 >/dev/null 2>&1 || true
juliaup default 1.11.5

TNS_PATH="${ROOT}/TensorNetworkQuantumSimulator.jl"
if [[ ! -d "${TNS_PATH}" ]]; then
  echo "Missing vendored TensorNetworkQuantumSimulator at ${TNS_PATH}"
  echo "Place the latest JoeyT1994/TensorNetworkQuantumSimulator.jl checkout there before running examples."
  exit 1
fi

echo "Setting up TNMP_test project..."
julia +1.11.5 --project="${ROOT}" -e '
using Pkg
Pkg.develop(path=joinpath(pwd(), "TensorNetworkQuantumSimulator.jl"))
Pkg.add("Dictionaries")
Pkg.add(name="OMEinsumContractionOrders", version="1.2.2")
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
'

echo "Setting up TensorNetworkQuantumSimulator project..."
julia +1.11.5 --project="${TNS_PATH}" -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
'

cat <<EOF

TNMP_test environment is ready.

Add Julia to your shell (once):
  export PATH="${JULIAUP_BIN}:\$PATH"

Examples:
  julia +1.11.5 --project=${ROOT} ${ROOT}/examples/random_double_layer_marginal.jl
  julia +1.11.5 --project=${ROOT} ${ROOT}/examples/boundarymps_random_double_layer.jl

boundary_mp tests:
  julia +1.11.5 --project=${TNS_PATH} ${ROOT}/boundary_mp/test/runtests.jl

Main test suite:
  julia +1.11.5 --project=${ROOT} ${ROOT}/test/runtests.jl
EOF
