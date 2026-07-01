# Boundary-MPS Center Marginal Prototype

Computes the center single-site marginal of a uniform-random real tensor-network state on an
`L×L` grid, via three routes, and checks the two boundary-MPS routes converge to exact.

Targets the local latest checkout of
[`JoeyT1994/TensorNetworkQuantumSimulator.jl`](https://github.com/JoeyT1994/TensorNetworkQuantumSimulator.jl),
which provides `BoundaryMPSCache` and exposes the TreeSA contraction order through
`OMEinsumContractionOrders`. Dependencies: TensorNetworkQuantumSimulator,
ITensors/ITensorMPS, Julia stdlib.

The package is vendored in this repository at `TNMP_test/TensorNetworkQuantumSimulator.jl` and
declared as a path dependency of the main `TNMP_test` project, so commands below use the
`TNMP_test` project.

## API note (vs. the old version)

The TreeSA contraction order is no longer requested with `alg = "tree_sa"`. It is provided
by `OMEinsumContractionOrders` and selected through the `omeinsum`
backend:

```julia
using TensorNetworkQuantumSimulator: contraction_sequence, TreeSA
seq = contraction_sequence(tensors; alg = "omeinsum", optimizer = TreeSA())
```

`TreeSA` (and the other optimizers `GreedyMethod`, `SABipartite`, `Treewidth`,
`ExactTreewidth`, `HyperND`) are exported by `TensorNetworkQuantumSimulator`.

## Modules (`src/`)

- `TNQSBoundaryMP.jl` — uniform-state builder, projected double-layer norm network,
  marginal normalization, and `bmps_marginal` (thin wrapper over TensorNetworkQuantumSimulator
  BMPS with the zipup message update).
- `ExactEnvFullUpdateBMPS.jl` — `full_update_marginal`: TensorNetworkQuantumSimulator BMPS
  forward sweep with a
  custom `Algorithm"full_update_exact_env"` whose message compression is an
  environment-weighted ALS, where the environment is the far side of the cut contracted
  exactly via TreeSA into a rank-`2L` tensor.
- `ExactSolver.jl` — `exact_marginal`: exact contraction of each projected network via
  TreeSA.

## Run tests

```bash
julia --project=TNMP_test -e 'using Pkg; Pkg.instantiate()'
julia --project=TNMP_test TNMP_test/boundary_mp/test/runtests.jl
```

## Sweep script

```bash
julia --project=TNMP_test \
    TNMP_test/boundary_mp/scripts/uniform_double_layer_sweep.jl --L=4 --bmps-chis=1,2,4,8
```
