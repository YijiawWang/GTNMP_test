# Boundary-MPS Center Marginal Prototype

Computes the center single-site marginal of a uniform-random real tensor-network state on an
`L×L` grid, via three routes, and checks the two boundary-MPS routes converge to exact.

Targets **`TensorNetworkQuantumSimulator.jl` on the `treesa` branch (v0.3.10)**
([xuanzhaogao/TensorNetworkQuantumSimulator.jl @ treesa](https://github.com/xuanzhaogao/TensorNetworkQuantumSimulator.jl/tree/treesa)),
which provides `BoundaryMPSCache` and exposes the TreeSA contraction order through
`OMEinsumContractionOrders`. Dependencies: TNQS, ITensors/ITensorMPS, Julia stdlib.

The package is vendored next to this prototype at `../TensorNetworkQuantumSimulator.jl`, so
all commands below use `--project` pointing at that checkout. Make sure it is on the `treesa`
branch (`git -C TNMP_test/TensorNetworkQuantumSimulator.jl checkout treesa`).

## API note (vs. the old version)

The TreeSA contraction order is no longer requested with `alg = "tree_sa"`. On the `treesa`
branch it is provided by `OMEinsumContractionOrders` and selected through the `omeinsum`
backend:

```julia
using TensorNetworkQuantumSimulator: contraction_sequence, TreeSA
seq = contraction_sequence(tensors; alg = "omeinsum", optimizer = TreeSA())
```

`TreeSA` (and the other optimizers `GreedyMethod`, `SABipartite`, `Treewidth`,
`ExactTreewidth`, `HyperND`) are exported by `TensorNetworkQuantumSimulator`.

## Modules (`src/`)

- `TNQSBoundaryMP.jl` — uniform-state builder, projected double-layer norm network,
  marginal normalization, and `bmps_marginal` (thin wrapper over TNQS BMPS with the SVD /
  `ITensorMPS` message update).
- `ExactEnvFullUpdateBMPS.jl` — `full_update_marginal`: TNQS BMPS forward sweep with a
  custom `Algorithm"full_update_exact_env"` whose message compression is an
  environment-weighted ALS, where the environment is the far side of the cut contracted
  exactly via TreeSA into a rank-`2L` tensor.
- `ExactSolver.jl` — `exact_marginal`: exact contraction of each projected network via
  TreeSA.

## Run tests

```bash
julia --project=TNMP_test/TensorNetworkQuantumSimulator.jl -e 'using Pkg; Pkg.instantiate()'
julia --project=TNMP_test/TensorNetworkQuantumSimulator.jl TNMP_test/boundary_mp/test/runtests.jl
```

## Sweep script

```bash
julia --project=TNMP_test/TensorNetworkQuantumSimulator.jl \
    TNMP_test/boundary_mp/scripts/uniform_double_layer_sweep.jl --L=4 --bmps-chis=1,2,4,8
```
