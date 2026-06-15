# TNMP Test

Minimal TNMP extraction with runnable double-layer marginal examples.

## Setup

1. Clone this repository.
2. Install [TensorNetworkQuantumSimulator](https://github.com/ITensor/TensorNetworkQuantumSimulator) into the project environment:

```bash
julia --project=TNMP_test -e 'using Pkg; Pkg.develop(path="path/to/TensorNetworkQuantumSimulator")'
```

If you use the monorepo layout with `TensorNetworkQuantumSimulator_q.jl` as a sibling directory, the examples will find it automatically.

You can also point to a custom checkout:

```bash
export TNQS_PROJECT=/path/to/TensorNetworkQuantumSimulator
```

## Examples

TNMP message passing on a small random PEPS:

```bash
julia --project=TNMP_test TNMP_test/examples/random_double_layer_marginal.jl
```

Boundary-MPS centre-site marginal (full workflow: build double-layer network, project centre spins, sweep `bmps_chi`):

```bash
julia --project=TNMP_test TNMP_test/examples/boundarymps_random_double_layer.jl
```

With custom parameters:

```bash
julia --project=TNMP_test TNMP_test/examples/boundarymps_random_double_layer.jl \
  --L 8 --chi 4 --bmps-chi-max 64 --bmps-epsilon 1e-4
```

## Tests

```bash
julia --project=TNMP_test TNMP_test/test/runtests.jl
```
