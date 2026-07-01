# TNMP Test

Minimal TNMP extraction with runnable double-layer marginal examples.

## Setup

Run the bootstrap script (installs Julia 1.11.5, instantiates packages, links the local
TensorNetworkQuantumSimulator checkout from
[`JoeyT1994/TensorNetworkQuantumSimulator.jl`](https://github.com/JoeyT1994/TensorNetworkQuantumSimulator.jl)):

```bash
./setup.sh
```

Or manually:

```bash
julia --project=TNMP_test -e 'using Pkg; Pkg.develop(path="TNMP_test/TensorNetworkQuantumSimulator.jl"); Pkg.instantiate()'
```

TensorNetworkQuantumSimulator is provided by the local checkout at
`TNMP_test/TensorNetworkQuantumSimulator.jl`. That checkout is the latest local copy of
[`JoeyT1994/TensorNetworkQuantumSimulator.jl`](https://github.com/JoeyT1994/TensorNetworkQuantumSimulator.jl)
and is declared in `Project.toml` through `[sources]`.

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

boundary_mp tests:

```bash
julia --project=TNMP_test TNMP_test/boundary_mp/test/runtests.jl
```
