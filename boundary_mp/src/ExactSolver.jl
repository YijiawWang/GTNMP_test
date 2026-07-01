module ExactSolver

# Exact contraction of a projected norm network into a scalar, using the TreeSA
# contraction-order optimiser shipped in TensorNetworkQuantumSimulator.jl. TreeSA comes from
# OMEinsumContractionOrders and is selected via the `alg = "omeinsum"` backend with
# `optimizer = TreeSA()`.
# Depends only on TensorNetworkQuantumSimulator + ITensors + the shared builder in TNQSBoundaryMP.

using ITensors: ITensor
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: contraction_sequence, TreeSA
using NamedGraphs: vertices
using ..TNQSBoundaryMP: UniformState, marginal

export exact_scalar, exact_marginal

function exact_scalar(tn)
    ts = ITensor[tn[v] for v in vertices(tn)]
    seq = contraction_sequence(ts; alg = "omeinsum", optimizer = TreeSA())
    return contract(ts; sequence = seq)[]
end

exact_marginal(state::UniformState, center) = marginal(state, center, exact_scalar)

end # module
