# Core, model-agnostic tensor-network state interface for TNMP.
#
# This file holds only the pieces the message-passing algorithm needs from a
# state: the `TensorNetworkState` container, its accessors, and the double-layer
# norm / marginal factor builders. Concrete state *constructors* (random PEPS,
# TFIM imaginary-time, ...) live with the models in `examples/state_models.jl`
# so that new models can be added without touching the solver.

struct TensorNetworkState{V}
    tensors::Dict{V, ITensor}
    siteinds::Dict{V, Vector{Index}}
    graph::NamedGraph{V}
end

graph(psi::TensorNetworkState) = psi.graph
siteinds(psi::TensorNetworkState) = psi.siteinds
siteinds(psi::TensorNetworkState, v) = psi.siteinds[v]
Base.getindex(psi::TensorNetworkState, v) = psi.tensors[v]

reverse_edge(e::NamedEdge) = NamedEdge(dst(e) => src(e))

virtualinds(psi::TensorNetworkState, e::NamedEdge) = commoninds(psi[src(e)], psi[dst(e)])

function traced_norm_factors(psi::TensorNetworkState, v)
    sinds = siteinds(psi, v)
    ket = psi[v]
    bra = replaceinds(dag(prime(ket)), prime.(sinds), sinds)
    return ITensor[ket, bra]
end

function fixed_site_norm_factors(psi::TensorNetworkState, v, state::Integer)
    sind = only(siteinds(psi, v))
    ket = psi[v] * onehot(sind => state)
    bra = dag(prime(psi[v])) * onehot(prime(sind) => state)
    return ITensor[ket, bra]
end

function norm_factors(psi::TensorNetworkState, verts)
    factors = ITensor[]
    for v in verts
        append!(factors, traced_norm_factors(psi, v))
    end
    return factors
end

function marginal_factors(psi::TensorNetworkState, verts, target, state::Integer)
    factors = ITensor[]
    for v in verts
        append!(factors, v == target ? fixed_site_norm_factors(psi, v, state) : traced_norm_factors(psi, v))
    end
    return factors
end

function normalize_weights(weights::Vector{Float64})
    total = sum(weights)
    iszero(total) && throw(ArgumentError("Cannot normalize a zero marginal"))
    return weights ./ total
end
