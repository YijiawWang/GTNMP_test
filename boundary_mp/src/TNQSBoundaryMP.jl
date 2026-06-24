module TNQSBoundaryMP

# Targets TensorNetworkQuantumSimulator.jl on the `treesa` branch (v0.3.10), which ships the
# TreeSA contraction order through OMEinsumContractionOrders (selected via the `omeinsum`
# backend in src/contraction_sequences.jl).
# Allowed dependencies: TensorNetworkQuantumSimulator, ITensors/ITensorMPS, stdlib.

using LinearAlgebra
using Random
using Dictionaries: Dictionary, set!
using ITensors: Index, ITensor, dag, prime, onehot, replaceinds, dim
using ITensorMPS
import NamedGraphs
using NamedGraphs: edges, neighbors, vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: Algorithm, partitionfunction

export UniformState, uniform_state, center_vertex,
    projected_norm_network, normalize_marginal, marginal, bmps_marginal

struct UniformState
    L::Int
    physical_dim::Int
    bond_dim::Int
    graph::Any
    phys::Dict{Tuple{Int,Int},Index}
    ketbond::Dict{Any,Index}
    kets::Dict{Tuple{Int,Int},ITensor}
end

function uniform_state(
        rng::AbstractRNG, L::Integer;
        bond_dim::Integer = 2, physical_dim::Integer = 2, lo::Real = -0.3, hi::Real = 0.5,
    )
    L > 0 || throw(ArgumentError("L must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    lo < hi || throw(ArgumentError("lo must be smaller than hi"))
    g = named_grid((Int(L), Int(L)))
    phys = Dict(v => Index(Int(physical_dim), "phys,v=$v") for v in vertices(g))
    ketbond = Dict{Any,Index}()
    for e in edges(g)
        idx = Index(Int(bond_dim), "ket,e=$e")
        ketbond[e] = idx
        ketbond[reverse(e)] = idx
    end
    kets = Dict{Tuple{Int,Int},ITensor}()
    width = Float64(hi) - Float64(lo)
    for v in vertices(g)
        local_inds = Index[phys[v]]
        for nb in neighbors(g, v)
            push!(local_inds, ketbond[NamedGraphs.NamedEdge(v => nb)])
        end
        data = Float64(lo) .+ width .* rand(rng, map(dim, local_inds)...)
        kets[v] = ITensor(data, local_inds...)
    end
    return UniformState(Int(L), Int(physical_dim), Int(bond_dim), g, phys, ketbond, kets)
end

center_vertex(state::UniformState) = ((state.L + 1) ÷ 2, (state.L + 1) ÷ 2)

# Double-layer norm network <psi| P_s |psi> as a flat TensorNetwork on the LxL grid.
# Each edge carries TWO separate indices: the ket bond b[e] and the bra bond b[e]'.
# Physical indices are summed (non-center) or pinned to s (center), so the network has
# no open indices and contracts to a scalar; each row-cut exposes 2L indices.
function projected_norm_network(state::UniformState, center, s::Integer)
    1 <= s <= state.physical_dim ||
        throw(ArgumentError("s must be in 1:$(state.physical_dim), got $s"))
    g = state.graph
    dict = Dictionary{Tuple{Int,Int},ITensor}()
    for v in vertices(g)
        ket = state.kets[v]
        bra = dag(prime(ket))
        if v == center
            ket_v = ket * onehot(state.phys[v] => s)
            bra_v = bra * onehot(prime(state.phys[v]) => s)
        else
            ket_v = ket
            bra_v = replaceinds(bra, prime(state.phys[v]) => state.phys[v])
        end
        set!(dict, v, ket_v * bra_v)
    end
    return TensorNetwork(dict, g)
end

function normalize_marginal(weights::AbstractVector)
    w = abs.(real.(weights))
    total = sum(w)
    total > 0 || throw(ArgumentError("cannot normalize zero marginal weights: $weights"))
    return w ./ total
end

# Generic driver: contract_scalar maps a projected TensorNetwork to a number.
function marginal(state::UniformState, center, contract_scalar)
    weights = [real(contract_scalar(projected_norm_network(state, center, s)))
               for s in 1:state.physical_dim]
    return normalize_marginal(weights)
end

# Shallow wrapper over TNQS BMPS with the ITensorMPS (SVD) message update.
function bmps_marginal(
        state::UniformState, center, chi::Integer;
        cutoff::Real = 1.0e-12, maxiter::Integer = 1,
    )
    f = function (tn)
        cache = BoundaryMPSCache(tn, Int(chi); partition_by = "row")
        cache = update(
            cache; alg = "bp", maxiter = Int(maxiter),
            message_update_alg = Algorithm("ITensorMPS"; cutoff = Float64(cutoff), normalize = true),
            tolerance = nothing,
        )
        return partitionfunction(cache)
    end
    return marginal(state, center, f)
end

end # module
