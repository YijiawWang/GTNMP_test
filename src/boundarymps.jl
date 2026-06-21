module TNMPBoundaryMPS

# Boundary-MPS double-layer marginals / partition function for the random PEPS
# states built by `tnmp.jl`. The actual contraction is delegated to
# TensorNetworkQuantumSimulator (TNQS): the double-layer norm network is wrapped
# in a `BoundaryMPSCache`, converged with `update`, and read out via
# `partitionfunction` / `free_energy`.
#
# TNQS is resolved in this order:
#   1. already installed in the active Julia environment
#   2. ENV["TNQS_PROJECT"] if set
#   3. sibling checkout at ../TensorNetworkQuantumSimulator_q.jl (monorepo layout)

const _SRC_DIR = @__DIR__
const _ROOT = normpath(joinpath(_SRC_DIR, ".."))

function resolve_tnqs_project()
    env = get(ENV, "TNQS_PROJECT", "")
    !isempty(env) && return normpath(env)
    sibling = normpath(joinpath(_ROOT, "..", "TensorNetworkQuantumSimulator_q.jl"))
    isdir(sibling) && return sibling
    return nothing
end

const _TNQS_PROJECT = resolve_tnqs_project()
if _TNQS_PROJECT !== nothing && !in(_TNQS_PROJECT, LOAD_PATH)
    pushfirst!(LOAD_PATH, _TNQS_PROJECT)
end

include(joinpath(_SRC_DIR, "tnmp.jl"))

using Dictionaries: Dictionary
using ITensors: ITensors, dim
using NamedGraphs: vertices
using TensorNetworkQuantumSimulator

# Re-export the TNMP state machinery so callers can `using .TNMPBoundaryMPS`
# and build instances without separately including `tnmp.jl` (this keeps a
# single `TNMPTest` module / `TensorNetworkState` type in play).
export TNMPTest,
    boundarymps_partition_function,
    boundarymps_marginal_weights,
    sweep_boundarymps_marginal

ITensors.disable_warn_order()

function normalize_log_weights(log_weights::Vector{Float64})
    isempty(log_weights) && throw(ArgumentError("Cannot normalize an empty marginal"))
    offset = maximum(log_weights)
    !isfinite(offset) && throw(ArgumentError("Cannot normalize non-finite log weights: $log_weights"))
    weights = exp.(log_weights .- offset)
    return TNMPTest.normalize_weights(weights)
end

function to_tnqs_double_layer_network(psi::TNMPTest.TensorNetworkState)
    vs = collect(vertices(TNMPTest.graph(psi)))
    tensors = Dictionary(vs, [reduce(*, TNMPTest.traced_norm_factors(psi, v)) for v in vs])
    return TensorNetwork(tensors, TNMPTest.graph(psi))
end

function to_tnqs_projected_network(psi::TNMPTest.TensorNetworkState, target, state::Integer)
    vs = collect(vertices(TNMPTest.graph(psi)))
    tensors = Dictionary(
        vs,
        [
            v == target ?
                reduce(*, TNMPTest.fixed_site_norm_factors(psi, v, state)) :
                reduce(*, TNMPTest.traced_norm_factors(psi, v))
            for v in vs
        ],
    )
    return TensorNetwork(tensors, TNMPTest.graph(psi))
end

function boundarymps_partition_function(
        psi::TNMPTest.TensorNetworkState,
        bmps_chi::Integer;
        partition_by::AbstractString = "row",
        bmps_update_kwargs = (;),
    )
    double_layer = to_tnqs_double_layer_network(psi)
    cache = BoundaryMPSCache(double_layer, bmps_chi; partition_by)
    return TensorNetworkQuantumSimulator.partitionfunction(
        TensorNetworkQuantumSimulator.update(cache; bmps_update_kwargs...),
    )
end

function boundarymps_marginal_weights(
        psi::TNMPTest.TensorNetworkState,
        target,
        bmps_chi::Integer;
        partition_by::AbstractString = "row",
        bmps_update_kwargs = (;),
    )
    d = dim(only(TNMPTest.siteinds(psi, target)))
    log_weights = Float64[]
    for s in 1:d
        net = to_tnqs_projected_network(psi, target, s)
        cache = BoundaryMPSCache(net, bmps_chi; partition_by)
        value = TensorNetworkQuantumSimulator.free_energy(
            TensorNetworkQuantumSimulator.update(cache; bmps_update_kwargs...),
        )
        push!(log_weights, real(value))
    end
    return normalize_log_weights(log_weights)
end

function next_bmps_chi(chi::Integer, chi_max::Integer)
    chi >= chi_max && return nothing
    nxt = min(chi * 2, chi_max)
    return nxt == chi ? nothing : nxt
end

function sweep_boundarymps_marginal(
        psi::TNMPTest.TensorNetworkState,
        center;
        chi_min::Integer = 1,
        chi_max::Integer,
        epsilon::Real,
        partition_by::AbstractString = "row",
        bmps_update_kwargs = (;),
    )
    history = NamedTuple[]
    chi_min > 0 || throw(ArgumentError("chi_min must be positive, got $chi_min"))
    chi_max >= chi_min ||
        throw(ArgumentError("chi_max must be >= chi_min, got chi_min=$chi_min chi_max=$chi_max"))
    chi = chi_min
    prev_marg = nothing
    converged_at = nothing

    while chi <= chi_max
        marg = boundarymps_marginal_weights(
            psi, center, chi;
            partition_by, bmps_update_kwargs,
        )
        delta_prev = prev_marg === nothing ? Inf : sum(abs.(marg .- prev_marg))
        push!(history, (; bmps_chi = chi, marginal = collect(marg), l1_delta_vs_prev = delta_prev))

        if prev_marg !== nothing && delta_prev < epsilon
            converged_at = chi
            break
        end

        prev_marg = marg
        nxt = next_bmps_chi(chi, chi_max)
        nxt === nothing && break
        chi = nxt
    end

    final_entry = history[end]
    return (
        history = history,
        converged_bmps_chi = converged_at,
        final_bmps_chi = final_entry.bmps_chi,
        final_marginal = final_entry.marginal,
    )
end

end # module
