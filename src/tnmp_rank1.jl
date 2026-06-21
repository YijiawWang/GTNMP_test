module TNMPRank1

# Rank-1 (vector) message variant of the double-layer TNMP cache.
#
# The neighborhood / region / cavity machinery (subdivision graph, region
# windows, double-layer norm factors, exact marginal) is reused unchanged from
# `tnmp.jl`. The only thing that differs is the *message*:
#
#   * `tnmp.jl` (rank-2, the "merged" figure): one ITensor per boundary edge
#     carrying both the ket bond index `b` and the bra bond index `b'`, i.e. a
#     `D x D` matrix `M[b, b']`. Plugging matrices into a cavity keeps the ket
#     and bra layers coupled through `D^2` boundary bonds -> expensive.
#
#   * here (rank-1, the "open indices" figure): the ket leg and bra leg of a
#     physical boundary edge are treated as two *independent* boundary bonds,
#     each carrying its own rank-1 vector message (`x[b]` on the ket leg,
#     `y[b']` on the bra leg). To compute the message on a leg we contract the
#     cavity leaving *only that one leg* open and close the opposite-layer leg
#     of the same edge with its current message, exactly like the original
#     vector-message TNMP applied to `2k` bonds instead of `k`.

include("tnmp.jl")

using ITensors: ITensor, dim, onehot, prime
using NamedGraphs: NamedEdge, NamedGraph, src, dst, vertices
using Random: AbstractRNG, MersenneTwister, rand
import ITensors

import .TNMPTest:
    TensorNetworkState,
    virtualinds,
    norm_factors,
    marginal_factors,
    incoming_boundary_edges,
    reverse_edge,
    subdivision_graph,
    region_vertices,
    first_order_region,
    contraction_sequence,
    contract_all,
    contraction_sc,
    normalize_weights,
    normalize_message,
    message_difference,
    log_message_passing_progress,
    message_passing_nthreads,
    _ensure_contraction_sequence!,
    prewarm_contraction_sequences!,
    # re-exported public helpers so a script only needs `using .TNMPRank1`
    random_alpha_state,
    random_state,
    random_uniform_complex_state,
    weak_entangled_biased_circuit_state,
    tfim_imaginary_time_state,
    spin_glass_pair_factor_state,
    fully_frustrated_pair_factor_state,
    exact_marginal,
    graph,
    siteinds

export TNMPRank1Cache,
    run_message_passing!,
    tnmp_marginal,
    contraction_sc,
    first_order_region,
    incoming_boundary_edges,
    marginal_tensors,
    message_tensors,
    random_alpha_state,
    random_state,
    random_uniform_complex_state,
    weak_entangled_biased_circuit_state,
    tfim_imaginary_time_state,
    spin_glass_pair_factor_state,
    fully_frustrated_pair_factor_state,
    exact_marginal,
    exact_marginal_single_layer,
    run_message_passing_single_layer!,
    tnmp_marginal_single_layer,
    graph,
    siteinds

# ---------------------------------------------------------------------------
# Rank-1 message primitives
# ---------------------------------------------------------------------------

# A uniform rank-1 message living on a single layer of edge `e`:
#   layer = :ket -> vector over the ket bond index/indices  b
#   layer = :bra -> vector over the bra bond index/indices  prime(b)
function default_message_vec(psi::TensorNetworkState, e::NamedEdge, layer::Symbol)
    linds = collect(virtualinds(psi, e))
    leg_inds = layer === :ket ? linds : prime.(linds)
    dims = ntuple(k -> dim(leg_inds[k]), length(leg_inds))
    return ITensor(ones(Float64, dims...), leg_inds...)
end

# Independent random rank-1 messages for ket and bra legs of the same edge.
function random_message_vec(psi::TensorNetworkState, e::NamedEdge, layer::Symbol, rng::AbstractRNG)
    linds = collect(virtualinds(psi, e))
    leg_inds = layer === :ket ? linds : prime.(linds)
    dims = ntuple(k -> dim(leg_inds[k]), length(leg_inds))
    return ITensor(rand(rng, Float64, dims...), leg_inds...)
end

function scalar_weight_value(z::ITensor)
    w = real(z[])
    if w < 0 && abs(w) < 1e-12
        return 0.0
    end
    return w < 0 ? abs(w) : w
end

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

struct TNMPRank1Cache
    network::TensorNetworkState
    gp::NamedGraph
    canonical::Dict{Tuple{Any, Any}, NamedEdge}
    L::Int
    regions::Dict{Any, Vector{Any}}
    # keyed by (center_node, directed boundary edge, layer)
    messages::Dict{Tuple{Any, NamedEdge, Symbol}, ITensor}
    contraction_sequences::Dict{Any, Any}
    normalize::Symbol
    sequence_lock::ReentrantLock
end

# See `TNMPCache`: `region_fn(gp, g, node)` selects the neighborhood. Defaults
# to the L*L grid window; pass `(gp, g, node) -> first_order_region(g, node)`
# for the graph-based first-order neighborhood (then `L` is ignored).
function TNMPRank1Cache(
        psi::TensorNetworkState,
        L::Integer;
        normalize::Symbol = :l1sum,
        region_fn = (gp, g, node) -> region_vertices(gp, node, L),
    )
    normalize in (:l2, :l1sum) ||
        throw(ArgumentError("normalize must be :l2 or :l1sum"))
    gp, canonical = subdivision_graph(graph(psi))
    regions = Dict{Any, Vector{Any}}()
    for node in vertices(gp)
        regions[node] = region_fn(gp, graph(psi), node)
    end
    return TNMPRank1Cache(
        psi,
        gp,
        canonical,
        Int(L),
        regions,
        Dict{Tuple{Any, NamedEdge, Symbol}, ITensor}(),
        Dict{Any, Any}(),
        normalize,
        ReentrantLock(),
    )
end

normalize_msg(cache::TNMPRank1Cache, message::ITensor) = normalize_message(message, cache.normalize)

function contract_cached!(cache::TNMPRank1Cache, key, tensors::Vector{<:ITensor})
    isempty(tensors) && return ITensor(1.0)
    length(tensors) == 1 && return only(tensors)
    seq = _ensure_contraction_sequence!(
        cache.contraction_sequences, cache.sequence_lock, key, tensors,
    )
    ITensors.disable_warn_order()
    return ITensors.contract(tensors; sequence = seq)
end

function get_message(cache::TNMPRank1Cache, center_node, e::NamedEdge, layer::Symbol)
    haskey(cache.messages, (center_node, e, layer)) &&
        return cache.messages[(center_node, e, layer)]
    return normalize_msg(cache, default_message_vec(cache.network, e, layer))
end

cavity_vertices(cache::TNMPRank1Cache, a_node, center_node) =
    setdiff(cache.regions[a_node], cache.regions[center_node])

# ---------------------------------------------------------------------------
# Single-leg message update
# ---------------------------------------------------------------------------

# Compute the rank-1 message on one layer (`layer`) of the directed boundary
# edge `in_edge` flowing into the region centred on `center_node`.
#
# Contract the cavity double-layer TN of the common neighborhood. Every *other*
# boundary edge of the cavity is closed by both of its rank-1 vector messages
# (ket and bra). For the open edge itself we leave the requested layer's leg
# open and close the opposite-layer leg with its current message, so the result
# is a single vector over the open leg.
# The tensors contracted to produce the rank-1 message on `layer` of `in_edge`
# into the region centred on `center_node`: the cavity norm factors, both
# rank-1 vector messages on every other boundary edge, and (on the open edge)
# the opposite-layer message that closes it so a single open leg remains.
# Returns an empty vector when the cavity is empty (message defaults to ones).
# This is the exact tensor list whose `sc` the message-update step costs.
function message_tensors(cache::TNMPRank1Cache, center_node, in_edge::NamedEdge, layer::Symbol)
    psi = cache.network
    g = graph(psi)
    a_node = (:bond, cache.canonical[(src(in_edge), dst(in_edge))])
    cav_vs = cavity_vertices(cache, a_node, center_node)
    isempty(cav_vs) && return ITensor[]

    open_edge = reverse_edge(in_edge)
    other_layer = layer === :ket ? :bra : :ket
    bedges = incoming_boundary_edges(g, cav_vs)
    tensors = norm_factors(psi, cav_vs)
    for e in bedges
        if e == open_edge
            push!(tensors, get_message(cache, center_node, in_edge, other_layer))
        else
            push!(tensors, get_message(cache, a_node, e, :ket))
            push!(tensors, get_message(cache, a_node, e, :bra))
        end
    end
    return tensors
end

function compute_message(cache::TNMPRank1Cache, center_node, in_edge::NamedEdge, layer::Symbol)
    tensors = message_tensors(cache, center_node, in_edge, layer)
    isempty(tensors) && return normalize_msg(cache, default_message_vec(cache.network, in_edge, layer))
    key = (:message, center_node, in_edge, layer)
    return normalize_msg(cache, contract_cached!(cache, key, tensors))
end

# ---------------------------------------------------------------------------
# Message passing
# ---------------------------------------------------------------------------

function message_keys(cache::TNMPRank1Cache, center_node)
    g = graph(cache.network)
    ks = Tuple{Any, NamedEdge, Symbol}[]
    for e in incoming_boundary_edges(g, cache.regions[center_node])
        for layer in (:ket, :bra)
            push!(ks, (center_node, e, layer))
        end
    end
    return ks
end

function init_messages!(cache::TNMPRank1Cache, keys_to_update; rng::AbstractRNG = MersenneTwister(0))
    psi = cache.network
    for (center_node, e, layer) in keys_to_update
        cache.messages[(center_node, e, layer)] =
            normalize_msg(cache, random_message_vec(psi, e, layer, rng))
    end
    return cache
end

function rank1_message_prewarm_specs(
        cache::TNMPRank1Cache,
        keys_to_update::Vector{Tuple{Any, NamedEdge, Symbol}},
    )
    specs = Tuple{Any, Vector{ITensor}}[]
    for (center_node, e, layer) in keys_to_update
        tensors = message_tensors(cache, center_node, e, layer)
        isempty(tensors) && continue
        push!(specs, ((:message, center_node, e, layer), tensors))
    end
    return specs
end

function prewarm_message_contraction_sequences!(
        cache::TNMPRank1Cache,
        keys_to_update::Vector{Tuple{Any, NamedEdge, Symbol}};
        nthreads::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
    )
    specs = rank1_message_prewarm_specs(cache, keys_to_update)
    label = isempty(progress_label) ? "prewarm" : "$(progress_label)/prewarm"
    return prewarm_contraction_sequences!(
        cache.contraction_sequences, cache.sequence_lock, specs;
        nthreads = nthreads, progress_label = label,
    )
end

function _message_passing_step_rank1!(cache::TNMPRank1Cache, keys_to_update)
    final_diff = 0.0
    for (center_node, e, layer) in keys_to_update
        previous = cache.messages[(center_node, e, layer)]
        new_message = compute_message(cache, center_node, e, layer)
        cache.messages[(center_node, e, layer)] = new_message
        final_diff = max(final_diff, message_difference(new_message, previous))
    end
    return final_diff
end

function _message_passing_step_single_layer!(cache::TNMPRank1Cache, keys_to_update)
    final_diff = 0.0
    for (center_node, e) in keys_to_update
        previous = cache.messages[(center_node, e, :ket)]
        new_message = compute_message_single_layer(cache, center_node, e)
        cache.messages[(center_node, e, :ket)] = new_message
        final_diff = max(final_diff, message_difference(new_message, previous))
    end
    return final_diff
end

# Iterate a given set of message keys to a fixed point. Used both for the
# global bond-region messages and (locally) for the site-region messages that
# feed the marginal.
function iterate_messages!(cache::TNMPRank1Cache, keys_to_update; max_iter::Integer, tol::Real,
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        nthreads::Union{Nothing, Integer} = nothing,
    )
    nt = message_passing_nthreads(nthreads)
    prewarm_message_contraction_sequences!(cache, keys_to_update;
        nthreads = nt,
        progress_label = progress_label,
    )
    converged = false
    iterations = Int(max_iter)
    final_diff = Inf
    for it in 1:max_iter
        final_diff = _message_passing_step_rank1!(cache, keys_to_update)
        if final_diff <= tol
            converged = true
            iterations = it
            log_message_passing_progress(;
                progress_interval, progress_label,
                iteration = it, max_iter, final_diff, n_keys = length(keys_to_update),
                converged = true,
            )
            break
        end
        log_message_passing_progress(;
            progress_interval, progress_label,
            iteration = it, max_iter, final_diff, n_keys = length(keys_to_update),
        )
    end
    return (; converged, iterations, final_diff)
end

function run_message_passing!(cache::TNMPRank1Cache; max_iter::Integer = 100, tol::Real = 1e-6,
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        nthreads::Union{Nothing, Integer} = nothing,
        message_init_rng::AbstractRNG = MersenneTwister(0),
    )
    keys_to_update = Tuple{Any, NamedEdge, Symbol}[]
    for node in vertices(cache.gp)
        first(node) == :bond || continue
        append!(keys_to_update, message_keys(cache, node))
    end
    init_messages!(cache, keys_to_update; rng = message_init_rng)
    return iterate_messages!(cache, keys_to_update;
        max_iter = max_iter, tol = tol,
        progress_interval = progress_interval, progress_label = progress_label,
        nthreads = nthreads,
    )
end

# ---------------------------------------------------------------------------
# Marginal
# ---------------------------------------------------------------------------

function converge_center_messages!(cache::TNMPRank1Cache, center_node; max_iter::Integer, tol::Real,
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        nthreads::Union{Nothing, Integer} = nothing,
        message_init_rng::AbstractRNG = MersenneTwister(0),
    )
    ks = message_keys(cache, center_node)
    for (cn, e, layer) in ks
        haskey(cache.messages, (cn, e, layer)) ||
            (cache.messages[(cn, e, layer)] =
                normalize_msg(cache, random_message_vec(cache.network, e, layer, message_init_rng)))
    end
    return iterate_messages!(cache, ks;
        max_iter = max_iter, tol = tol,
        progress_interval = progress_interval, progress_label = progress_label,
        nthreads = nthreads,
    )
end

# The tensors contracted to produce the (unnormalised) weight of `target` in
# `state`: the neighborhood norm factors (with `target` fixed) plus both rank-1
# vector messages on every boundary edge. Assumes the center-region messages
# are already converged. This is the tensor list whose `sc` the marginal costs.
function marginal_tensors(cache::TNMPRank1Cache, target, state::Integer)
    psi = cache.network
    center_node = (:site, target)
    region = cache.regions[center_node]
    tensors = marginal_factors(psi, region, target, state)
    for e in incoming_boundary_edges(graph(psi), region)
        push!(tensors, get_message(cache, center_node, e, :ket))
        push!(tensors, get_message(cache, center_node, e, :bra))
    end
    return tensors
end

function tnmp_marginal(cache::TNMPRank1Cache, target; max_iter::Integer = 100, tol::Real = 1e-8,
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        nthreads::Union{Nothing, Integer} = nothing,
        message_init_rng::AbstractRNG = MersenneTwister(0),
    )
    psi = cache.network
    center_node = (:site, target)
    converge_center_messages!(cache, center_node;
        max_iter = max_iter, tol = tol,
        progress_interval = progress_interval,
        progress_label = isempty(progress_label) ? "center-mp" : progress_label,
        nthreads = nthreads,
        message_init_rng = message_init_rng,
    )

    d = dim(only(siteinds(psi, target)))
    weights = Float64[]
    for state in 1:d
        progress_interval !== nothing && progress_interval > 0 &&
            (println("[$(progress_label)] contracting marginal state $(state)/$(d)"); flush(stdout))
        z = contract_cached!(cache, (:marginal, target, state), marginal_tensors(cache, target, state))
        push!(weights, scalar_weight_value(z))
    end
    return normalize_weights(weights)
end

# ---------------------------------------------------------------------------
# Single-layer variant
# ---------------------------------------------------------------------------
#
# On the *double-layer* (norm) network every graph edge closes a 4-cycle
# (ket-bond -> physical-trace -> bra-bond -> physical-trace), so even on a tree
# the doubled network is loopy and the rank-1 (mean-field) messages above are
# only approximate.
#
# The *single-layer* network instead has one tensor per vertex (the bare ket,
# no bra) with its physical leg summed over — so it is loopless exactly when the
# underlying graph is a tree. Then a single rank-1 vector message per bond is
# EXACT. These helpers reuse the same region / cavity machinery, but with
# single-layer factors and one `:ket` message per bond (no bra layer, nothing to
# close on the open edge — the open bond is simply left open).

# Single-layer factor for vertex `v`: the site tensor with its physical leg
# summed (closed with the all-ones vector), or fixed to `state` when `v` is the
# marginal target.
function single_layer_factor(psi::TensorNetworkState, v)
    s = only(siteinds(psi, v))
    return psi[v] * ITensor(ones(Float64, dim(s)), s)
end

function single_layer_factor(psi::TensorNetworkState, v, target, state::Integer)
    v == target || return single_layer_factor(psi, v)
    s = only(siteinds(psi, v))
    return psi[v] * onehot(s => state)
end

single_layer_factors(psi::TensorNetworkState, verts) =
    ITensor[single_layer_factor(psi, v) for v in verts]
single_layer_factors(psi::TensorNetworkState, verts, target, state::Integer) =
    ITensor[single_layer_factor(psi, v, target, state) for v in verts]

# Single rank-1 message on the (ket) bond of `in_edge` flowing into the region
# centred on `center_node`: contract the cavity single-layer factors closed by
# the current message on every other boundary bond, leaving the open bond open.
function compute_message_single_layer(cache::TNMPRank1Cache, center_node, in_edge::NamedEdge)
    psi = cache.network
    g = graph(psi)
    a_node = (:bond, cache.canonical[(src(in_edge), dst(in_edge))])
    cav_vs = cavity_vertices(cache, a_node, center_node)
    isempty(cav_vs) && return normalize_msg(cache, default_message_vec(psi, in_edge, :ket))

    open_edge = reverse_edge(in_edge)
    tensors = single_layer_factors(psi, cav_vs)
    for e in incoming_boundary_edges(g, cav_vs)
        e == open_edge && continue
        push!(tensors, get_message(cache, a_node, e, :ket))
    end
    key = (:sl_message, center_node, in_edge)
    return normalize_msg(cache, contract_cached!(cache, key, tensors))
end

function iterate_messages_single_layer!(cache::TNMPRank1Cache, keys_to_update; max_iter::Integer, tol::Real,
        nthreads::Union{Nothing, Integer} = nothing,
    )
    converged = false
    iterations = Int(max_iter)
    final_diff = Inf
    for it in 1:max_iter
        final_diff = _message_passing_step_single_layer!(cache, keys_to_update)
        if final_diff <= tol
            converged = true
            iterations = it
            break
        end
    end
    return (; converged, iterations, final_diff)
end

function run_message_passing_single_layer!(cache::TNMPRank1Cache; max_iter::Integer = 100, tol::Real = 1e-6,
        nthreads::Union{Nothing, Integer} = nothing,
    )
    g = graph(cache.network)
    keys_to_update = Tuple{Any, NamedEdge}[]
    for node in vertices(cache.gp)
        first(node) == :bond || continue
        for e in incoming_boundary_edges(g, cache.regions[node])
            push!(keys_to_update, (node, e))
        end
    end
    for (center_node, e) in keys_to_update
        cache.messages[(center_node, e, :ket)] =
            normalize_msg(cache, default_message_vec(cache.network, e, :ket))
    end
    return iterate_messages_single_layer!(cache, keys_to_update;
        max_iter = max_iter, tol = tol, nthreads = nthreads,
    )
end

function converge_center_messages_single_layer!(cache::TNMPRank1Cache, center_node; max_iter::Integer, tol::Real)
    g = graph(cache.network)
    ks = Tuple{Any, NamedEdge}[(center_node, e) for e in incoming_boundary_edges(g, cache.regions[center_node])]
    for (cn, e) in ks
        haskey(cache.messages, (cn, e, :ket)) ||
            (cache.messages[(cn, e, :ket)] = normalize_msg(cache, default_message_vec(cache.network, e, :ket)))
    end
    return iterate_messages_single_layer!(cache, ks; max_iter = max_iter, tol = tol)
end

function tnmp_marginal_single_layer(cache::TNMPRank1Cache, target; max_iter::Integer = 100, tol::Real = 1e-8)
    psi = cache.network
    center_node = (:site, target)
    converge_center_messages_single_layer!(cache, center_node; max_iter = max_iter, tol = tol)

    region = cache.regions[center_node]
    bedges = incoming_boundary_edges(graph(psi), region)
    d = dim(only(siteinds(psi, target)))
    weights = Float64[]
    for state in 1:d
        tensors = single_layer_factors(psi, region, target, state)
        for e in bedges
            push!(tensors, get_message(cache, center_node, e, :ket))
        end
        z = contract_cached!(cache, (:sl_marginal, target, state), tensors)
        push!(weights, real(z[]))
    end
    return normalize_weights(weights)
end

# Exact single-layer marginal: contract the whole single-layer network with the
# target physical leg fixed and every other physical leg summed.
function exact_marginal_single_layer(psi::TensorNetworkState, target)
    verts = collect(vertices(graph(psi)))
    d = dim(only(siteinds(psi, target)))
    weights = Float64[]
    for state in 1:d
        z = contract_all(single_layer_factors(psi, verts, target, state))
        push!(weights, real(z[]))
    end
    return normalize_weights(weights)
end

end
