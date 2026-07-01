# TNMP cache and low-level helpers for rank-2 message passing.

function default_message(psi::TensorNetworkState, e::NamedEdge)
    linds = collect(virtualinds(psi, e))
    return delta(Index[linds...; prime.(dag.(linds))...])
end

struct TNMPCache
    network::TensorNetworkState
    gp::NamedGraph
    canonical::Dict{Tuple{Any, Any}, NamedEdge}
    L::Int
    regions::Dict{Any, Vector{Any}}
    messages::Dict{Tuple{Any, NamedEdge}, ITensor}
    contraction_sequences::Dict{Any, Any}
    normalize::Symbol
    sequence_lock::ReentrantLock
end

# `region_fn(gp, g, node)` returns the neighborhood (a vector of original-graph
# vertices) for a subdivision-graph `node`. It is model-dependent and therefore
# required: build it at the model level, e.g. `region_fn = grid_region_fn(L)`
# (the L*L grid window) or `region_fn = first_order_region_fn()` (graph-based
# first-order neighborhood) from `examples/neighborhoods.jl`. `L` is recorded on
# the cache for reference but the neighborhood itself comes entirely from
# `region_fn`.
function TNMPCache(
        psi::TensorNetworkState,
        L::Integer;
        normalize::Symbol = :l2,
        region_fn,
    )
    normalize in (:l2, :l1sum) ||
        throw(ArgumentError("normalize must be :l2 or :l1sum"))
    gp, canonical = subdivision_graph(graph(psi))
    regions = Dict{Any, Vector{Any}}()
    for node in vertices(gp)
        regions[node] = region_fn(gp, graph(psi), node)
    end
    return TNMPCache(
        psi,
        gp,
        canonical,
        Int(L),
        regions,
        Dict{Tuple{Any, NamedEdge}, ITensor}(),
        Dict{Any, Any}(),
        normalize,
        ReentrantLock(),
    )
end

function get_contraction_sequence!(cache::TNMPCache, key, tensors::Vector{<:ITensor})
    return _ensure_contraction_sequence!(
        cache.contraction_sequences, cache.sequence_lock, key, tensors,
    )
end

function contract_all!(cache::TNMPCache, key, tensors::Vector{<:ITensor}; complexity_probe = nothing)
    isempty(tensors) && return ITensor(1.0)
    length(tensors) == 1 && return only(tensors)
    ITensors.disable_warn_order()
    seq = _ensure_contraction_sequence!(
        cache.contraction_sequences, cache.sequence_lock, key, tensors;
        complexity_probe,
    )
    return ITensors.contract(tensors; sequence = seq)
end

function scalar_weight!(cache::TNMPCache, key, tensors::Vector{<:ITensor}; complexity_probe = nothing)
    z = contract_all!(cache, key, tensors; complexity_probe)[]
    w = real(z)
    if w < 0 && abs(w) < 1e-12
        return 0.0
    end
    return w < 0 ? abs(w) : w
end

function normalize_message(message::ITensor, mode::Symbol)
    if mode === :l1sum
        s = sum(ITensors.array(message))
        return iszero(s) ? message : message / s
    end
    n = norm(message)
    return iszero(n) ? message : message / n
end

normalize_message(cache::TNMPCache, message::ITensor) = normalize_message(message, cache.normalize)

function init_messages!(cache::TNMPCache)
    psi = cache.network
    g = graph(psi)
    for node in vertices(cache.gp)
        first(node) == :bond || continue
        for e in incoming_boundary_edges(g, cache.regions[node])
            cache.messages[(node, e)] = normalize_message(cache, default_message(psi, e))
        end
    end
    return cache
end

function get_message(cache::TNMPCache, center_node, e::NamedEdge)
    return get(cache.messages, (center_node, e), normalize_message(cache, default_message(cache.network, e)))
end

cavity_vertices(cache::TNMPCache, a_node, center_node) =
    setdiff(cache.regions[a_node], cache.regions[center_node])
