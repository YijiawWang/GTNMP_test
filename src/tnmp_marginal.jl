# TNMP marginal readout.

function marginal_tensors(cache::TNMPCache, target, state::Integer)
    psi = cache.network
    center_node = (:site, target)
    region = cache.regions[center_node]
    tensors = marginal_factors(psi, region, target, state)
    for e in incoming_boundary_edges(graph(psi), region)
        push!(tensors, get_message(cache, center_node, e))
    end
    return tensors
end

function rank2_marginal_prewarm_spec(cache::TNMPCache, target)
    psi = cache.network
    center_node = (:site, target)
    region = cache.regions[center_node]
    tensors = marginal_factors(psi, region, target, 1)
    for e in incoming_boundary_edges(graph(psi), region)
        push!(tensors, default_message(psi, e))
    end
    return ((:marginal, target), tensors)
end

function prewarm_rank2_marginal_sequence!(
        cache::TNMPCache,
        target;
        nthreads::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        complexity_probe = nothing,
    )
    spec = rank2_marginal_prewarm_spec(cache, target)
    label = isempty(progress_label) ? "marginal/prewarm" : "$(progress_label)/marginal/prewarm"
    return prewarm_contraction_sequences!(
        cache.contraction_sequences, cache.sequence_lock, Tuple{Any, Vector{ITensor}}[spec];
        nthreads = nthreads, progress_label = label, complexity_probe,
    )
end

function tnmp_marginal(cache::TNMPCache, target;
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        nthreads::Union{Nothing, Integer} = nothing,
        complexity_probe = nothing,
    )
    psi = cache.network
    center_node = (:site, target)
    region = cache.regions[center_node]
    d = dim(only(siteinds(psi, target)))

    # Bond-node messages are already converged; derive the site-region boundary
    # messages from them in a single pass before reading the marginal.
    compute_site_messages!(cache, target; complexity_probe)

    prewarm_rank2_marginal_sequence!(cache, target;
        nthreads = message_passing_nthreads(nthreads),
        progress_label = progress_label,
        complexity_probe,
    )

    weights = Float64[]
    for state in 1:d
        progress_interval !== nothing && progress_interval > 0 &&
            (println("[$(progress_label)] contracting marginal state $(state)/$(d)"); flush(stdout))
        push!(weights, scalar_weight!(cache, (:marginal, target), marginal_tensors(cache, target, state)))
    end
    return normalize_weights(weights)
end
