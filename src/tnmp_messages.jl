# Rank-2 TNMP message updates.

function message_tensors(cache::TNMPCache, center_node, in_edge::NamedEdge)
    psi = cache.network
    g = graph(psi)
    a_node = (:bond, cache.canonical[(src(in_edge), dst(in_edge))])
    cav_vs = cavity_vertices(cache, a_node, center_node)
    isempty(cav_vs) && return ITensor[]

    open_edge = reverse_edge(in_edge)
    bedges = incoming_boundary_edges(g, cav_vs)
    tensors = norm_factors(psi, cav_vs)
    for e in bedges
        e == open_edge && continue
        push!(tensors, get_message(cache, a_node, e))
    end
    return tensors
end

function rank2_message_prewarm_specs(
        cache::TNMPCache,
        keys_to_update::Vector{Tuple{Any, NamedEdge}},
    )
    specs = Tuple{Any, Vector{ITensor}}[]
    for (center_node, e) in keys_to_update
        tensors = message_tensors(cache, center_node, e)
        isempty(tensors) && continue
        push!(specs, ((:message, center_node, e), tensors))
    end
    return specs
end

function prewarm_message_contraction_sequences!(
        cache::TNMPCache,
        keys_to_update::Vector{Tuple{Any, NamedEdge}};
        nthreads::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        complexity_probe = nothing,
    )
    specs = rank2_message_prewarm_specs(cache, keys_to_update)
    label = isempty(progress_label) ? "prewarm" : "$(progress_label)/prewarm"
    n = prewarm_contraction_sequences!(
        cache.contraction_sequences, cache.sequence_lock, specs;
        nthreads = nthreads, progress_label = label, complexity_probe,
    )
    return n
end

# `rank2_marginal_prewarm_spec` / `prewarm_rank2_marginal_sequence!` live in
# `tnmp_marginal.jl` (the marginal readout file) and are reused here.

function prewarm_rank2_tnmp_sequences!(
        cache::TNMPCache,
        target;
        nthreads::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        complexity_probe = nothing,
    )
    psi = cache.network
    g = graph(psi)
    keys_to_update = Tuple{Any, NamedEdge}[]
    for node in vertices(cache.gp)
        first(node) == :bond || continue
        for e in incoming_boundary_edges(g, cache.regions[node])
            push!(keys_to_update, (node, e))
        end
    end
    # The site-region boundary messages computed by `compute_site_messages!`
    # after bond convergence. Their contraction order only depends on the index
    # structure, so it is valid to pre-warm them now (with default messages).
    center_node = (:site, target)
    for e in incoming_boundary_edges(g, cache.regions[center_node])
        push!(keys_to_update, (center_node, e))
    end
    prewarm_message_contraction_sequences!(cache, keys_to_update;
        nthreads = nthreads, progress_label = progress_label, complexity_probe,
    )
    prewarm_rank2_marginal_sequence!(cache, target;
        nthreads = nthreads, progress_label = progress_label, complexity_probe,
    )
    return cache
end

function compute_message(cache::TNMPCache, center_node, in_edge::NamedEdge; complexity_probe = nothing)
    tensors = message_tensors(cache, center_node, in_edge)
    isempty(tensors) && return normalize_message(cache, default_message(cache.network, in_edge))
    key = (:message, center_node, in_edge)
    return normalize_message(cache, contract_all!(cache, key, tensors; complexity_probe))
end

# After the bond-node messages have converged, the messages flowing into a
# *site* (tensor) region — which the marginal readout closes its boundary with —
# are obtained in a single pass from the converged bond-node messages: each
# incoming boundary edge of the site region is closed by contracting the
# corresponding bond cavity (`compute_message` with `center_node = (:site, v)`).
# Site messages never feed back into one another, so one pass is exact (unlike
# the iterated bond messages).
function compute_site_messages!(cache::TNMPCache, target; complexity_probe = nothing)
    g = graph(cache.network)
    center_node = (:site, target)
    for e in incoming_boundary_edges(g, cache.regions[center_node])
        cache.messages[(center_node, e)] =
            compute_message(cache, center_node, e; complexity_probe)
    end
    return cache
end

function message_difference(a::ITensor, b::ITensor)
    na, nb = norm(a), norm(b)
    (iszero(na) || iszero(nb)) && return 0.0
    fidelity = abs2(dot(a, b) / (na * nb))
    return max(0.0, 1 - fidelity)
end

function log_message_passing_progress(;
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        iteration::Integer,
        max_iter::Integer,
        final_diff::Real,
        n_keys::Integer,
        converged::Bool = false,
    )
    progress_interval === nothing && return
    progress_interval <= 0 && return
    iteration == 1 || iteration % progress_interval == 0 || converged || return
    prefix = isempty(progress_label) ? "" : "[$(progress_label)] "
    suffix = converged ? " converged" : ""
    println("$(prefix)iter $(iteration)/$(max_iter), diff=$(final_diff), keys=$(n_keys)$(suffix)")
    flush(stdout)
    return nothing
end

function _message_passing_step_rank2!(cache::TNMPCache, keys_to_update; complexity_probe = nothing)
    final_diff = 0.0
    for (center_node, e) in keys_to_update
        previous = cache.messages[(center_node, e)]
        new_message = compute_message(cache, center_node, e; complexity_probe)
        cache.messages[(center_node, e)] = new_message
        final_diff = max(final_diff, message_difference(new_message, previous))
    end
    return final_diff
end

function run_message_passing!(cache::TNMPCache; max_iter::Integer = 100, tol::Real = 1e-6,
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        nthreads::Union{Nothing, Integer} = nothing,
        complexity_probe = nothing,
        skip_prewarm::Bool = false,
    )
    psi = cache.network
    g = graph(psi)
    init_messages!(cache)

    keys_to_update = Tuple{Any, NamedEdge}[]
    for node in vertices(cache.gp)
        first(node) == :bond || continue
        for e in incoming_boundary_edges(g, cache.regions[node])
            push!(keys_to_update, (node, e))
        end
    end

    nt = message_passing_nthreads(nthreads)
    skip_prewarm || prewarm_message_contraction_sequences!(cache, keys_to_update;
        nthreads = nt,
        progress_label = progress_label,
        complexity_probe,
    )

    converged = false
    iterations = Int(max_iter)
    final_diff = Inf
    for it in 1:max_iter
        final_diff = _message_passing_step_rank2!(cache, keys_to_update; complexity_probe)
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
