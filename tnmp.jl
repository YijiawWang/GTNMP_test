module TNMPTest

using OMEinsumContractionOrders:
    EinCode, NestedEinsum, SlicedEinsum, TreeSA, contraction_complexity, optimize_code
using ITensors:
    ITensor,
    Index,
    Algorithm,
    commoninds,
    dag,
    delta,
    dim,
    eachindval,
    inds,
    norm,
    onehot,
    prime,
    replaceinds
using Base.Threads
using LinearAlgebra: dot
using NamedGraphs: NamedEdge, NamedGraph, add_edge!, dst, edges, neighbors, src, vertices
using Random: AbstractRNG, rand
import ITensors

# `nthreads <= 0` or `nothing` -> use all Julia threads (`Threads.nthreads()`).
message_passing_nthreads(nthreads::Union{Nothing, Integer} = nothing) =
    nthreads === nothing || Int(nthreads) <= 0 ? Threads.nthreads() : Int(nthreads)

export TNMPCache,
    contraction_sc,
    exact_marginal,
    first_order_region,
    graph,
    incoming_boundary_edges,
    marginal_tensors,
    message_tensors,
    random_alpha_state,
    random_state,
    random_uniform_complex_state,
    frustrated_copy_noise_state,
    run_message_passing!,
    siteinds,
    tnmp_marginal

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

function random_entry(rng::AbstractRNG, ::Type{ComplexF64})
    return complex(randn(rng), randn(rng))
end

function random_entry(rng::AbstractRNG, ::Type{Float64})
    return randn(rng)
end

# Independent Uniform(lo, hi) draws for the real and imaginary parts.
function random_uniform_complex_entry(rng::AbstractRNG; lo::Real = 0.0, hi::Real = 1.0)
    lo_f = Float64(lo)
    width = Float64(hi - lo)
    return complex(lo_f + width * rand(rng), lo_f + width * rand(rng))
end

function random_state(
        rng::AbstractRNG,
        g::NamedGraph;
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
        element_type::Type = ComplexF64,
    )
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(physical_dim, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(bond_dim, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[reverse_edge(e)] = ind
    end

    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        local_inds = Index[sitedict[v]]
        for vn in neighbors(g, v)
            push!(local_inds, edgeinds[NamedEdge(v => vn)])
        end
        tensor = ITensor(element_type, local_inds)
        for iv in eachindval(tensor)
            tensor[iv...] = random_entry(rng, element_type)
        end
        tensors[v] = tensor
    end

    return TensorNetworkState(tensors, Dict(v => Index[sitedict[v]] for v in vs), g)
end

# Random complex PEPS: each entry is complex(re ~ Uniform(lo, hi), im ~ Uniform(lo, hi))
# with independent real/imag draws. The double-layer norm network is built from
# these ket tensors via `traced_norm_factors`.
function random_uniform_complex_state(
        rng::AbstractRNG,
        g::NamedGraph;
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
        lo::Real = 0.0,
        hi::Real = 1.0,
    )
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))
    lo <= hi || throw(ArgumentError("require lo <= hi, got lo=$lo hi=$hi"))

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(physical_dim, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(bond_dim, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[reverse_edge(e)] = ind
    end

    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        local_inds = Index[sitedict[v]]
        for vn in neighbors(g, v)
            push!(local_inds, edgeinds[NamedEdge(v => vn)])
        end
        tensor = ITensor(ComplexF64, local_inds)
        for iv in eachindval(tensor)
            tensor[iv...] = random_uniform_complex_entry(rng; lo, hi)
        end
        tensors[v] = tensor
    end

    return TensorNetworkState(tensors, Dict(v => Index[sitedict[v]] for v in vs), g)
end

# Random PEPS from arXiv:2604.24760: each ket entry is Uniform(-alpha, 1-alpha).
# The double-layer norm network is built from these ket tensors via `traced_norm_factors`.
function random_alpha_state(
        rng::AbstractRNG,
        g::NamedGraph;
        alpha::Real = 0.5,
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
    )
    0 <= alpha <= 1 || throw(ArgumentError("alpha must be in [0, 1], got $alpha"))
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(physical_dim, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(bond_dim, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[reverse_edge(e)] = ind
    end

    lo = -Float64(alpha)
    width = 1.0
    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        local_inds = Index[sitedict[v]]
        for vn in neighbors(g, v)
            push!(local_inds, edgeinds[NamedEdge(v => vn)])
        end
        tensor = ITensor(ComplexF64, local_inds)
        for iv in eachindval(tensor)
            tensor[iv...] = complex(lo + width * rand(rng))
        end
        tensors[v] = tensor
    end

    return TensorNetworkState(tensors, Dict(v => Index[sitedict[v]] for v in vs), g)
end

include(joinpath(@__DIR__, "..", "scripts", "frustrated_copy_peps.jl"))

function frustrated_copy_noise_state(
        rng::AbstractRNG,
        g::NamedGraph;
        eps::Real,
        physical_dim::Integer = 2,
        bond_dim::Integer = 8,
    )
    tensors, siteinds_dict = build_frustrated_copy_noise_tensors(
        rng, g;
        eps = eps,
        physical_dim = physical_dim,
        bond_dim = bond_dim,
    )
    return TensorNetworkState(tensors, siteinds_dict, g)
end

virtualinds(psi::TensorNetworkState, e::NamedEdge) = commoninds(psi[src(e)], psi[dst(e)])

function default_message(psi::TensorNetworkState, e::NamedEdge)
    linds = collect(virtualinds(psi, e))
    return delta(Index[linds...; prime.(dag.(linds))...])
end

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

function to_eincode(tensors::Vector{<:ITensor})
    ixs = [Any[ind for ind in inds(tensor)] for tensor in tensors]
    size_dict = Dict{Any, Int}()
    counts = Dict{Any, Int}()
    for ix in ixs, ind in ix
        size_dict[ind] = dim(ind)
        counts[ind] = get(counts, ind, 0) + 1
    end
    iy = Any[ind for (ind, c) in counts if c == 1]
    return EinCode(ixs, iy), size_dict
end

# Convert an OMEinsumContractionOrders tree (leaves = 1-based tensor indices)
# into the nested-Vector format that ITensors `contract(...; sequence)` expects.
nested_to_sequence(ne::NestedEinsum) =
    ne.tensorindex >= 1 ? ne.tensorindex : Any[nested_to_sequence(c) for c in ne.args]
nested_to_sequence(se::SlicedEinsum) = nested_to_sequence(se.eins)

function contraction_sequence(tensors::Vector{<:ITensor})
    length(tensors) == 1 && return 1
    code, size_dict = to_eincode(tensors)
    optcode = optimize_code(code, size_dict, TreeSA())
    return nested_to_sequence(optcode)
end

# Space complexity (sc) of contracting `tensors`: log2 of the number of
# elements in the largest *intermediate* tensor along a TreeSA-optimised order.
# The TreeSA settings match `docs/figures/contraction_sc.jl` so the numbers can
# be compared directly against `docs/figures/contraction_sc_results.md`.
function contraction_sc(
        tensors::Vector{<:ITensor};
        ntrials::Integer = 20,
        niters::Integer = 60,
        βs = 1.0:1.0:18.0,
    )
    length(tensors) <= 1 && return 0.0
    code, size_dict = to_eincode(tensors)
    optcode = optimize_code(code, size_dict, TreeSA(; ntrials = ntrials, niters = niters, βs = βs))
    return contraction_complexity(optcode, size_dict).sc
end

function contract_all(tensors::Vector{<:ITensor}; sequence = nothing)
    isempty(tensors) && return ITensor(1.0)
    length(tensors) == 1 && return only(tensors)
    ITensors.disable_warn_order()
    seq = something(sequence, contraction_sequence(tensors))
    return ITensors.contract(tensors; sequence = seq)
end

function scalar_weight(tensors::Vector{<:ITensor}; sequence = nothing)
    z = contract_all(tensors; sequence = sequence)[]
    w = real(z)
    if w < 0 && abs(w) < 1e-12
        return 0.0
    end
    return w < 0 ? abs(w) : w
end

function normalize_weights(weights::Vector{Float64})
    total = sum(weights)
    iszero(total) && throw(ArgumentError("Cannot normalize a zero marginal"))
    return weights ./ total
end

function exact_marginal(psi::TensorNetworkState, target)
    verts = collect(vertices(graph(psi)))
    d = dim(only(siteinds(psi, target)))
    weights = [scalar_weight(marginal_factors(psi, verts, target, state)) for state in 1:d]
    return normalize_weights(weights)
end

function subdivision_graph(g::NamedGraph)
    canonical = Dict{Tuple{Any, Any}, NamedEdge}()
    for e in edges(g)
        canonical[(src(e), dst(e))] = e
        canonical[(dst(e), src(e))] = e
    end

    nodes = Any[]
    for v in vertices(g)
        push!(nodes, (:site, v))
    end
    for e in edges(g)
        push!(nodes, (:bond, e))
    end

    gp = NamedGraph(nodes)
    for e in edges(g)
        add_edge!(gp, (:site, src(e)), (:bond, e))
        add_edge!(gp, (:site, dst(e)), (:bond, e))
    end
    return gp, canonical
end

# Sub-lattice neighborhood on the (pre-subdivision) grid whose vertices are (x, y).
# A site gets the L*L block centered on it; a bond gets the L*(L-1) block centered
# on the bond (L sites across the bond, L-1 sites along it). L must be odd.
function region_vertices(gp::NamedGraph, center_node, L::Integer)
    isodd(L) || throw(ArgumentError("L must be odd"))
    r = (L - 1) ÷ 2
    if first(center_node) === :site
        (cx, cy) = last(center_node)
        xlo, xhi, ylo, yhi = cx - r, cx + r, cy - r, cy + r
    else
        e = last(center_node)
        (x1, y1), (x2, y2) = src(e), dst(e)
        if x1 == x2
            ya, yb = minmax(y1, y2)
            xlo, xhi = x1 - r, x1 + r
            ylo, yhi = ya - (r - 1), yb + (r - 1)
        else
            xa, xb = minmax(x1, x2)
            xlo, xhi = xa - (r - 1), xb + (r - 1)
            ylo, yhi = y1 - r, y1 + r
        end
    end
    region = Any[]
    for node in vertices(gp)
        first(node) === :site || continue
        (x, y) = last(node)
        if xlo <= x <= xhi && ylo <= y <= yhi
            push!(region, last(node))
        end
    end
    return region
end

# Graph-based neighborhood that needs no grid coordinates, so it works on any
# graph (e.g. a tree): a node's own original-graph vertices plus every
# first-order (graph-distance-1) neighbor. For a site node (:site, v) this is
# {v} ∪ neighbors(g, v); for a bond node (:bond, e) it is the union of the
# first-order neighborhoods of the two endpoints of e.
function first_order_region(g::NamedGraph, center_node)
    seeds = first(center_node) === :site ? Any[last(center_node)] :
        Any[src(last(center_node)), dst(last(center_node))]
    region = Any[]
    seen = Set{Any}()
    for s in seeds
        for u in (s, neighbors(g, s)...)
            if !(u in seen)
                push!(seen, u)
                push!(region, u)
            end
        end
    end
    return region
end

function incoming_boundary_edges(g::NamedGraph, region)
    region_set = Set(region)
    bedges = NamedEdge[]
    for e in edges(g)
        u, v = src(e), dst(e)
        u_inside = u in region_set
        v_inside = v in region_set
        u_inside == v_inside && continue
        push!(bedges, u_inside ? NamedEdge(v => u) : NamedEdge(u => v))
    end
    return bedges
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
# vertices) for a subdivision-graph `node`. The default is the L*L grid window
# (`region_vertices`); pass e.g. `(gp, g, node) -> first_order_region(g, node)`
# to use the graph-based first-order neighborhood instead (and `L` is ignored).
function TNMPCache(
        psi::TensorNetworkState,
        L::Integer;
        normalize::Symbol = :l2,
        region_fn = (gp, g, node) -> region_vertices(gp, node, L),
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

# Compute TreeSA outside the lock so independent keys can optimize in parallel.
function _ensure_contraction_sequence!(
        sequences::Dict{Any, Any},
        seq_lock::ReentrantLock,
        key,
        tensors::Vector{<:ITensor},
    )
    if haskey(sequences, key)
        return sequences[key]
    end
    seq = contraction_sequence(tensors)
    lock(seq_lock) do
        if haskey(sequences, key)
            return sequences[key]
        end
        sequences[key] = seq
        return seq
    end
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

function prewarm_contraction_sequences!(
        sequences::Dict{Any, Any},
        seq_lock::ReentrantLock,
        specs::Vector{Tuple{Any, Vector{ITensor}}},
        ;
        nthreads::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
    )
    isempty(specs) && return 0
    nt = message_passing_nthreads(nthreads)
    n = length(specs)
    if nt <= 1
        for (key, tensors) in specs
            _ensure_contraction_sequence!(sequences, seq_lock, key, tensors)
        end
    else
        Threads.@threads for i in 1:n
            key, tensors = specs[i]
            _ensure_contraction_sequence!(sequences, seq_lock, key, tensors)
        end
    end
    if !isempty(progress_label)
        println("[$progress_label] pre-warmed $n contraction sequence(s)")
        flush(stdout)
    end
    return n
end

function prewarm_message_contraction_sequences!(
        cache::TNMPCache,
        keys_to_update::Vector{Tuple{Any, NamedEdge}};
        nthreads::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
    )
    specs = rank2_message_prewarm_specs(cache, keys_to_update)
    label = isempty(progress_label) ? "prewarm" : "$(progress_label)/prewarm"
    return prewarm_contraction_sequences!(
        cache.contraction_sequences, cache.sequence_lock, specs;
        nthreads = nthreads, progress_label = label,
    )
end

function contract_all!(cache::TNMPCache, key, tensors::Vector{<:ITensor})
    isempty(tensors) && return ITensor(1.0)
    length(tensors) == 1 && return only(tensors)
    ITensors.disable_warn_order()
    return ITensors.contract(tensors; sequence = get_contraction_sequence!(cache, key, tensors))
end

function scalar_weight!(cache::TNMPCache, key, tensors::Vector{<:ITensor})
    z = contract_all!(cache, key, tensors)[]
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

# The double-layer tensors contracted to produce the message flowing across
# `in_edge` into the region centred on `center_node`: the cavity norm factors
# plus the current messages on every other boundary edge of the cavity. Returns
# an empty vector when the cavity is empty (the message defaults to the identity
# and no contraction is needed). This is the exact tensor list whose `sc` the
# message-update step costs.
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

function compute_message(cache::TNMPCache, center_node, in_edge::NamedEdge)
    tensors = message_tensors(cache, center_node, in_edge)
    isempty(tensors) && return normalize_message(cache, default_message(cache.network, in_edge))
    key = (:message, center_node, in_edge)
    return normalize_message(cache, contract_all!(cache, key, tensors))
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

function _message_passing_step_rank2!(cache::TNMPCache, keys_to_update)
    final_diff = 0.0
    for (center_node, e) in keys_to_update
        previous = cache.messages[(center_node, e)]
        new_message = compute_message(cache, center_node, e)
        cache.messages[(center_node, e)] = new_message
        final_diff = max(final_diff, message_difference(new_message, previous))
    end
    return final_diff
end

function run_message_passing!(cache::TNMPCache; max_iter::Integer = 100, tol::Real = 1e-6,
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        nthreads::Union{Nothing, Integer} = nothing,
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
    prewarm_message_contraction_sequences!(cache, keys_to_update;
        nthreads = nt,
        progress_label = progress_label,
    )

    converged = false
    iterations = Int(max_iter)
    final_diff = Inf
    for it in 1:max_iter
        final_diff = _message_passing_step_rank2!(cache, keys_to_update)
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

# The double-layer tensors contracted to produce the (unnormalised) weight of
# `target` being in `state`: the neighborhood norm factors (with `target` fixed
# to `state`) plus the converged incoming messages. This is the exact tensor
# list whose `sc` the final marginal contraction costs.
function marginal_tensors(cache::TNMPCache, target, state::Integer)
    psi = cache.network
    center_node = (:site, target)
    region = cache.regions[center_node]
    tensors = marginal_factors(psi, region, target, state)
    for e in incoming_boundary_edges(graph(psi), region)
        push!(tensors, compute_message(cache, center_node, e))
    end
    return tensors
end

function rank2_marginal_prewarm_specs(cache::TNMPCache, target)
    psi = cache.network
    center_node = (:site, target)
    region = cache.regions[center_node]
    incoming = [
        compute_message(cache, center_node, e)
        for e in incoming_boundary_edges(graph(psi), region)
    ]
    d = dim(only(siteinds(psi, target)))
    specs = Tuple{Any, Vector{ITensor}}[]
    for state in 1:d
        tensors = marginal_factors(psi, region, target, state)
        append!(tensors, incoming)
        push!(specs, ((:marginal, target, state), tensors))
    end
    return specs
end

function tnmp_marginal(cache::TNMPCache, target;
        progress_interval::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        nthreads::Union{Nothing, Integer} = nothing,
    )
    psi = cache.network
    center_node = (:site, target)
    region = cache.regions[center_node]
    incoming = [compute_message(cache, center_node, e) for e in incoming_boundary_edges(graph(psi), region)]
    d = dim(only(siteinds(psi, target)))

    nt = message_passing_nthreads(nthreads)
    prewarm_label = isempty(progress_label) ? "marginal/prewarm" : "$(progress_label)/prewarm"
    prewarm_contraction_sequences!(
        cache.contraction_sequences, cache.sequence_lock,
        rank2_marginal_prewarm_specs(cache, target);
        nthreads = nt, progress_label = prewarm_label,
    )

    weights = Float64[]
    for state in 1:d
        progress_interval !== nothing && progress_interval > 0 &&
            (println("[$(progress_label)] contracting marginal state $(state)/$(d)"); flush(stdout))
        tensors = marginal_factors(psi, region, target, state)
        append!(tensors, incoming)
        push!(weights, scalar_weight!(cache, (:marginal, target, state), tensors))
    end
    return normalize_weights(weights)
end

end
