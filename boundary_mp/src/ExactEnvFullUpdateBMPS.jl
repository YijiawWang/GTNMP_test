module ExactEnvFullUpdateBMPS

# Full-update boundary-MPS compression weighted by the EXACT rank-2L environment.
# The forward sweep is inherited from TensorNetworkQuantumSimulator.jl's BoundaryMPSCache;
# only the message-compression step is replaced. The environment is the far side of the cut
# contracted exactly via TreeSA (the `omeinsum` backend with `optimizer = TreeSA()`). Allowed
# deps: TensorNetworkQuantumSimulator + ITensors + stdlib.

using ITensors
using ITensors: ITensor, Index, dim, commoninds, commonind, dag, svd, inds,
    prime, noprime, replaceind, replaceinds, sim, delta, combiner
using ITensorMPS
using ITensorMPS: apply, linsolve, truncate
using NamedGraphs: vertices, neighbors
using NamedGraphs.GraphsExtensions: src, dst
using NamedGraphs.PartitionedGraphs:
    PartitionedGraphs, partitions_graph, PartitionVertex, parent
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator:
    Algorithm, BoundaryMPSCache, partitionfunction, contraction_sequence, TreeSA,
    set_interpartition_message!, prev_partitionedge, supergraph, network,
    generic_apply, mps_bond_dimension
using ITensors: @Algorithm_str
import TensorNetworkQuantumSimulator: update_message!, set_default_kwargs
using ..TNQSBoundaryMP: UniformState, marginal

export full_update_marginal, full_update_compress

# Merged-D² messages carry ket+bra site legs; fuse to one index per site for linsolve.
function _fuse_sites!(ψ::MPS, site_inds)
    combiners = Vector{Union{ITensor, Nothing}}(undef, length(ψ))
    for i in 1:length(ψ)
        si = site_inds[i]
        if length(si) <= 1
            combiners[i] = nothing
            continue
        end
        C = combiner(si...; tags = "SiteFuse_$i")
        ψ[i] = ψ[i] * C
        combiners[i] = C
    end
    return combiners
end

function _unfuse_sites!(ψ::MPS, combiners)
    for i in 1:length(ψ)
        C = combiners[i]
        isnothing(C) || (ψ[i] = ψ[i] * dag(C))
    end
    return ψ
end

function _fused_site_inds(ψ::MPS, site_inds, combiners)
    return [
        isnothing(combiners[i]) ? site_inds[i] : [only(ITensorMPS.siteinds(ψ, i))]
        for i in 1:length(ψ)
    ]
end

function full_update_compress(
        target::MPS, W::MPO;
        maxdim::Integer, nsweeps::Integer = 4, cutoff::Real = 1.0e-12, regularization::Real = 1.0e-10,
    )
    b = apply(W, target; cutoff = Float64(cutoff))
    for i in 1:length(b)
        b[i] = noprime(b[i])
    end
    c0 = truncate(copy(target); maxdim = Int(maxdim), cutoff = Float64(cutoff))
    c = linsolve(
        W, b, c0, Float64(regularization), 1.0;
        nsweeps = Int(nsweeps), maxdim = Int(maxdim), cutoff = Float64(cutoff),
        updater_kwargs = (; ishermitian = true, tol = 1.0e-8, krylovdim = 30, maxiter = 30),
    )
    return c
end

# ---- exact rank-2L environment -------------------------------------------------------
function _far_vertices(cache::BoundaryMPSCache, pe)
    sg = supergraph(cache)
    pg = partitions_graph(sg)
    srcp = parent(src(pe)); dstp = parent(dst(pe))
    visited = Set([srcp]); stack = [dstp]; comp = eltype(stack)[]
    while !isempty(stack)
        p = pop!(stack)
        p in visited && continue
        push!(visited, p); push!(comp, p)
        for nb in neighbors(pg, p); nb in visited || push!(stack, nb); end
    end
    return reduce(vcat, [collect(vertices(sg, PartitionVertex(p))) for p in comp])
end

function _tensor_to_mps(E::ITensor, site_inds)
    N = length(site_inds)
    ts = ITensor[]; rest = E; linkprev = nothing
    for i in 1:(N - 1)
        leftinds = i == 1 ? site_inds[i] : vcat(site_inds[i], Index[linkprev])
        # All env links share the tag "envlink" so that `prime(e1; tags = "envlink")` in
        # `_squared_env_mpo` matches *every* env bond. ITensors tag matching is exact-token,
        # so a per-link tag like "envlink_$i" would not be matched by "envlink" (the prime
        # would be a no-op), collapsing the squared-env metric to a separable product.
        U, S, V = svd(rest, leftinds...; lefttags = "envlink")
        push!(ts, U); rest = S * V; linkprev = commonind(U, S)
    end
    push!(ts, rest)
    return MPS(ts)
end

function _exact_env_mps(cache::BoundaryMPSCache, pe, target::MPS)
    vs = _far_vertices(cache, pe)
    ts = ITensor[copy(network(cache)[v]) for v in vs]
    seq = contraction_sequence(ts; alg = "omeinsum", optimizer = TreeSA())
    E = contract(ts; sequence = seq)
    site_inds = [collect(ITensorMPS.siteinds(target, i)) for i in 1:length(target)]
    return _tensor_to_mps(E, site_inds)
end

const _ENV_MEMO = IdDict{Any, Dict{Vector{Int}, ITensor}}()
_env_memo(cache) = get!(() -> Dict{Vector{Int}, ITensor}(), _ENV_MEMO, cache)

function _far_rows(cache::BoundaryMPSCache, pe)
    pg = partitions_graph(supergraph(cache))
    srcp = parent(src(pe)); dstp = parent(dst(pe))
    visited = Set([srcp]); stack = [dstp]; rows = Int[]
    while !isempty(stack)
        p = pop!(stack); p in visited && continue
        push!(visited, p); push!(rows, p)
        for nb in neighbors(pg, p); nb in visited || push!(stack, nb); end
    end
    return sort(rows; rev = true)
end

_row_tensors(cache, row::Int) =
    ITensor[copy(network(cache)[v]) for v in vertices(supergraph(cache), PartitionVertex(row))]

# Partition row (Int) that contains the center vertex.
function _center_row(cache::BoundaryMPSCache, center)
    sg = supergraph(cache)
    for p in vertices(partitions_graph(sg))
        center in vertices(sg, PartitionVertex(p)) && return p
    end
    error("center vertex $center not found in any partition")
end

# Exact far-side environment tensor for `rows`, built bottom-up and memoised.
#
# Two memo tiers:
# - `shared_env` (cross-s): a far region that EXCLUDES the center row is byte-identical
#   across the d marginal branches — only the center tensor changes with s, and the cut
#   bonds are the same `state` `Index` objects — so it is built once and reused for all s.
# - `_env_memo(cache)` (within-s): center-INCLUDING regions are s-dependent, so they stay
#   keyed by the per-s cache identity.
function _env_tensor_cached(
        cache::BoundaryMPSCache, rows::Vector{Int};
        shared_env::Union{Nothing, Dict{Vector{Int}, ITensor}} = nothing, center_row = nothing,
    )
    isempty(rows) && return ITensor(1.0)
    shareable = !isnothing(shared_env) && !isnothing(center_row) && !(center_row in rows)
    shareable && haskey(shared_env, rows) && return shared_env[rows]
    memo = _env_memo(cache)
    !shareable && haskey(memo, rows) && return memo[rows]

    row_ts = _row_tensors(cache, rows[1])
    ts = if length(rows) == 1
        row_ts
    else
        sub = _env_tensor_cached(cache, rows[2:end]; shared_env, center_row)
        vcat(row_ts, ITensor[sub])
    end
    seq = contraction_sequence(ts; alg = "omeinsum", optimizer = TreeSA())
    E = contract(ts; sequence = seq)
    shareable ? (shared_env[rows] = E) : (memo[rows] = E)
    return E
end

function _exact_env_mps_cached(
        cache::BoundaryMPSCache, pe, target::MPS;
        shared_env = nothing, center_row = nothing,
    )
    E = _env_tensor_cached(cache, _far_rows(cache, pe); shared_env, center_row)
    site_inds = [collect(ITensorMPS.siteinds(target, i)) for i in 1:length(target)]
    return _tensor_to_mps(E, site_inds)
end

function _squared_env_mpo(env::MPS, site_inds, row_site_inds)
    env = copy(env)
    env_combiners = _fuse_sites!(env, site_inds)
    local_fused = _fused_site_inds(env, site_inds, env_combiners)
    N = length(env)
    ts = ITensor[]
    for i in 1:N
        e1 = env[i]
        e2 = prime(e1; tags = "envlink")
        s_local = only(local_fused[i])
        s_row = only(row_site_inds[i])
        sa = sim(s_local); sb = sim(s_local)
        e1 = replaceind(e1, s_local => sa)
        e2 = replaceind(e2, s_local => sb)
        Wi = e1 * e2 * delta(sa, sb, s_local, prime(s_local))
        Wi = replaceinds(Wi, s_local => s_row, prime(s_local) => prime(s_row))
        push!(ts, Wi)
    end
    for i in 1:(N - 1)
        link2 = commoninds(ts[i], ts[i + 1])
        C = combiner(link2...; tags = "Wlink_$i")
        ts[i] = ts[i] * C
        ts[i + 1] = ts[i + 1] * dag(C)
    end
    return MPO(ts)
end

# ---- TensorNetworkQuantumSimulator BMPS integration ----------------------------------
function set_default_kwargs(alg::Algorithm"full_update_exact_env", cache::BoundaryMPSCache)
    nsweeps = get(alg.kwargs, :nsweeps, 4)
    cutoff = get(alg.kwargs, :cutoff, 1.0e-12)
    regularization = get(alg.kwargs, :regularization, 1.0e-10)
    normalize = get(alg.kwargs, :normalize, true)
    shared_env = get(alg.kwargs, :shared_env, nothing)
    center = get(alg.kwargs, :center, nothing)
    return Algorithm(
        "full_update_exact_env";
        nsweeps, cutoff, regularization, normalize, shared_env, center,
    )
end

function update_message!(
        alg::Algorithm"full_update_exact_env", cache::BoundaryMPSCache, pe;
        maxdim::Integer = mps_bond_dimension(cache),
    )
    prev_pe = prev_partitionedge(cache, pe)
    local_alg = set_default_kwargs(
        Algorithm("zipup"; cutoff = alg.kwargs.cutoff, normalize = alg.kwargs.normalize), cache,
    )
    isnothing(prev_pe) && return update_message!(local_alg, cache, pe; maxdim)

    mpo, mps, right_inds = TensorNetworkQuantumSimulator._bmps_apply_inputs(cache, pe)
    raw = MPS(generic_apply(
        mpo, mps, right_inds;
        cutoff = alg.kwargs.cutoff, normalize = false, maxdim = typemax(Int),
    ))
    shared_env = get(alg.kwargs, :shared_env, nothing)
    center = get(alg.kwargs, :center, nothing)
    center_row = isnothing(center) ? nothing : _center_row(cache, center)
    env = _exact_env_mps_cached(cache, pe, raw; shared_env, center_row)
    site_inds = [collect(ITensorMPS.siteinds(raw, i)) for i in 1:length(raw)]
    raw_fused = copy(raw)
    combiners = _fuse_sites!(raw_fused, site_inds)
    row_site_inds = _fused_site_inds(raw_fused, site_inds, combiners)
    W = _squared_env_mpo(env, site_inds, row_site_inds)
    compressed = full_update_compress(
        raw_fused, W; maxdim, nsweeps = alg.kwargs.nsweeps,
        cutoff = alg.kwargs.cutoff, regularization = alg.kwargs.regularization,
    )
    compressed = _unfuse_sites!(compressed, combiners)
    alg.kwargs.normalize && (compressed = ITensors.normalize(compressed))
    return set_interpartition_message!(cache, [compressed[i] for i in 1:length(compressed)], pe)
end

function full_update_marginal(
        state::UniformState, center, chi::Integer;
        nsweeps::Integer = 4, cutoff::Real = 1.0e-12, regularization::Real = 1.0e-10, maxiter::Integer = 1,
    )
    # Cross-s far-side environment cache: shared by the d marginal branches (`marginal`
    # calls `f` once per physical value, all closing over this dict). Far regions that
    # exclude the center row are identical across s, so they are contracted once.
    shared_env = Dict{Vector{Int}, ITensor}()
    f = function (tn)
        cache = BoundaryMPSCache(tn, Int(chi); partition_by = "row")
        cache = update(
            cache; alg = "bp", maxiter = Int(maxiter),
            message_update_alg = Algorithm(
                "full_update_exact_env"; nsweeps = Int(nsweeps),
                cutoff = Float64(cutoff), regularization = Float64(regularization), normalize = true,
                shared_env = shared_env, center = center,
            ),
            tolerance = nothing,
        )
        return partitionfunction(cache)
    end
    return marginal(state, center, f)
end

end # module
