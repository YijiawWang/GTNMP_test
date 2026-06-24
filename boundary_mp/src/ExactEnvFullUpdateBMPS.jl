module ExactEnvFullUpdateBMPS

# Full-update boundary-MPS compression weighted by the EXACT rank-2L environment.
# The forward sweep is inherited from TensorNetworkQuantumSimulator.jl (treesa branch,
# v0.3.10) BoundaryMPSCache; only the message-compression step is replaced. The environment
# is the far side of the cut contracted exactly via TreeSA (the `omeinsum` backend with
# `optimizer = TreeSA()`). Allowed deps: TNQS + ITensors + stdlib.

using LinearAlgebra
using ITensors
using ITensors: ITensor, Index, dim, commoninds, commonind, dag, svd, inds
using ITensorMPS
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

# ---- environment-weighted ALS (real, dense-block) -----------------------------------
# Ordered index list (left_link, site_inds..., right_link) for site `i`, links via commoninds.
function _ordered_inds(mps::MPS, site_inds, i)
    N = length(mps)
    order = Index[]
    i == 1 || push!(order, only(commoninds(mps[i - 1], mps[i])))
    append!(order, site_inds[i])
    i == N || push!(order, only(commoninds(mps[i], mps[i + 1])))
    return order
end

function _block(mps::MPS, site_inds, i)
    order = _ordered_inds(mps, site_inds, i)
    a = Array(mps[i], order...)
    N = length(mps)
    ld = i == 1 ? 1 : dim(order[1])
    rd = i == N ? 1 : dim(order[end])
    sd = prod(dim.(site_inds[i]))
    return reshape(a, ld, sd, rd)
end

_blocks(mps::MPS, site_inds) = [_block(mps, site_inds, i) for i in 1:length(mps)]

# w_i = b_i ⊗ b_i on bonds, squared on phys (so weight = |env|^2 along the chain).
function _hadamard_square(env_blocks)
    map(env_blocks) do t
        ld, pd, rd = size(t)
        out = zeros(Float64, ld * ld, pd, rd * rd)
        for l1 in 1:ld, l2 in 1:ld, p in 1:pd, r1 in 1:rd, r2 in 1:rd
            out[l1 + (l2 - 1) * ld, p, r1 + (r2 - 1) * rd] = real(t[l1, p, r1] * t[l2, p, r2])
        end
        m = maximum(abs, out)
        m == 0 ? out : out ./ m
    end
end

_loc(l, p, r, ld, pd) = l + (p - 1) * ld + (r - 1) * ld * pd

function _normal(left, w, right, dims)
    ld, pd, rd = dims
    n = ld * pd * rd
    A = zeros(Float64, n, n)
    for l in 1:ld, lp in 1:ld, p in 1:pd, r in 1:rd, rp in 1:rd,
            wl in 1:size(w, 1), wr in 1:size(w, 3)
        A[_loc(l, p, r, ld, pd), _loc(lp, p, rp, ld, pd)] +=
            left[l, lp, wl] * w[wl, p, wr] * right[r, rp, wr]
    end
    return A
end

function _rhs(left, t, w, right, dims)
    ld, pd, rd = dims
    b = zeros(Float64, ld * pd * rd)
    for l in 1:ld, p in 1:pd, r in 1:rd, tl in 1:size(t, 1), tr in 1:size(t, 3),
            wl in 1:size(w, 1), wr in 1:size(w, 3)
        b[_loc(l, p, r, ld, pd)] += left[l, tl, wl] * t[tl, p, tr] * w[wl, p, wr] * right[r, tr, wr]
    end
    return b
end

function _cl_normal(env, c, w)
    nx = zeros(Float64, size(c, 3), size(c, 3), size(w, 3))
    for l in 1:size(c, 1), lp in 1:size(c, 1), wl in 1:size(w, 1),
            p in 1:size(c, 2), r in 1:size(c, 3), rp in 1:size(c, 3), wr in 1:size(w, 3)
        nx[r, rp, wr] += env[l, lp, wl] * c[l, p, r] * c[lp, p, rp] * w[wl, p, wr]
    end
    return nx
end

function _cr_normal(c, w, env)
    nx = zeros(Float64, size(c, 1), size(c, 1), size(w, 1))
    for l in 1:size(c, 1), lp in 1:size(c, 1), wl in 1:size(w, 1),
            p in 1:size(c, 2), r in 1:size(c, 3), rp in 1:size(c, 3), wr in 1:size(w, 3)
        nx[l, lp, wl] += c[l, p, r] * c[lp, p, rp] * w[wl, p, wr] * env[r, rp, wr]
    end
    return nx
end

function _cl_mixed(env, c, t, w)
    nx = zeros(Float64, size(c, 3), size(t, 3), size(w, 3))
    for cl in 1:size(c, 1), tl in 1:size(t, 1), wl in 1:size(w, 1),
            p in 1:size(c, 2), cr in 1:size(c, 3), tr in 1:size(t, 3), wr in 1:size(w, 3)
        nx[cr, tr, wr] += env[cl, tl, wl] * c[cl, p, cr] * t[tl, p, tr] * w[wl, p, wr]
    end
    return nx
end

function _cr_mixed(c, t, w, env)
    nx = zeros(Float64, size(c, 1), size(t, 1), size(w, 1))
    for cl in 1:size(c, 1), tl in 1:size(t, 1), wl in 1:size(w, 1),
            p in 1:size(c, 2), cr in 1:size(c, 3), tr in 1:size(t, 3), wr in 1:size(w, 3)
        nx[cl, tl, wl] += c[cl, p, cr] * t[tl, p, tr] * w[wl, p, wr] * env[cr, tr, wr]
    end
    return nx
end

function _ridge_solve(A, b, reg)
    reg == 0 && return Hermitian(A) \ b
    s = real(tr(A)) / size(A, 1)
    (!isfinite(s) || s <= 0) && (s = 1.0)
    return (Hermitian(A) + reg * s * I(size(A, 1))) \ b
end

function _cl_normal_fold(comp, w, upto)
    e = ones(Float64, 1, 1, 1)
    for k in 1:upto; e = _cl_normal(e, comp[k], w[k]); end
    return e
end
function _cr_normal_fold(comp, w, from)
    e = ones(Float64, 1, 1, 1)
    for k in length(comp):-1:from; e = _cr_normal(comp[k], w[k], e); end
    return e
end
function _cl_mixed_fold(comp, t, w, upto)
    e = ones(Float64, 1, 1, 1)
    for k in 1:upto; e = _cl_mixed(e, comp[k], t[k], w[k]); end
    return e
end
function _cr_mixed_fold(comp, t, w, from)
    e = ones(Float64, 1, 1, 1)
    for k in length(comp):-1:from; e = _cr_mixed(comp[k], t[k], w[k], e); end
    return e
end

function _opt!(comp, t, w, i, reg)
    ln = _cl_normal_fold(comp, w, i - 1)
    rn = _cr_normal_fold(comp, w, i + 1)
    lb = _cl_mixed_fold(comp, t, w, i - 1)
    rb = _cr_mixed_fold(comp, t, w, i + 1)
    A = _normal(ln, w[i], rn, size(comp[i]))
    b = _rhs(lb, t[i], w[i], rb, size(comp[i]))
    x = _ridge_solve(A, b, reg)
    comp[i] = reshape(x, size(comp[i]))
    return comp
end

function _to_mps(blocks, site_inds)
    N = length(blocks)
    links = [Index(size(blocks[i], 3), "fu_link_$i") for i in 1:(N - 1)]
    ts = ITensor[]
    for i in 1:N
        b = blocks[i]; si = site_inds[i]; ld, sd, rd = size(b)
        ord = Index[]; dims = Int[]
        if i > 1; push!(ord, dag(links[i - 1])); push!(dims, dim(links[i - 1])); end
        for s in si; push!(ord, s); push!(dims, dim(s)); end
        if i < N; push!(ord, links[i]); push!(dims, dim(links[i])); end
        push!(ts, ITensor(reshape(b, dims...), ord...))
    end
    return MPS(ts)
end

function full_update_compress(
        target::MPS, env::MPS;
        maxdim::Integer, nsweeps::Integer = 4, cutoff::Real = 1.0e-12, regularization::Real = 1.0e-10,
    )
    site_inds = [collect(ITensorMPS.siteinds(target, i)) for i in 1:length(target)]
    t = _blocks(target, site_inds)
    w = _hadamard_square(_blocks(env, site_inds))
    init = ITensorMPS.truncate(copy(target); maxdim = Int(maxdim), cutoff = Float64(cutoff))
    comp = _blocks(init, site_inds)
    N = length(comp)
    for _ in 1:Int(nsweeps)
        for i in 1:N; _opt!(comp, t, w, i, Float64(regularization)); end
        for i in N:-1:1; _opt!(comp, t, w, i, Float64(regularization)); end
    end
    return _to_mps(comp, site_inds)
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
        U, S, V = svd(rest, leftinds...; lefttags = "envlink_$i")
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

# ---- TNQS BMPS integration -----------------------------------------------------------
function set_default_kwargs(alg::Algorithm"full_update_exact_env", cache::BoundaryMPSCache)
    nsweeps = get(alg.kwargs, :nsweeps, 4)
    cutoff = get(alg.kwargs, :cutoff, 1.0e-12)
    regularization = get(alg.kwargs, :regularization, 1.0e-10)
    normalize = get(alg.kwargs, :normalize, true)
    return Algorithm("full_update_exact_env"; nsweeps, cutoff, regularization, normalize)
end

function update_message!(
        alg::Algorithm"full_update_exact_env", cache::BoundaryMPSCache, pe;
        maxdim::Integer = mps_bond_dimension(cache),
    )
    prev_pe = prev_partitionedge(cache, pe)
    local_alg = set_default_kwargs(
        Algorithm("ITensorMPS"; cutoff = alg.kwargs.cutoff, normalize = alg.kwargs.normalize), cache,
    )
    isnothing(prev_pe) && return update_message!(local_alg, cache, pe; maxdim)

    O = ITensorMPS.MPO(cache, src(pe))
    M = ITensorMPS.MPS(cache, prev_pe)
    raw = generic_apply(O, M; cutoff = alg.kwargs.cutoff, normalize = false, maxdim = typemax(Int))
    env = _exact_env_mps(cache, pe, raw)
    compressed = full_update_compress(
        raw, env; maxdim, nsweeps = alg.kwargs.nsweeps,
        cutoff = alg.kwargs.cutoff, regularization = alg.kwargs.regularization,
    )
    alg.kwargs.normalize && (compressed = ITensors.normalize(compressed))
    return set_interpartition_message!(cache, compressed, pe)
end

function full_update_marginal(
        state::UniformState, center, chi::Integer;
        nsweeps::Integer = 4, cutoff::Real = 1.0e-12, regularization::Real = 1.0e-10, maxiter::Integer = 1,
    )
    f = function (tn)
        cache = BoundaryMPSCache(tn, Int(chi); partition_by = "row")
        cache = update(
            cache; alg = "bp", maxiter = Int(maxiter),
            message_update_alg = Algorithm(
                "full_update_exact_env"; nsweeps = Int(nsweeps),
                cutoff = Float64(cutoff), regularization = Float64(regularization), normalize = true,
            ),
            tolerance = nothing,
        )
        return partitionfunction(cache)
    end
    return marginal(state, center, f)
end

end # module
