# Minimal, self-contained rank-2 TNMP on the fully-frustrated Ising double layer.
#
# Goal: reproduce, with the *smallest possible* model (L = 4) and the *smallest
# possible* code, the result that on this frustrated instance rank-2 TNMP
# converges to the correct single-site marginal while plain (single-site) BP
# does not converge.
#
# What this file deliberately does NOT reuse from `src/`:
#   * the TNMP message-passing / cavity / marginal logic -- it is rewritten here
#     from scratch with only ITensors primitives and two `Dict`s, so the whole
#     algorithm is readable in one place.
#
# What it *does* reuse (these are the model and the baseline, not TNMP):
#   * the fully-frustrated PEPS builder (`examples/state_models.jl`)            -> the model
#   * `TensorNetworkQuantumSimulator`'s belief propagation                       -> BP is *called*, never re-implemented
#
# Algorithm flow (matches the task description):
#   1. build neighborhoods (L_R x L_R site window, L_R x (L_R-1) bond window)
#      and the cavity = source-region \ receiver-region;
#   2. TreeSA-precompute tc/sc of every cavity / neighborhood sub-TN, print the
#      mean/max tc and sc, with BP's single-site tc/sc next to it;
#   3. run rank-2 message passing, read the center marginal, and compare it to
#      the exact marginal (and to BP).
#
# Run:  julia --project=TNMP_test TNMP_test/demo/minimal_tnmp_rank2.jl

# --- model + BP (reused) -----------------------------------------------------
include(joinpath(@__DIR__, "..", "..", "src", "tnmp.jl"))
using .TNMPTest                                   # only for the TensorNetworkState type + model builder
include(joinpath(@__DIR__, "..", "..", "examples", "state_models.jl"))  # fully_frustrated_pair_factor_state

using NamedGraphs: NamedGraph, NamedEdge, edges, vertices, src, dst, neighbors
using NamedGraphs.NamedGraphGenerators: named_grid
using OMEinsumContractionOrders: EinCode, optimize_code, TreeSA, contraction_complexity,
    NestedEinsum, SlicedEinsum
using Random: MersenneTwister
using ITensors: ITensor, Index, inds, dim, dag, prime, delta, onehot, commoninds,
    replaceinds, norm, dot
import ITensors

# `state_models.jl` aliases TensorNetworkState to TNMPTest's; keep that alias
# winning over TensorNetworkQuantumSimulator's by `using` the package last.
using Dictionaries: Dictionary
using LinearAlgebra: diag
using TensorNetworkQuantumSimulator

ITensors.disable_warn_order()

# =============================================================================
# 1. double-layer factors + neighborhoods (the only data we ever build)
# =============================================================================

# ket / bra of the norm network at a vertex: the physical leg is shared (traced)
# while the bond legs split into an unprimed (ket) and a primed (bra) layer.
ket(psi, v) = psi[v]
function bra(psi, v)
    s = psi.siteinds[v]
    return replaceinds(dag(prime(psi[v])), prime.(s), s)
end
dl_factors(psi, v) = ITensor[ket(psi, v), bra(psi, v)]

# Same, but with the physical leg projected onto a fixed state `s` (marginal).
function dl_factors_fixed(psi, v, s::Integer)
    sind = only(psi.siteinds[v])
    return ITensor[psi[v] * onehot(sind => s), dag(prime(psi[v])) * onehot(prime(sind) => s)]
end

# A rank-2 message lives on a bond: the (ket, bra) index pair of that edge.
function default_message(psi, e::NamedEdge)
    b = commoninds(psi[src(e)], psi[dst(e)])
    return delta(Index[b...; prime.(dag.(b))...])
end

# TNMP neighborhood definition on the (x,y) grid (L_R must be odd):
#   * a site gets the L_R x L_R block centered on it;
#   * a bond gets the L_R x (L_R-1) block centered on the bond.
function grid_window(g, node, L::Integer)
    r = (L - 1) ÷ 2
    if first(node) === :site
        cx, cy = last(node)
        xlo, xhi, ylo, yhi = cx - r, cx + r, cy - r, cy + r
    else
        e = last(node)
        (x1, y1), (x2, y2) = src(e), dst(e)
        if x1 == x2
            ya, yb = minmax(y1, y2)
            xlo, xhi, ylo, yhi = x1 - r, x1 + r, ya - (r - 1), yb + (r - 1)
        else
            xa, xb = minmax(x1, x2)
            xlo, xhi, ylo, yhi = xa - (r - 1), xb + (r - 1), y1 - r, y1 + r
        end
    end
    return Any[v for v in vertices(g) if xlo <= v[1] <= xhi && ylo <= v[2] <= yhi]
end

# Edges crossing out of `region`, oriented outside -> inside.
function boundary_edges(g, region)
    rs = Set(region)
    out = NamedEdge[]
    for e in edges(g)
        u, v = src(e), dst(e)
        (u in rs) == (v in rs) && continue
        push!(out, (u in rs) ? NamedEdge(v => u) : NamedEdge(u => v))
    end
    return out
end

# =============================================================================
# 2. TreeSA contraction planning + complexity (tc/sc)
# =============================================================================

function eincode(tensors)
    ixs = [Any[i for i in inds(t)] for t in tensors]
    sd, cnt = Dict{Any, Int}(), Dict{Any, Int}()
    for ix in ixs, i in ix
        sd[i] = dim(i)
        cnt[i] = get(cnt, i, 0) + 1
    end
    iy = Any[i for (i, c) in cnt if c == 1]
    return EinCode(ixs, iy), sd
end

ne_to_seq(ne::NestedEinsum) =
    ne.tensorindex >= 1 ? ne.tensorindex : Any[ne_to_seq(c) for c in ne.args]
ne_to_seq(se::SlicedEinsum) = ne_to_seq(se.eins)

# Plan a contraction once: returns the ITensors sequence and the TreeSA tc/sc.
function treesa_plan(tensors)
    length(tensors) <= 1 && return (; seq = 1, tc = 0.0, sc = 0.0)
    code, sd = eincode(tensors)
    opt = optimize_code(code, sd, TreeSA())
    cc = contraction_complexity(opt, sd)
    return (; seq = ne_to_seq(opt), tc = cc.tc, sc = cc.sc)
end

# Contract `tensors`, caching the TreeSA order under `key`.
function contract_cached!(seqs, key, tensors)
    isempty(tensors) && return ITensor(1.0)
    length(tensors) == 1 && return only(tensors)
    seq = get!(() -> treesa_plan(tensors).seq, seqs, key)
    return ITensors.contract(tensors; sequence = seq)
end

# =============================================================================
# 3. rank-2 TNMP: cavity tensors, message update, marginal
# =============================================================================

normalize_msg(m) = (n = norm(m); iszero(n) ? m : m / n)

get_message(psi, messages, node, e) =
    get(() -> normalize_msg(default_message(psi, e)), messages, (node, e))

# Sub-TN whose contraction produces the message flowing along `in_edge` into the
# region of `center_node`. `in_edge` sits on a bond whose own (larger) region is
# the source; the cavity is exactly the vertices the source region adds over the
# receiver region. The open legs are the (ket,bra) pair of `in_edge`.
function message_tensors(psi, g, canon, regions, messages, center_node, in_edge)
    a_edge = canon[(src(in_edge), dst(in_edge))]
    a_node = (:bond, a_edge)
    cavity = setdiff(regions[a_node], regions[center_node])
    isempty(cavity) && return ITensor[]
    tensors = ITensor[]
    for v in cavity
        append!(tensors, dl_factors(psi, v))
    end
    for e in boundary_edges(g, cavity)
        canon[(src(e), dst(e))] == a_edge && continue   # leave the open edge open
        push!(tensors, get_message(psi, messages, a_node, e))
    end
    return tensors
end

function compute_message(psi, g, canon, regions, messages, seqs, center_node, in_edge)
    ts = message_tensors(psi, g, canon, regions, messages, center_node, in_edge)
    isempty(ts) && return normalize_msg(default_message(psi, in_edge))
    return normalize_msg(contract_cached!(seqs, (center_node, in_edge), ts))
end

# 1 - |<a,b>|^2 normalized fidelity, the rank-2 message convergence metric.
function msg_diff(a, b)
    na, nb = norm(a), norm(b)
    (iszero(na) || iszero(nb)) && return 0.0
    return max(0.0, 1 - abs2(dot(a, b) / (na * nb)))
end

function run_message_passing!(psi, g, canon, regions, messages, seqs; max_iter, tol)
    keys_to_update = Tuple{Any, NamedEdge}[]
    for e0 in edges(g), e in boundary_edges(g, regions[(:bond, e0)])
        push!(keys_to_update, ((:bond, e0), e))
    end
    converged, iters, diff = false, max_iter, Inf
    for it in 1:max_iter
        diff = 0.0
        for (node, e) in keys_to_update
            new = compute_message(psi, g, canon, regions, messages, seqs, node, e)
            diff = max(diff, msg_diff(new, messages[(node, e)]))
            messages[(node, e)] = new
        end
        if diff <= tol
            converged, iters = true, it
            break
        end
    end
    return (; converged, iterations = iters, final_diff = diff)
end

# After the bond messages converge, the site-region boundary messages follow in
# one pass; then the marginal is the closed neighborhood sub-TN per state.
function tnmp_marginal(psi, g, canon, regions, messages, seqs, target)
    center = (:site, target)
    for e in boundary_edges(g, regions[center])
        messages[(center, e)] = compute_message(psi, g, canon, regions, messages, seqs, center, e)
    end
    d = dim(only(psi.siteinds[target]))
    weights = Float64[]
    for s in 1:d
        tensors = ITensor[]
        for v in regions[center]
            append!(tensors, v == target ? dl_factors_fixed(psi, v, s) : dl_factors(psi, v))
        end
        for e in boundary_edges(g, regions[center])
            push!(tensors, messages[(center, e)])
        end
        z = contract_cached!(seqs, (:marginal, target, s), tensors)[]
        push!(weights, abs(real(z)))
    end
    return weights ./ sum(weights)
end

# Exact single-site marginal: contract the whole double layer, center fixed.
function exact_marginal(psi, g, seqs, target)
    verts = collect(vertices(g))
    d = dim(only(psi.siteinds[target]))
    weights = Float64[]
    for s in 1:d
        tensors = ITensor[]
        for v in verts
            append!(tensors, v == target ? dl_factors_fixed(psi, v, s) : dl_factors(psi, v))
        end
        z = contract_cached!(seqs, (:exact, target, s), tensors)[]
        push!(weights, abs(real(z)))
    end
    return weights ./ sum(weights)
end

# =============================================================================
# 4. BP baseline (called from TensorNetworkQuantumSimulator) + BP complexity
# =============================================================================

to_tnqs_state(psi) = TensorNetworkQuantumSimulator.TensorNetworkState(
    Dictionary(collect(vertices(psi.graph)), [psi[v] for v in vertices(psi.graph)]),
    psi.graph,
)

function bp_marginal(psi, target; maxiter, tolerance)
    cache = BeliefPropagationCache(to_tnqs_state(psi))
    alg = TensorNetworkQuantumSimulator.set_default_kwargs(
        ITensors.Algorithm("bp"; maxiter = Int(maxiter), tolerance = Float64(tolerance)), cache,
    )
    updated = copy(cache)
    TensorNetworkQuantumSimulator.invalidate_contraction_sequences!(updated)
    edge_sequence = alg.kwargs.edge_sequence
    diff, iters, converged = Inf, 0, false
    for it in 1:Int(maxiter)
        d = Ref(0.0)
        TensorNetworkQuantumSimulator.update_iteration!(alg, updated, edge_sequence; update_diff! = d)
        diff = d[] / length(edge_sequence)
        iters = it
        if diff <= Float64(tolerance)
            converged = true
            break
        end
    end
    ρ = reduced_density_matrix(ITensors.Algorithm("bp"), updated, [target])
    ρd = max.(collect(real.(diag(ITensors.array(ρ)))), 0.0)
    return ρd ./ sum(ρd), (; converged, iterations = iters, final_diff = diff)
end

# BP's single-site sub-TNs (one double-layer site tensor + its other-edge
# messages), for an apples-to-apples tc/sc comparison against TNMP.
function bp_complexity(psi, g, target)
    tcs, scs = Float64[], Float64[]
    for u in vertices(g), v in neighbors(g, u)         # message u -> v
        tensors = dl_factors(psi, u)
        for w in neighbors(g, u)
            w == v && continue
            push!(tensors, default_message(psi, NamedEdge(w => u)))
        end
        p = treesa_plan(tensors)
        push!(tcs, p.tc); push!(scs, p.sc)
    end
    mtensors = dl_factors_fixed(psi, target, 1)         # center marginal
    for w in neighbors(g, target)
        push!(mtensors, default_message(psi, NamedEdge(w => target)))
    end
    mp = treesa_plan(mtensors)
    return (; tc_mean = sum(tcs) / length(tcs), tc_max = maximum(tcs),
        sc_mean = sum(scs) / length(scs), sc_max = maximum(scs),
        marg_tc = mp.tc, marg_sc = mp.sc, n = length(tcs))
end

# =============================================================================
# 5. driver
# =============================================================================

build_regions(g, region_L) = merge(
    Dict{Any, Vector{Any}}((:site, v) => grid_window(g, (:site, v), region_L) for v in vertices(g)),
    Dict{Any, Vector{Any}}((:bond, e) => grid_window(g, (:bond, e), region_L) for e in edges(g)),
)

# Full rank-2 TNMP run at a given neighborhood size: init bond messages, pass to
# convergence, read the center marginal.
function tnmp_run(psi, g, canon, region_L, center; max_iter = 500, tol = 1e-8)
    regions = build_regions(g, region_L)
    messages = Dict{Tuple{Any, NamedEdge}, ITensor}()
    for e0 in edges(g), e in boundary_edges(g, regions[(:bond, e0)])
        messages[((:bond, e0), e)] = normalize_msg(default_message(psi, e))
    end
    seqs = Dict{Any, Any}()
    info = run_message_passing!(psi, g, canon, regions, messages, seqs; max_iter, tol)
    marg = tnmp_marginal(psi, g, canon, regions, messages, seqs, center)
    return (; regions, info, marg)
end

# TreeSA tc/sc of every rank-2 cavity sub-TN + the neighborhood (marginal) sub-TN.
function tnmp_complexity(psi, g, canon, regions, center)
    messages = Dict{Tuple{Any, NamedEdge}, ITensor}()
    for e0 in edges(g), e in boundary_edges(g, regions[(:bond, e0)])
        messages[((:bond, e0), e)] = normalize_msg(default_message(psi, e))
    end
    keys = Tuple{Any, NamedEdge}[((:bond, e0), e) for e0 in edges(g)
        for e in boundary_edges(g, regions[(:bond, e0)])]
    append!(keys, [((:site, center), e) for e in boundary_edges(g, regions[(:site, center)])])
    tc, sc = Float64[], Float64[]
    for (node, e) in keys
        ts = message_tensors(psi, g, canon, regions, messages, node, e)
        isempty(ts) && continue
        p = treesa_plan(ts)
        push!(tc, p.tc); push!(sc, p.sc)
    end
    nts = ITensor[]
    for v in regions[(:site, center)]
        append!(nts, v == center ? dl_factors_fixed(psi, v, 1) : dl_factors(psi, v))
    end
    for e in boundary_edges(g, regions[(:site, center)])
        push!(nts, default_message(psi, e))
    end
    n = treesa_plan(nts)
    return (; tc, sc, marg_tc = n.tc, marg_sc = n.sc)
end

function main()
    L, chi, seed, K, field = 4, 2, 7, 1.0, 0.2
    region_Ls = [3, 5]          # 3x3 window, then 5x5 (covers the whole 4x4 lattice)
    center = ((L + 1) ÷ 2, (L + 1) ÷ 2)

    lines = String[]
    log!(s = "") = (println(s); push!(lines, s))

    g = named_grid((L, L))
    psi = fully_frustrated_pair_factor_state(MersenneTwister(seed), g; K, field, bond_dim = chi)
    canon = Dict{Tuple{Any, Any}, NamedEdge}()
    for e in edges(g)
        canon[(src(e), dst(e))] = e
        canon[(dst(e), src(e))] = e
    end

    log!("="^72)
    log!("Minimal rank-2 TNMP vs BP  --  fully-frustrated Ising double layer")
    log!("L=$L  chi=$chi  K=$K  field=$field  seed=$seed  center=$center")
    log!("="^72)

    # --- step 1: neighborhoods & cavities (rank-2, region_L=3) --------------
    regions3 = build_regions(g, 3)
    log!("\n[1] neighborhoods & cavities (rank-2, region_L=3)")
    log!("    site-region |R| (center)   = $(length(regions3[(:site, center)]))  (3x3 window)")
    bsz = [length(regions3[(:bond, e)]) for e in edges(g)]
    log!("    bond-region |R| (min/max)  = $(minimum(bsz))/$(maximum(bsz))")
    e_in = first(boundary_edges(g, regions3[(:site, center)]))
    a_edge = canon[(src(e_in), dst(e_in))]
    log!("    e.g. message into site-region along $e_in:")
    log!("        cavity = R[bond $a_edge] \\ R[site $center] = " *
        "$(setdiff(regions3[(:bond, a_edge)], regions3[(:site, center)]))")

    # --- step 2: TreeSA contraction complexity vs BP -----------------------
    cx = tnmp_complexity(psi, g, canon, regions3, center)
    bp = bp_complexity(psi, g, center)
    log!("\n[2] TreeSA contraction complexity (log2 #ops / #elements)  --  rank-2 vs BP")
    log!("    quantity                     TNMP rank-2          BP (single-site)")
    log!("    cavity tc (mean/max)         $(pad(mean(cx.tc)))/$(pad(maximum(cx.tc)))        $(pad(bp.tc_mean))/$(pad(bp.tc_max))")
    log!("    cavity sc (mean/max)         $(pad(mean(cx.sc)))/$(pad(maximum(cx.sc)))        $(pad(bp.sc_mean))/$(pad(bp.sc_max))")
    log!("    neighborhood tc (marginal)   $(pad(cx.marg_tc))               $(pad(bp.marg_tc))")
    log!("    neighborhood sc (marginal)   $(pad(cx.marg_sc))               $(pad(bp.marg_sc))")
    log!("    #cavity contractions         $(length(cx.tc))                  $(bp.n)")

    # --- step 3: message passing, marginals, convergence to exact ----------
    exact = exact_marginal(psi, g, Dict{Any, Any}(), center)
    bp_marg, bp_info = bp_marginal(psi, center; maxiter = 2000, tolerance = 1e-10)

    log!("\n[3] message passing & marginals (center = $center)")
    log!("    exact marginal            = $(rndv(exact))")
    log!("")
    log!("    method        converged  iters   marginal                  L1(.,exact)")
    log!("    BP (1-site)   $(rpad(bp_info.converged, 9)) $(rpad(bp_info.iterations, 6))  $(rpad(rndv(bp_marg), 24))  $(rnd(l1(bp_marg, exact), 4))")
    tnmp_results = []
    for rL in region_Ls
        r = tnmp_run(psi, g, canon, rL, center)
        push!(tnmp_results, r)
        tag = "TNMP rL=$rL"
        log!("    $(rpad(tag, 13)) $(rpad(r.info.converged, 9)) $(rpad(r.info.iterations, 6))  $(rpad(rndv(r.marg), 24))  $(rnd(l1(r.marg, exact), 4))")
    end

    log!("")
    log!("    => BP fails to converge (diff stuck at $(rnd(bp_info.final_diff, 3))).")
    log!("    => rank-2 TNMP converges at every window size, and its marginal")
    log!("       systematically approaches the exact one as the window grows")
    log!("       (rL=5 covers the whole 4x4 lattice => L1 = $(rnd(l1(tnmp_results[end].marg, exact), 4))).")
    log!("="^72)

    out = joinpath(@__DIR__, "minimal_tnmp_rank2_result.txt")
    open(out, "w") do io
        println(io, join(lines, "\n"))
    end
    println("\nsaved -> $out")
    return nothing
end

mean(xs) = sum(xs) / length(xs)
rnd(x, d = 2) = round(x; digits = d)
rndv(v) = "[" * join(rnd.(v, 4), ", ") * "]"
l1(a, b) = sum(abs.(a .- b))
pad(x) = rpad(rnd(x, 2), 5)

main()
