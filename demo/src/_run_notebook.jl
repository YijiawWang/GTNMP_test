cd(@__DIR__)
# --- cell 2 ---
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

using Dictionaries: Dictionary
using LinearAlgebra: diag
using TensorNetworkQuantumSimulator

ITensors.disable_warn_order()
println("__CELL_2_END__")

# --- cell 4 ---
mean(xs) = sum(xs) / length(xs)
rnd(x, d = 2) = round(x; digits = d)
rndv(v) = "[" * join(rnd.(v, 4), ", ") * "]"
l1(a, b) = sum(abs.(a .- b))
pad(x) = rpad(rnd(x, 2), 5)

# --- the minimal model -------------------------------------------------------
L, chi, seed, K, field = 4, 2, 7, 1.0, 0.2
center = ((L + 1) ÷ 2, (L + 1) ÷ 2)
g = named_grid((L, L))
psi = fully_frustrated_pair_factor_state(MersenneTwister(seed), g; K, field, bond_dim = chi)
canon = Dict{Tuple{Any, Any}, NamedEdge}()
for e in edges(g)
    canon[(src(e), dst(e))] = e
    canon[(dst(e), src(e))] = e
end
println("fully-frustrated model:  L=$L  chi=$chi  K=$K  field=$field  seed=$seed")
println("grid = $L x $L  ($(length(collect(vertices(g)))) sites, $(length(collect(edges(g)))) bonds)   center = $center")
println("__CELL_4_END__")

# --- cell 6 ---
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

# Assign a neighborhood region to every site/bond node of the graph.
build_regions(g, region_L) = merge(
    Dict{Any, Vector{Any}}((:site, v) => grid_window(g, (:site, v), region_L) for v in vertices(g)),
    Dict{Any, Vector{Any}}((:bond, e) => grid_window(g, (:bond, e), region_L) for e in edges(g)),
)
println("__CELL_6_END__")

# --- cell 8 ---
regions = build_regions(g, 3)
println("site-region |R| (center)  = $(length(regions[(:site, center)]))   (3x3 window)")
bsz = [length(regions[(:bond, e)]) for e in edges(g)]
println("bond-region |R| (min/max) = $(minimum(bsz))/$(maximum(bsz))\n")

# ASCII schematic of the LxL lattice: `glyph(v)` returns each vertex's character.
function draw_lattice(glyph; legend = "")
    for x in 1:L
        println("   " * join((string(glyph((x, y))) for y in 1:L), "---"))
        x < L && println("   " * join(fill("|", L), "   "))
    end
    isempty(legend) || println("\n   " * legend)
end

reg = Set(regions[(:site, center)])
println("neighborhood  R[site $center]  (3x3 window):")
draw_lattice(v -> v == center ? 'C' : (v in reg ? 'o' : '.');
    legend = "C = center,   o = R[site] neighborhood,   . = outside")

e_in   = first(boundary_edges(g, regions[(:site, center)]))   # one incoming message edge
a_edge = canon[(src(e_in), dst(e_in))]                        # the bond it lives on (the source)
cavity = setdiff(regions[(:bond, a_edge)], regions[(:site, center)])
cav    = Set(cavity)
println("\ncavity for the message into $center along in_edge = $e_in :")
println("   cavity = R[bond $a_edge] \\ R[site $center] = $(sort(cavity))\n")
draw_lattice(v -> v == center ? 'C' : (v in cav ? '*' : (v in reg ? 'o' : '.'));
    legend = "C = center,   o = receiver R[site],   * = cavity = R[bond] \\ R[site],   . = outside")

println("\nboundary edges into R[site $center] (outside -> inside) = the incoming messages:")
for e in boundary_edges(g, regions[(:site, center)])
    println("   $e")
end
println("__CELL_8_END__")

# --- cell 10 ---
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
println("__CELL_10_END__")

# --- cell 12 ---
nbhd = ITensor[]
for v in regions[(:site, center)]
    append!(nbhd, v == center ? dl_factors_fixed(psi, v, 1) : dl_factors(psi, v))
end
for e in boundary_edges(g, regions[(:site, center)])
    push!(nbhd, default_message(psi, e))
end
plan = treesa_plan(nbhd)
println("center neighborhood sub-TN:  $(length(nbhd)) tensors")
println("   TreeSA  tc = $(rnd(plan.tc))  (log2 #ops),   sc = $(rnd(plan.sc))  (log2 max intermediate)")
println("__CELL_12_END__")

# --- cell 14 ---
normalize_msg(m) = (n = norm(m); iszero(n) ? m : m / n)

get_message(psi, messages, node, e) =
    get(() -> normalize_msg(default_message(psi, e)), messages, (node, e))

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

# --- thin drivers ------------------------------------------------------------
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
println("__CELL_14_END__")

# --- cell 16 ---
# (a) the cavity sub-TN for one incoming message, planned with TreeSA
seqs     = Dict{Any, Any}()
messages = Dict{Tuple{Any, NamedEdge}, ITensor}()
for e0 in edges(g), e in boundary_edges(g, regions[(:bond, e0)])
    messages[((:bond, e0), e)] = normalize_msg(default_message(psi, e))
end
cav_ts = message_tensors(psi, g, canon, regions, messages, (:site, center), e_in)
cplan  = treesa_plan(cav_ts)
println("cavity sub-TN for $e_in -> $center :  $(length(cav_ts)) tensors,  tc = $(rnd(cplan.tc)),  sc = $(rnd(cplan.sc))")

# (b) run rank-2 message passing to a fixed point, then read the marginal
info  = run_message_passing!(psi, g, canon, regions, messages, seqs; max_iter = 500, tol = 1e-8)
marg  = tnmp_marginal(psi, g, canon, regions, messages, seqs, center)
exact = exact_marginal(psi, g, Dict{Any, Any}(), center)
println("\nmessage passing (region_L = 3):  converged=$(info.converged)  iters=$(info.iterations)  diff=$(rnd(info.final_diff, 6))")
println("   TNMP marginal  = $(rndv(marg))")
println("   exact marginal = $(rndv(exact))    (L1 = $(rnd(l1(marg, exact), 4)))")
println("__CELL_16_END__")

# --- cell 18 ---
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
println("__CELL_18_END__")

# --- cell 20 ---
bp_marg, bp_info = bp_marginal(psi, center; maxiter = 2000, tolerance = 1e-10)
println("BP:  converged=$(bp_info.converged)  iters=$(bp_info.iterations)  diff=$(rnd(bp_info.final_diff, 4))")
println("   BP marginal    = $(rndv(bp_marg))    (L1 = $(rnd(l1(bp_marg, exact), 4)))")
println("   exact marginal = $(rndv(exact))\n")

cx = tnmp_complexity(psi, g, canon, regions, center)
bp = bp_complexity(psi, g, center)
println("TreeSA complexity (log2)     TNMP rank-2          BP (single-site)")
println("cavity tc (mean/max)         $(pad(mean(cx.tc)))/$(pad(maximum(cx.tc)))        $(pad(bp.tc_mean))/$(pad(bp.tc_max))")
println("cavity sc (mean/max)         $(pad(mean(cx.sc)))/$(pad(maximum(cx.sc)))        $(pad(bp.sc_mean))/$(pad(bp.sc_max))")
println("neighborhood tc (marginal)   $(pad(cx.marg_tc))               $(pad(bp.marg_tc))")
println("neighborhood sc (marginal)   $(pad(cx.marg_sc))               $(pad(bp.marg_sc))")
println("#cavity contractions         $(length(cx.tc))                  $(bp.n)")
println("__CELL_20_END__")

# --- cell 22 ---
println("exact marginal            = $(rndv(exact))\n")
println("method        converged  iters   marginal                  L1(.,exact)")
println("BP (1-site)   $(rpad(bp_info.converged, 9)) $(rpad(bp_info.iterations, 6))  $(rpad(rndv(bp_marg), 24))  $(rnd(l1(bp_marg, exact), 4))")
for rL in [3, 5]
    r = tnmp_run(psi, g, canon, rL, center)
    tag = "TNMP rL=$rL"
    println("$(rpad(tag, 13)) $(rpad(r.info.converged, 9)) $(rpad(r.info.iterations, 6))  $(rpad(rndv(r.marg), 24))  $(rnd(l1(r.marg, exact), 4))")
end
println("__CELL_22_END__")
