# Rank-1 (single-layer) TNMP workflow library for the
# `minimal_tnmp_single_layer.ipynb` demo.
#
# This is the *single-layer* counterpart of `tnmp_workflow.jl`. There the object
# we contract is the double layer ⟨ψ|ψ⟩ = ∑_s |ψ(s)|² (an artificial stress
# test). Here we contract the *single layer* ∑_s ψ(s) directly, which is the
# genuine classical fully-frustrated Ising partition function — so the marginals
# read off are the true classical single-site marginals.
#
# The only structural differences from the rank-2 file are:
#   * each vertex contributes ONE tensor (the spin-summed classical factor),
#     not a (ket, bra) pair;
#   * a message lives on a SINGLE bond index (rank-1 vector), not on a
#     (ket, bra) index pair (rank-2);
#   * the BP baseline runs on a single-layer `TensorNetwork` (one factor per
#     vertex), not on the ⟨ψ|ψ⟩ norm network of a `TensorNetworkState`.
#
# Everything else — neighborhood / cavity construction, TreeSA planning,
# message passing, the convergence metric — is line-for-line the same as the
# rank-2 workflow.
#
# Public surface used by the notebook:
#   * model builder                                 -> `build_demo_model`
#   * single-layer factors / messages / windows     -> `sl_factors`, `default_message`,
#                                                       `grid_window`, `boundary_edges`,
#                                                       `build_regions`
#   * TreeSA contraction planning + complexity       -> `treesa_plan`, `contract_cached!`
#   * rank-1 TNMP cavity / message / marginal        -> `message_tensors`, `compute_message`,
#                                                       `run_message_passing!`, `tnmp_marginal`,
#                                                       `exact_marginal`, `tnmp_run`, `tnmp_complexity`
#   * single-layer BP baseline (called, never re-implemented)
#                                                    -> `bp_marginal`, `bp_complexity`
# Reporting / visualisation helpers (shared with the rank-2 demo):
#   `region_contents`, `describe_region`, `einsum_equation`, `draw_region`.

# --- model + BP (reused) -----------------------------------------------------
include(joinpath(@__DIR__, "..", "..", "src", "tnmp.jl"))
using .TNMPTest                                   # only for the TensorNetworkState type + model helpers
include(joinpath(@__DIR__, "..", "..", "examples", "state_models.jl"))  # signed_pair_features, fully_frustrated_square_couplings, ...

using NamedGraphs: NamedGraph, NamedEdge, edges, vertices, src, dst, neighbors
using NamedGraphs.NamedGraphGenerators: named_grid
using OMEinsumContractionOrders: EinCode, optimize_code, TreeSA, contraction_complexity,
    NestedEinsum, SlicedEinsum, getixsv, getiyv
using Random: MersenneTwister
using ITensors: ITensor, Index, inds, dim, dag, prime, delta, onehot, commoninds,
    replaceinds, norm, dot
import ITensors

# `state_models.jl` aliases TensorNetworkState to TNMPTest's; keep that alias
# winning over TensorNetworkQuantumSimulator's by `using` the package last.
using Dictionaries: Dictionary
using LinearAlgebra: diag
using TensorNetworkQuantumSimulator

using Luxor

ITensors.disable_warn_order()

# --- formatting helpers ------------------------------------------------------
mean(xs) = sum(xs) / length(xs)
rnd(x, d = 2) = round(x; digits = d)
rndv(v) = "[" * join(rnd.(v, 4), ", ") * "]"
l1(a, b) = sum(abs.(a .- b))
pad(x) = rpad(rnd(x, 2), 5)

# Human-readable site / bond labels for printing (avoid NamedEdge's `=>` syntax).
site_label(v) = "($(v[1]), $(v[2]))"
bond_label(u, v) = "$(site_label(u))-$(site_label(v))"
bond_label(e::NamedEdge) = bond_label(src(e), dst(e))
bond_labels(edges) = join(bond_label.(edges), ", ")

# =============================================================================
# 0. the running model  (classical fully-frustrated Ising, single layer)
# =============================================================================

# Single-layer classical fully-frustrated Ising network on the LxL grid.
#
# This is a *standalone* classical model, unrelated to the rank-2 demo. We
# factorise each 2x2 Ising transfer matrix across its bond at *full* strength K,
#   W_J(s, t) = exp(K · J_e · z(s) z(t)) = ∑_{b=1}^{χ} F_s[b] F_t[b],
# (`signed_pair_features`, exact even though F is complex for frustrated bonds),
# and put a local bias a(s) = exp(±h·z(s)) on each site. The amplitude
#   ψ(s) = ∏_v a(s_v) ∏_{e=(u,v)} W_{J_e}(s_u, s_v)
# is then exactly the classical fully-frustrated Ising Boltzmann weight at the
# parameters (K, h) you pass in — contracting the single layer ∑_s ψ(s) gives the
# partition function Z, and fixing one spin gives its exact classical marginal.
function single_layer_ff_ising_state(rng, g; K = 1.0, field = 0.2, bond_dim = 2)
    couplings = fully_frustrated_square_couplings(g)
    edge_features = Dict{Any, Matrix{ComplexF64}}()
    for e in edges(g)
        features = signed_pair_features(Float64(K), couplings[e], bond_dim)  # full K
        edge_features[e] = features
        edge_features[perturbed_product_reverse_edge(e)] = features
    end
    # state 1 -> z=-1, state 2 -> z=+1 ; full field h.
    h = Float64(field)
    local_amplitudes = [exp(-h), exp(h)]
    tensors, siteinds_dict = build_edge_feature_peps_tensors(
        g; local_amplitudes, edge_features, physical_dim = 2, bond_dim,
    )
    return TensorNetworkState(tensors, siteinds_dict, g)
end

# Build the fully-frustrated LxL classical Ising single layer used throughout
# the demo. Returns everything the notebook needs as a NamedTuple.
function build_demo_model(; L = 4, chi = 2, seed = 7, K = 1.2, field = 0.1)
    center = ((L + 1) ÷ 2, (L + 1) ÷ 2)
    g = named_grid((L, L))
    psi = single_layer_ff_ising_state(MersenneTwister(seed), g; K, field, bond_dim = chi)
    canon = Dict{Tuple{Any, Any}, NamedEdge}()
    for e in edges(g)
        canon[(src(e), dst(e))] = e
        canon[(dst(e), src(e))] = e
    end
    return (; L, chi, seed, K, field, center, g, psi, canon)
end

# =============================================================================
# 1. single-layer factors + neighborhoods (the only data we ever build)
# =============================================================================

# The all-ones vector on an index `i` (the rank-1 copy/trace tensor): contracting
# it with the physical leg sums the classical factor over the spin.
ones_index(i::Index) = ITensor(ones(Float64, dim(i)), i)

# Single-layer factor at a vertex: the classical site tensor with its physical
# spin SUMMED, leaving only the bond legs. The rank-2 demo returns a (ket, bra)
# pair here; the single layer returns just one tensor.
sl_factors(psi, v) = ITensor[psi[v] * ones_index(only(psi.siteinds[v]))]

# Same, but with the physical leg projected onto a fixed spin `s` (marginal).
sl_factors_fixed(psi, v, s::Integer) = ITensor[psi[v] * onehot(only(psi.siteinds[v]) => s)]

# A rank-1 message lives on a bond: the single (shared) bond index of that edge.
function default_message(psi, e::NamedEdge)
    b = commoninds(psi[src(e)], psi[dst(e)])
    return delta(b...)
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

# =============================================================================
# 2. TreeSA contraction planning + complexity (tc/sc) + einsum equation
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

# Excel-style label for an integer: 1->a, 26->z, 27->aa, ...
function _index_label(n::Int)
    s = ""
    while n > 0
        n, r = divrem(n - 1, 26)
        s = string('a' + r) * s
    end
    return s
end

# Human-readable einsum equation of the sub-TN spanned by `tensors`, e.g.
# "ab,bc,cd -> ad".  Each distinct ITensor `Index` is mapped to a short label;
# open (uncontracted) indices form the right-hand side.
function einsum_equation(tensors)
    isempty(tensors) && return "(empty)"
    code, _ = eincode(tensors)
    ixs, iy = getixsv(code), getiyv(code)
    labels = Dict{Any, String}()
    n = Ref(0)
    lab(i) = get!(() -> _index_label(n[] += 1), labels, i)
    lhs = join((isempty(ix) ? "1" : join(lab(i) for i in ix) for ix in ixs), ",")
    rhs = join((lab(i) for i in iy))
    return "$lhs -> $rhs"
end

# =============================================================================
# 3. rank-1 TNMP: cavity tensors, message update, marginal
# =============================================================================

normalize_msg(m) = (n = norm(m); iszero(n) ? m : m / n)

get_message(psi, messages, node, e) =
    get(() -> normalize_msg(default_message(psi, e)), messages, (node, e))

# Initial bond-node messages (one normalized rank-1 delta per boundary edge).
function init_messages(psi, g, regions)
    messages = Dict{Tuple{Any, NamedEdge}, ITensor}()
    for e0 in edges(g), e in boundary_edges(g, regions[(:bond, e0)])
        messages[((:bond, e0), e)] = normalize_msg(default_message(psi, e))
    end
    return messages
end

# Sub-TN whose contraction produces the message flowing along `in_edge` into the
# region of `center_node`. `in_edge` sits on a bond whose own (larger) region is
# the source; the cavity is exactly the vertices the source region adds over the
# receiver region. The open leg is the (single) bond index of `in_edge`.
function message_tensors(psi, g, canon, regions, messages, center_node, in_edge)
    a_edge = canon[(src(in_edge), dst(in_edge))]
    a_node = (:bond, a_edge)
    cavity = setdiff(regions[a_node], regions[center_node])
    isempty(cavity) && return ITensor[]
    tensors = ITensor[]
    for v in cavity
        append!(tensors, sl_factors(psi, v))
    end
    for e in boundary_edges(g, cavity)
        canon[(src(e), dst(e))] == a_edge && continue   # leave the open edge open
        push!(tensors, get_message(psi, messages, a_node, e))
    end
    return tensors
end

# The cavity vertices for a message into `center_node` along `in_edge`.
function cavity_vertices(canon, regions, center_node, in_edge)
    a_edge = canon[(src(in_edge), dst(in_edge))]
    return setdiff(regions[(:bond, a_edge)], regions[center_node])
end

# Does the (graph-level) subgraph induced on `verts` contain a cycle? Rank-1
# message passing is only faithful when each cavity's sites induce a *forest* in
# g (no loop) — the cavity sub-TN must be tree-like. Union-find flags a loop the
# instant a bond connects two sites already in the same component.
function cavity_has_loop(g, verts)
    vs = Set(verts)
    parent = Dict{Any, Any}(v => v for v in vs)
    function root(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    for e in edges(g)
        u, v = src(e), dst(e)
        (u in vs && v in vs) || continue
        ru, rv = root(u), root(v)
        ru == rv && return true            # this bond closes a cycle
        parent[ru] = rv
    end
    return false
end

# Verify that every cavity ever contracted is loop-free at the graph level. The
# checked set mirrors `tnmp_complexity`: one cavity per (bond region, boundary
# edge) used in message passing, plus the center site region's marginal
# cavities. Returns how many cavities were checked / empty, the offending
# (receiver-node, in-edge) keys, and `ok = isempty(looped)`.
function verify_cavities_acyclic(g, canon, regions, center)
    keys = Tuple{Any, NamedEdge}[]
    for e0 in edges(g), e in boundary_edges(g, regions[(:bond, e0)])
        push!(keys, ((:bond, e0), e))
    end
    for e in boundary_edges(g, regions[(:site, center)])
        push!(keys, ((:site, center), e))
    end
    looped, n_checked, n_empty = Tuple{Any, NamedEdge}[], 0, 0
    for (node, e) in keys
        cav = cavity_vertices(canon, regions, node, e)
        if isempty(cav)
            n_empty += 1
            continue
        end
        n_checked += 1
        cavity_has_loop(g, cav) && push!(looped, (node, e))
    end
    return (; n_checked, n_empty, looped, ok = isempty(looped))
end

function compute_message(psi, g, canon, regions, messages, seqs, center_node, in_edge)
    ts = message_tensors(psi, g, canon, regions, messages, center_node, in_edge)
    isempty(ts) && return normalize_msg(default_message(psi, in_edge))
    return normalize_msg(contract_cached!(seqs, (center_node, in_edge), ts))
end

# 1 - |<a,b>|^2 normalized fidelity, the message convergence metric.
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
# one pass; then the marginal is the closed neighborhood sub-TN per spin.
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
            append!(tensors, v == target ? sl_factors_fixed(psi, v, s) : sl_factors(psi, v))
        end
        for e in boundary_edges(g, regions[center])
            push!(tensors, messages[(center, e)])
        end
        z = contract_cached!(seqs, (:marginal, target, s), tensors)[]
        push!(weights, abs(real(z)))
    end
    return weights ./ sum(weights)
end

# Tensors of the closed center-neighborhood sub-TN (center spin fixed to `s`,
# default messages on the boundary). Used for the marginal complexity/eq.
function neighborhood_tensors(psi, g, regions, target; s = 1)
    tensors = ITensor[]
    for v in regions[(:site, target)]
        append!(tensors, v == target ? sl_factors_fixed(psi, v, s) : sl_factors(psi, v))
    end
    for e in boundary_edges(g, regions[(:site, target)])
        push!(tensors, default_message(psi, e))
    end
    return tensors
end

# Exact single-site marginal: contract the whole single layer, center fixed.
function exact_marginal(psi, g, seqs, target)
    verts = collect(vertices(g))
    d = dim(only(psi.siteinds[target]))
    weights = Float64[]
    for s in 1:d
        tensors = ITensor[]
        for v in verts
            append!(tensors, v == target ? sl_factors_fixed(psi, v, s) : sl_factors(psi, v))
        end
        z = contract_cached!(seqs, (:exact, target, s), tensors)[]
        push!(weights, abs(real(z)))
    end
    return weights ./ sum(weights)
end

# Full rank-1 TNMP run at a given neighborhood size: init bond messages, pass to
# convergence, read the center marginal.
function tnmp_run(psi, g, canon, region_L, center; max_iter = 500, tol = 1e-8)
    regions = build_regions(g, region_L)
    messages = init_messages(psi, g, regions)
    seqs = Dict{Any, Any}()
    info = run_message_passing!(psi, g, canon, regions, messages, seqs; max_iter, tol)
    marg = tnmp_marginal(psi, g, canon, regions, messages, seqs, center)
    return (; regions, info, marg)
end

# TreeSA tc/sc of every rank-1 cavity sub-TN + the neighborhood (marginal) sub-TN.
function tnmp_complexity(psi, g, canon, regions, center)
    messages = init_messages(psi, g, regions)
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
    n = treesa_plan(neighborhood_tensors(psi, g, regions, center))
    return (; tc, sc, marg_tc = n.tc, marg_sc = n.sc)
end

# =============================================================================
# 4. single-layer BP baseline (called from TensorNetworkQuantumSimulator)
# =============================================================================

# The single-layer network as a TNQS `TensorNetwork` (one spin-summed classical
# factor per vertex). On a `TensorNetwork`, TNQS belief propagation uses
# `bp_factors(tn, v) = [tn[v]]` (single layer) and rank-1 delta messages on the
# bond indices — i.e. textbook BP on the classical model.
to_single_layer_tn(psi) = TensorNetworkQuantumSimulator.TensorNetwork(
    Dictionary(collect(vertices(psi.graph)), [only(sl_factors(psi, v)) for v in vertices(psi.graph)]),
    psi.graph,
)

function bp_marginal(psi, g, target; maxiter, tolerance)
    tn = to_single_layer_tn(psi)
    cache = BeliefPropagationCache(tn)
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
    # Single-site belief: the spin-open target factor closed with its incoming
    # converged rank-1 messages, evaluated per spin.
    incoming = TensorNetworkQuantumSimulator.incoming_messages(updated, [target])
    d = dim(only(psi.siteinds[target]))
    weights = Float64[]
    for s in 1:d
        tensors = ITensor[only(sl_factors_fixed(psi, target, s)); incoming...]
        z = ITensors.contract(tensors)[]
        push!(weights, abs(real(z)))
    end
    return weights ./ sum(weights), (; converged, iterations = iters, final_diff = diff)
end

# BP's single-site sub-TNs (one single-layer site tensor + its other-edge
# messages), for an apples-to-apples tc/sc comparison against TNMP.
function bp_complexity(psi, g, target)
    tcs, scs = Float64[], Float64[]
    for u in vertices(g), v in neighbors(g, u)         # message u -> v
        tensors = sl_factors(psi, u)
        for w in neighbors(g, u)
            w == v && continue
            push!(tensors, default_message(psi, NamedEdge(w => u)))
        end
        p = treesa_plan(tensors)
        push!(tcs, p.tc); push!(scs, p.sc)
    end
    mtensors = sl_factors_fixed(psi, target, 1)         # center marginal
    for w in neighbors(g, target)
        push!(mtensors, default_message(psi, NamedEdge(w => target)))
    end
    mp = treesa_plan(mtensors)
    return (; tc_mean = sum(tcs) / length(tcs), tc_max = maximum(tcs),
        sc_mean = sum(scs) / length(scs), sc_max = maximum(scs),
        marg_tc = mp.tc, marg_sc = mp.sc, n = length(tcs))
end

# =============================================================================
# 5. reporting helpers: what tensors/bonds a region contains
# =============================================================================

# Classify the edges of `g` w.r.t. a region (a collection of vertices):
#   * internal bonds : both endpoints inside  (contracted)
#   * boundary bonds : exactly one endpoint inside (carry messages / open legs)
function region_contents(g, region)
    rs = Set(region)
    sites = sort(collect(rs))
    internal = NamedEdge[]
    boundary = NamedEdge[]
    for e in edges(g)
        u, v = src(e), dst(e)
        both = (u in rs) && (v in rs)
        one = (u in rs) ⊻ (v in rs)
        both && push!(internal, e)
        one && push!(boundary, e)
    end
    return (; sites, internal_bonds = internal, boundary_bonds = boundary)
end

# Pretty-print which single-layer tensors and bonds a region contains.
function describe_region(name, g, region)
    c = region_contents(g, region)
    n_sites = length(c.sites)
    println("$name:  $n_sites sites  ->  $n_sites single-layer tensors (one per site)")
    println("   sites          : ", join(string.(c.sites), ", "))
    println("   internal bonds ($(length(c.internal_bonds)), contracted): ",
        bond_labels(c.internal_bonds))
    println("   boundary bonds ($(length(c.boundary_bonds)), open / messages): ",
        bond_labels(c.boundary_bonds))
    return c
end

# =============================================================================
# 6. visualisation: the LxL lattice with a region highlighted
# =============================================================================

# Draw the LxL grid `g`. Vertices in `region` are filled circles joined by solid
# bonds; vertices outside are hollow circles and any bond touching the outside is
# dashed. `center` (if given) is drawn in red, `cavity` vertices in blue.
# Returns the path written to `filename` (side effect: displays inline in IJulia).
function draw_region(g, L, region; center = nothing, cavity = nothing,
        region_color = "black", title = "", filename = joinpath(@__DIR__, "region_single_layer.png"))
    reg = Set(region)
    cav = cavity === nothing ? Set{Any}() : Set{Any}(cavity)
    spacing = 95.0
    margin = 70.0
    top = 46.0
    W = round(Int, 2margin + (L - 1) * spacing)
    H = round(Int, 2margin + (L - 1) * spacing + top)
    pos(v) = Point(margin + (v[2] - 1) * spacing, top + margin + (v[1] - 1) * spacing)

    d = Drawing(W, H, :png)   # in-memory; we write the file ourselves below
    background("white")
    sethue("black")
    if !isempty(title)
        fontsize(17)
        text(title, Point(W / 2, 24); halign = :center, valign = :middle)
    end

    # bonds first, under the nodes
    setline(2.2)
    for e in edges(g)
        u, v = src(e), dst(e)
        if (u in reg) && (v in reg)
            sethue(region_color); setdash("solid")
        else
            sethue("gray65"); setdash("dashed")
        end
        line(pos(u), pos(v), :stroke)
    end
    setdash("solid")

    # nodes
    r = 19.0
    for v in vertices(g)
        p = pos(v)
        iscenter = center !== nothing && v == center
        if v in reg
            sethue(iscenter ? "firebrick" : (v in cav ? "royalblue" : region_color))
            circle(p, r, :fill)
            sethue("white")
        else
            setline(iscenter ? 3.0 : 2.2)
            sethue(iscenter ? "firebrick" : "black")
            circle(p, r, :stroke)
            sethue(iscenter ? "firebrick" : "black")
        end
        fontsize(12)
        text("$(v[1]),$(v[2])", p; halign = :center, valign = :middle)
    end
    finish()
    # Snapshot PNG bytes once (Luxor's IJulia MIME handler reads the buffer as a
    # binary-unsafe String, so we keep our own copy for the manual file write).
    seekstart(d.buffer)
    png_bytes = read(d.buffer)
    d.buffer = IOBuffer(png_bytes)
    if !isempty(filename)
        open(filename, "w") do io
            write(io, png_bytes)
        end
    end
    # Explicit MIME display is the most reliable path in IJulia / VS Code notebooks.
    if !isempty(png_bytes)
        try
            display(MIME("image/png"), png_bytes)
        catch
            # Non-interactive scripts can ignore display failures.
        end
    end
    return filename
end
