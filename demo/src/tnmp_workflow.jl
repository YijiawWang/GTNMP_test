# Rank-2 TNMP workflow library for the `minimal_tnmp_rank2.ipynb` demo.
#
# All the *machinery* of the demo lives here so the notebook only has to
# `include` this file and call into it:
#
#   * the fully-frustrated model builder           -> `build_demo_model`
#   * double-layer factors / messages / windows    -> `dl_factors`, `default_message`,
#                                                      `grid_window`, `boundary_edges`,
#                                                      `build_regions`
#   * TreeSA contraction planning + complexity      -> `treesa_plan`, `contract_cached!`
#   * rank-2 TNMP cavity / message / marginal       -> `message_tensors`, `compute_message`,
#                                                      `run_message_passing!`, `tnmp_marginal`,
#                                                      `exact_marginal`, `tnmp_run`, `tnmp_complexity`
#   * BP baseline (called, never re-implemented)    -> `bp_marginal`, `bp_complexity`
#
# Reporting / visualisation helpers used by the notebook:
#   * `region_contents`  -> the sites / internal bonds / boundary bonds of a region
#   * `describe_region`  -> pretty-print which tensors and bonds a region contains
#   * `einsum_equation`  -> the einsum string of a sub-TN contraction
#   * `draw_region`      -> a Luxor figure of the LxL lattice (region = filled circle +
#                           solid bonds, outside = hollow circle + dashed bonds)

# --- model + BP (reused) -----------------------------------------------------
include(joinpath(@__DIR__, "..", "..", "src", "tnmp.jl"))
using .TNMPTest                                   # only for the TensorNetworkState type + model builder
include(joinpath(@__DIR__, "..", "..", "examples", "state_models.jl"))  # fully_frustrated_pair_factor_state

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
# 0. the running model
# =============================================================================

# Build the fully-frustrated L x L Ising double layer used throughout the demo.
# Returns everything the notebook needs as a NamedTuple.
function build_demo_model(; L = 4, chi = 2, seed = 7, K = 1.0, field = 0.2)
    center = ((L + 1) ÷ 2, (L + 1) ÷ 2)
    g = named_grid((L, L))
    psi = fully_frustrated_pair_factor_state(MersenneTwister(seed), g; K, field, bond_dim = chi)
    canon = Dict{Tuple{Any, Any}, NamedEdge}()
    for e in edges(g)
        canon[(src(e), dst(e))] = e
        canon[(dst(e), src(e))] = e
    end
    return (; L, chi, seed, K, field, center, g, psi, canon)
end

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
# 3. rank-2 TNMP: cavity tensors, message update, marginal
# =============================================================================

normalize_msg(m) = (n = norm(m); iszero(n) ? m : m / n)

get_message(psi, messages, node, e) =
    get(() -> normalize_msg(default_message(psi, e)), messages, (node, e))

# Initial bond-node messages (one normalized rank-2 delta per boundary edge).
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

# The cavity vertices for a message into `center_node` along `in_edge`.
function cavity_vertices(canon, regions, center_node, in_edge)
    a_edge = canon[(src(in_edge), dst(in_edge))]
    return setdiff(regions[(:bond, a_edge)], regions[center_node])
end

# Does the single-layer (graph-level) subgraph induced on `verts` contain a
# cycle? Rank-2 message passing is only faithful when each cavity's sites induce
# a *forest* in g (no loop) — on each layer of the double-layer TN the cavity
# sub-TN must be tree-like. Union-find flags a loop the instant a bond connects
# two sites already in the same component.
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

# Tensors of the closed center-neighborhood sub-TN (center physical leg fixed to
# `s`, default messages on the boundary). Used for the marginal complexity/eq.
function neighborhood_tensors(psi, g, regions, target; s = 1)
    tensors = ITensor[]
    for v in regions[(:site, target)]
        append!(tensors, v == target ? dl_factors_fixed(psi, v, s) : dl_factors(psi, v))
    end
    for e in boundary_edges(g, regions[(:site, target)])
        push!(tensors, default_message(psi, e))
    end
    return tensors
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

# Full rank-2 TNMP run at a given neighborhood size: init bond messages, pass to
# convergence, read the center marginal.
function tnmp_run(psi, g, canon, region_L, center; max_iter = 500, tol = 1e-8)
    regions = build_regions(g, region_L)
    messages = init_messages(psi, g, regions)
    seqs = Dict{Any, Any}()
    info = run_message_passing!(psi, g, canon, regions, messages, seqs; max_iter, tol)
    marg = tnmp_marginal(psi, g, canon, regions, messages, seqs, center)
    return (; regions, info, marg)
end

# TreeSA tc/sc of every rank-2 cavity sub-TN + the neighborhood (marginal) sub-TN.
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

# Pretty-print which double-layer tensors and bonds a region contains.
function describe_region(name, g, region)
    c = region_contents(g, region)
    n_sites = length(c.sites)
    println("$name:  $n_sites sites  ->  $(2 * n_sites) double-layer tensors (ket+bra each)")
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
        region_color = "black", title = "", filename = joinpath(@__DIR__, "region.png"))
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
    # Snapshot PNG bytes once; Luxor's IJulia MIME handler reads the buffer as
    # a (binary-unsafe) String and the manual file write below used to leave an
    # empty buffer, so notebooks showed `"image/png": ""`.
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
