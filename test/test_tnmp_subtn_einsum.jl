# Pin down the sub-tensor-networks that rank-2 TNMP contracts, on a small L = 5
# grid (physical dim D = 2, virtual bond dim χ = 2), by writing the einsum
# equations out *explicitly* and testing them for equivalence against what the
# solver actually assembles.
#
# Three contractions are covered:
#   * the message update for the bond (3,3)-(4,3) -> bond (2,3)-(3,3),
#   * the message update for the bond (1,3)-(1,4) -> bond (1,2)-(2,2),
#   * the marginal readout at the grid center (3,3).
#
# Index notation used in the explicit equations (stable, lattice-based):
#   * P(x,y)            : physical (trace) leg of site (x,y)  -- ket/bra shared
#   * K(a,b)-(c,d)      : ket-layer virtual leg of bond (a,b)-(c,d)
#   * B(a,b)-(c,d)      : bra-layer virtual leg of bond (a,b)-(c,d)
# Edges are written endpoint-sorted, so orientation never matters. Two einsum
# equations are "equivalent" iff they describe the same contraction hypergraph:
# the same multiset of operands (each operand identified by its set of leg
# labels) and the same set of open output legs. Operand order and the arbitrary
# einsum symbol names are irrelevant.
#
# This file is guarded so it runs both standalone and from runtests.jl.

if !isdefined(@__MODULE__, :TNMPTest)
    include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))
    using .TNMPTest
end
if !isdefined(@__MODULE__, :random_state)
    include(joinpath(@__DIR__, "..", "examples", "state_models.jl"))
end
if !isdefined(@__MODULE__, :grid_region_fn)
    include(joinpath(@__DIR__, "..", "examples", "neighborhoods.jl"))
end

using NamedGraphs: NamedEdge, src, dst, edges, vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister
using Test: @test, @testset
import ITensors

# --- canonical lattice labels for an einsum leg ------------------------------

_vlabel(v) = "($(v[1]),$(v[2]))"
function _elabel(e)
    a, b = src(e), dst(e)
    lo, hi = a <= b ? (a, b) : (b, a)
    return "$(_vlabel(lo))-$(_vlabel(hi))"
end

# Maps from a (prime-0) bond index -> its graph edge and a physical index -> its
# vertex, so any ITensors `Index` can be turned into a lattice label.
function _lattice_maps(psi)
    g = TNMPTest.graph(psi)
    bondedge = Dict{ITensors.Index, Any}()
    for e in edges(g)
        for ind in TNMPTest.virtualinds(psi, e)
            bondedge[ITensors.noprime(ind)] = e
        end
    end
    physvertex = Dict{ITensors.Index, Any}()
    for v in vertices(g)
        physvertex[only(TNMPTest.siteinds(psi, v))] = v
    end
    return bondedge, physvertex
end

function _leg_label(ind, bondedge, physvertex)
    if ITensors.hastags(ind, "phys")
        return "P$(_vlabel(physvertex[ITensors.noprime(ind)]))"
    end
    e = bondedge[ITensors.noprime(ind)]
    layer = ITensors.plev(ind) == 0 ? "K" : "B"
    return "$(layer)$(_elabel(e))"
end

# Indices that appear exactly once across all operands stay open (the output).
function _open_indices(tensors)
    counts = Dict{ITensors.Index, Int}()
    for t in tensors, ind in ITensors.inds(t)
        counts[ind] = get(counts, ind, 0) + 1
    end
    return [ind for (ind, c) in counts if c == 1]
end

# --- einsum-equation normal form (operand-order / symbol independent) --------

# Canonical signature: (sorted list of sorted-operand-leg-lists, sorted outputs).
const EinsumSig = Tuple{Vector{Vector{String}}, Vector{String}}

function computed_signature(tensors, psi)::EinsumSig
    bondedge, physvertex = _lattice_maps(psi)
    operands = Vector{String}[
        sort(String[_leg_label(ind, bondedge, physvertex) for ind in ITensors.inds(t)])
        for t in tensors
    ]
    outs = sort(String[_leg_label(ind, bondedge, physvertex) for ind in _open_indices(tensors)])
    return (sort(operands), outs)
end

# Parse an explicit einsum string. One operand per line (a trailing comma is
# tolerated), legs inside an operand are whitespace-separated, and the line
# beginning with `->` lists the open legs (none == a scalar). Leg labels may
# themselves contain commas (e.g. `K(4,1)-(4,2)`), so operands are delimited by
# newlines rather than commas.
function parse_einsum(eq::AbstractString)::EinsumSig
    operands = Vector{String}[]
    outs = String[]
    for raw in split(eq, '\n')
        line = strip(raw)
        isempty(line) && continue
        if startswith(line, "->")
            outs = sort(String.(split(strip(line[3:end]))))
        else
            push!(operands, sort(String.(split(rstrip(line, ',')))))
        end
    end
    return (sort(operands), outs)
end

function _print_signature(io, sig::EinsumSig)
    ops, outs = sig
    println(io, "  ", join([join(o, " ") for o in ops], ",\n  "))
    println(io, "  -> ", isempty(outs) ? "(scalar)" : join(outs, " "))
end

# Compare the explicit equation against the assembled one; print a readable diff
# on mismatch. Returns the boolean for `@test`.
function einsum_equivalent(title, expected_eq, tensors, psi)
    expected = parse_einsum(expected_eq)
    computed = computed_signature(tensors, psi)
    ok = expected == computed
    println("\n", "="^78)
    println(title, ok ? "   [equivalent]" : "   [MISMATCH]")
    println("="^78)
    println("explicit einsum equation:")
    _print_signature(stdout, expected)
    if !ok
        exp_ops, exp_out = expected
        got_ops, got_out = computed
        println("operands only in explicit eq: ", setdiff(exp_ops, got_ops))
        println("operands only in computed eq: ", setdiff(got_ops, exp_ops))
        exp_out == got_out || println("outputs differ: explicit=$(exp_out) computed=$(got_out)")
    end
    return ok
end

# --- locate the message-passing key (center bond node + incoming edge) -------
#
# A rank-2 message "src_bond -> recv_bond" is stored under the key
# `(center_node, in_edge)` where `center_node = (:bond, recv_bond)` and `in_edge`
# is the boundary edge of the receiver's region whose bond node is `src_bond`.
function message_tensors_for(cache, recv_pair, src_pair)
    recv_edge = cache.canonical[recv_pair]
    src_edge = cache.canonical[src_pair]
    center_node = (:bond, recv_edge)
    g = TNMPTest.graph(cache.network)
    for e in TNMPTest.incoming_boundary_edges(g, cache.regions[center_node])
        if cache.canonical[(src(e), dst(e))] == src_edge
            cavity = TNMPTest.cavity_vertices(cache, (:bond, src_edge), center_node)
            return TNMPTest.message_tensors(cache, center_node, e), cavity
        end
    end
    error("no incoming boundary edge from bond $(src_pair) into region of bond $(recv_pair)")
end

# --- explicitly written-out einsum equations ---------------------------------

# Message: bond (3,3)-(4,3) -> bond (2,3)-(3,3). Cavity = the x = 4 column that
# the (3,3)-(4,3) window adds over the receiving (2,3)-(3,3) window.
const EQ_MESSAGE_1 = """
P(4,2) K(4,1)-(4,2) K(3,2)-(4,2) K(4,2)-(5,2) K(4,2)-(4,3),
P(4,2) B(4,1)-(4,2) B(3,2)-(4,2) B(4,2)-(5,2) B(4,2)-(4,3),
P(4,3) K(4,2)-(4,3) K(3,3)-(4,3) K(4,3)-(5,3) K(4,3)-(4,4),
P(4,3) B(4,2)-(4,3) B(3,3)-(4,3) B(4,3)-(5,3) B(4,3)-(4,4),
P(4,4) K(4,3)-(4,4) K(3,4)-(4,4) K(4,4)-(5,4) K(4,4)-(4,5),
P(4,4) B(4,3)-(4,4) B(3,4)-(4,4) B(4,4)-(5,4) B(4,4)-(4,5),
K(4,1)-(4,2) B(4,1)-(4,2),
K(3,2)-(4,2) B(3,2)-(4,2),
K(4,2)-(5,2) B(4,2)-(5,2),
K(4,3)-(5,3) B(4,3)-(5,3),
K(3,4)-(4,4) B(3,4)-(4,4),
K(4,4)-(5,4) B(4,4)-(5,4),
K(4,4)-(4,5) B(4,4)-(4,5),
-> K(3,3)-(4,3) B(3,3)-(4,3)
"""

# Message: bond (1,3)-(1,4) -> bond (1,2)-(2,2). Cavity = {(1,4),(2,4)}.
const EQ_MESSAGE_2 = """
P(1,4) K(1,3)-(1,4) K(1,4)-(2,4) K(1,4)-(1,5),
P(1,4) B(1,3)-(1,4) B(1,4)-(2,4) B(1,4)-(1,5),
P(2,4) K(2,3)-(2,4) K(1,4)-(2,4) K(2,4)-(3,4) K(2,4)-(2,5),
P(2,4) B(2,3)-(2,4) B(1,4)-(2,4) B(2,4)-(3,4) B(2,4)-(2,5),
K(2,3)-(2,4) B(2,3)-(2,4),
K(1,4)-(1,5) B(1,4)-(1,5),
K(2,4)-(3,4) B(2,4)-(3,4),
K(2,4)-(2,5) B(2,4)-(2,5),
-> K(1,3)-(1,4) B(1,3)-(1,4)
"""

# Marginal at center (3,3): the full {2,3,4}×{2,3,4} window. The center (3,3)
# carries no physical leg (it is projected onto a fixed state), and the network
# is fully closed (scalar).
const EQ_MARGINAL = """
P(2,2) K(2,1)-(2,2) K(1,2)-(2,2) K(2,2)-(3,2) K(2,2)-(2,3),
P(2,2) B(2,1)-(2,2) B(1,2)-(2,2) B(2,2)-(3,2) B(2,2)-(2,3),
P(3,2) K(3,1)-(3,2) K(2,2)-(3,2) K(3,2)-(4,2) K(3,2)-(3,3),
P(3,2) B(3,1)-(3,2) B(2,2)-(3,2) B(3,2)-(4,2) B(3,2)-(3,3),
P(4,2) K(4,1)-(4,2) K(3,2)-(4,2) K(4,2)-(5,2) K(4,2)-(4,3),
P(4,2) B(4,1)-(4,2) B(3,2)-(4,2) B(4,2)-(5,2) B(4,2)-(4,3),
P(2,3) K(2,2)-(2,3) K(1,3)-(2,3) K(2,3)-(3,3) K(2,3)-(2,4),
P(2,3) B(2,2)-(2,3) B(1,3)-(2,3) B(2,3)-(3,3) B(2,3)-(2,4),
K(3,2)-(3,3) K(2,3)-(3,3) K(3,3)-(4,3) K(3,3)-(3,4),
B(3,2)-(3,3) B(2,3)-(3,3) B(3,3)-(4,3) B(3,3)-(3,4),
P(4,3) K(4,2)-(4,3) K(3,3)-(4,3) K(4,3)-(5,3) K(4,3)-(4,4),
P(4,3) B(4,2)-(4,3) B(3,3)-(4,3) B(4,3)-(5,3) B(4,3)-(4,4),
P(2,4) K(2,3)-(2,4) K(1,4)-(2,4) K(2,4)-(3,4) K(2,4)-(2,5),
P(2,4) B(2,3)-(2,4) B(1,4)-(2,4) B(2,4)-(3,4) B(2,4)-(2,5),
P(3,4) K(3,3)-(3,4) K(2,4)-(3,4) K(3,4)-(4,4) K(3,4)-(3,5),
P(3,4) B(3,3)-(3,4) B(2,4)-(3,4) B(3,4)-(4,4) B(3,4)-(3,5),
P(4,4) K(4,3)-(4,4) K(3,4)-(4,4) K(4,4)-(5,4) K(4,4)-(4,5),
P(4,4) B(4,3)-(4,4) B(3,4)-(4,4) B(4,4)-(5,4) B(4,4)-(4,5),
K(2,1)-(2,2) B(2,1)-(2,2),
K(3,1)-(3,2) B(3,1)-(3,2),
K(4,1)-(4,2) B(4,1)-(4,2),
K(1,2)-(2,2) B(1,2)-(2,2),
K(4,2)-(5,2) B(4,2)-(5,2),
K(1,3)-(2,3) B(1,3)-(2,3),
K(4,3)-(5,3) B(4,3)-(5,3),
K(1,4)-(2,4) B(1,4)-(2,4),
K(2,4)-(2,5) B(2,4)-(2,5),
K(3,4)-(3,5) B(3,4)-(3,5),
K(4,4)-(5,4) B(4,4)-(5,4),
K(4,4)-(4,5) B(4,4)-(4,5),
->
"""

@testset "rank-2 TNMP sub-TN einsum equations (L=5, χ=2, D=2)" begin
    g = named_grid((5, 5))
    center = (3, 3)
    rng = MersenneTwister(20260629)
    psi = random_state(rng, g; physical_dim = 2, bond_dim = 2)
    cache = TNMPCache(psi, 3; region_fn = grid_region_fn(3))

    @testset "message update: bond (3,3)-(4,3) -> bond (2,3)-(3,3)" begin
        tensors, cavity = message_tensors_for(cache, ((2, 3), (3, 3)), ((3, 3), (4, 3)))
        @test Set(cavity) == Set(Any[(4, 2), (4, 3), (4, 4)])
        @test einsum_equivalent(
            "MESSAGE  bond (3,3)-(4,3)  ->  bond (2,3)-(3,3)", EQ_MESSAGE_1, tensors, psi,
        )
    end

    @testset "message update: bond (1,3)-(1,4) -> bond (1,2)-(2,2)" begin
        tensors, cavity = message_tensors_for(cache, ((1, 2), (2, 2)), ((1, 3), (1, 4)))
        @test Set(cavity) == Set(Any[(1, 4), (2, 4)])
        @test einsum_equivalent(
            "MESSAGE  bond (1,3)-(1,4)  ->  bond (1,2)-(2,2)", EQ_MESSAGE_2, tensors, psi,
        )
    end

    @testset "marginal readout at center (3,3)" begin
        region = cache.regions[(:site, center)]
        @test Set(region) == Set(Any[(x, y) for x in 2:4 for y in 2:4])
        tensors = TNMPTest.marginal_tensors(cache, center, 1)
        @test einsum_equivalent(
            "MARGINAL  center site (3,3)  (state = 1)", EQ_MARGINAL, tensors, psi,
        )
    end
end
