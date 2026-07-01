# Pin down the sub-tensor-networks that GenericMessagePassing's (single-layer)
# TNMP engine contracts, on the same small L = 5 grid used by the rank-2 ITensors
# test (`test/test_tnmp_subtn_einsum.jl`), with the CUSTOM 3x3 / 3x2 neighborhoods
# from `run_tnmp.jl` (region_L = 3 -> R = 1). We write the einsum equations out
# *explicitly* and test them for equivalence against what `tnbp_precompute`
# actually assembles (`eins[(w,v)]` for messages, `mars_eins[v]` for marginals).
#
# Two message updates and one marginal are covered (mirroring the ITensors test):
#   * message  bond (3,3)-(4,3) -> bond (2,3)-(3,3),
#   * message  bond (1,3)-(1,4) -> bond (1,2)-(2,2),
#   * marginal readout at the grid center (3,3).
#
# KEY DIFFERENCE FROM THE RANK-2 ITensors TEST: this is a *single-layer* network
# (one tensor per site, bond dim = physical dim = q), so there is no ket/bra
# doubling. Index notation used in the explicit equations (lattice-based):
#   * P(x,y)        : the (open) physical leg of site (x,y)
#   * (a,b)-(c,d)   : the virtual bond leg of edge (a,b)-(c,d), endpoint-sorted
#
# How the GMP engine builds a message `(w -> v)` (see src/tnmp.jl):
#   The cavity = neighborhood(w) \ neighborhood(v). For each node in the cavity:
#     - a site factor contributes its full site tensor;
#     - a boundary bond that points *away* from v (`boundary_with_message`)
#       contributes an incoming rank-1 message on that single bond leg;
#     - a site factor on the open boundary (`boundary_without_message`)
#       additionally contributes a rank-1 "ones" cap on its dangling leg toward
#       v's region (this traces / leaves-open that leg).
#   The open output leg is the receiving bond `w` itself.
# The marginal at a site contracts its whole 3x3 window: every site tensor in the
# window plus one incoming rank-1 message on each bond leaving the window, with
# the center site's physical leg P(center) left open.
#
# Two einsum equations are "equivalent" iff they describe the same contraction
# hypergraph: the same multiset of operands (each identified by its set of leg
# labels) and the same set of open output legs. Operand order and the arbitrary
# einsum symbol names are irrelevant.

if !isdefined(@__MODULE__, :build_potts_code)
    include(joinpath(@__DIR__, "run_tnmp.jl"))
end

using GenericMessagePassing: TNBPConfig
const GMP = GenericMessagePassing
using OMEinsum: getixsv, getiyv
using Test: @test, @testset

# --- canonical lattice labels for an einsum leg ------------------------------

_vlabel(v) = "($(v[1]),$(v[2]))"
function _blabel(pair)
    a, b = pair
    lo, hi = a <= b ? (a, b) : (b, a)
    return "$(_vlabel(lo))-$(_vlabel(hi))"
end

# Turn an integer einsum label (intcode space) into its stable lattice label,
# using `label_meaning` (orig label -> (:phys,v) / (:bond,(a,b))) and `idict`
# (intcode int -> orig label).
function _leg_label(i, label_meaning, idict)
    m = label_meaning[idict[i]]
    return m[1] === :phys ? "P$(_vlabel(m[2]))" : _blabel(m[2])
end

# --- einsum-equation normal form (operand-order / symbol independent) --------

const EinsumSig = Tuple{Vector{Vector{String}}, Vector{String}}

# Read the assembled (possibly optimized) einsum back as a signature. `getixsv`
# returns the leaf operand index lists, `getiyv` the open output legs.
function computed_signature(eincode, label_meaning, idict)::EinsumSig
    operands = Vector{String}[
        sort(String[_leg_label(i, label_meaning, idict) for i in ix])
        for ix in getixsv(eincode)
    ]
    outs = sort(String[_leg_label(i, label_meaning, idict) for i in getiyv(eincode)])
    return (sort(operands), outs)
end

# Parse an explicit einsum string. One operand per line (a trailing comma is
# tolerated), legs inside an operand are whitespace-separated, and the line
# beginning with `->` lists the open legs (none == a scalar). Leg labels may
# themselves contain commas (e.g. `(4,1)-(4,2)`), so operands are delimited by
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

function einsum_equivalent(title, expected_eq, eincode, label_meaning, idict)
    expected = parse_einsum(expected_eq)
    computed = computed_signature(eincode, label_meaning, idict)
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

# --- locate nodes -----------------------------------------------------------

# intcode int of the bond variable for edge (a,b).
function bond_int(reg, label_meaning, a, b)
    key = minmax(a, b)
    for (lab, m) in label_meaning
        m[1] === :bond && m[2] == key && return reg.lab2int[lab]
    end
    error("bond $key not found")
end

# The cavity (neighborhood(w) \ neighborhood(v)) site coordinates of message w->v.
function cavity_sites(reg, factor_coord, w, v)
    local_region = setdiff(reg.neibs[w], reg.neibs[v])
    return Set(factor_coord[u - reg.nvars] for u in local_region if u > reg.nvars)
end

# Site coordinates of a variable's neighborhood window.
function window_sites(reg, factor_coord, v)
    return Set(factor_coord[u - reg.nvars] for u in reg.neibs[v] if u > reg.nvars)
end

# --- explicitly written-out einsum equations ---------------------------------

# Message: bond (3,3)-(4,3) -> bond (2,3)-(3,3). Cavity = the x = 4 column that
# the (3,3)-(4,3) window adds over the receiving (2,3)-(3,3) window. The three
# cavity site tensors are closed by incoming messages on every bond leaving to
# the far/outer side, and by "ones" caps on the legs toward the receiver (the
# output bond (3,3)-(4,3) is left open).
const EQ_MESSAGE_1 = """
P(4,2) (4,1)-(4,2) (3,2)-(4,2) (4,2)-(5,2) (4,2)-(4,3),
P(4,3) (4,2)-(4,3) (3,3)-(4,3) (4,3)-(5,3) (4,3)-(4,4),
P(4,4) (4,3)-(4,4) (3,4)-(4,4) (4,4)-(5,4) (4,4)-(4,5),
(4,1)-(4,2),
(4,2)-(5,2),
(4,3)-(5,3),
(4,4)-(5,4),
(4,4)-(4,5),
(3,2)-(4,2),
(3,3)-(4,3),
(3,4)-(4,4),
-> (3,3)-(4,3)
"""

# Message: bond (1,3)-(1,4) -> bond (1,2)-(2,2). Cavity = {(1,4),(2,4)}.
const EQ_MESSAGE_2 = """
P(1,4) (1,3)-(1,4) (1,4)-(2,4) (1,4)-(1,5),
P(2,4) (2,3)-(2,4) (1,4)-(2,4) (2,4)-(3,4) (2,4)-(2,5),
(1,4)-(1,5),
(2,4)-(3,4),
(2,4)-(2,5),
(1,3)-(1,4),
(2,3)-(2,4),
-> (1,3)-(1,4)
"""

# Marginal at center (3,3): the full {2,3,4}x{2,3,4} window. Every site tensor in
# the window, plus one incoming rank-1 message on each of the 12 bonds leaving
# the window. The center physical leg P(3,3) is the open output.
const EQ_MARGINAL = """
P(2,2) (2,1)-(2,2) (1,2)-(2,2) (2,2)-(3,2) (2,2)-(2,3),
P(3,2) (3,1)-(3,2) (2,2)-(3,2) (3,2)-(4,2) (3,2)-(3,3),
P(4,2) (4,1)-(4,2) (3,2)-(4,2) (4,2)-(5,2) (4,2)-(4,3),
P(2,3) (2,2)-(2,3) (1,3)-(2,3) (2,3)-(3,3) (2,3)-(2,4),
P(3,3) (3,2)-(3,3) (2,3)-(3,3) (3,3)-(4,3) (3,3)-(3,4),
P(4,3) (4,2)-(4,3) (3,3)-(4,3) (4,3)-(5,3) (4,3)-(4,4),
P(2,4) (2,3)-(2,4) (1,4)-(2,4) (2,4)-(3,4) (2,4)-(2,5),
P(3,4) (3,3)-(3,4) (2,4)-(3,4) (3,4)-(4,4) (3,4)-(3,5),
P(4,4) (4,3)-(4,4) (3,4)-(4,4) (4,4)-(5,4) (4,4)-(4,5),
(2,1)-(2,2),
(3,1)-(3,2),
(4,1)-(4,2),
(1,2)-(2,2),
(1,3)-(2,3),
(1,4)-(2,4),
(4,2)-(5,2),
(4,3)-(5,3),
(4,4)-(5,4),
(2,4)-(2,5),
(3,4)-(3,5),
(4,4)-(4,5),
-> P(3,3)
"""

@testset "single-layer TNMP sub-TN einsum equations (GMP, L=5, q=2, R=1)" begin
    p = PottsParams(5, 2, 0.3, default_field(2), (3, 3), "ferro")
    R = 1
    code, tensors, label_meaning, factor_coord, phys_label = build_potts_code(p)
    reg = build_potts_regions(code, tensors, label_meaning, factor_coord, p.L, R)
    cfg = TNBPConfig(max_iter = 1, error = 1e-8, damping = 0.0,
        random_order = false, verbose = false)
    _, eins, _, mars_eins, _ =
        GMP.tnbp_precompute(reg.fg, reg.icode, tensors, reg.neibs, reg.boundaries, cfg.optimizer)
    idict = reg.idict

    @testset "message update: bond (3,3)-(4,3) -> bond (2,3)-(3,3)" begin
        w = bond_int(reg, label_meaning, (3, 3), (4, 3))
        v = bond_int(reg, label_meaning, (2, 3), (3, 3))
        @test cavity_sites(reg, factor_coord, w, v) == Set(Any[(4, 2), (4, 3), (4, 4)])
        @test einsum_equivalent(
            "MESSAGE  bond (3,3)-(4,3)  ->  bond (2,3)-(3,3)",
            EQ_MESSAGE_1, eins[(w, v)], label_meaning, idict,
        )
    end

    @testset "message update: bond (1,3)-(1,4) -> bond (1,2)-(2,2)" begin
        w = bond_int(reg, label_meaning, (1, 3), (1, 4))
        v = bond_int(reg, label_meaning, (1, 2), (2, 2))
        @test cavity_sites(reg, factor_coord, w, v) == Set(Any[(1, 4), (2, 4)])
        @test einsum_equivalent(
            "MESSAGE  bond (1,3)-(1,4)  ->  bond (1,2)-(2,2)",
            EQ_MESSAGE_2, eins[(w, v)], label_meaning, idict,
        )
    end

    @testset "marginal readout at center (3,3)" begin
        cint = reg.lab2int[phys_label[(3, 3)]]
        @test window_sites(reg, factor_coord, cint) ==
              Set(Any[(x, y) for x in 2:4 for y in 2:4])
        @test einsum_equivalent(
            "MARGINAL  center site (3,3)",
            EQ_MARGINAL, mars_eins[cint], label_meaning, idict,
        )
    end
end
