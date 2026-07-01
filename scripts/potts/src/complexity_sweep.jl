#!/usr/bin/env julia
# Single-layer Potts TNMP cavity + neighborhood contraction-complexity sweep,
# the q-state analogue of `contraction_process_tn/tnmp_rank2_complexity_sweep.jl`
# (there the swept knob is the bond dim chi; here it is the Potts dimension q,
# which IS the single-layer bond dim).
#
# Crucially, this script does NOT assemble any cavity/neighborhood einsum by
# hand. It builds the exact same regions used by the real marginal run
# (`build_potts_regions`, shared with run_tnmp.jl) and hands them to the TNMP
# package routine `GenericMessagePassing.tnbp_precompute`, which is the function
# `run_tnmp.jl` actually uses to construct + TreeSA-optimize every cavity message
# contraction (`eins`) and every neighborhood marginal contraction (`mars_eins`).
# We then only READ tc/sc off those optimized codes -- message passing
# (`tnbp_update!`) and the numerical marginal contraction are switched off.
#
# Run with:
#   julia --project=../../../env_gmp complexity_sweep.jl [--L 10 --region-L 3 \
#         --q-list 2,3,4,5,6,7,8 --couplings ferro --treesa-ntrials 2 --treesa-niters 10]

using GenericMessagePassing
const GMP = GenericMessagePassing
using GenericMessagePassing: FactorGraph, TNBPConfig
using OMEinsum: getixsv, getiyv, get_size_dict
using OMEinsum.OMEinsumContractionOrders: TreeSA, contraction_complexity
using Statistics: mean
using Serialization: serialize
using Printf: @sprintf
using Dates: now

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "run_tnmp.jl"))

const DEFAULT_Q_VALUES = [2, 3, 4, 5, 6, 7, 8]

progress(msg) = (println("[progress $(now())] $msg"); flush(stdout))

aggregate(values::AbstractVector{<:Real}) =
    isempty(values) ? (; count = 0, min = NaN, max = NaN, mean = NaN) :
    (; count = length(values), min = minimum(values), max = maximum(values), mean = mean(values))

fmt(x) = isnan(x) ? "-" : @sprintf("%.3f", x)

# Generalized NxN sub-lattice window, supporting BOTH odd and even side lengths.
# For side N the offsets are a = (N-1)÷2 (lower) and b = N÷2 (upper), so the
# window spans `c-a : c+b` (N sites). For odd N this is exactly the centered
# 2R+1 window used by `site_block`/`bond_block` in run_tnmp.jl (a = b = R); for
# even N the window is the canonical N sites with the extra site placed on the
# +x/+y side. The bond window keeps the established "N across x (N-1) along"
# shape (a-1 / b-1 along-bond extensions reduce to R-1 for odd N).
window_offsets(region_L::Int) = ((region_L - 1) ÷ 2, region_L ÷ 2)

function site_block_n(c::Tuple{Int,Int}, L::Int, region_L::Int)
    a, b = window_offsets(region_L)
    S = Set{Tuple{Int,Int}}()
    for x in (c[1] - a):(c[1] + b), y in (c[2] - a):(c[2] + b)
        (1 <= x <= L && 1 <= y <= L) && push!(S, (x, y))
    end
    return S
end

function bond_block_n(p1::Tuple{Int,Int}, p2::Tuple{Int,Int}, L::Int, region_L::Int)
    a, b = window_offsets(region_L)
    (x1, y1) = p1
    (x2, y2) = p2
    S = Set{Tuple{Int,Int}}()
    if x1 == x2  # vertical bond: span endpoints along y, widen across x
        ya, yb = minmax(y1, y2)
        xlo, xhi = x1 - a, x1 + b
        ylo, yhi = ya - (a - 1), yb + (b - 1)
    else        # horizontal bond
        xa, xb = minmax(x1, x2)
        xlo, xhi = xa - (a - 1), xb + (b - 1)
        ylo, yhi = y1 - a, y1 + b
    end
    for x in xlo:xhi, y in ylo:yhi
        (1 <= x <= L && 1 <= y <= L) && push!(S, (x, y))
    end
    return S
end

# Region/boundary builder identical in structure to `build_potts_regions`
# (run_tnmp.jl), but parameterised by an arbitrary (even or odd) `region_L`
# instead of a single integer radius R, so 4x4 / 6x6 windows are supported.
# For odd `region_L` it reproduces `build_potts_regions` exactly. The cavity
# (`eins`) and neighborhood (`mars_eins`) einsums are still constructed and
# TreeSA-optimized by the package routine `tnbp_precompute`.
function build_potts_regions_n(code, tensors, label_meaning, factor_coord, L, region_L)
    icode, idict = GMP.intcode(code)
    ixs = getixsv(icode)
    iy = getiyv(icode)
    hyper = IncidenceList(Dict([i => ix for (i, ix) in enumerate(ixs)]); openedges = iy)
    fg = FactorGraph(hyper)
    nvars = fg.num_vars
    ntot = nvars + length(tensors)
    lab2int = Dict(idict[i] => i for i in 1:nvars)

    var_meaning(u) = label_meaning[idict[u]]
    function member(u::Int, S)
        if u <= nvars
            m = var_meaning(u)
            if m[1] === :phys
                return m[2] in S
            else
                a, b = m[2]
                return (a in S) || (b in S)
            end
        else
            return factor_coord[u - nvars] in S
        end
    end
    function region_of(u::Int)
        m = var_meaning(u)
        return m[1] === :phys ? site_block_n(m[2], L, region_L) :
               bond_block_n(m[2][1], m[2][2], L, region_L)
    end

    neibs = Dict{Int,Vector{Int}}()
    boundaries = Dict{Int,Vector{Int}}()
    for v in 1:nvars
        S = region_of(v)
        nb = Int[u for u in 1:ntot if member(u, S)]
        neibs[v] = nb
        boundaries[v] = GMP.open_boundaries(fg, nb)
    end
    return (; fg, icode, idict, ixs, nvars, ntot, lab2int, var_meaning, neibs, boundaries)
end

# Run tnbp_precompute (the real TNMP constructor) for one q and read off the
# per-contraction tc/sc of every cavity message and every site-neighborhood
# marginal. No iteration, no numerical contraction.
function complexity_rows_for_q(p::PottsParams, region_L::Int, optimizer)
    code, tensors, label_meaning, factor_coord, phys_label = build_potts_code(p; sizes_only = true)
    reg = build_potts_regions_n(code, tensors, label_meaning, factor_coord, p.L, region_L)
    size_dict = get_size_dict(reg.ixs, tensors)

    # THE call: the package builds + TreeSA-optimizes all cavity (`eins`) and
    # neighborhood (`mars_eins`) contractions exactly as a real TNMP run would.
    _messages, eins, _ptensors, mars_eins, _mars_tensors =
        GMP.tnbp_precompute(reg.fg, reg.icode, tensors, reg.neibs, reg.boundaries, optimizer)

    cavity = NamedTuple[]
    for ((w, v), optcode) in eins
        cc = contraction_complexity(optcode, size_dict)
        push!(cavity, (; kind = "cavity", receiver = v, source = w, sc = cc.sc, tc = cc.tc))
    end

    # Neighborhood = the site (phys) marginal contractions. The bond-variable
    # marginals also live in mars_eins but TNMP reads physical-site marginals,
    # so we report those (the genuine 3x3 neighborhood readouts).
    neighborhood = NamedTuple[]
    for (v, optcode) in mars_eins
        reg.var_meaning(v)[1] === :phys || continue
        cc = contraction_complexity(optcode, size_dict)
        is_center = v == reg.lab2int[phys_label[p.center]]
        push!(neighborhood, (; kind = "neighborhood", site = v, is_center, sc = cc.sc, tc = cc.tc))
    end

    return cavity, neighborhood
end

function summarize(q::Int, cavity, neighborhood)
    center = filter(r -> r.is_center, neighborhood)
    return (;
        q = q,
        n_cavity = length(cavity),
        n_neighborhood = length(neighborhood),
        cavity_tc = aggregate([r.tc for r in cavity]),
        cavity_sc = aggregate([r.sc for r in cavity]),
        neighborhood_tc = aggregate([r.tc for r in neighborhood]),
        neighborhood_sc = aggregate([r.sc for r in neighborhood]),
        center_tc = isempty(center) ? NaN : only(center).tc,
        center_sc = isempty(center) ? NaN : only(center).sc,
        all_tc = aggregate([r.tc for r in vcat(cavity, neighborhood)]),
        all_sc = aggregate([r.sc for r in vcat(cavity, neighborhood)]),
    )
end

function write_markdown(path, summaries, meta)
    open(path, "w") do io
        println(io, "# Single-layer Potts TNMP cavity + neighborhood contraction complexity (TreeSA)")
        println(io)
        println(io, "Generated by [`complexity_sweep.jl`](complexity_sweep.jl).")
        println(io)
        println(io, "Model: single-layer classical q-state Potts on an open $(meta.L)x$(meta.L) ")
        println(io, "square lattice (bond dim = q), $(meta.region_L)x$(meta.region_L) sub-lattice ")
        println(io, "neighborhood, couplings = `$(meta.couplings)`. ")
        println(io)
        println(io, "Sub-TNs are built by the package routine `GenericMessagePassing.tnbp_precompute` ")
        println(io, "(the same call `run_tnmp.jl` uses); message passing and the numerical marginal ")
        println(io, "contraction are switched off, so only the cavity (`eins`) and neighborhood ")
        println(io, "(`mars_eins`) TreeSA contraction orders are measured. Complexity is independent ")
        println(io, "of the coupling K / field / tensor values.")
        println(io)
        println(io, "`tc` = log2(#multiply-adds), `sc` = log2(largest intermediate size).")
        println(io)

        println(io, "## Cavity (all message contractions, per-q)")
        println(io)
        println(io, "| q | #cavity | tc min | tc mean | tc max | sc min | sc mean | sc max |")
        println(io, "|--:|--------:|-------:|--------:|-------:|-------:|--------:|-------:|")
        for s in summaries
            println(io, "| $(s.q) | $(s.n_cavity) | $(fmt(s.cavity_tc.min)) | $(fmt(s.cavity_tc.mean)) | $(fmt(s.cavity_tc.max)) | $(fmt(s.cavity_sc.min)) | $(fmt(s.cavity_sc.mean)) | $(fmt(s.cavity_sc.max)) |")
        end
        println(io)

        println(io, "## Neighborhood (all site marginal contractions, per-q)")
        println(io)
        println(io, "| q | #neighborhood | tc min | tc mean | tc max | sc min | sc mean | sc max | center tc | center sc |")
        println(io, "|--:|--------------:|-------:|--------:|-------:|-------:|--------:|-------:|----------:|----------:|")
        for s in summaries
            println(io, "| $(s.q) | $(s.n_neighborhood) | $(fmt(s.neighborhood_tc.min)) | $(fmt(s.neighborhood_tc.mean)) | $(fmt(s.neighborhood_tc.max)) | $(fmt(s.neighborhood_sc.min)) | $(fmt(s.neighborhood_sc.mean)) | $(fmt(s.neighborhood_sc.max)) | $(fmt(s.center_tc)) | $(fmt(s.center_sc)) |")
        end
        println(io)

        println(io, "## Combined (cavity + neighborhood)")
        println(io)
        println(io, "| q | #total | tc min | tc mean | tc max | sc min | sc mean | sc max |")
        println(io, "|--:|-------:|-------:|--------:|-------:|-------:|--------:|-------:|")
        for s in summaries
            ntotal = s.n_cavity + s.n_neighborhood
            println(io, "| $(s.q) | $(ntotal) | $(fmt(s.all_tc.min)) | $(fmt(s.all_tc.mean)) | $(fmt(s.all_tc.max)) | $(fmt(s.all_sc.min)) | $(fmt(s.all_sc.mean)) | $(fmt(s.all_sc.max)) |")
        end
        println(io)
    end
    return path
end

function main()
    args = ARGS
    L = parse_int_opt(args, "L", 10)
    region_L = parse_int_opt(args, "region-L", 3)
    region_L >= 1 || error("region-L must be >= 1")
    couplings = parse_opt(args, "couplings", "ferro")
    q_str = parse_opt(args, "q-list", nothing)
    q_values = q_str === nothing ? DEFAULT_Q_VALUES : [parse(Int, strip(s)) for s in split(q_str, ",")]
    ntrials = parse_int_opt(args, "treesa-ntrials", 2)
    niters = parse_int_opt(args, "treesa-niters", 10)
    sc_target = parse_int_opt(args, "treesa-sc-target", 20)
    out_md = parse_opt(args, "out", joinpath(@__DIR__, "..", "results", "potts_complexity_sweep.md"))
    out_jls = replace(out_md, r"\.md$" => ".jls")
    mkpath(dirname(out_md))

    optimizer = TreeSA(; ntrials = ntrials, niters = niters, sc_target = sc_target, βs = 0.1:0.1:15.0)

    progress("potts complexity sweep: L=$L region_L=$region_L couplings=$couplings " *
             "q_values=$q_values treesa(ntrials=$ntrials,niters=$niters,sc_target=$sc_target)")

    summaries = NamedTuple[]
    all_rows = Dict{Int,Any}()
    for q in q_values
        # K / field do not affect contraction complexity; use harmless defaults.
        p = PottsParams(L, q, 1.0, default_field(q), grid_center(L), couplings)
        progress("q=$q: calling tnbp_precompute (region_L=$region_L window)")
        cavity, neighborhood = complexity_rows_for_q(p, region_L, optimizer)
        all_rows[q] = (; cavity, neighborhood)
        s = summarize(q, cavity, neighborhood)
        push!(summaries, s)
        progress("q=$q -> cavity n=$(s.n_cavity) sc[min/mean/max]=" *
                 "$(fmt(s.cavity_sc.min))/$(fmt(s.cavity_sc.mean))/$(fmt(s.cavity_sc.max)) " *
                 "tc[max]=$(fmt(s.cavity_tc.max)) | neighborhood n=$(s.n_neighborhood) " *
                 "center sc=$(fmt(s.center_sc)) tc=$(fmt(s.center_tc))")
    end

    meta = (; L, region_L, couplings, q_values, ntrials, niters)
    write_markdown(out_md, summaries, meta)
    open(out_jls, "w") do io
        serialize(io, (; meta, summaries, rows = all_rows))
    end
    progress("wrote $out_md")
    progress("wrote $out_jls")
    println("\nsaved markdown -> $out_md")
    println("saved jls      -> $out_jls")
    return summaries
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
