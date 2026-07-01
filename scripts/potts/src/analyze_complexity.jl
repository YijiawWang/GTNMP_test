#!/usr/bin/env julia
# TNMP cavity / neighborhood vs BP: loop structure and TreeSA sc/tc for the
# single-layer frustrated Potts model (potts_compare setup).
#
#   julia --project=../../../env_gmp analyze_complexity.jl [--L 10 --q 3 --couplings frustrated]

using GenericMessagePassing
const GMP = GenericMessagePassing
using GenericMessagePassing: FactorGraph, TNBPConfig
using OMEinsum: EinCode, getixsv, getiyv, get_size_dict
using OMEinsum.OMEinsumContractionOrders: IncidenceList, TreeSA, contraction_complexity, optimize_code
using Printf: @printf, @sprintf

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "run_tnmp.jl"))

function build_tnmp_regions(code, tensors, label_meaning, factor_coord, L, R)
    icode, idict = GMP.intcode(code)
    ixs = getixsv(icode)
    hyper = IncidenceList(Dict([i => ix for (i, ix) in enumerate(ixs)]); openedges = getiyv(icode))
    fg = FactorGraph(hyper)
    nvars = fg.num_vars
    ntot = nvars + length(tensors)
    lab2int = Dict(idict[i] => i for i in 1:nvars)

    var_meaning(u) = label_meaning[idict[u]]
    member(u, S) = u <= nvars ?
        (begin m = var_meaning(u); m[1] === :phys ? m[2] in S : (m[2][1] in S || m[2][2] in S) end) :
        (factor_coord[u - nvars] in S)
    region_of(u) = begin m = var_meaning(u); m[1] === :phys ? site_block(m[2], L, R) :
        bond_block(m[2][1], m[2][2], L, R) end

    neibs = Dict{Int,Vector{Int}}()
    boundaries = Dict{Int,Vector{Int}}()
    for v in 1:nvars
        S = region_of(v)
        nb = Int[u for u in 1:ntot if member(u, S)]
        neibs[v] = nb
        boundaries[v] = GMP.open_boundaries(fg, nb)
    end
    return (; fg, icode, ixs, idict, lab2int, neibs, boundaries, nvars, ntot, var_meaning, factor_coord)
end

# Cyclomatic number of the variable adjacency graph induced by tensor ixs.
function subtn_cycle_info(ixs::Vector{Vector{Int}})
    labels = sort(unique(vcat(ixs...)))
    pos = Dict(l => i for (i, l) in enumerate(labels))
    edges = Set{Tuple{Int,Int}}()
    for ix in ixs
        for i in 1:length(ix), j in (i + 1):length(ix)
            ai, bi = pos[ix[i]], pos[ix[j]]
            if ai > bi
                ai, bi = bi, ai
            end
            push!(edges, (ai, bi))
        end
    end
    V = length(labels)
    E = length(edges)
    # Build adjacency for component count.
    adj = [Int[] for _ in 1:V]
    for (a, b) in edges
        push!(adj[a], b); push!(adj[b], a)
    end
    seen = falses(V)
    n_comp = 0
    for v in 1:V
        if seen[v]; continue; end
        n_comp += 1
        stack = [v]
        seen[v] = true
        while !isempty(stack)
            u = pop!(stack)
            for w in adj[u]
                if !seen[w]
                    seen[w] = true
                    push!(stack, w)
                end
            end
        end
    end
    n_cycles = max(0, E - V + n_comp)
    return (;
        n_tensors = length(ixs),
        n_labels = V,
        n_independent_cycles = n_cycles,
        has_cycle = n_cycles > 0,
    )
end

function cc_metrics(eincode::EinCode, size_dict; opt)
    optcode = optimize_code(eincode, size_dict, opt)
    cc = contraction_complexity(optcode, size_dict)
    cyc = subtn_cycle_info(getixsv(eincode))
    return (;
        sc = cc.sc,
        tc = cc.tc,
        n_tensors = cyc.n_tensors,
        n_labels = cyc.n_labels,
        has_cycle = cyc.has_cycle,
        n_independent_cycles = cyc.n_independent_cycles,
    )
end

function describe_node(var_meaning, u)
    m = var_meaning(u)
    m[1] === :phys ? "phys$(m[2])" : "bond$(m[2][1])-$(m[2][2])"
end

function local_iys_lookup(fg, neibs)
    d = Dict{Tuple{Int,Int},Vector{Int}}()
    for v in keys(neibs)
        for w in GMP.open_boundaries(fg, neibs[v])
            if GMP.is_factor(fg, w)
                d[(w, v)] = [u for u in GMP.neighbors(fg, w) if u ∉ neibs[v]]
            else
                d[(w, v)] = [w]
            end
        end
    end
    return d
end

function cavity_eincode(reg, w, v, local_iys)
    fg, ixs = reg.fg, reg.ixs
    neib_v, neib_w = reg.neibs[v], reg.neibs[w]
    boundary_w = reg.boundaries[w]
    local_region = setdiff(neib_w, neib_v)
    local_ixs = Vector{Vector{Int}}()
    bwm = intersect(boundary_w, local_region)
    bwo = setdiff(GMP.open_boundaries(fg, local_region), bwm)
    for b in local_region
        b == w && continue
        GMP.is_factor(fg, b) && push!(local_ixs, ixs[b - fg.num_vars])
        if b ∈ bwm
            push!(local_ixs, local_iys[(b, w)])
        elseif b ∈ bwo
            if GMP.is_factor(fg, b)
                push!(local_ixs, [u for u in GMP.neighbors(fg, b) if u ∉ local_region])
            else
                push!(local_ixs, [b])
            end
        end
    end
    return EinCode(local_ixs, local_iys[(w, v)]), local_region
end

function main()
    args = ARGS
    p = parse_potts_params(args)
    region_L = parse_int_opt(args, "region-L", 3)
    R = (region_L - 1) ÷ 2
    opt = TreeSA(; ntrials = 10, niters = 40, βs = 1.0:1.0:20.0)

    code, tensors, label_meaning, factor_coord, phys_label = build_potts_code(p)
    reg = build_tnmp_regions(code, tensors, label_meaning, factor_coord, p.L, R)
    cint = reg.lab2int[phys_label[p.center]]
    size_dict = get_size_dict(reg.ixs, tensors)
    local_iys = local_iys_lookup(reg.fg, reg.neibs)

    center_tidx = only(i for (i, c) in enumerate(factor_coord) if c == p.center)

    println("="^72)
    println("Potts TNMP/BP complexity  L=$(p.L) q=$(p.q) K=$(p.coupling) " *
            "couplings=$(p.couplings) region_L=$region_L center=$(p.center)")
    println("="^72)

    println("\n--- TNMP cavity updates (w -> center phys) ---")
    println(@sprintf("%-16s %5s %5s %5s %6s %6s %5s",
        "boundary w", "ntens", "ncyc", "loop", "sc", "tc", "n_cav"))
    cavity_scs, cavity_tcs, cavity_loops = Float64[], Float64[], Bool[]
    for w in reg.boundaries[cint]
        ec, local_region = cavity_eincode(reg, w, cint, local_iys)
        m = cc_metrics(ec, size_dict; opt)
        push!(cavity_scs, m.sc); push!(cavity_tcs, m.tc); push!(cavity_loops, m.has_cycle)
        println(@sprintf("%-16s %5d %5d %5s %6.1f %6.1f %5d",
            describe_node(reg.var_meaning, w), m.n_tensors, m.n_independent_cycles,
            m.has_cycle ? "yes" : "no", m.sc, m.tc, length(local_region)))
    end

    println("\n--- TNMP center neighborhood (marginal) ---")
    marg_ixs = Vector{Int}[]
    for w in reg.neibs[cint]
        w ∈ reg.boundaries[cint] && push!(marg_ixs, [w])
        GMP.is_factor(reg.fg, w) && push!(marg_ixs, reg.ixs[w - reg.fg.num_vars])
    end
    marg_m = cc_metrics(EinCode(marg_ixs, [cint]), size_dict; opt)
    @printf("ntens=%d ncyc=%d loop=%s sc=%.1f tc=%.1f\n",
        marg_m.n_tensors, marg_m.n_independent_cycles,
        marg_m.has_cycle ? "yes" : "no", marg_m.sc, marg_m.tc)

    println("\n--- GMP BP on same TN (single site factor at center) ---")
    hyper = IncidenceList(Dict([i => ix for (i, ix) in enumerate(reg.ixs)]))
    ids = collect(keys(hyper.v2e))
    e2v_plans, _ = GMP.bp_precompute_plans(hyper, reg.ixs, ids, size_dict; optimizer = opt)

    bp_scs, bp_tcs, bp_loops = Float64[], Float64[], Bool[]
    for ((e, v), plan) in e2v_plans
        v == center_tidx || continue
        local_ixs = Vector{Int}[]
        for key in plan.arg_keys
            push!(local_ixs, key[2] == 0 ? reg.ixs[key[1]] : [key[1]])
        end
        m = cc_metrics(EinCode(local_ixs, [e]), size_dict; opt)
        push!(bp_scs, m.sc); push!(bp_tcs, m.tc); push!(bp_loops, m.has_cycle)
        @printf("e2v edge %-14s ntens=%d ncyc=%d loop=%s sc=%.1f tc=%.1f\n",
            describe_node(reg.var_meaning, e), m.n_tensors, m.n_independent_cycles,
            m.has_cycle ? "yes" : "no", m.sc, m.tc)
    end

    bp_marg_ixs = Vector{Int}[[e] for e in reg.ixs[center_tidx]]
    push!(bp_marg_ixs, reg.ixs[center_tidx])
    bp_marg = cc_metrics(EinCode(bp_marg_ixs, [cint]), size_dict; opt)
    @printf("marginal ntens=%d ncyc=%d loop=%s sc=%.1f tc=%.1f\n",
        bp_marg.n_tensors, bp_marg.n_independent_cycles,
        bp_marg.has_cycle ? "yes" : "no", bp_marg.sc, bp_marg.tc)

    println("\n--- Summary ---")
    @printf("TNMP cavity  sc=[%.1f,%.1f] tc=[%.1f,%.1f] has_loop=%s\n",
        minimum(cavity_scs), maximum(cavity_scs),
        minimum(cavity_tcs), maximum(cavity_tcs), any(cavity_loops) ? "yes" : "no")
    @printf("TNMP marg    sc=%.1f tc=%.1f has_loop=%s\n",
        marg_m.sc, marg_m.tc, marg_m.has_cycle ? "yes" : "no")
    @printf("BP e2v       sc=[%.1f,%.1f] tc=[%.1f,%.1f] has_loop=%s\n",
        isempty(bp_scs) ? NaN : minimum(bp_scs), isempty(bp_scs) ? NaN : maximum(bp_scs),
        isempty(bp_tcs) ? NaN : minimum(bp_tcs), isempty(bp_tcs) ? NaN : maximum(bp_tcs),
        any(bp_loops) ? "yes" : "no")
    @printf("BP marg      sc=%.1f tc=%.1f has_loop=%s\n",
        bp_marg.sc, bp_marg.tc, bp_marg.has_cycle ? "yes" : "no")
    println("="^72)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
