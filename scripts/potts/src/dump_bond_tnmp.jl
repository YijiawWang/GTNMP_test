#!/usr/bin/env julia
using GenericMessagePassing
const GMP = GenericMessagePassing
using GenericMessagePassing: FactorGraph, TNBPConfig
using OMEinsum: getixsv, getiyv
using OMEinsum.OMEinsumContractionOrders: IncidenceList
using Printf

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "run_tnmp.jl"))

bond_key(a, b) = minmax(a, b)

function find_bond_var(label_meaning, idict, a, b)
    key = bond_key(a, b)
    lab2int = Dict(idict[i] => i for i in 1:length(idict))
    for (lab, m) in label_meaning
        m[1] === :bond && m[2] == key && return lab2int[lab]
    end
    error("bond $key not found")
end

function describe_node(u, var_meaning)
    m = var_meaning(u)
    m[1] === :phys && return "phys$(m[2])"
    m[1] === :bond && return "bond$(m[2][1])-$(m[2][2])"
    return string(m)
end

function run_tnmp_with_messages(p; region_L=3, max_iter=200, tol=1e-8)
    R = (region_L - 1) ÷ 2
    code, tensors, label_meaning, factor_coord, _ = build_potts_code(p)
    icode, idict = GMP.intcode(code)
    ixs = getixsv(icode)
    iy = getiyv(icode)
    hyper = IncidenceList(Dict([i => ix for (i, ix) in enumerate(ixs)]); openedges = iy)
    fg = FactorGraph(hyper)
    nvars = fg.num_vars
    ntot = nvars + length(tensors)
    var_meaning(u) = label_meaning[idict[u]]
    member(u, S) = begin
        if u <= nvars
            m = var_meaning(u)
            m[1] === :phys && return m[2] in S
            a, b = m[2]
            return (a in S) || (b in S)
        else
            return factor_coord[u - nvars] in S
        end
    end
    region_of(u) = begin
        m = var_meaning(u)
        m[1] === :phys ? site_block(m[2], p.L, R) : bond_block(m[2][1], m[2][2], p.L, R)
    end
    neibs = Dict{Int,Vector{Int}}()
    boundaries = Dict{Int,Vector{Int}}()
    for v in 1:nvars
        S = region_of(v)
        neibs[v] = Int[u for u in 1:ntot if member(u, S)]
        boundaries[v] = GMP.open_boundaries(fg, neibs[v])
    end
    cfg = TNBPConfig(max_iter=max_iter, error=tol, damping=0.0, random_order=false, verbose=false)
    messages, eins, ptensors, _, _ =
        GMP.tnbp_precompute(fg, icode, tensors, neibs, boundaries, cfg.optimizer)
    finalerr = Inf
    iters = 0
    for i in 1:max_iter
        finalerr = GMP.tnbp_update!(messages, eins, ptensors, cfg)
        iters = i
        finalerr < tol && break
    end
    return (; messages, var_meaning, boundaries, converged=finalerr < tol, iters, finalerr)
end

function main()
    p = parse_potts_params(ARGS)
    bx = parse_int_opt(ARGS, "bx", 3)
    by = parse_int_opt(ARGS, "by", 3)
    ax = parse_int_opt(ARGS, "ax", 4)
    ay = parse_int_opt(ARGS, "ay", 3)
    out = parse_opt(ARGS, "out", joinpath(@__DIR__, "..", "results", "bond_tnmp.txt"))
    max_iter = parse_int_opt(ARGS, "max-iter", 200)
    tol = parse_float_opt(ARGS, "tol", 1e-8)
    bond_a, bond_b = (bx, by), (ax, ay)

    tnmp = run_tnmp_with_messages(p; max_iter=max_iter, tol=tol)
    code, _, label_meaning, _, _ = build_potts_code(p)
    _, idict = GMP.intcode(code)
    bond_int = find_bond_var(label_meaning, idict, bond_a, bond_b)

    rows = Tuple{String,Bool,Vector{Float64}}[]
    for ((w, v), msg) in tnmp.messages
        w == bond_int || continue
        vec = collect(Float64, msg)
        vec ./= sum(vec)
        push!(rows, (describe_node(v, tnmp.var_meaning), bond_int in tnmp.boundaries[v], vec))
    end
    sort!(rows, by = x -> x[1])

    open(out, "w") do io
        println(io, "method=tnmp")
        println(io, "L=$(p.L)")
        println(io, "q=$(p.q)")
        println(io, "coupling=$(p.coupling)")
        println(io, "couplings=$(p.couplings)")
        println(io, "bond=$(bond_a[1]),$(bond_a[2]),$(bond_b[1]),$(bond_b[2])")
        println(io, "converged=$(tnmp.converged)")
        println(io, "iters=$(tnmp.iters)")
        println(io, "finalerr=$(tnmp.finalerr)")
        println(io, "n_incoming=$(length(rows))")
        for (reg, onbnd, vec) in rows
            println(io, "incoming region=$reg open_boundary=$onbnd message=$(join(vec, ","))")
        end
    end
    println("saved TNMP bond messages -> $out ($(length(rows)) incoming)")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
