#!/usr/bin/env julia
# Print TreeSA contraction order for BP sub-TNs (correct open-leg model).

using Pkg
Pkg.activate(normpath(@__DIR__, ".."); io = devnull)

using OMEinsumContractionOrders:
    EinCode, optimize_code, contraction_complexity, TreeSA

include(joinpath(@__DIR__, "bp_contraction_sc.jl"))

function merge_legs(la::Set{Int}, lb::Set{Int})
    return setdiff(union(la, lb), intersect(la, lb))
end

function subtree_legs(code, ixs)
    code.tensorindex >= 1 && return Set(ixs[code.tensorindex])
    return merge_legs(subtree_legs(code.args[1], ixs), subtree_legs(code.args[2], ixs))
end

function walk_code(code, nodes, dims, ixs, ec, steps; indent = 0)
    pad = "  " ^ indent
    if code.tensorindex >= 1
        i = code.tensorindex
        legs = ixs[i]
        ds = [dims[l] for l in legs]
        l2 = sum(log2, ds)
        println(pad, "leaf[$i] $(nodes[i])  dims=$ds  log2=$(round(l2; digits = 3))")
        return
    end
    println(pad, "contract {")
    walk_code(code.args[1], nodes, dims, ixs, ec, steps; indent = indent + 1)
    walk_code(code.args[2], nodes, dims, ixs, ec, steps; indent = indent + 1)
    ol = sort(collect(subtree_legs(code, ixs)))
    if isempty(ol)
        push!(steps, (; step = length(steps) + 1, out_dims = Int[], log2 = 0.0))
        println(pad, "}  ==>  scalar  [step $(length(steps))]")
        return
    end
    ds = [dims[l] for l in ol]
    l2 = sum(log2, ds)
    open_mark = any(l -> l in ec.iy, ol) ? " (contains open legs)" : ""
    push!(steps, (; step = length(steps) + 1, out_dims = ds, log2 = l2))
    println(pad, "}  ==>  out_dims=$ds  log2=$(round(l2; digits = 3))$open_mark  [step $(length(steps))]")
    return
end

function analyze(name, builder, chi)
    nb = builder(chi)
    ec, dims, _ = eincode(nb)
    nodes = collect(keys(nb.node_inds))
    opt = TreeSA(; ntrials = 20, niters = 60, βs = 1.0:1.0:18.0)
    code = optimize_code(ec, dims, opt)
    cc = contraction_complexity(code, dims)

    println("\n", "=" ^ 70)
    println("$name  χ=$chi  |  open outputs iy=$(ec.iy)  dims=$([dims[l] for l in ec.iy])")
    println("=" ^ 70)
    for (i, n) in enumerate(nodes)
        legs = ec.ixs[i]
        ds = [dims[l] for l in legs]
        println("  [$i] $n  dims=$ds  log2=$(round(sum(log2, ds); digits = 3))")
    end

    steps = []
    println("\nContraction tree:")
    walk_code(code, nodes, dims, ec.ixs, ec, steps)

    println("\nSteps:")
    for s in steps
        mark = isapprox(s.log2, cc.sc; atol = 1e-9) ? "  <-- PEAK sc" : ""
        println("  step $(s.step): out_dims=$(s.out_dims)  log2=$(round(s.log2; digits = 3))$mark")
    end
    println("\nTreeSA: sc=$(round(cc.sc; digits = 3))  tc=$(round(cc.tc; digits = 3))")
end

analyze("bp_cavity", build_bp_cavity, 4)
analyze("bp_cavity", build_bp_cavity, 8)
