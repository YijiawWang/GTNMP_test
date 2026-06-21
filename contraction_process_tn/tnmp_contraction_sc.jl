#!/usr/bin/env julia
# TreeSA contraction complexity for TNMP sub-TNs (rank-1 / rank-2 cavity & neighborhood).
# Red triangles and center ▲/▼ are open external legs (not tensors).
# Blue/black boundary triangles / diamonds are incoming message tensors.

using Pkg
Pkg.activate(normpath(@__DIR__, ".."); io = devnull)

using OMEinsumContractionOrders:
    EinCode, optimize_code, contraction_complexity, TreeSA, peak_memory
using Printf: @printf

const DIM_GREEN = 2

mutable struct NetBuilder
    node_inds::Dict{Any, Vector{Int}}
    dims::Dict{Int, Int}
    nedge::Int
    ntri::Int
end
NetBuilder() = NetBuilder(Dict{Any, Vector{Int}}(), Dict{Int, Int}(), 0, 0)
node!(nb::NetBuilder, n) = get!(nb.node_inds, n, Int[])

function edge!(nb::NetBuilder, u, v, d::Int)
    nb.nedge += 1
    e = nb.nedge
    nb.dims[e] = d
    push!(node!(nb, u), e)
    push!(node!(nb, v), e)
    return e
end

function open_leg!(nb::NetBuilder, node, d::Int)
    nb.nedge += 1
    e = nb.nedge
    nb.dims[e] = d
    push!(node!(nb, node), e)
    return e
end

function caps!(nb::NetBuilder, c, k::Int, d::Int)
    for _ in 1:k
        nb.ntri += 1
        edge!(nb, c, (:tri, nb.ntri), d)
    end
end

function merged!(nb::NetBuilder, a, b, d::Int)
    nb.ntri += 1
    t = (:tri, nb.ntri)
    edge!(nb, a, t, d)
    edge!(nb, b, t, d)
    return t
end

function eincode(nb::NetBuilder)
    nodes = collect(keys(nb.node_inds))
    ixs = [nb.node_inds[n] for n in nodes]
    counts = Dict{Int, Int}()
    for ix in ixs, l in ix
        counts[l] = get(counts, l, 0) + 1
    end
    iy = sort([l for (l, c) in counts if c == 1])
    return EinCode(ixs, iy), nb.dims, length(nodes)
end

max_input_log2(ec::EinCode, dims::Dict{Int, Int}) =
    maximum(ix -> isempty(ix) ? 0.0 : sum(l -> log2(dims[l]), ix), ec.ixs)

boundary_count(i, j, n) = (i == 1) + (i == n) + (j == 1) + (j == n)

function build_rank1_cavity(strip_len::Int, dim_chi::Int; dim_green::Int = DIM_GREEN)
    nb = NetBuilder()
    top = [(:t, i) for i in 1:strip_len]
    bot = [(:b, i) for i in 1:strip_len]
    for row in (top, bot)
        for i in 1:(strip_len - 1)
            edge!(nb, row[i], row[i + 1], dim_chi)
        end
    end
    for i in 1:strip_len
        edge!(nb, top[i], bot[i], dim_green)
    end
    mid = (strip_len + 1) ÷ 2
    for i in 1:strip_len
        if i == mid
            open_leg!(nb, top[i], dim_chi)
            caps!(nb, top[i], 1, dim_chi)
            open_leg!(nb, bot[i], dim_chi)
            caps!(nb, bot[i], 1, dim_chi)
        else
            caps!(nb, top[i], 3, dim_chi)
            caps!(nb, bot[i], 3, dim_chi)
        end
    end
    return nb
end

function build_rank1_neighborhood(n::Int, dim_chi::Int; dim_green::Int = DIM_GREEN, dim_center::Int = dim_chi)
    nb = NetBuilder()
    T(i, j) = (:T, i, j)
    B(i, j) = (:B, i, j)
    cx = cy = (n + 1) ÷ 2
    for lay in (:T, :B)
        N(i, j) = (lay, i, j)
        for j in 1:n, i in 1:(n - 1)
            edge!(nb, N(i, j), N(i + 1, j), dim_chi)
        end
        for i in 1:n, j in 1:(n - 1)
            edge!(nb, N(i, j), N(i, j + 1), dim_chi)
        end
    end
    for j in 1:n, i in 1:n
        if (i, j) == (cx, cy)
            open_leg!(nb, T(i, j), dim_center)
            open_leg!(nb, B(i, j), dim_center)
        else
            edge!(nb, T(i, j), B(i, j), dim_green)
        end
    end
    for j in 1:n, i in 1:n
        k = boundary_count(i, j, n)
        caps!(nb, T(i, j), k, dim_chi)
        caps!(nb, B(i, j), k, dim_chi)
    end
    return nb
end

function build_rank2_cavity(strip_len::Int, dim_chi::Int; dim_green::Int = DIM_GREEN)
    nb = NetBuilder()
    top = [(:t, i) for i in 1:strip_len]
    bot = [(:b, i) for i in 1:strip_len]
    for row in (top, bot)
        for i in 1:(strip_len - 1)
            edge!(nb, row[i], row[i + 1], dim_chi)
        end
    end
    for i in 1:strip_len
        edge!(nb, top[i], bot[i], dim_green)
    end
    mid = (strip_len + 1) ÷ 2
    for i in 1:strip_len
        if i == mid
            merged!(nb, top[i], bot[i], dim_chi)
            open_leg!(nb, top[i], dim_chi)
            open_leg!(nb, bot[i], dim_chi)
        else
            for _ in 1:3
                merged!(nb, top[i], bot[i], dim_chi)
            end
        end
    end
    return nb
end

function build_rank2_neighborhood(n::Int, dim_chi::Int; dim_green::Int = DIM_GREEN, dim_center::Int = dim_chi)
    nb = NetBuilder()
    T(i, j) = (:T, i, j)
    B(i, j) = (:B, i, j)
    cx = cy = (n + 1) ÷ 2
    for lay in (:T, :B)
        N(i, j) = (lay, i, j)
        for j in 1:n, i in 1:(n - 1)
            edge!(nb, N(i, j), N(i + 1, j), dim_chi)
        end
        for i in 1:n, j in 1:(n - 1)
            edge!(nb, N(i, j), N(i, j + 1), dim_chi)
        end
    end
    for j in 1:n, i in 1:n
        if (i, j) == (cx, cy)
            open_leg!(nb, T(i, j), dim_center)
            open_leg!(nb, B(i, j), dim_center)
        else
            edge!(nb, T(i, j), B(i, j), dim_green)
        end
    end
    for j in 1:n, i in 1:n
        for _ in 1:boundary_count(i, j, n)
            merged!(nb, T(i, j), B(i, j), dim_chi)
        end
    end
    return nb
end

function report(name::String, nb::NetBuilder; opt = TreeSA(; ntrials = 20, niters = 60, βs = 1.0:1.0:18.0))
    ec, dims, nnodes = eincode(nb)
    code = optimize_code(ec, dims, opt)
    cc = contraction_complexity(code, dims)
    return (;
        name,
        tensors = nnodes,
        open_legs = length(ec.iy),
        max_input = max_input_log2(ec, dims),
        sc = cc.sc,
        peak_memory = log2(peak_memory(code, dims)),
        tc = cc.tc,
    )
end

function main()
    chi_values = [4, 8, 16, 32]
    grid_sizes = [3, 5]

    println("=== TNMP sub-TN contraction complexity (TreeSA) ===")
    println("green bond = dim $DIM_GREEN, black bond = dim χ")
    println("red ▲/▼ = open external legs (not tensors)\n")

    builders = (
        ("rank1_cavity", (strip, chi) -> build_rank1_cavity(strip, chi)),
        ("rank1_neighborhood", (n, chi) -> build_rank1_neighborhood(n, chi)),
        ("rank2_cavity", (strip, chi) -> build_rank2_cavity(strip, chi)),
        ("rank2_neighborhood", (n, chi) -> build_rank2_neighborhood(n, chi)),
    )

    results = Dict{String, Vector{NamedTuple}}()
    for n in grid_sizes, chi in chi_values, (label, builder) in builders
        key = occursin("cavity", label) ? "$(label)_$(n)x1" : "$(label)_$(n)x$(n)"
        nb = builder(n, chi)
        r = report(label, nb)
        push!(get!(results, key, NamedTuple[]), (; n, chi, r...))
        grid_label = occursin("cavity", label) ? "$(n)×1" : "$(n)×$(n)"
        @printf(
            "%-22s grid=%s χ=%2d | tensors=%3d | open=%d | sc=2^%-5.1f | tc=2^%.1f\n",
            label,
            grid_label,
            chi,
            r.tensors,
            r.open_legs,
            r.sc,
            r.tc,
        )
    end

    md_path = joinpath(@__DIR__, "tnmp_contraction_sc_results.md")
    open(md_path, "w") do io
        println(io, "# TNMP contraction complexity (TreeSA)\n")
        println(io, "Generated by [`tnmp_contraction_sc.jl`](tnmp_contraction_sc.jl).\n")
        println(io, "## Conventions\n")
        println(io, "- **Green bond**: dim = 2")
        println(io, "- **Black bond χ**: dim ∈ {4, 8, 16, 32}")
        println(io, "- **Red cavity opening / center ▲/▼**: open external legs on T/B (not tensors)")
        println(io, "- **Blue/black boundary triangles / diamonds**: incoming message tensors")
        println(io, "- Optimizer: TreeSA (`ntrials=20`, `niters=60`, `βs=1:1:18`)\n")

        for chi in chi_values
            println(io, "## χ = $chi\n")
            println(io, "| Figure | Grid | tensors | sc | tc |")
            println(io, "|--------|------|--------:|---:|---:|")
            for n in grid_sizes, label in ("rank1_cavity", "rank1_neighborhood", "rank2_cavity", "rank2_neighborhood")
                key = occursin("cavity", label) ? "$(label)_$(n)x1" : "$(label)_$(n)x$(n)"
                row = only(filter(r -> r.chi == chi, results[key]))
                grid_label = occursin("cavity", label) ? "$(n)×1" : "$(n)×$(n)"
                @printf(io, "| %s | %s | %d | %.3f | %.3f |\n", label, grid_label, row.tensors, row.sc, row.tc)
            end
            println(io)
        end

        for n in grid_sizes
            cavity_label = "$(n)×1 cavity / $(n)×$(n) neighborhood"
            println(io, "### N = $n ($cavity_label)\n")
            println(io, "| Figure | χ=4 sc | χ=4 tc | χ=8 sc | χ=8 tc | χ=16 sc | χ=16 tc | χ=32 sc | χ=32 tc |")
            println(io, "|--------|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|")
            for label in ("rank1_cavity", "rank1_neighborhood", "rank2_cavity", "rank2_neighborhood")
                key = occursin("cavity", label) ? "$(label)_$(n)x1" : "$(label)_$(n)x$(n)"
                vals = String[]
                for chi in chi_values
                    row = only(filter(r -> r.chi == chi, results[key]))
                    push!(vals, string(round(row.sc; digits = 3)))
                    push!(vals, string(round(row.tc; digits = 3)))
                end
                println(io, "| $label | " * join(vals, " | ") * " |")
            end
            println(io)
        end
    end
    println("\nWrote $md_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
