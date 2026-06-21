#!/usr/bin/env julia
# TreeSA contraction complexity for GBP sub-TNs (cavity / neighborhood).
# Network topology matches plot_4x4x2_tensor_network.py with
#   --group-boundary-sides
# and the figure conventions in gbp_cavity.png / gbp_neighborhood.png.

using Pkg
Pkg.activate(normpath(@__DIR__, ".."); io = devnull)

using OMEinsumContractionOrders:
    EinCode, optimize_code, contraction_complexity, TreeSA, peak_memory
using Printf: @printf, @sprintf

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

# Rank-1 incoming message leaf (blue triangle in figures).
function caps!(nb::NetBuilder, c, k::Int, d::Int)
    for _ in 1:k
        nb.ntri += 1
        edge!(nb, c, (:tri, nb.ntri), d)
    end
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

function max_input_log2(ec::EinCode, dims::Dict{Int, Int})
    return maximum(ix -> isempty(ix) ? 0.0 : sum(l -> log2(dims[l]), ix), ec.ixs)
end

function center_xy(n::Int)
    cx = (n - 1) ÷ 2
    cy = (n - 1) ÷ 2
    return cx, cy
end

function add_lattice_core!(nb::NetBuilder, n::Int, dim_chi::Int, dim_green::Int; break_center::Bool = false)
    T(x, y) = (:T, x, y)
    B(x, y) = (:B, x, y)
    cx, cy = center_xy(n)

    for y in 0:(n - 1), x in 0:(n - 2)
        edge!(nb, T(x, y), T(x + 1, y), dim_chi)
        edge!(nb, B(x, y), B(x + 1, y), dim_chi)
    end
    for x in 0:(n - 1), y in 0:(n - 2)
        edge!(nb, T(x, y), T(x, y + 1), dim_chi)
        edge!(nb, B(x, y), B(x, y + 1), dim_chi)
    end
    for y in 0:(n - 1), x in 0:(n - 1)
        if break_center && x == cx && y == cy
            open_leg!(nb, T(x, y), dim_chi)
            open_leg!(nb, B(x, y), dim_chi)
        else
            edge!(nb, T(x, y), B(x, y), dim_green)
        end
    end
    return nb
end

function merged_side!(nb::NetBuilder, n::Int, direction::Symbol, dim_chi::Int)
    t = (:side, direction)
    if direction === :xp
        for y in 0:(n - 1)
            edge!(nb, (:T, n - 1, y), t, dim_chi)
            edge!(nb, (:B, n - 1, y), t, dim_chi)
        end
    elseif direction === :xm
        for y in 0:(n - 1)
            edge!(nb, (:T, 0, y), t, dim_chi)
            edge!(nb, (:B, 0, y), t, dim_chi)
        end
    elseif direction === :yp
        for x in 0:(n - 1)
            edge!(nb, (:T, x, n - 1), t, dim_chi)
            edge!(nb, (:B, x, n - 1), t, dim_chi)
        end
    elseif direction === :ym
        for x in 0:(n - 1)
            edge!(nb, (:T, x, 0), t, dim_chi)
            edge!(nb, (:B, x, 0), t, dim_chi)
        end
    else
        error("unknown direction $direction")
    end
    return t
end

function open_side!(nb::NetBuilder, n::Int, direction::Symbol, dim_chi::Int)
    if direction === :xm
        for y in 0:(n - 1)
            open_leg!(nb, (:T, 0, y), dim_chi)
            open_leg!(nb, (:B, 0, y), dim_chi)
        end
    elseif direction === :xp
        for y in 0:(n - 1)
            open_leg!(nb, (:T, n - 1, y), dim_chi)
            open_leg!(nb, (:B, n - 1, y), dim_chi)
        end
    elseif direction === :yp
        for x in 0:(n - 1)
            open_leg!(nb, (:T, x, n - 1), dim_chi)
            open_leg!(nb, (:B, x, n - 1), dim_chi)
        end
    elseif direction === :ym
        for x in 0:(n - 1)
            open_leg!(nb, (:T, x, 0), dim_chi)
            open_leg!(nb, (:B, x, 0), dim_chi)
        end
    else
        error("unknown direction $direction")
    end
end

# GBP message-update cavity: one open face (x−); red triangles = open external legs
# on T/B (not tensors).  Other three faces carry one merged rank-(2n) message each.
function build_gbp_cavity(n::Int, dim_chi::Int; dim_green::Int = DIM_GREEN)
    nb = NetBuilder()
    add_lattice_core!(nb, n, dim_chi, dim_green; break_center = false)
    open_side!(nb, n, :xm, dim_chi)
    for dir in (:xp, :yp, :ym)
        merged_side!(nb, n, dir, dim_chi)
    end
    return nb
end

# GBP marginal neighborhood: all four faces grouped; center inter-layer bond
# broken into open external legs on T/B (▲/▼ in figure, not tensors).
function build_gbp_neighborhood(n::Int, dim_chi::Int; dim_green::Int = DIM_GREEN)
    nb = NetBuilder()
    add_lattice_core!(nb, n, dim_chi, dim_green; break_center = true)
    for dir in (:xp, :xm, :yp, :ym)
        merged_side!(nb, n, dir, dim_chi)
    end
    return nb
end

function leaf_count(name::String, n::Int)
    site_tensors = 2 * n * n
    merged = name == "gbp_cavity" ? 3 : 4
    return site_tensors + merged
end

function report(name::String, nb::NetBuilder; opt = TreeSA(; ntrials = 20, niters = 60, βs = 1.0:1.0:18.0))
    ec, dims, nnodes = eincode(nb)
    code = optimize_code(ec, dims, opt)
    cc = contraction_complexity(code, dims)
    sc_in = max_input_log2(ec, dims)
    pm = log2(peak_memory(code, dims))
    return (;
        name,
        tensors = nnodes,
        max_input = sc_in,
        sc = cc.sc,
        peak_memory = pm,
        tc = cc.tc,
    )
end

function main()
    chi_values = [8, 16, 32]
    grid_sizes = [2, 3]

    println("=== GBP sub-TN contraction complexity (TreeSA) ===")
    println("green bond = dim $DIM_GREEN, black bond = dim χ")
    println("cavity: open x− face (external legs on T/B, rank-2n result)")
    println("neighborhood: all four faces merged; center ▲/▼ = open external legs\n")

    results = Dict{String, Vector{NamedTuple}}()
    for n in grid_sizes, chi in chi_values
        for (builder, label) in (
            (build_gbp_cavity, "gbp_cavity"),
            (build_gbp_neighborhood, "gbp_neighborhood"),
        )
            key = "$(label)_$(n)x$(n)"
            r = report(label, builder(n, chi))
            push!(get!(results, key, NamedTuple[]), (; n, chi, r...))
            @printf(
                "%-22s n=%d χ=%2d | tensors=%3d | max_input=2^%-5.1f | sc=2^%-5.1f | peak=2^%-5.1f | tc=2^%.1f\n",
                label,
                n,
                chi,
                r.tensors,
                r.max_input,
                r.sc,
                r.peak_memory,
                r.tc,
            )
        end
    end

    md_path = joinpath(@__DIR__, "gbp_contraction_sc_results.md")
    open(md_path, "w") do io
        println(io, "# GBP contraction complexity (TreeSA)\n")
        println(io, "Generated by [`gbp_contraction_sc.jl`](gbp_contraction_sc.jl).\n")
        println(io, "## Conventions\n")
        println(io, "- **Green bond** (inter-layer physical leg): dim = 2")
        println(io, "- **Black bond** (virtual bond χ): dim ∈ {8, 16, 32}")
        println(io, "- **Cavity** (`gbp_cavity.png`): n×n×2 lattice; x− face open (red triangles = external legs on T/B, not tensors); centre inter-layer bond intact (green, dim 2); other three faces one merged boundary tensor each")
        println(io, "- **Neighborhood** (`gbp_neighborhood.png`): n×n×2 lattice; four grouped face messages; center inter-layer bond → open external legs on T/B (▲/▼, not tensors)")
        println(io, "- **sc** = log₂(largest intermediate tensor elements)")
        println(io, "- **tc** = log₂(total contraction FLOPs)")
        println(io, "- Optimizer: TreeSA (`ntrials=20`, `niters=60`, `βs=1:1:18`)\n")

        for chi in chi_values
            println(io, "## χ = $chi\n")
            println(io, "| Figure | Grid | tensors | sc | tc | sc / log₂(χ) |")
            println(io, "|--------|------|--------:|---:|---:|-------------:|")
            for n in grid_sizes
                for label in ("gbp_cavity", "gbp_neighborhood")
                    key = "$(label)_$(n)x$(n)"
                    row = only(filter(r -> r.chi == chi, results[key]))
                    sc_ratio = row.sc / log2(chi)
                    @printf(
                        io,
                        "| %s | %d×%d | %d | %.3f | %.3f | %.2f |\n",
                        label,
                        n,
                        n,
                        row.tensors,
                        row.sc,
                        row.tc,
                        sc_ratio,
                    )
                end
            end
            println(io)
        end

        println(io, "## Summary: sc vs χ\n")
        for n in grid_sizes
            println(io, "### n = $n ($(n)×$(n) neighborhood / cavity)\n")
            println(io, "| Figure | χ=8 | χ=16 | χ=32 |")
            println(io, "|--------|----:|-----:|-----:|")
            for label in ("gbp_cavity", "gbp_neighborhood")
                key = "$(label)_$(n)x$(n)"
                vals = [only(filter(r -> r.chi == chi, results[key])).sc for chi in chi_values]
                @printf(io, "| %s | %.3f | %.3f | %.3f |\n", label, vals...)
            end
            println(io)
        end
    end
    println("\nWrote $md_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
