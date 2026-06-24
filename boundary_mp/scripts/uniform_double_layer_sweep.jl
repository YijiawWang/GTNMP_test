#!/usr/bin/env julia
# Center-site marginal of a uniform-random real state via exact (TreeSA), SVD-BMPS, and
# exact-environment full-update BMPS. Depends on TensorNetworkQuantumSimulator.jl on the
# `treesa` branch (v0.3.10). Run:
#   julia --project=TNMP_test/TensorNetworkQuantumSimulator.jl \
#       TNMP_test/boundary_mp/scripts/uniform_double_layer_sweep.jl

using Printf
using Random

const _SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(_SRC, "TNQSBoundaryMP.jl"));         using .TNQSBoundaryMP
include(joinpath(_SRC, "ExactSolver.jl"));            using .ExactSolver
include(joinpath(_SRC, "ExactEnvFullUpdateBMPS.jl")); using .ExactEnvFullUpdateBMPS

function parse_arg(args, name, default)
    flag = "--$name"
    for i in eachindex(args)
        args[i] == flag && return args[i + 1]
        startswith(args[i], "$flag=") && return args[i][(length(flag) + 2):end]
    end
    return default
end
parse_chis(t) = parse.(Int, split(t, ","))

function csv_row(values)
    join(string.(values), ",")
end

function write_sweep_csv(
    path,
    L,
    bond_dim,
    physical_dim,
    lo,
    hi,
    seed,
    nsweeps,
    chis,
    p_exact,
    rows,
)
    d = length(p_exact)
    marginal_cols = ["marginal_$i" for i in 0:(d - 1)]
    header = join(
        vcat(
            ["L", "bond_dim", "physical_dim", "lo", "hi", "seed", "nsweeps", "method", "bmps_chi", "seconds", "l1_error_vs_exact"],
            marginal_cols,
        ),
        ",",
    )
    open(path, "w") do io
        println(io, header)
        meta = (L, bond_dim, physical_dim, lo, hi, seed, nsweeps)
        for row in rows
            method, chi, seconds, l1_err, p = row
            chi_str = chi === nothing ? "" : string(chi)
            println(io, csv_row(vcat(collect(meta), [method, chi_str, seconds, l1_err], p)))
        end
    end
end

function main(args = ARGS)
    L = parse(Int, parse_arg(args, "L", "4"))
    bond_dim = parse(Int, parse_arg(args, "bond-dim", "2"))
    physical_dim = parse(Int, parse_arg(args, "physical-dim", "2"))
    lo = parse(Float64, parse_arg(args, "lo", "-0.3"))
    hi = parse(Float64, parse_arg(args, "hi", "0.5"))
    seed = parse(Int, parse_arg(args, "seed", "1234"))
    chis = parse_chis(parse_arg(args, "bmps-chis", "1,2,4,8"))
    nsweeps = parse(Int, parse_arg(args, "nsweeps", "4"))
    output = parse_arg(args, "output", "")

    state = uniform_state(MersenneTwister(seed), L; bond_dim, physical_dim, lo, hi)
    c = center_vertex(state)
    t_exact = @elapsed p_exact = exact_marginal(state, c)

    println("center marginal sweep (uniform double-layer norm network)")
    println("L=$L  single_layer_bond=$bond_dim  double_layer_bond=$(bond_dim^2)  d=$physical_dim")
    println("entry_distribution=Uniform($lo,$hi)  seed=$seed  nsweeps=$nsweeps")
    @printf("exact_marginal=[%s]\n", join((@sprintf("%.10f", x) for x in p_exact), ", "))
    println()
    println("method,bmps_chi,marginal_1,l1_error_vs_exact,seconds")
    l1(p) = sum(abs.(p .- p_exact))
    rows = Tuple{String, Union{Int, Nothing}, Float64, Float64, Vector{Float64}}[]
    push!(rows, ("exact", nothing, t_exact, 0.0, collect(p_exact)))
    for chi in chis
        t1 = @elapsed p_svd = bmps_marginal(state, c, chi)
        @printf("svd,%d,%.10f,%.3e,%.3f\n", chi, p_svd[1], l1(p_svd), t1)
        push!(rows, ("svd", chi, t1, l1(p_svd), collect(p_svd)))
        t2 = @elapsed p_fu = full_update_marginal(state, c, chi; nsweeps)
        @printf("full_update,%d,%.10f,%.3e,%.3f\n", chi, p_fu[1], l1(p_fu), t2)
        push!(rows, ("full_update", chi, t2, l1(p_fu), collect(p_fu)))
    end
    if !isempty(output)
        write_sweep_csv(output, L, bond_dim, physical_dim, lo, hi, seed, nsweeps, chis, p_exact, rows)
        println()
        println("wrote CSV: $output")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
