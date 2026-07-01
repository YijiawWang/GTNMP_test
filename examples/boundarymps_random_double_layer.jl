#!/usr/bin/env julia
# Boundary-MPS centre-site marginal on a random double-layer PEPS.
#
# This example only handles the *instance* (random alpha PEPS) and *parameters*
# (CLI flags / output). The boundary-MPS solver itself lives in
# `TNMP_test/src/boundarymps.jl` (module `TNMPBoundaryMPS`).
#
# Self-contained example for TNMP_test. After running `./setup.sh`, run:
#
#   julia --project=TNMP_test TNMP_test/examples/boundarymps_random_double_layer.jl
#
# Optional CLI flags:
#   --L 10 --chi 8 --seed 7 --alpha 0.5
#   --bmps-chi-min 2 --bmps-chi-max 64 --bmps-epsilon 1e-4 --bmps-partition-by row
#   --output path/to/result.jls
#
# TensorNetworkQuantumSimulator is resolved from the active project dependency declared in
# `Project.toml`, which points at the local `TensorNetworkQuantumSimulator.jl` checkout.

module TNMPBoundaryMPSDemo

export BoundaryMPSConfig,
    parse_boundarymps_config,
    run_boundarymps_marginal

const _ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(_ROOT, "src", "boundarymps.jl"))
using .TNMPBoundaryMPS

include(joinpath(_ROOT, "examples", "state_models.jl"))

using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister
using Serialization: serialize

Base.@kwdef struct BoundaryMPSConfig
    L::Int = 6
    alpha::Float64 = 0.5
    chi::Int = 4
    seed::Int = 7
    bmps_chi_min::Int = 1
    bmps_chi_max::Int = 32
    bmps_epsilon::Float64 = 1e-4
    bmps_partition_by::String = "row"
    output::String = ""
end

function grid_center(L::Integer)
    c = (L + 1) ÷ 2
    return (c, c)
end

function parse_int_option(args::Vector{String}, key::AbstractString, default::Integer)
    flag = "--$key"
    for i in eachindex(args)
        args[i] == flag && return parse(Int, args[i + 1])
        startswith(args[i], "$flag=") && return parse(Int, args[i][(length(flag) + 2):end])
    end
    return default
end

function parse_float_option(args::Vector{String}, key::AbstractString, default::Float64)
    flag = "--$key"
    for i in eachindex(args)
        args[i] == flag && return parse(Float64, args[i + 1])
        startswith(args[i], "$flag=") && return parse(Float64, args[i][(length(flag) + 2):end])
    end
    return default
end

function parse_string_option(args::Vector{String}, key::AbstractString, default::AbstractString)
    flag = "--$key"
    for i in eachindex(args)
        args[i] == flag && return args[i + 1]
        startswith(args[i], "$flag=") && return args[i][(length(flag) + 2):end]
    end
    return default
end

function parse_boundarymps_config(args::Vector{String} = ARGS)
    return BoundaryMPSConfig(
        L = parse_int_option(args, "L", 6),
        alpha = parse_float_option(args, "alpha", 0.5),
        chi = parse_int_option(args, "chi", 4),
        seed = parse_int_option(args, "seed", 7),
        bmps_chi_min = parse_int_option(args, "bmps-chi-min", 1),
        bmps_chi_max = parse_int_option(args, "bmps-chi-max", 32),
        bmps_epsilon = parse_float_option(args, "bmps-epsilon", 1e-4),
        bmps_partition_by = parse_string_option(args, "bmps-partition-by", "row"),
        output = parse_string_option(args, "output", ""),
    )
end

function default_output_path(cfg::BoundaryMPSConfig)
    tag = "L$(cfg.L)_alpha$(cfg.alpha)_chi$(cfg.chi)_seed$(cfg.seed)_boundarymps.jls"
    return joinpath(_ROOT, "results", tag)
end

function save_result(path::AbstractString, payload)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(tmp, "w") do io
        serialize(io, payload)
    end
    mv(tmp, path; force = true)
    println("saved result -> $path")
    flush(stdout)
    return path
end

function run_boundarymps_marginal(cfg::BoundaryMPSConfig = BoundaryMPSConfig())
    rng = MersenneTwister(cfg.seed)
    g = named_grid((cfg.L, cfg.L))
    center = grid_center(cfg.L)

    println("sampling random alpha PEPS: L=$(cfg.L), chi=$(cfg.chi), alpha=$(cfg.alpha), center=$center")
    psi = random_alpha_state(
        rng, g;
        alpha = cfg.alpha,
        physical_dim = 2,
        bond_dim = cfg.chi,
    )

    println("sweeping boundary-MPS bond dimension up to $(cfg.bmps_chi_max) (epsilon=$(cfg.bmps_epsilon))")
    result = sweep_boundarymps_marginal(
        psi, center;
        chi_min = cfg.bmps_chi_min,
        chi_max = cfg.bmps_chi_max,
        epsilon = cfg.bmps_epsilon,
        partition_by = cfg.bmps_partition_by,
    )

    payload = Dict(
        "algorithm" => "boundarymps",
        "L" => cfg.L,
        "alpha" => cfg.alpha,
        "chi" => cfg.chi,
        "seed" => cfg.seed,
        "center" => collect(center),
        "bmps_chi_min" => cfg.bmps_chi_min,
        "bmps_chi_max" => cfg.bmps_chi_max,
        "bmps_epsilon" => cfg.bmps_epsilon,
        "bmps_partition_by" => cfg.bmps_partition_by,
        "converged_bmps_chi" => result.converged_bmps_chi,
        "final_bmps_chi" => result.final_bmps_chi,
        "final_marginal" => result.final_marginal,
        "history" => [
            Dict(
                "bmps_chi" => entry.bmps_chi,
                "marginal" => entry.marginal,
                "l1_delta_vs_prev" => entry.l1_delta_vs_prev,
            )
            for entry in result.history
        ],
    )

    out = isempty(cfg.output) ? default_output_path(cfg) : cfg.output
    save_result(out, payload)

    println("algorithm = boundarymps")
    println("center = $center")
    println("converged_bmps_chi = $(result.converged_bmps_chi)")
    for entry in result.history
        println(
            "bmps_chi = $(entry.bmps_chi), " *
            "marginal = $(entry.marginal), " *
            "l1_delta_vs_prev = $(entry.l1_delta_vs_prev)",
        )
    end
    return payload
end

function main(args::Vector{String} = ARGS)
    return run_boundarymps_marginal(parse_boundarymps_config(args))
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    TNMPBoundaryMPSDemo.main()
end
