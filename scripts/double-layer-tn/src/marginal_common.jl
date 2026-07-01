# Shared helpers for the uniform-complex double-layer benchmarks.

using Dates: Dates, now
using Random: MersenneTwister
using Serialization: serialize

const MARGINAL_RESULTS_DIR = joinpath(@__DIR__, "..", "results")

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

function progress_log(msg::AbstractString)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    println("[progress $ts] ", msg)
    flush(stdout)
    return nothing
end

Base.@kwdef struct UniformDoubleLayerRunConfig
    L::Int = 10
    chi::Int = 8
    seed::Int = 7
    uniform_lo::Float64 = -0.5
    uniform_hi::Float64 = 0.5
    output::String = ""
    bp_max_iter::Int = 2000
    bp_tol::Float64 = 1e-10
    tnmp_region_L::Int = 3
    tnmp_max_iter::Int = 500
    tnmp_tol::Float64 = 1e-8
    # When false, only TreeSA is run (cavity + neighborhood contraction orders /
    # tc / sc are recorded) and the actual message passing + marginal contraction
    # are skipped. Useful for probing complexity at large chi without paying for
    # the contractions.
    tnmp_run::Bool = true
    bmps_chi_min::Int = 1
    bmps_chi_max::Int = 256
    bmps_epsilon::Float64 = 1e-4
    bmps_partition_by::String = "row"
    progress_interval::Int = 10
    tnmp_nthreads::Int = 0
    tnmp_complexity_output::String = ""
end

# `random_uniform_complex_state` is provided by `examples/state_models.jl`,
# which the run scripts include after `src/tnmp.jl`. It is resolved at call time
# (this helper is only invoked from inside the run functions).
function sample_uniform_double_layer_peps(rng::MersenneTwister, g, cfg::UniformDoubleLayerRunConfig, tnmp = nothing)
    cfg.uniform_lo <= cfg.uniform_hi ||
        throw(ArgumentError("uniform_lo must be <= uniform_hi"))
    return random_uniform_complex_state(
        rng, g;
        physical_dim = 2,
        bond_dim = cfg.chi,
        lo = cfg.uniform_lo,
        hi = cfg.uniform_hi,
    )
end

function uniform_result_tag(cfg::UniformDoubleLayerRunConfig)
    return "uniform_complex_lo$(cfg.uniform_lo)_hi$(cfg.uniform_hi)"
end

function tnmp_progress_kwargs(cfg::UniformDoubleLayerRunConfig, label::AbstractString)
    kwargs = NamedTuple()
    if cfg.progress_interval > 0
        kwargs = (;
            progress_interval = cfg.progress_interval,
            progress_label = label,
        )
    end
    return (; kwargs..., nthreads = cfg.tnmp_nthreads)
end

function parse_uniform_double_layer_config(args::Vector{String} = ARGS)
    return UniformDoubleLayerRunConfig(
        L = parse_int_option(args, "L", 10),
        chi = parse_int_option(args, "chi", 8),
        seed = parse_int_option(args, "seed", 7),
        uniform_lo = parse_float_option(args, "uniform-lo", -0.5),
        uniform_hi = parse_float_option(args, "uniform-hi", 0.5),
        output = parse_string_option(args, "output", ""),
        bp_max_iter = parse_int_option(args, "bp-max-iter", 2000),
        bp_tol = parse_float_option(args, "bp-tol", 1e-10),
        tnmp_region_L = parse_int_option(args, "tnmp-region-L", 3),
        tnmp_max_iter = parse_int_option(args, "tnmp-max-iter", 500),
        tnmp_tol = parse_float_option(args, "tnmp-tol", 1e-8),
        tnmp_run = parse_int_option(args, "tnmp-run", 1) != 0,
        bmps_chi_min = parse_int_option(args, "bmps-chi-min", 1),
        bmps_chi_max = parse_int_option(args, "bmps-chi-max", 256),
        bmps_epsilon = parse_float_option(args, "bmps-epsilon", 1e-4),
        bmps_partition_by = parse_string_option(args, "bmps-partition-by", "row"),
        progress_interval = parse_int_option(args, "progress-interval", 10),
        tnmp_nthreads = parse_int_option(args, "tnmp-nthreads", 0),
        tnmp_complexity_output = parse_string_option(args, "tnmp-complexity-output", ""),
    )
end

function default_result_path(cfg::UniformDoubleLayerRunConfig, algorithm::AbstractString)
    tag = "L$(cfg.L)_$(uniform_result_tag(cfg))_chi$(cfg.chi)_seed$(cfg.seed)_$(algorithm).jls"
    return joinpath(MARGINAL_RESULTS_DIR, tag)
end

function resolve_output_path(cfg::UniformDoubleLayerRunConfig, algorithm::AbstractString)
    path = isempty(cfg.output) ? default_result_path(cfg, algorithm) : cfg.output
    mkpath(dirname(path))
    return path
end

function save_marginal_result(path::AbstractString, payload)
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
