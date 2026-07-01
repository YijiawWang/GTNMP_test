#!/usr/bin/env julia
# CATN contraction of a PEPS double-layer single-site marginal.
#
# This mirrors the BP / TNMP benchmark instances: build the same random PEPS,
# pin the center physical state to each value, contract the resulting closed
# double-layer network with CATN, and normalize the two weights.

const CATN_ENV = joinpath(@__DIR__, "..", "..", "..", "env_catn")
isdir(CATN_ENV) && push!(LOAD_PATH, CATN_ENV)

using CATN: TensorNetwork, contraction!
using ITensors: ITensors, Index, inds
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs: vertices
using Random: MersenneTwister

include(joinpath(@__DIR__, "marginal_common.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "tnmp.jl"))

using .TNMPTest

include(joinpath(@__DIR__, "..", "..", "..", "examples", "state_models.jl"))

function sample_double_layer_peps(rng::MersenneTwister, g, init_mode::AbstractString, cfg)
    if init_mode == "alpha"
        return random_alpha_state(
            rng,
            g;
            alpha = cfg.alpha,
            physical_dim = cfg.physical_dim,
            bond_dim = cfg.chi,
        )
    elseif init_mode == "uniform_complex"
        return random_uniform_complex_state(
            rng,
            g;
            physical_dim = cfg.physical_dim,
            bond_dim = cfg.chi,
            lo = cfg.uniform_lo,
            hi = cfg.uniform_hi,
        )
    else
        throw(ArgumentError("unsupported init_mode=$init_mode"))
    end
end

function double_layer_result_tag(cfg, algorithm::AbstractString)
    if cfg.init_mode == "alpha"
        return "L$(cfg.L)_alpha$(compact_number(cfg.alpha))_chi$(cfg.chi)_seed$(cfg.seed)_$(algorithm).jls"
    elseif cfg.init_mode == "uniform_complex"
        return "L$(cfg.L)_uniform_complex_lo$(compact_number(cfg.uniform_lo))_hi$(compact_number(cfg.uniform_hi))_chi$(cfg.chi)_seed$(cfg.seed)_$(algorithm).jls"
    else
        throw(ArgumentError("unsupported init_mode=$(cfg.init_mode)"))
    end
end

function compact_number(x::Real)
    s = string(x)
    return endswith(s, ".0") ? s[1:(end - 2)] : s
end

function default_catn_output(cfg)
    return joinpath(MARGINAL_RESULTS_DIR, double_layer_result_tag(cfg, "catn"))
end

function index_label!(labels::Dict{Index,Int}, ind::Index)
    return get!(labels, ind) do
        length(labels) + 1
    end
end

function catn_tensors_and_ixs(factors::Vector{ITensors.ITensor})
    labels = Dict{Index,Int}()
    tensors = Array{ComplexF64}[]
    ixs = Vector{Int}[]

    for factor in factors
        local_inds = collect(inds(factor))
        push!(tensors, Array{ComplexF64}(ITensors.array(factor, local_inds...)))
        push!(ixs, [index_label!(labels, ind) for ind in local_inds])
    end

    counts = Dict{Int,Int}()
    for local_ixs in ixs, ix in local_ixs
        counts[ix] = get(counts, ix, 0) + 1
    end
    dangling = sort([ix => n for (ix, n) in counts if n != 2]; by = first)
    isempty(dangling) ||
        throw(ArgumentError("CATN expects a closed network; non-pair labels: $dangling"))

    return tensors, ixs
end

function catn_log_weight(factors; dmax::Int, catn_chi::Int)
    tensors, ixs = catn_tensors_and_ixs(factors)
    tn = TensorNetwork(tensors, ixs; Dmax = dmax, chi = catn_chi, select = 1, compress = true)
    lnZ, err, psi = contraction!(tn)
    return lnZ, err, psi
end

function normalize_catn_weights(log_weights::Vector{ComplexF64}, phases::Vector{ComplexF64})
    real_logs = real.(log_weights)
    offset = maximum(real_logs)
    weights = exp.(real_logs .- offset) .* phases
    probs = real.(weights ./ sum(weights))
    return TNMPTest.normalize_weights(max.(probs, 0.0))
end

function parse_catn_config(args::Vector{String} = ARGS)
    return (;
        L = parse_int_option(args, "L", 10),
        chi = parse_int_option(args, "chi", 16),
        seed = parse_int_option(args, "seed", 7),
        physical_dim = parse_int_option(args, "physical-dim", 2),
        init_mode = parse_string_option(args, "init-mode", "alpha"),
        alpha = parse_float_option(args, "alpha", 0.0),
        uniform_lo = parse_float_option(args, "uniform-lo", -0.5),
        uniform_hi = parse_float_option(args, "uniform-hi", 0.5),
        catn_dmax = parse_int_option(args, "catn-dmax", 64),
        catn_chi = parse_int_option(args, "catn-chi", 64),
        output = parse_string_option(args, "output", ""),
    )
end

function run_catn_double_layer_marginal(cfg)
    progress_log(
        "catn double-layer start: L=$(cfg.L), chi=$(cfg.chi), seed=$(cfg.seed), " *
        "init_mode=$(cfg.init_mode), alpha=$(cfg.alpha), uniform_lo=$(cfg.uniform_lo), " *
        "uniform_hi=$(cfg.uniform_hi), Dmax=$(cfg.catn_dmax), catn_chi=$(cfg.catn_chi)",
    )

    rng = MersenneTwister(cfg.seed)
    g = named_grid((cfg.L, cfg.L))
    center = grid_center(cfg.L)
    verts = collect(vertices(g))

    progress_log("catn: sampling PEPS at center=$center")
    psi = sample_double_layer_peps(rng, g, cfg.init_mode, cfg)

    log_weights = ComplexF64[]
    phases = ComplexF64[]
    errs = Float64[]
    elapsed_by_state = Float64[]

    for state in 1:cfg.physical_dim
        progress_log("catn: building pinned double-layer factors state=$state")
        factors = TNMPTest.marginal_factors(psi, verts, center, state)
        t0 = time()
        lnZ, err, phase = catn_log_weight(
            factors;
            dmax = cfg.catn_dmax,
            catn_chi = cfg.catn_chi,
        )
        elapsed = time() - t0
        push!(log_weights, ComplexF64(lnZ))
        push!(phases, ComplexF64(phase))
        push!(errs, Float64(real(err)))
        push!(elapsed_by_state, elapsed)
        progress_log(
            "catn: state=$state lnZ=$(real(lnZ)) phase=$phase " *
            "trunc_err=$(real(err)) elapsed=$(round(elapsed; digits=3))s",
        )
    end

    marginal = normalize_catn_weights(log_weights, phases)
    total_elapsed = sum(elapsed_by_state)
    progress_log("catn: marginal=$marginal total_elapsed=$(round(total_elapsed; digits=3))s")

    payload = Dict(
        "algorithm" => "catn",
        "L" => cfg.L,
        "chi" => cfg.chi,
        "seed" => cfg.seed,
        "center" => collect(center),
        "physical_dim" => cfg.physical_dim,
        "init_mode" => cfg.init_mode,
        "alpha" => cfg.alpha,
        "uniform_lo" => cfg.uniform_lo,
        "uniform_hi" => cfg.uniform_hi,
        "catn_dmax" => cfg.catn_dmax,
        "catn_chi" => cfg.catn_chi,
        "log_weights" => collect(log_weights),
        "phases" => collect(phases),
        "trunc_err" => maximum(errs),
        "elapsed_by_state" => round.(elapsed_by_state; digits = 4),
        "elapsed" => round(total_elapsed; digits = 4),
        "marginal" => collect(marginal),
    )

    out = isempty(cfg.output) ? default_catn_output(cfg) : cfg.output
    progress_log("catn: saving -> $out")
    save_marginal_result(out, payload)

    println("algorithm = catn")
    println("center = $center")
    println("catn_dmax = $(cfg.catn_dmax)")
    println("catn_chi = $(cfg.catn_chi)")
    println("catn_trunc_err = $(payload["trunc_err"])")
    println("catn_marginal = $marginal")
    progress_log("catn done")
    return payload
end

function main()
    cfg = parse_catn_config()
    return run_catn_double_layer_marginal(cfg)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
