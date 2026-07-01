#!/usr/bin/env julia
# Uniform-complex double-layer rank-2 TNMP marginal benchmark.
#
# Run with:
#   julia --project=TNMP_test scripts/double-layer-tn/src/run_tnmp_rank2_marginal.jl [options]

include(joinpath(@__DIR__, "marginal_common.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "tnmp.jl"))

using .TNMPTest
using Base.Threads
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister

include(joinpath(@__DIR__, "..", "..", "..", "examples", "state_models.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "examples", "neighborhoods.jl"))

function run_tnmp_rank2_marginal(cfg::UniformDoubleLayerRunConfig)
    progress_log(
        "tnmp_rank2 start: L=$(cfg.L), chi=$(cfg.chi), seed=$(cfg.seed), " *
        "uniform_lo=$(cfg.uniform_lo), uniform_hi=$(cfg.uniform_hi), " *
        "region_L=$(cfg.tnmp_region_L), max_iter=$(cfg.tnmp_max_iter), " *
        "run=$(cfg.tnmp_run) (false = TreeSA / complexity probe only), " *
        "threads=$(cfg.tnmp_nthreads <= 0 ? Threads.nthreads() : cfg.tnmp_nthreads)",
    )

    rng = MersenneTwister(cfg.seed)
    g = named_grid((cfg.L, cfg.L))
    center = grid_center(cfg.L)

    progress_log("tnmp_rank2: sampling uniform-complex PEPS at center=$center")
    psi = sample_uniform_double_layer_peps(rng, g, cfg)
    cache = TNMPCache(psi, cfg.tnmp_region_L; normalize = :l2, region_fn = grid_region_fn(cfg.tnmp_region_L))
    complexity_probe = isempty(cfg.tnmp_complexity_output) ? nothing : TNMPComplexityProbe()

    # Run TreeSA up front for BOTH the cavity message contractions and the
    # center neighborhood (marginal) contraction. This records the cavity and
    # neighborhood tc/sc before — and independently of — any message passing, so
    # at large chi we can read off the complexity without paying for the actual
    # contractions (see `cfg.tnmp_run`).
    progress_log("tnmp_rank2: pre-warming cavity + neighborhood contraction orders (TreeSA)")
    prewarm_kwargs = (; nthreads = cfg.tnmp_nthreads, progress_label = "tnmp_rank2/prewarm")
    complexity_probe === nothing ||
        (prewarm_kwargs = (; prewarm_kwargs..., complexity_probe))
    prewarm_rank2_tnmp_sequences!(cache, center; prewarm_kwargs...)

    info = nothing
    p_tnmp = nothing
    if cfg.tnmp_run
        progress_log("tnmp_rank2: global message passing (iterate)")
        global_mp_kwargs = tnmp_progress_kwargs(cfg, "tnmp_rank2/global-mp")
        complexity_probe === nothing ||
            (global_mp_kwargs = (; global_mp_kwargs..., complexity_probe))
        info = run_message_passing!(cache;
            max_iter = cfg.tnmp_max_iter,
            tol = cfg.tnmp_tol,
            skip_prewarm = true,
            global_mp_kwargs...,
        )
        progress_log(
            "tnmp_rank2: global MP finished converged=$(info.converged) " *
            "iterations=$(info.iterations) final_diff=$(info.final_diff)",
        )

        progress_log("tnmp_rank2: center marginal")
        marginal_kwargs = tnmp_progress_kwargs(cfg, "tnmp_rank2/marginal")
        complexity_probe === nothing ||
            (marginal_kwargs = (; marginal_kwargs..., complexity_probe))
        p_tnmp = tnmp_marginal(cache, center; marginal_kwargs...)
    else
        progress_log(
            "tnmp_rank2: tnmp_run=false -> skipping message passing + marginal " *
            "(cavity + neighborhood complexity already recorded)",
        )
    end

    payload = Dict(
        "algorithm" => "tnmp_rank2",
        "L" => cfg.L,
        "uniform_lo" => cfg.uniform_lo,
        "uniform_hi" => cfg.uniform_hi,
        "chi" => cfg.chi,
        "seed" => cfg.seed,
        "center" => collect(center),
        "tnmp_region_L" => cfg.tnmp_region_L,
        "tnmp_max_iter" => cfg.tnmp_max_iter,
        "tnmp_tol" => cfg.tnmp_tol,
        "tnmp_run" => cfg.tnmp_run,
        "tnmp_complexity_output" => cfg.tnmp_complexity_output,
        "converged" => info === nothing ? nothing : info.converged,
        "iterations" => info === nothing ? nothing : info.iterations,
        "final_diff" => info === nothing ? nothing : info.final_diff,
        "marginal" => p_tnmp === nothing ? nothing : collect(p_tnmp),
    )

    out = resolve_output_path(cfg, "tnmp_rank2")
    progress_log("tnmp_rank2: saving -> $out")
    save_marginal_result(out, payload)
    if complexity_probe !== nothing
        save_complexity_probe(cfg.tnmp_complexity_output, complexity_probe)
        println("tnmp_complexity_rows = $(length(complexity_rows(complexity_probe)))")
    end

    println("algorithm = tnmp_rank2")
    println("center = $center")
    println("tnmp_run = $(cfg.tnmp_run)")
    if cfg.tnmp_run
        println("tnmp_converged = $(info.converged)")
        println("tnmp_iterations = $(info.iterations)")
        println("tnmp_final_diff = $(info.final_diff)")
        println("tnmp_marginal = $p_tnmp")
    else
        println("tnmp_marginal = (skipped: tnmp_run=false)")
    end
    progress_log("tnmp_rank2 done")
    return payload
end

function main()
    cfg = parse_uniform_double_layer_config()
    return run_tnmp_rank2_marginal(cfg)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
