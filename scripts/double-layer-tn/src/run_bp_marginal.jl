#!/usr/bin/env julia
# Uniform-complex double-layer single-site BP marginal benchmark.
#
# Run with:
#   julia --project=TNMP_test scripts/double-layer-tn/src/run_bp_marginal.jl [options]

include(joinpath(@__DIR__, "marginal_common.jl"))

using Dictionaries: Dictionary
using ITensors: ITensors, dim, Algorithm
using LinearAlgebra: diag
using NamedGraphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister
using Base.Threads
using TensorNetworkQuantumSimulator

include(joinpath(@__DIR__, "..", "..", "..", "src", "tnmp.jl"))
using .TNMPTest
include(joinpath(@__DIR__, "..", "..", "..", "examples", "state_models.jl"))

ITensors.disable_warn_order()

function normalize_log_weights(log_weights::Vector{Float64})
    isempty(log_weights) && throw(ArgumentError("Cannot normalize an empty marginal"))
    offset = maximum(log_weights)
    if !isfinite(offset)
        throw(ArgumentError("Cannot normalize non-finite log weights: $log_weights"))
    end
    weights = exp.(log_weights .- offset)
    return TNMPTest.normalize_weights(weights)
end

function to_tnqs_state(psi::TNMPTest.TensorNetworkState)
    g = TNMPTest.graph(psi)
    vs = collect(vertices(g))
    tensors = Dictionary(vs, [psi[v] for v in vs])
    return TensorNetworkQuantumSimulator.TensorNetworkState(tensors, g)
end

function bp_marginal_weights(
    psi::TNMPTest.TensorNetworkState,
    target;
    maxiter::Integer,
    tolerance::Real,
    progress_label::AbstractString = "",
)
    tnqs_psi = to_tnqs_state(psi)
    !isempty(progress_label) &&
        (println("[$progress_label] running BP on full double-layer state"); flush(stdout))
    cache = BeliefPropagationCache(tnqs_psi)
    updated, info = update_bp_with_info(cache; maxiter, tolerance)
    ρ = reduced_density_matrix(Algorithm("bp"), updated, [target])
    ρ_diag = collect(real.(diag(ITensors.array(ρ))))
    ρ_diag = max.(ρ_diag, 0.0)
    return TNMPTest.normalize_weights(ρ_diag), info
end

function update_bp_with_info(cache; maxiter::Integer, tolerance::Real)
    alg = TensorNetworkQuantumSimulator.set_default_kwargs(
        ITensors.Algorithm("bp"; maxiter = Int(maxiter), tolerance = Float64(tolerance)),
        cache,
    )
    updated = copy(cache)
    TensorNetworkQuantumSimulator.invalidate_contraction_sequences!(updated)

    final_diff = Inf
    iterations = 0
    converged = false
    edge_sequence = alg.kwargs.edge_sequence
    for it in 1:Int(maxiter)
        diff = Ref(0.0)
        TensorNetworkQuantumSimulator.update_iteration!(
            alg, updated, edge_sequence; update_diff! = diff,
        )
        final_diff = diff[] / length(edge_sequence)
        iterations = it
        if final_diff <= Float64(tolerance)
            converged = true
            break
        end
    end

    TensorNetworkQuantumSimulator.invalidate_contraction_sequences!(updated)
    return updated, (; converged, iterations, final_diff)
end

function run_bp_marginal(cfg::UniformDoubleLayerRunConfig)
    progress_log(
        "bp start: L=$(cfg.L), chi=$(cfg.chi), seed=$(cfg.seed), " *
        "uniform_lo=$(cfg.uniform_lo), uniform_hi=$(cfg.uniform_hi), " *
        "max_iter=$(cfg.bp_max_iter), " *
        "threads=$(cfg.tnmp_nthreads <= 0 ? Threads.nthreads() : cfg.tnmp_nthreads)",
    )

    rng = MersenneTwister(cfg.seed)
    g = named_grid((cfg.L, cfg.L))
    center = grid_center(cfg.L)

    progress_log("bp: sampling uniform-complex PEPS at center=$center")
    ψ = sample_uniform_double_layer_peps(rng, g, cfg, TNMPTest)

    progress_log("bp: computing double-layer marginal")
    bp_marg, bp_info = bp_marginal_weights(
        ψ, center;
        maxiter = cfg.bp_max_iter,
        tolerance = cfg.bp_tol,
        progress_label = cfg.progress_interval > 0 ? "bp" : "",
    )
    progress_log(
        "bp: double-layer BP finished converged=$(bp_info.converged) " *
        "iterations=$(bp_info.iterations) final_diff=$(bp_info.final_diff)",
    )

    payload = Dict(
        "algorithm" => "bp",
        "L" => cfg.L,
        "uniform_lo" => cfg.uniform_lo,
        "uniform_hi" => cfg.uniform_hi,
        "chi" => cfg.chi,
        "seed" => cfg.seed,
        "center" => collect(center),
        "converged" => bp_info.converged,
        "iterations" => bp_info.iterations,
        "final_diff" => bp_info.final_diff,
        "bp_max_iter" => cfg.bp_max_iter,
        "bp_tol" => cfg.bp_tol,
        "marginal" => collect(bp_marg),
    )

    out = resolve_output_path(cfg, "bp")
    progress_log("bp: saving -> $out")
    save_marginal_result(out, payload)

    println("algorithm = bp")
    println("center = $center")
    println("bp_converged = $(bp_info.converged)")
    println("bp_iterations = $(bp_info.iterations)")
    println("bp_final_diff = $(bp_info.final_diff)")
    println("bp_max_iter = $(cfg.bp_max_iter)")
    println("bp_tolerance = $(cfg.bp_tol)")
    println("bp_marginal = $bp_marg")
    progress_log("bp done")
    return payload
end

function main()
    cfg = parse_uniform_double_layer_config()
    return run_bp_marginal(cfg)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
