#!/usr/bin/env julia
# Uniform-complex double-layer boundary-MPS marginal benchmark.
#
# Estimates the centre marginal by projecting |s><s| bond dimension until
# successive marginals differ by less than epsilon.
#
# Run with:
#   julia --project=TNMP_test scripts/double-layer-tn/src/run_boundarymps_marginal.jl [options]

include(joinpath(@__DIR__, "marginal_common.jl"))

using Dictionaries: Dictionary
using ITensors: ITensors, dim
using NamedGraphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator: freenergy

# Guard the include so an already-loaded `TNMPTest` (e.g. when this script is
# pulled in by the test suite) is reused rather than replaced. Re-including the
# module would create a second `TNMPTest`/`TensorNetworkState`/`TNMPCache` type
# with the same name, breaking method dispatch for callers holding the original.
if !isdefined(@__MODULE__, :TNMPTest)
    include(joinpath(@__DIR__, "..", "..", "..", "src", "tnmp.jl"))
end
using .TNMPTest
if !isdefined(@__MODULE__, :random_uniform_complex_state)
    include(joinpath(@__DIR__, "..", "..", "..", "examples", "state_models.jl"))
end

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

function to_tnqs_projected_network(psi::TNMPTest.TensorNetworkState, target, state::Integer)
    vs = collect(vertices(TNMPTest.graph(psi)))
    tensors = Dictionary(
        vs,
        [
            v == target ?
                reduce(*, TNMPTest.fixed_site_norm_factors(psi, v, state)) :
                reduce(*, TNMPTest.traced_norm_factors(psi, v))
            for v in vs
        ],
    )
    return TensorNetwork(tensors, TNMPTest.graph(psi))
end

function boundarymps_marginal_weights(
        psi::TNMPTest.TensorNetworkState,
        target,
        bmps_chi::Integer;
        partition_by::AbstractString,
        bmps_update_kwargs,
        progress_label::AbstractString = "",
    )
    d = dim(only(TNMPTest.siteinds(psi, target)))
    log_weights = Float64[]
    for s in 1:d
        !isempty(progress_label) &&
            (println("[$progress_label] spin state $s/$d"); flush(stdout))
        net = to_tnqs_projected_network(psi, target, s)
        cache = BoundaryMPSCache(net, bmps_chi; partition_by)
        value = freenergy(update(cache; bmps_update_kwargs...))
        push!(log_weights, real(value))
    end
    return normalize_log_weights(log_weights)
end

function next_bmps_chi(chi::Integer, chi_max::Integer)
    chi >= chi_max && return nothing
    nxt = min(chi * 2, chi_max)
    return nxt == chi ? nothing : nxt
end

function boundarymps_result_tuple(
        history::Vector{Dict{String, Any}},
        converged_at,
        complete::Bool,
    )
    final_entry = history[end]
    return (
        history = history,
        converged_bmps_chi = converged_at,
        final_bmps_chi = final_entry["bmps_chi"],
        final_marginal = final_entry["marginal"],
        complete = complete,
    )
end

function build_boundarymps_payload(cfg::UniformDoubleLayerRunConfig, center, result)
    return Dict(
        "algorithm" => "boundarymps",
        "L" => cfg.L,
        "uniform_lo" => cfg.uniform_lo,
        "uniform_hi" => cfg.uniform_hi,
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
        "history" => result.history,
        "complete" => result.complete,
    )
end

function sweep_boundarymps_marginal(
        psi::TNMPTest.TensorNetworkState,
        center;
        chi_min::Integer = 1,
        chi_max::Integer,
        epsilon::Real,
        partition_by::AbstractString,
        bmps_update_kwargs,
        progress_interval::Integer = 0,
        on_step = nothing,
    )
    history = Dict{String, Any}[]
    chi_min > 0 || throw(ArgumentError("chi_min must be positive, got $chi_min"))
    chi_max >= chi_min ||
        throw(ArgumentError("chi_max must be >= chi_min, got chi_min=$chi_min chi_max=$chi_max"))
    chi = chi_min
    prev_marg = nothing
    converged_at = nothing
    step = 0

    while chi <= chi_max
        step += 1
        progress_interval > 0 &&
            progress_log("boundarymps: sweep step $step, bmps_chi=$chi")
        marg = boundarymps_marginal_weights(
            psi, center, chi;
            partition_by, bmps_update_kwargs,
            progress_label = progress_interval > 0 ? "boundarymps/chi=$chi" : "",
        )
        delta_prev = prev_marg === nothing ? Inf : sum(abs.(marg .- prev_marg))
        push!(history, Dict(
            "bmps_chi" => chi,
            "marginal" => collect(marg),
            "l1_delta_vs_prev" => delta_prev,
        ))
        progress_interval > 0 &&
            progress_log(
                "boundarymps: chi=$chi marginal=$marg l1_delta_vs_prev=$delta_prev",
            )

        partial = boundarymps_result_tuple(history, converged_at, false)
        on_step !== nothing && on_step(partial)

        if prev_marg !== nothing && delta_prev < epsilon
            converged_at = chi
            progress_interval > 0 &&
                progress_log("boundarymps: converged at chi=$chi (delta=$delta_prev < epsilon=$epsilon)")
            break
        end

        prev_marg = marg
        nxt = next_bmps_chi(chi, chi_max)
        nxt === nothing && break
        chi = nxt
    end

    return boundarymps_result_tuple(history, converged_at, true)
end

function run_boundarymps_marginal(cfg::UniformDoubleLayerRunConfig)
    progress_log(
        "boundarymps start: L=$(cfg.L), chi=$(cfg.chi), seed=$(cfg.seed), " *
        "uniform_lo=$(cfg.uniform_lo), uniform_hi=$(cfg.uniform_hi), " *
        "bmps_chi_min=$(cfg.bmps_chi_min), " *
        "bmps_chi_max=$(cfg.bmps_chi_max), epsilon=$(cfg.bmps_epsilon)",
    )

    rng = MersenneTwister(cfg.seed)
    g = named_grid((cfg.L, cfg.L))
    center = grid_center(cfg.L)

    progress_log("boundarymps: sampling uniform-complex PEPS at center=$center")
    psi = sample_uniform_double_layer_peps(rng, g, cfg, TNMPTest)

    out = resolve_output_path(cfg, "boundarymps")
    checkpoint_save!(partial) = begin
        progress_log("boundarymps: checkpoint bmps_chi=$(partial.final_bmps_chi) -> $out")
        save_marginal_result(out, build_boundarymps_payload(cfg, center, partial))
    end

    bmps_update_kwargs = (;)
    progress_log("boundarymps: sweeping boundary-MPS chi")
    result = sweep_boundarymps_marginal(
        psi, center;
        chi_min = cfg.bmps_chi_min,
        chi_max = cfg.bmps_chi_max,
        epsilon = cfg.bmps_epsilon,
        partition_by = cfg.bmps_partition_by,
        bmps_update_kwargs,
        progress_interval = cfg.progress_interval,
        on_step = checkpoint_save!,
    )

    payload = build_boundarymps_payload(cfg, center, result)
    progress_log("boundarymps: saving final -> $out")
    save_marginal_result(out, payload)

    println("algorithm = boundarymps")
    println("center = $center")
    println("bmps_epsilon = $(cfg.bmps_epsilon)")
    println("converged_bmps_chi = $(result.converged_bmps_chi)")
    for entry in result.history
        println(
            "bmps_chi = $(entry["bmps_chi"]), " *
            "marginal = $(entry["marginal"]), " *
            "l1_delta_vs_prev = $(entry["l1_delta_vs_prev"])",
        )
    end
    progress_log("boundarymps done")
    return payload
end

function main()
    cfg = parse_uniform_double_layer_config()
    return run_boundarymps_marginal(cfg)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
