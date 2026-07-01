# Fully-frustrated Ising double-layer marginal: BP vs TNMP rank-2.
#
# Builds the fully-frustrated pair-factor PEPS (see `double_layer_ising.jl`)
# and contrasts standard belief propagation (BP, single-site regions, run via
# TensorNetworkQuantumSimulator) against rank-2 TNMP on L*L grid windows. On
# this frustrated instance BP fails to converge while TNMP rank-2 does, so the
# two single-site marginals disagree.
#
# Quick local run (defaults reproduce the K=1.0 batch entry):
#   julia --project=TNMP_test TNMP_test/examples/fully_frustrated_bp_tnmp_rank2.jl
#
# Override via ARGS (key=value), e.g.:
#   julia ... fully_frustrated_bp_tnmp_rank2.jl L=10 chi=2 K=1.0 field=0.2 seed=7

include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))

using .TNMPTest
using NamedGraphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister

# Include the model/neighborhood definitions *before* `using
# TensorNetworkQuantumSimulator`: state_models.jl aliases the unexported
# `TensorNetworkState` to `TNMPTest.TensorNetworkState`, and that alias must win
# over the one TensorNetworkQuantumSimulator exports.
include(joinpath(@__DIR__, "state_models.jl"))
include(joinpath(@__DIR__, "neighborhoods.jl"))

using Dictionaries: Dictionary
using ITensors: ITensors, Algorithm
using LinearAlgebra: diag
using TensorNetworkQuantumSimulator

ITensors.disable_warn_order()

# Wrap a `TNMPTest.TensorNetworkState` into the TNS state BP operates on.
function to_tnqs_state(psi::TNMPTest.TensorNetworkState)
    g = TNMPTest.graph(psi)
    vs = collect(vertices(g))
    tensors = Dictionary(vs, [psi[v] for v in vs])
    return TensorNetworkQuantumSimulator.TensorNetworkState(tensors, g)
end

# Standard single-site BP on the double-layer norm network, mirroring
# `scripts/double-layer-tn/src/run_bp_marginal.jl`. Returns the normalized center marginal plus
# convergence info.
function bp_marginal(psi::TNMPTest.TensorNetworkState, target; maxiter::Integer, tolerance::Real)
    cache = BeliefPropagationCache(to_tnqs_state(psi))
    alg = TensorNetworkQuantumSimulator.set_default_kwargs(
        ITensors.Algorithm("bp"; maxiter = Int(maxiter), tolerance = Float64(tolerance)),
        cache,
    )
    updated = copy(cache)
    TensorNetworkQuantumSimulator.invalidate_contraction_sequences!(updated)

    edge_sequence = alg.kwargs.edge_sequence
    final_diff = Inf
    iterations = 0
    converged = false
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

    ρ = reduced_density_matrix(Algorithm("bp"), updated, [target])
    ρ_diag = max.(collect(real.(diag(ITensors.array(ρ)))), 0.0)
    return TNMPTest.normalize_weights(ρ_diag), (; converged, iterations, final_diff)
end

function parse_arg(args, key, default::T) where {T}
    for a in args
        startswith(a, "$key=") || continue
        raw = a[(length(key) + 2):end]
        return T <: Integer ? parse(Int, raw) : (T <: AbstractFloat ? parse(Float64, raw) : raw)
    end
    return default
end

function main(args = ARGS)
    L = parse_arg(args, "L", 10)
    chi = parse_arg(args, "chi", 2)
    seed = parse_arg(args, "seed", 7)
    K = parse_arg(args, "K", 1.0)
    field = parse_arg(args, "field", 0.2)
    region_L = parse_arg(args, "region-L", 3)
    bp_max_iter = parse_arg(args, "bp-max-iter", 2000)
    bp_tol = parse_arg(args, "bp-tol", 1e-10)
    tnmp_max_iter = parse_arg(args, "tnmp-max-iter", 500)
    tnmp_tol = parse_arg(args, "tnmp-tol", 1e-8)

    center = ((L + 1) ÷ 2, (L + 1) ÷ 2)
    rng = MersenneTwister(seed)
    g = named_grid((L, L))
    psi = fully_frustrated_pair_factor_state(rng, g; K, field, bond_dim = chi)

    println("=== fully-frustrated K=$K field=$field L=$L chi=$chi seed=$seed center=$center ===")

    println("\n--- BP (single-site, TensorNetworkQuantumSimulator) ---")
    bp_marg, bp_info = bp_marginal(psi, center; maxiter = bp_max_iter, tolerance = bp_tol)
    println("bp_converged = $(bp_info.converged)")
    println("bp_iterations = $(bp_info.iterations)")
    println("bp_final_diff = $(bp_info.final_diff)")
    println("bp_marginal = $bp_marg")

    println("\n--- TNMP rank-2 (region_L=$region_L) ---")
    cache = TNMPCache(psi, region_L; normalize = :l2, region_fn = grid_region_fn(region_L))
    tnmp_info = run_message_passing!(cache; max_iter = tnmp_max_iter, tol = tnmp_tol)
    tnmp_marg = tnmp_marginal(cache, center)
    println("tnmp_converged = $(tnmp_info.converged)")
    println("tnmp_iterations = $(tnmp_info.iterations)")
    println("tnmp_final_diff = $(tnmp_info.final_diff)")
    println("tnmp_marginal = $tnmp_marg")

    println("\nl1_distance(bp, tnmp) = $(sum(abs.(bp_marg .- tnmp_marg)))")
    return (; bp_marg, bp_info, tnmp_marg, tnmp_info)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
