# Double-layer TNMP rank-2 marginal on a random complex PEPS.
#
# For full benchmarks (rank1 / rank2 / boundary-MPS, chi=8,16), use run_all:
#   JL_INIT_MODE=uniform_complex JULIA_NUM_THREADS=64 scripts/run_all_marginals.sh
#
# Quick local smoke test:
#   julia --project=TNMP_test TNMP_test/examples/random_uniform_complex_double_layer_marginal.jl

include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))

using .TNMPTest
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister

function main()
    rng = MersenneTwister(7)
    g = named_grid((3, 3))
    center = (2, 2)
    lo = -0.5
    hi = 0.5
    max_iter = 50
    tol = 1e-8

    psi = random_uniform_complex_state(rng, g; physical_dim = 2, bond_dim = 2, lo, hi)
    cache = TNMPCache(psi, 3; normalize = :l2)
    info = run_message_passing!(cache; max_iter, tol)

    p_tnmp = tnmp_marginal(cache, center)
    p_exact = exact_marginal(psi, center)

    println("initialization = uniform_complex")
    println("uniform_range = [$lo, $hi) for real and imag independently")
    println("center = $center")
    println("tnmp_converged = $(info.converged)")
    println("tnmp_iterations = $(info.iterations)")
    println("tnmp_final_diff = $(info.final_diff)")
    println("tnmp_marginal = $p_tnmp")
    println("exact_marginal = $p_exact")
    println("l1_error = $(sum(abs.(p_tnmp .- p_exact)))")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
