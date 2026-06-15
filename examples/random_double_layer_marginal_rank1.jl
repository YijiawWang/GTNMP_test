include(joinpath(@__DIR__, "..", "src", "tnmp_rank1.jl"))

using .TNMPRank1
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister

function main()
    rng = MersenneTwister(7)
    message_init_rng = MersenneTwister(7)
    g = named_grid((3, 3))
    center = (2, 2)
    alpha = 0.5

    psi = random_alpha_state(rng, g; alpha, physical_dim = 2, bond_dim = 2)
    cache = TNMPRank1Cache(psi, 3; normalize = :l1sum)
    info = run_message_passing!(cache; max_iter = 50, tol = 1e-8, message_init_rng = message_init_rng)

    p_tnmp = tnmp_marginal(cache, center; message_init_rng = message_init_rng)
    p_exact = exact_marginal(psi, center)

    println("center = $center")
    println("tnmp_rank1_converged = $(info.converged)")
    println("tnmp_rank1_iterations = $(info.iterations)")
    println("tnmp_rank1_final_diff = $(info.final_diff)")
    println("tnmp_rank1_marginal = $p_tnmp")
    println("exact_marginal = $p_exact")
    println("l1_error = $(sum(abs.(p_tnmp .- p_exact)))")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
