# q-state Potts model marginal example.
#
# Embeds a classical ferromagnetic q-state Potts model on a grid as a real PEPS
# (see `potts_model_state` in state_models.jl) and compares the single-site
# marginal obtained from rank-2 TNMP message passing against the exact marginal
# computed by an exact (TreeSA-optimised) contraction of the double-layer norm
# network.
#
# Run with:
#   julia --project=. examples/potts_model_marginal.jl

include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))

using .TNMPTest
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs: edges, src, dst, vertices
using Random: MersenneTwister

include(joinpath(@__DIR__, "state_models.jl"))
include(joinpath(@__DIR__, "neighborhoods.jl"))

# Brute-force classical Potts marginal at `target` on grid graph `g`, used as an
# independent cross-check that the PEPS encoding is faithful.
function brute_force_potts_marginal(g, target; q, coupling, field)
    vs = collect(vertices(g))
    idx = Dict(v => i for (i, v) in enumerate(vs))
    es = collect(edges(g))
    fld = field === nothing ? zeros(Float64, q) : collect(Float64, field)
    weights = zeros(Float64, q)
    config = ones(Int, length(vs))
    function recurse(pos)
        if pos > length(vs)
            logw = 0.0
            for v in vs
                logw += fld[config[idx[v]]]
            end
            for e in es
                if config[idx[src(e)]] == config[idx[dst(e)]]
                    logw += coupling
                end
            end
            weights[config[idx[target]]] += exp(logw)
            return
        end
        for s in 1:q
            config[pos] = s
            recurse(pos + 1)
        end
    end
    recurse(1)
    return weights ./ sum(weights)
end

function run_case(g, center, L; q, coupling, field, brute = false)
    rng = MersenneTwister(2026)
    psi = potts_model_state(rng, g; q, coupling, field)

    cache = TNMPCache(psi, L; normalize = :l1sum, region_fn = grid_region_fn(L))
    info = run_message_passing!(cache; max_iter = 100, tol = 1e-10)

    p_tnmp = tnmp_marginal(cache, center)
    p_exact = exact_marginal(psi, center)

    println("grid = $(maximum(first.(vertices(g))))x$(maximum(last.(vertices(g)))), " *
            "center = $center, window L = $L")
    println("  tnmp_converged  = $(info.converged) (", info.iterations, " iters)")
    println("  tnmp_marginal   = $p_tnmp")
    println("  exact_marginal  = $p_exact")
    println("  l1(tnmp, exact) = $(sum(abs.(p_tnmp .- p_exact)))")
    if brute
        p_brute = brute_force_potts_marginal(g, center; q, coupling, field)
        println("  brute_force     = $p_brute")
        println("  l1(exact, brute)= $(sum(abs.(p_exact .- p_brute)))")
    end
    println()
end

function main()
    q = 3
    coupling = 0.3
    field = [0.6, 0.0, -0.2]  # break the q-fold symmetry so the marginal is non-trivial

    println("q-state Potts model marginal (q = $q, coupling = $coupling, field = $field)\n")

    # Exact regime: L = 3 around (2, 2) covers the whole 3x3 grid, plus an
    # independent brute-force classical cross-check.
    run_case(named_grid((3, 3)), (2, 2), 3; q, coupling, field, brute = true)

    # Approximate regime: a local L = 3 window on a larger grid is no longer
    # exact, but rank-2 TNMP stays very close to the exact TreeSA contraction.
    run_case(named_grid((5, 5)), (3, 3), 3; q, coupling, field)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
