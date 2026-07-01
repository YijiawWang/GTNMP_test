using NamedGraphs: edges, src, dst, vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister
using Test: @test, @testset

if !isdefined(@__MODULE__, :TNMPTest)
    include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))
    using .TNMPTest
end
if !isdefined(@__MODULE__, :potts_model_state)
    include(joinpath(@__DIR__, "..", "examples", "state_models.jl"))
end
if !isdefined(@__MODULE__, :grid_region_fn)
    include(joinpath(@__DIR__, "..", "examples", "neighborhoods.jl"))
end

# Independent brute-force classical q-state Potts marginal at `target`.
function brute_force_potts_marginal(g, target; q, coupling, field)
    vs = collect(vertices(g))
    idx = Dict(v => i for (i, v) in enumerate(vs))
    es = collect(edges(g))
    weights = zeros(Float64, q)
    config = ones(Int, length(vs))
    function recurse(pos)
        if pos > length(vs)
            logw = 0.0
            for v in vs
                logw += field[config[idx[v]]]
            end
            for e in es
                config[idx[src(e)]] == config[idx[dst(e)]] && (logw += coupling)
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

@testset "q-state Potts model marginals" begin
    q = 3
    coupling = 0.3
    field = [0.6, 0.0, -0.2]

    # The PEPS encoding must reproduce the classical Potts marginal exactly when
    # the double-layer norm network is contracted in full.
    @testset "exact double layer == classical Potts marginal" begin
        g = named_grid((3, 3))
        center = (2, 2)
        psi = potts_model_state(MersenneTwister(2026), g; q, coupling, field)

        p_exact = exact_marginal(psi, center)
        p_brute = brute_force_potts_marginal(g, center; q, coupling, field)

        @test sum(p_exact) ≈ 1 atol = 1e-12
        @test p_exact ≈ p_brute atol = 1e-10
    end

    # A window that covers the whole lattice makes rank-2 TNMP exact.
    @testset "TNMP exact when the window covers the grid" begin
        g = named_grid((3, 3))
        center = (2, 2)
        psi = potts_model_state(MersenneTwister(2026), g; q, coupling, field)

        cache = TNMPCache(psi, 3; normalize = :l1sum, region_fn = grid_region_fn(3))
        info = run_message_passing!(cache; max_iter = 100, tol = 1e-10)
        p_tnmp = tnmp_marginal(cache, center)
        p_exact = exact_marginal(psi, center)

        @test info.converged
        @test sum(p_tnmp) ≈ 1 atol = 1e-12
        @test p_tnmp ≈ p_exact atol = 1e-8
    end

    # A local window on a larger lattice is only approximate, but stays close to
    # the exact (TreeSA) contraction in the weakly-correlated regime.
    @testset "TNMP close to exact for a local window" begin
        g = named_grid((5, 5))
        center = (3, 3)
        psi = potts_model_state(MersenneTwister(2026), g; q, coupling, field)

        cache = TNMPCache(psi, 3; normalize = :l1sum, region_fn = grid_region_fn(3))
        run_message_passing!(cache; max_iter = 100, tol = 1e-10)
        p_tnmp = tnmp_marginal(cache, center)
        p_exact = exact_marginal(psi, center)

        @test sum(p_tnmp) ≈ 1 atol = 1e-12
        @test sum(abs.(p_tnmp .- p_exact)) < 0.05
    end
end
