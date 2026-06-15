include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))
include(joinpath(@__DIR__, "..", "src", "tnmp_rank1.jl"))

using .TNMPTest
using NamedGraphs: NamedGraph, add_edge!, vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister
using Test: @test, @testset

l1_error(p, q) = sum(abs.(p .- q))

@testset "TNMP marginals on a grid" begin
    @testset "exact when the L*L sub-lattice covers the whole grid" begin
        rng = MersenneTwister(20260611)
        g = named_grid((3, 3))
        center = (2, 2)
        psi = random_state(rng, g; physical_dim = 2, bond_dim = 2)

        exact = exact_marginal(psi, center)
        # L = 3 around (2, 2) spans x, y in [1, 3], i.e. the entire 3x3 grid.
        cache = TNMPCache(psi, 3)
        info = run_message_passing!(cache; max_iter = 50, tol = 1e-10)
        tnmp = tnmp_marginal(cache, center)

        @test info.converged
        @test sum(exact) ≈ 1 atol = 1e-12
        @test sum(tnmp) ≈ 1 atol = 1e-12
        @test tnmp ≈ exact atol = 1e-8
    end

    @testset "larger L drives the approximation toward exact" begin
        # A 3x5 grid keeps the exact contraction cheap (min dimension 3) while
        # leaving room for L = 3 to be a genuinely local (approximate) window.
        rng = MersenneTwister(20260612)
        g = named_grid((3, 5))
        center = (2, 3)
        psi = random_state(rng, g; physical_dim = 2, bond_dim = 2)

        exact = exact_marginal(psi, center)

        # L = 3 at (2, 3) covers x in [1, 3], y in [2, 4] -> misses y = 1, 5 (approximate).
        cache3 = TNMPCache(psi, 3)
        run_message_passing!(cache3; max_iter = 100, tol = 1e-10)
        tnmp3 = tnmp_marginal(cache3, center)

        # L = 5 at (2, 3) covers y in [1, 5] -> the whole 3x5 grid (exact).
        cache5 = TNMPCache(psi, 5)
        run_message_passing!(cache5; max_iter = 100, tol = 1e-10)
        tnmp5 = tnmp_marginal(cache5, center)

        @test sum(tnmp3) ≈ 1 atol = 1e-12
        @test sum(tnmp5) ≈ 1 atol = 1e-12
        @test l1_error(tnmp5, exact) <= 1e-8
        @test l1_error(tnmp5, exact) <= l1_error(tnmp3, exact)
    end
end

include("test_contraction_sc.jl")
include("test_first_order_tree.jl")
include("test_boundarymps_log_normalization.jl")
include("test_boundarymps_random_double_layer.jl")
include("test_real_double_layer.jl")
