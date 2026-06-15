@eval module $(gensym())
using Random
using Test
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensors: dag, prime, norm

include(joinpath(@__DIR__, "..", "..", "real_double_layer", "independent_double_layer.jl"))
using .IndependentDoubleLayer

@testset "RealDoubleLayer independent sampling" begin
    rng1 = MersenneTwister(7)
    g = named_grid((3, 3))
    rdl1 = sample_independent_uniform_complex(rng1, g; bond_dim = 2)
    rng2 = MersenneTwister(7)
    rdl2 = sample_independent_uniform_complex(rng2, g; bond_dim = 2)

    p1 = exact_marginal(rdl1, (2, 2))
    p2 = exact_marginal(rdl2, (2, 2))
    @test p1 ≈ p2
    @test rdl1 !== rdl2
end

@testset "RealDoubleLayer exact marginal" begin
    rng = MersenneTwister(7)
    g = named_grid((3, 3))
    rdl = sample_independent_uniform_complex(rng, g; bond_dim = 2)
    p = exact_marginal(rdl, (2, 2))
    @test length(p) == 2
    @test abs(sum(p) - 1) < 1e-10
    @test all(p .>= 0)
end

end
