@eval module $(gensym())
using Random
using Test
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensors: ITensor, Index

include(joinpath(@__DIR__, "..", "..", "real_double_layer", "independent_double_layer.jl"))
using .IndependentDoubleLayer
using .IndependentDoubleLayer.TNMPTest: TensorNetworkState

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

@testset "RealDoubleLayer complex projection weights use L1 magnitude" begin
    g = named_grid((1, 1))
    v = (1, 1)
    ket_s = Index(2, "ket-phys")
    bra_s = Index(2, "bra-phys")

    ket_t = ITensor(ComplexF64, ket_s)
    ket_t[ket_s => 1] = 1 + 1im
    ket_t[ket_s => 2] = 2 + 0im

    bra_t = ITensor(ComplexF64, bra_s)
    bra_t[bra_s => 1] = 0 + 1im
    bra_t[bra_s => 2] = 1 + 0im

    rdl = RealDoubleLayerState(
        TensorNetworkState(Dict(v => ket_t), Dict(v => Index[ket_s]), g),
        TensorNetworkState(Dict(v => bra_t), Dict(v => Index[bra_s]), g),
    )

    expected_weights = [abs(1 - 1im), abs(2 + 0im)]
    expected = expected_weights ./ sum(expected_weights)
    @test exact_marginal(rdl, v) ≈ expected
end

end
