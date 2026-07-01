include(joinpath(@__DIR__, "..", "scripts", "double-layer-tn", "src", "run_boundarymps_marginal.jl"))

using .TNMPTest
using Test: @test, @testset

@testset "Boundary MPS log-domain marginal normalization" begin
    marginal = Main.normalize_log_weights([1000.0, 999.0])

    @test all(isfinite, marginal)
    @test sum(marginal) ≈ 1.0
    @test marginal ≈ [1 / (1 + exp(-1)), exp(-1) / (1 + exp(-1))]
end
