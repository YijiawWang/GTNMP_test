include(joinpath(@__DIR__, "..", "examples", "boundarymps_random_double_layer.jl"))

using .TNMPBoundaryMPSDemo
using Test: @test, @testset

@testset "Boundary MPS production marginal sweep" begin
    out = joinpath(mktempdir(), "test_boundarymps.jls")
    cfg = TNMPBoundaryMPSDemo.BoundaryMPSConfig(
        L = 4,
        chi = 2,
        seed = 7,
        bmps_chi_max = 8,
        bmps_epsilon = 1e-2,
        output = out,
    )

    payload = TNMPBoundaryMPSDemo.run_boundarymps_marginal(cfg)

    @test payload["algorithm"] == "boundarymps"
    @test isfile(out)

    history = payload["history"]
    @test !isempty(history)
    @test history[1]["l1_delta_vs_prev"] == Inf

    for entry in history
        marginal = entry["marginal"]
        @test all(isfinite, marginal)
        @test sum(marginal) ≈ 1.0
    end

    final_marginal = payload["final_marginal"]
    @test all(isfinite, final_marginal)
    @test sum(final_marginal) ≈ 1.0
    @test payload["final_bmps_chi"] <= cfg.bmps_chi_max
end

@testset "Boundary MPS sweep can start at bmps_chi=2" begin
    out = joinpath(mktempdir(), "test_boundarymps_chi2.jls")
    cfg = TNMPBoundaryMPSDemo.BoundaryMPSConfig(
        L = 4,
        chi = 2,
        seed = 7,
        bmps_chi_min = 2,
        bmps_chi_max = 2,
        bmps_epsilon = 1e-2,
        output = out,
    )

    payload = TNMPBoundaryMPSDemo.run_boundarymps_marginal(cfg)

    @test payload["history"][1]["bmps_chi"] == 2
    @test payload["final_bmps_chi"] == 2
end
