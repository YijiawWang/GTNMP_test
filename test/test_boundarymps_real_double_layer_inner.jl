include(joinpath(@__DIR__, "..", "..", "real_double_layer", "run_boundarymps_marginal.jl"))

using .RealDLBoundaryMPSDemo
using .RealDLBoundaryMPSDemo.RealDLBoundaryMPS
using .RealDLBoundaryMPSDemo.RealDLBoundaryMPS.IndependentDoubleLayer: exact_marginal
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister
using Serialization: deserialize
using Test: @test, @testset

@testset "Real double-layer Boundary MPS production marginal sweep" begin
    out = joinpath(mktempdir(), "test_realdl_boundarymps.jls")
    cfg = RealDLBoundaryMPSDemo.RealDLBoundaryMPSConfig(
        L = 2,
        chi = 2,
        seed = 7,
        bmps_chi_min = 1,
        bmps_chi_max = 1,
        bmps_epsilon = 1e-2,
        output = out,
    )

    payload = RealDLBoundaryMPSDemo.run_boundarymps_marginal(cfg)

    @test payload["algorithm"] == "realdl_boundarymps"
    @test payload["double_layer_mode"] == "independent"
    @test payload["ket_seed"] == cfg.seed
    @test payload["bra_seed"] == cfg.seed
    @test isfile(out)
    @test deserialize(out)["algorithm"] == "realdl_boundarymps"

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
    @test payload["final_bmps_chi"] == cfg.bmps_chi_max
end

@testset "Real double-layer Boundary MPS matches exact TreeSA contraction" begin
    rng = MersenneTwister(7)
    cfg = RealDLBoundaryMPSDemo.RealDLBoundaryMPSConfig(
        L = 2,
        chi = 2,
        seed = 7,
        bmps_chi_max = 4,
    )
    center = (1, 1)
    rdl = sample_independent_peps(rng, named_grid((cfg.L, cfg.L)), cfg)

    # `exact_marginal` contracts all real double-layer factors through
    # TNMP_test's TreeSA contraction-sequence search.
    exact = exact_marginal(rdl, center)
    bmps = boundarymps_marginal_weights(rdl, center, cfg.bmps_chi_max)

    @test sum(exact) ≈ 1.0
    @test sum(bmps) ≈ 1.0
    @test bmps ≈ exact atol = 1e-10
end
