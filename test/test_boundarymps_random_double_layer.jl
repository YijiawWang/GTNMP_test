using Test: @test, @testset

@testset "Boundary MPS random double-layer test script" begin
    project = normpath(joinpath(@__DIR__, ".."))
    inner_test = joinpath(@__DIR__, "test_boundarymps_random_double_layer_inner.jl")
    @test run(`$(Base.julia_cmd()) --project=$project $inner_test`).exitcode == 0
end
