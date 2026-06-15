include(joinpath(@__DIR__, "..", "examples", "boundarymps_random_double_layer.jl"))

using .TNMPBoundaryMPSDemo
using Test: @test, @testset

@testset "Boundary MPS contracts random double-layer TN toward exact" begin
    # A 4x4 grid gives 4 partitions (rows), so the boundary-MPS sweep actually
    # exercises the iterative MPO x MPS apply/truncate path (with >2 partitions),
    # unlike a 2-row grid where only a single MPO truncation runs.
    result = TNMPBoundaryMPSDemo.boundarymps_convergence(
        grid_dims = (4, 4),
        seed = 7,
        physical_dim = 2,
        bond_dim = 2,
        mps_bond_dimensions = [1, 2, 4, 8, 16],
    )

    errors = [entry.abs_error for entry in result.estimates]
    scale = max(1, abs(result.exact))

    @test isfinite(result.exact)
    @test length(errors) == 5
    # Error should not grow as the bond dimension increases (allow a tiny
    # relative slack so that round-off near convergence can't cause flakiness).
    for i in 2:length(errors)
        @test errors[i] <= errors[i - 1] + 1e-8 * scale
    end
    # The truncation must matter: the smallest bond dimension is far from exact.
    @test errors[1] > 1e-3 * scale
    # The largest bond dimension recovers the exact contraction.
    @test errors[end] <= 1e-10 * scale
end
