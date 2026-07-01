# Pin down the TreeSA contraction complexity (tc/sc) recorded by the rank-2 TNMP
# complexity probe on a double-layer 2D-lattice tensor network.
#
# Setup (matches the figures elsewhere in the repo):
#   * physical leg (trace bond) D = 2,
#   * virtual / inter-layer bond  χ = 16,
#   * a 3×3 sub-lattice neighborhood (region_L = 3),
#   * center (3,3) of a 5×5 grid, so its 3×3 neighborhood sits in the interior
#     (x, y ∈ [2,4]) and the genuine interior cavity strips are exercised.
#
# The probe is driven through the real prewarm path
# (`prewarm_rank2_tnmp_sequences!` + `TNMPComplexityProbe` + `complexity_rows`),
# which TreeSA-optimizes every cavity (message) contraction and the center
# neighborhood (marginal) contraction without doing any numerical contraction.
# `index_only_double_layer_state` keeps the χ = 16 tensors as index-only ITensor
# storage, so only the index topology + bond dimensions feed TreeSA (tc/sc are
# independent of the actual tensor entries).
#
# Expected, at χ = 16 / D = 2 / 3×3 neighborhood:
#   * over all cavity regions: max tc = 25, max sc = 17,
#   * the center marginal:      tc = 42, sc = 32.
#
# This file is guarded so it runs both standalone and from runtests.jl.

if !isdefined(@__MODULE__, :TNMPTest)
    include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))
    using .TNMPTest
end
if !isdefined(@__MODULE__, :index_only_double_layer_state)
    include(joinpath(@__DIR__, "..", "examples", "state_models.jl"))
end
if !isdefined(@__MODULE__, :grid_region_fn)
    include(joinpath(@__DIR__, "..", "examples", "neighborhoods.jl"))
end

using NamedGraphs.NamedGraphGenerators: named_grid
using Random: seed!
using Test: @test, @testset

@testset "TNMP complexity probe (χ = 16, D = 2, 3×3 neighborhood)" begin
    chi = 16
    g = named_grid((5, 5))
    center = (3, 3)

    # Index-only double layer: tc/sc depend only on the index structure, so this
    # avoids allocating the dense 2·χ^4 ket tensors.
    psi = index_only_double_layer_state(g; physical_dim = 2, bond_dim = chi)
    cache = TNMPCache(psi, 3; region_fn = grid_region_fn(3))

    # Seed the global RNG that TreeSA draws from, so the optimized tc/sc are
    # reproducible run to run.
    seed!(20260630)

    probe = TNMPComplexityProbe()
    prewarm_rank2_tnmp_sequences!(cache, center; nthreads = 1, complexity_probe = probe)

    rows = complexity_rows(probe)
    cavity = filter(r -> r.kind == "cavity", rows)
    neighborhood = filter(r -> r.kind == "neighborhood", rows)

    @test !isempty(cavity)
    @test length(neighborhood) == 1

    # All cavity-region (message) contractions: the interior 3-wide strips peak
    # at tc = 25, sc = 17.
    @test maximum(r.sc for r in cavity) == 17
    @test round(Int, maximum(r.tc for r in cavity)) == 25

    # Center 3×3 neighborhood (marginal) contraction.
    marg = only(neighborhood)
    @test marg.sc == 32
    @test round(Int, marg.tc) == 42
end
