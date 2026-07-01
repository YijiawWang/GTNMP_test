# Test [1]: the per-step contraction space complexity (sc) of the TNMP message
# updates and final marginal must reproduce the figures tabulated in
# docs/figures/contraction_sc_results.md (χ = 8 column).
#
# Conventions matching that table: physical leg (green bond) = 2, virtual bond
# (black bond / inter-layer χ) = 8, L = 3 → a 3×3 sub-lattice neighborhood.
# The grid is 7×7 with center (4, 4) so the center's 3×3 neighborhood sits in
# the interior: its message cavities are the genuine interior strips drawn in
# the *cavity* figures (on a smaller grid they would be truncated at the
# boundary and cost less).

@testset "contraction sc matches docs/figures (χ = 8)" begin
    chi = 8
    g = named_grid((7, 7))
    center = (4, 4)

    @testset "rank2 (merged matrix messages)" begin
        rng = MersenneTwister(20260611)
        psi = random_state(rng, g; physical_dim = 2, bond_dim = chi)
        cache = TNMPTest.TNMPCache(psi, 3; region_fn = grid_region_fn(3))
        bedges = TNMPTest.incoming_boundary_edges(TNMPTest.graph(psi), cache.regions[(:site, center)])

        # Each message update contracts a cavity → md rank2_cavity (χ = 8) = 13.
        for e in bedges
            ts = TNMPTest.message_tensors(cache, (:site, center), e)
            @test TNMPTest.contraction_sc(ts) == 13
        end
        # Final marginal contracts the 3×3 neighborhood → md rank2_neighborhood = 24.
        @test TNMPTest.contraction_sc(TNMPTest.marginal_tensors(cache, center, 1)) == 24
    end

    @testset "rank1 (per-leg vector messages)" begin
        rng = MersenneTwister(20260611)
        psi = random_state(rng, g; physical_dim = 2, bond_dim = chi)
        cache = TNMPRank1.TNMPRank1Cache(psi, 3; region_fn = grid_region_fn(3))
        bedges = TNMPRank1.incoming_boundary_edges(TNMPRank1.graph(psi), cache.regions[(:site, center)])

        # Each message update: md rank1_cavity (χ = 8) = 9. The actual rank-1
        # update leaves the output leg open, so TreeSA finds an order at or below
        # the figure's (fully-capped) value — it fluctuates in 7–9 — so the md
        # value is asserted as an upper bound rather than an equality.
        for e in bedges
            for layer in (:ket, :bra)
                ts = TNMPRank1.message_tensors(cache, (:site, center), e, layer)
                @test TNMPRank1.contraction_sc(ts) <= 9
            end
        end
        # Final marginal contracts the 3×3 neighborhood → md rank1_neighborhood = 15.
        @test TNMPRank1.contraction_sc(TNMPRank1.marginal_tensors(cache, center, 1)) == 15
    end
end
