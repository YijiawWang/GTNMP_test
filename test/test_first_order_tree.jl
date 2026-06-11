# Test [2]: swap the L×L grid neighborhood for the graph-based *first-order*
# neighborhood (self + immediate graph neighbors) and check exactness on a tree.
#
# `random_state` needs a uniform vertex type, so the tree is built with integer
# labels rather than via `named_binary_tree` (whose vertices are ragged tuples).
#
#   * rank2 (merged *matrix* messages) keeps the ket/bra legs of each edge
#     coupled, so each tree edge becomes a single super-bond and the DOUBLE-layer
#     (norm) marginal is exact on a tree.
#   * rank1 (per-leg *vector* messages) is exact on a SINGLE-layer tree network:
#     with no bra layer the network has no 4-cycles, so one vector message per
#     bond reproduces the exact single-layer contraction. (On the double-layer
#     network rank1 would only be a mean-field approximation, because every edge
#     closes a ket-bond → physical → bra-bond → physical 4-cycle.)

# A small but non-trivial tree (depth 3, branching) on integer vertices.
function _tree_graph()
    g = NamedGraph(collect(1:11))
    for (a, b) in ((1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (4, 9), (7, 10), (7, 11))
        add_edge!(g, a, b)
    end
    return g
end

# Use the first-order neighborhood instead of the grid window (L is ignored).
_first_order(gp, g, node) = TNMPTest.first_order_region(g, node)

@testset "first-order neighborhood on a tree" begin
    @testset "rank2 matrix messages: exact on a double-layer tree" begin
        rng = MersenneTwister(20260613)
        g = _tree_graph()
        psi = TNMPTest.random_state(rng, g; physical_dim = 2, bond_dim = 3)
        cache = TNMPTest.TNMPCache(psi, 1; region_fn = _first_order)
        info = TNMPTest.run_message_passing!(cache; max_iter = 200, tol = 1e-12)

        @test info.converged
        for v in vertices(g)
            exact = TNMPTest.exact_marginal(psi, v)
            tnmp = TNMPTest.tnmp_marginal(cache, v)
            @test sum(tnmp) ≈ 1 atol = 1e-12
            @test tnmp ≈ exact atol = 1e-8
        end
    end

    @testset "rank1 vector messages: exact on a single-layer tree" begin
        # Real tensors so the single-layer contraction is a real weight (a complex
        # amplitude would not define a probability under real-part normalisation).
        rng = MersenneTwister(20260613)
        g = _tree_graph()
        psi = TNMPRank1.random_state(rng, g; physical_dim = 2, bond_dim = 3, element_type = Float64)
        cache = TNMPRank1.TNMPRank1Cache(psi, 1; region_fn = _first_order)
        info = TNMPRank1.run_message_passing_single_layer!(cache; max_iter = 200, tol = 1e-12)

        @test info.converged
        for v in vertices(g)
            exact = TNMPRank1.exact_marginal_single_layer(psi, v)
            tnmp = TNMPRank1.tnmp_marginal_single_layer(cache, v)
            @test sum(tnmp) ≈ 1 atol = 1e-12
            @test tnmp ≈ exact atol = 1e-8
        end
    end
end
