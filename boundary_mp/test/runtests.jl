using Test
using Random
using LinearAlgebra: norm
using TensorNetworkQuantumSimulator
using TensorNetworkQuantumSimulator:
    BoundaryMPSCache, supergraph, network, contraction_sequence, TreeSA, contract
using NamedGraphs.PartitionedGraphs: partitions_graph, PartitionEdge
using NamedGraphs: edges
using ITensors: ITensor, inds, disable_warn_order

const _SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(_SRC, "TNQSBoundaryMP.jl"))
using .TNQSBoundaryMP
include(joinpath(_SRC, "ExactSolver.jl"))
using .ExactSolver
include(joinpath(_SRC, "ExactEnvFullUpdateBMPS.jl"))
using .ExactEnvFullUpdateBMPS

@testset "builder produces a positive scalar projected network" begin
    state = uniform_state(MersenneTwister(1), 3; bond_dim = 2, physical_dim = 2)
    c = center_vertex(state)
    tn = projected_norm_network(state, c, 1)
    z = TensorNetworkQuantumSimulator.contract(tn; alg = "exact")
    @test isfinite(z)
    @test real(z) > 0
end

@testset "exact TreeSA marginal is a valid distribution" begin
    state = uniform_state(MersenneTwister(2), 3; bond_dim = 2, physical_dim = 2)
    c = center_vertex(state)
    p = exact_marginal(state, c)
    @test length(p) == 2
    @test sum(p) ≈ 1.0 rtol = 1e-12 atol = 1e-12
    @test all(>=(0), p)
    f_default = tn -> real(TensorNetworkQuantumSimulator.contract(tn; alg = "exact"))
    p_default = marginal(state, c, f_default)
    @test p ≈ p_default rtol = 1e-10 atol = 1e-12
end

@testset "SVD-BMPS marginal converges to exact" begin
    state = uniform_state(MersenneTwister(3), 3; bond_dim = 2, physical_dim = 2)
    c = center_vertex(state)
    p_exact = exact_marginal(state, c)
    l1(p) = sum(abs.(p .- p_exact))
    err_small = l1(bmps_marginal(state, c, 1))
    err_large = l1(bmps_marginal(state, c, 16))
    @test err_large <= 1e-6
    @test err_large <= err_small + 1e-12
end

@testset "exact-env full-update marginal converges to exact" begin
    state = uniform_state(MersenneTwister(4), 3; bond_dim = 2, physical_dim = 2)
    c = center_vertex(state)
    p_exact = exact_marginal(state, c)
    l1(p) = sum(abs.(p .- p_exact))
    err_small = l1(full_update_marginal(state, c, 1; nsweeps = 4))
    err_large = l1(full_update_marginal(state, c, 16; nsweeps = 4))
    @test err_large <= 1e-6
    @test err_large <= err_small + 1e-12
end

# Closed-form theoretical reference: for a product state (bond_dim = 1) the double-layer
# norm network factorises across sites, so every non-center factor cancels in the
# normalisation and the center single-site marginal is exactly
#     p(s) = a_s^2 / sum_{s'} a_{s'}^2,
# where a_s is the center tensor's physical amplitude (all virtual bonds have dimension 1).
function product_state_marginal(state, c)
    ket = state.kets[c]
    phys = state.phys[c]
    bonds = filter(!=(phys), collect(inds(ket)))
    a = [ket[phys => s, [b => 1 for b in bonds]...] for s in 1:state.physical_dim]
    w = abs2.(a)
    return w ./ sum(w)
end

@testset "FU marginal matches product-state closed form (bond_dim=1)" begin
    # bond_dim = 1 => double-layer bond dimension 1, so chi = 1 BMPS is lossless and the
    # full-update result must reproduce the analytic product-state marginal exactly.
    for (seed, d) in ((11, 2), (12, 3), (13, 2))
        state = uniform_state(MersenneTwister(seed), 3; bond_dim = 1, physical_dim = d)
        c = center_vertex(state)
        p_theory = product_state_marginal(state, c)
        p_fu = full_update_marginal(state, c, 1; nsweeps = 4)
        @test length(p_fu) == d
        @test sum(p_fu) ≈ 1.0 atol = 1e-12
        @test p_fu ≈ p_theory rtol = 1e-8 atol = 1e-10
    end
end

@testset "incremental env cache equals from-scratch env (L=4)" begin
    # The cache is a pure optimisation: the memoised incremental fold must reproduce the
    # from-scratch TreeSA far-side contraction *exactly* (up to round-off) on every cut,
    # in both sweep directions. This directly pins cache correctness without relying on the
    # χ-truncated marginal being lossless (which it is NOT at L=4, χ=bond_dim²).
    disable_warn_order()
    state = uniform_state(MersenneTwister(9), 4; bond_dim = 2, physical_dim = 2)
    c = center_vertex(state)
    tn = projected_norm_network(state, c, 1)
    cache = BoundaryMPSCache(tn, 4; partition_by = "row")
    pg = partitions_graph(supergraph(cache))
    maxdiff = 0.0
    for pe in edges(pg), ppe in (PartitionEdge(pe), PartitionEdge(reverse(pe)))
        vs = ExactEnvFullUpdateBMPS._far_vertices(cache, ppe)
        ts = ITensor[copy(network(cache)[v]) for v in vs]
        seq = contraction_sequence(ts; alg = "omeinsum", optimizer = TreeSA())
        E_ref = contract(ts; sequence = seq)
        E_cached = ExactEnvFullUpdateBMPS._env_tensor_cached(cache, ExactEnvFullUpdateBMPS._far_rows(cache, ppe))
        maxdiff = max(maxdiff, norm(E_ref - E_cached))
    end
    @test maxdiff <= 1e-10
end

@testset "FU marginal is lossless at full bond dimension" begin
    # The exact boundary message across one horizontal double-layer bond has dimension
    # bond_dim^2; choosing chi = bond_dim^2 makes the compression lossless, so the
    # full-update marginal must equal the exact (TreeSA) marginal up to round-off / ridge.
    for seed in (21, 22)
        state = uniform_state(MersenneTwister(seed), 3; bond_dim = 2, physical_dim = 2)
        c = center_vertex(state)
        p_exact = exact_marginal(state, c)
        p_fu = full_update_marginal(state, c, 4; nsweeps = 4)
        @test p_fu ≈ p_exact rtol = 1e-7 atol = 1e-9
    end
end
