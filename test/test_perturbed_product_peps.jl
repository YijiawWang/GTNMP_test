using ITensors: dim, inds
using NamedGraphs: NamedEdge, edges, src, dst

function first_bond_entry_probability(psi, v)
    local_inds = collect(inds(psi[v]))
    assignment0 = [local_inds[1] => 1]
    assignment1 = [local_inds[1] => 2]
    for ind in local_inds[2:end]
        push!(assignment0, ind => 1)
        push!(assignment1, ind => 1)
    end
    w0 = abs2(psi[v][assignment0...])
    w1 = abs2(psi[v][assignment1...])
    return w0 / (w0 + w1)
end

@testset "Perturbed product PEPS constructors" begin
    rng = MersenneTwister(20260619)
    g = named_grid((2, 2))
    center = (1, 1)

    weak = weak_entangled_biased_circuit_state(
        rng,
        g;
        physical_dim = 2,
        bond_dim = 4,
        theta = 0.4,
        phi = 0.0,
        depth = 0,
    )
    weak_tensor_inds = collect(inds(weak[center]))
    weak_p0 = first_bond_entry_probability(weak, center)

    @test dim(weak_tensor_inds[1]) == 2
    @test all(ind -> dim(ind) == 4, weak_tensor_inds[2:end])
    @test weak_p0 ≈ sin(0.4 / 2)^2 atol = 1e-12
    @test abs(weak_p0 - 0.5) > 0.25

    tfim = tfim_imaginary_time_state(
        rng,
        g;
        physical_dim = 2,
        bond_dim = 4,
        tau = 0.05,
        coupling_j = 0.0,
        field_h = 0.8,
        steps = 2,
    )
    tfim_tensor_inds = collect(inds(tfim[center]))
    tfim_p0 = first_bond_entry_probability(tfim, center)

    @test dim(tfim_tensor_inds[1]) == 2
    @test all(ind -> dim(ind) == 4, tfim_tensor_inds[2:end])
    tfim_field = 0.05 * 2 * 0.8
    expected_tfim_p0 = sinh(tfim_field)^2 / (sinh(tfim_field)^2 + cosh(tfim_field)^2)
    @test tfim_p0 ≈ expected_tfim_p0 atol = 1e-12
    @test abs(tfim_p0 - 0.5) > 0.35

    spin_glass = spin_glass_pair_factor_state(
        rng,
        g;
        physical_dim = 2,
        bond_dim = 4,
        beta = 0.0,
        bias = 0.2,
    )
    spin_glass_tensor_inds = collect(inds(spin_glass[center]))
    spin_glass_p0 = first_bond_entry_probability(spin_glass, center)

    @test dim(spin_glass_tensor_inds[1]) == 2
    @test all(ind -> dim(ind) == 4, spin_glass_tensor_inds[2:end])
    expected_spin_glass_p0 = exp(-2 * 0.2) / (exp(-2 * 0.2) + exp(2 * 0.2))
    @test spin_glass_p0 ≈ expected_spin_glass_p0 atol = 1e-12
    @test spin_glass_p0 < 0.5

    frustrated_g = named_grid((4, 4))
    couplings = fully_frustrated_square_couplings(frustrated_g)
    for x in 1:3, y in 1:3
        plaquette_sign =
            couplings[NamedEdge((x, y) => (x + 1, y))] *
            couplings[NamedEdge((x + 1, y) => (x + 1, y + 1))] *
            couplings[NamedEdge((x, y + 1) => (x + 1, y + 1))] *
            couplings[NamedEdge((x, y) => (x, y + 1))]
        @test plaquette_sign == -1
    end
    @test all(e -> (src(e)[1] == dst(e)[1] || couplings[e] == 1), edges(frustrated_g))

    fully_frustrated = fully_frustrated_pair_factor_state(
        rng,
        frustrated_g;
        physical_dim = 2,
        bond_dim = 4,
        K = 0.0,
        field = 0.2,
    )
    fully_frustrated_tensor_inds = collect(inds(fully_frustrated[(1, 1)]))
    fully_frustrated_p0 = first_bond_entry_probability(fully_frustrated, (1, 1))

    @test dim(fully_frustrated_tensor_inds[1]) == 2
    @test all(ind -> dim(ind) == 4, fully_frustrated_tensor_inds[2:end])
    expected_fully_frustrated_p0 = exp(-0.2) / (exp(-0.2) + exp(0.2))
    @test fully_frustrated_p0 ≈ expected_fully_frustrated_p0 atol = 1e-12
    @test fully_frustrated_p0 < 0.5
end
