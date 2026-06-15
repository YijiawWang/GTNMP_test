module TNMPBoundaryMPSDemo

const _ROOT = normpath(joinpath(@__DIR__, ".."))
const _TNQS_PROJECT = joinpath(_ROOT, "..", "TensorNetworkQuantumSimulator_q.jl")
if !in(_TNQS_PROJECT, LOAD_PATH)
    pushfirst!(LOAD_PATH, _TNQS_PROJECT)
end

using Dictionaries: Dictionary
using NamedGraphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Random: MersenneTwister
using TensorNetworkQuantumSimulator

include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))

function to_tnqs_double_layer_network(psi::TNMPTest.TensorNetworkState)
    vs = collect(vertices(TNMPTest.graph(psi)))
    tensors = Dictionary(vs, [reduce(*, TNMPTest.traced_norm_factors(psi, v)) for v in vs])
    return TensorNetworkQuantumSimulator.TensorNetwork(tensors, TNMPTest.graph(psi))
end

function random_tnmp_state(;
        grid_dims = (2, 3),
        seed::Integer = 7,
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
        element_type::Type = Float64,
    )
    rng = MersenneTwister(seed)
    graph = named_grid(grid_dims)
    return TNMPTest.random_state(rng, graph; physical_dim, bond_dim, element_type)
end

function exact_double_layer_contraction(psi::TNMPTest.TensorNetworkState)
    vs = collect(vertices(TNMPTest.graph(psi)))
    # Raw (signed/complex) contraction of the double-layer network, so it can be
    # compared directly against the boundary-MPS estimate. The double layer of a
    # state is the norm ⟨ψ|ψ⟩ and is real, but we deliberately avoid
    # `scalar_weight`'s abs/sign-flip so the comparison stays correct even when
    # the network is not a norm.
    return TNMPTest.contract_all(TNMPTest.norm_factors(psi, vs))[]
end

function boundarymps_convergence(;
        grid_dims = (2, 3),
        seed::Integer = 7,
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
        element_type::Type = Float64,
        mps_bond_dimensions = [1, 2, 4],
        partition_by = "row",
        bmps_update_kwargs = (;),
    )
    tnmp_state = random_tnmp_state(; grid_dims, seed, physical_dim, bond_dim, element_type)
    double_layer = to_tnqs_double_layer_network(tnmp_state)
    exact = exact_double_layer_contraction(tnmp_state)
    estimates = map(mps_bond_dimensions) do chi
        # Build the boundary-MPS cache directly so `partition_by` is honoured;
        # `contract(...; alg = "boundarymps")` does not expose it.
        cache = TensorNetworkQuantumSimulator.BoundaryMPSCache(
            double_layer, chi; partition_by,
        )
        value = TensorNetworkQuantumSimulator.partitionfunction(
            TensorNetworkQuantumSimulator.update(cache; bmps_update_kwargs...),
        )
        abs_error = abs(value - exact)
        rel_error = abs_error / max(1, abs(exact))
        return (; mps_bond_dimension = chi, value, abs_error, rel_error)
    end
    return (; exact, estimates)
end

function main()
    result = boundarymps_convergence()
    println("exact = $(result.exact)")
    for entry in result.estimates
        println(
            "mps_bond_dimension = $(entry.mps_bond_dimension), " *
            "boundarymps = $(entry.value), " *
            "abs_error = $(entry.abs_error), " *
            "rel_error = $(entry.rel_error)",
        )
    end
    return result
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end

end
