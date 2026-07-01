# Shared PEPS constructions from a biased all-|1> product state plus structured
# nearest-neighbor perturbations.
#
# Both constructors build a pair-feature PEPS
#
#   psi(s) = prod_v a_v(s_v) * prod_(u,v) K(s_u, s_v),
#
# with K(s,t) represented exactly by dense virtual features of dimension
# `bond_dim` (bond_dim >= 2). Extra virtual channels are a deterministic gauge
# spread of the same rank-2 pair factor, so chi=4 benchmarks still use
# four-dimensional dense bond legs while keeping the state interpretable.

using ITensors: ITensor, Index, dim, eachindval, norm
using LinearAlgebra: dot
using NamedGraphs: NamedEdge, dst, edges, neighbors, src, vertices
using Random: MersenneTwister, rand

perturbed_product_reverse_edge(e::NamedEdge) = NamedEdge(dst(e) => src(e))

function perturbed_product_site_indices(
        g;
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    physical_dim == 2 ||
        throw(ArgumentError("perturbed product PEPS currently require physical_dim = 2"))
    bond_dim >= 2 ||
        throw(ArgumentError("perturbed product PEPS require bond_dim >= 2, got $bond_dim"))

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(physical_dim, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(bond_dim, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[perturbed_product_reverse_edge(e)] = ind
    end

    return sitedict, edgeinds
end

function dense_pair_feature_basis(bond_dim::Integer)
    q1 = fill(1.0 / sqrt(bond_dim), bond_dim)
    q2 = collect(range(-1.0, 1.0; length = bond_dim))
    q2 .-= dot(q1, q2) .* q1
    n2 = sqrt(dot(q2, q2))
    n2 > 0 || throw(ArgumentError("cannot build feature basis for bond_dim=$bond_dim"))
    q2 ./= n2
    return q1, q2
end

function ferromagnetic_pair_features(pair_strength::Real, bond_dim::Integer)
    strength = Float64(pair_strength)
    strength >= 0 ||
        throw(ArgumentError("pair_strength must be non-negative, got $pair_strength"))

    # K = [exp(g) exp(-g); exp(-g) exp(g)] in the physical basis
    # state 1 -> |0>, state 2 -> |1>.  Its eigenvectors are uniform/staggered.
    λ_uniform = 2.0 * cosh(strength)
    λ_staggered = 2.0 * sinh(strength)
    q1, q2 = dense_pair_feature_basis(bond_dim)

    features = zeros(Float64, 2, bond_dim)
    for b in 1:bond_dim
        uniform = sqrt(λ_uniform) * q1[b] / sqrt(2.0)
        staggered = sqrt(max(λ_staggered, 0.0)) * q2[b] / sqrt(2.0)
        features[1, b] = uniform + staggered
        features[2, b] = uniform - staggered
    end
    return features
end

function signed_pair_features(beta::Real, coupling::Integer, bond_dim::Integer)
    coupling in (-1, 1) ||
        throw(ArgumentError("spin-glass coupling must be +1 or -1, got $coupling"))
    strength = Float64(beta) * coupling
    q1, q2 = dense_pair_feature_basis(bond_dim)

    λ_uniform = 2.0 * cosh(strength)
    λ_staggered = 2.0 * sinh(strength)
    sqrt_uniform = sqrt(complex(λ_uniform))
    sqrt_staggered = sqrt(complex(λ_staggered))

    features = zeros(ComplexF64, 2, bond_dim)
    for b in 1:bond_dim
        uniform = sqrt_uniform * q1[b] / sqrt(2.0)
        staggered = sqrt_staggered * q2[b] / sqrt(2.0)
        features[1, b] = uniform + staggered
        features[2, b] = uniform - staggered
    end
    return features
end

function normalize_local_amplitudes(local_amplitudes::AbstractVector{<:Number})
    length(local_amplitudes) == 2 ||
        throw(ArgumentError("expected two local amplitudes, got $(length(local_amplitudes))"))
    n = sqrt(sum(abs2, local_amplitudes))
    n > 0 || throw(ArgumentError("local amplitudes must not be all zero"))
    return ComplexF64.(local_amplitudes ./ n)
end

function build_pair_feature_peps_tensors(
        g;
        local_amplitudes::AbstractVector{<:Number},
        pair_strength::Real,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    sitedict, edgeinds = perturbed_product_site_indices(
        g; physical_dim = physical_dim, bond_dim = bond_dim,
    )
    features = ferromagnetic_pair_features(pair_strength, bond_dim)
    local_amp = normalize_local_amplitudes(local_amplitudes)

    vs = collect(vertices(g))
    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        neighbor_list = Tuple{Int, Int}[vn for vn in neighbors(g, v)]
        bond_inds = Index[edgeinds[NamedEdge(v => vn)] for vn in neighbor_list]
        local_inds = [sitedict[v]; bond_inds]
        tensor = ITensor(ComplexF64, local_inds...)
        dims = [dim(i) for i in local_inds]
        for idx in CartesianIndices(Tuple(dims))
            s = idx[1]
            value = local_amp[s]
            for k in 2:length(local_inds)
                value *= features[s, idx[k]]
            end
            tensor[(local_inds[i] => idx[i] for i in 1:length(local_inds))...] = value
        end
        n = norm(tensor)
        tensors[v] = n > 0 ? tensor / n : tensor
    end

    siteinds = Dict(v => Index[sitedict[v]] for v in vs)
    return tensors, siteinds
end

function spin_glass_couplings(rng, g)
    return Dict(e => (rand(rng) < 0.5 ? -1 : 1) for e in edges(g))
end

function fully_frustrated_square_couplings(g)
    couplings = Dict{Any, Int}()
    for e in edges(g)
        u, v = src(e), dst(e)
        horizontal = u[2] == v[2]
        vertical = u[1] == v[1]
        (horizontal || vertical) ||
            throw(ArgumentError("fully-frustrated couplings require square-grid nearest-neighbor edges, got $e"))
        couplings[e] = vertical && isodd(min(u[1], v[1])) ? -1 : 1
    end
    return couplings
end

function build_edge_feature_peps_tensors(
        g;
        local_amplitudes::AbstractVector{<:Number},
        edge_features::Dict,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    sitedict, edgeinds = perturbed_product_site_indices(
        g; physical_dim = physical_dim, bond_dim = bond_dim,
    )
    local_amp = normalize_local_amplitudes(local_amplitudes)

    vs = collect(vertices(g))
    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        neighbor_list = Tuple{Int, Int}[vn for vn in neighbors(g, v)]
        bond_edges = NamedEdge[NamedEdge(v => vn) for vn in neighbor_list]
        bond_inds = Index[edgeinds[e] for e in bond_edges]
        local_inds = [sitedict[v]; bond_inds]
        tensor = ITensor(ComplexF64, local_inds...)
        dims = [dim(i) for i in local_inds]
        for idx in CartesianIndices(Tuple(dims))
            s = idx[1]
            value = local_amp[s]
            for (k, e) in enumerate(bond_edges)
                value *= edge_features[e][s, idx[k + 1]]
            end
            tensor[(local_inds[i] => idx[i] for i in 1:length(local_inds))...] = value
        end
        n = norm(tensor)
        tensors[v] = n > 0 ? tensor / n : tensor
    end

    siteinds = Dict(v => Index[sitedict[v]] for v in vs)
    return tensors, siteinds
end

function spin_glass_pair_factor_tensors(
        rng,
        g;
        beta::Real = 0.8,
        bias::Real = 0.2,
        disorder_seed::Integer = 7,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    β = Float64(beta)
    β >= 0 || throw(ArgumentError("beta must be non-negative, got $beta"))

    coupling_rng = MersenneTwister(disorder_seed)
    couplings = spin_glass_couplings(coupling_rng, g)
    edge_features = Dict{Any, Matrix{ComplexF64}}()
    for e in edges(g)
        features = signed_pair_features(β, couplings[e], bond_dim)
        edge_features[e] = features
        edge_features[perturbed_product_reverse_edge(e)] = features
    end

    # state 1 -> z=-1 (|0>), state 2 -> z=+1 (|1>).
    h = Float64(bias)
    local_amplitudes = [exp(-h), exp(h)]
    return build_edge_feature_peps_tensors(
        g;
        local_amplitudes,
        edge_features,
        physical_dim,
        bond_dim,
    )
end

function fully_frustrated_pair_factor_tensors(
        rng,
        g;
        K::Real = 1.0,
        field::Real = 0.2,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    strength = Float64(K)
    strength >= 0 || throw(ArgumentError("K must be non-negative, got $K"))

    couplings = fully_frustrated_square_couplings(g)
    edge_features = Dict{Any, Matrix{ComplexF64}}()
    for e in edges(g)
        features = signed_pair_features(0.5 * strength, couplings[e], bond_dim)
        edge_features[e] = features
        edge_features[perturbed_product_reverse_edge(e)] = features
    end

    # state 1 -> z=-1 (|0>), state 2 -> z=+1 (|1>).
    h = Float64(field)
    local_amplitudes = [exp(-0.5 * h), exp(0.5 * h)]
    return build_edge_feature_peps_tensors(
        g;
        local_amplitudes,
        edge_features,
        physical_dim,
        bond_dim,
    )
end

function weak_entangled_biased_circuit_tensors(
        rng,
        g;
        theta::Real = 0.4,
        phi::Real = 0.15,
        depth::Integer = 2,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    depth >= 0 || throw(ArgumentError("depth must be non-negative, got $depth"))
    θ = Float64(theta)
    0 <= θ <= pi || throw(ArgumentError("theta must be in [0, pi], got $theta"))
    pair_strength = abs(Float64(phi)) * depth
    # Ry(theta)|1> has probabilities sin(theta/2)^2, cos(theta/2)^2.
    local_amplitudes = [sin(θ / 2), cos(θ / 2)]
    return build_pair_feature_peps_tensors(
        g;
        local_amplitudes,
        pair_strength,
        physical_dim,
        bond_dim,
    )
end

function tfim_imaginary_time_tensors(
        rng,
        g;
        tau::Real = 0.05,
        coupling_j::Real = 1.0,
        field_h::Real = 0.8,
        steps::Integer = 5,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    steps >= 0 || throw(ArgumentError("steps must be non-negative, got $steps"))
    τ = Float64(tau)
    τ >= 0 || throw(ArgumentError("tau must be non-negative, got $tau"))
    β = τ * steps
    pair_strength = β * Float64(coupling_j)
    pair_strength >= 0 ||
        throw(ArgumentError("tau * steps * coupling_j must be non-negative"))
    # exp(beta*h*X)|1> = sinh(beta*h)|0> + cosh(beta*h)|1>.
    field = β * Float64(field_h)
    local_amplitudes = [sinh(field), cosh(field)]
    return build_pair_feature_peps_tensors(
        g;
        local_amplitudes,
        pair_strength,
        physical_dim,
        bond_dim,
    )
end
