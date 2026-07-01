# Concrete state / PEPS constructors for the TNMP examples.
#
# These build `TNMPTest.TensorNetworkState` instances and live at the model
# level (outside the solver module) so new models can be added here without
# touching `src/`. Include this file *after* `src/tnmp.jl` so `TNMPTest` is
# already defined:
#
#   include(joinpath(@__DIR__, "..", "src", "tnmp.jl"))
#   using .TNMPTest
#   include(joinpath(@__DIR__, "..", "examples", "state_models.jl"))

using ITensors: ITensor, Index, eachindval, emptyITensor
using NamedGraphs: NamedGraph, NamedEdge, edges, neighbors, src, dst, vertices
using Random: AbstractRNG, rand

# Guarded so this file is safe to `include` more than once into the same module
# (e.g. a test runner includes it, then a run script included by a test includes
# it again). `const` here would otherwise error on the second include.
if !@isdefined(TensorNetworkState)
    const TensorNetworkState = TNMPTest.TensorNetworkState
end
if !@isdefined(reverse_edge)
    const reverse_edge = TNMPTest.reverse_edge
end

function random_entry(rng::AbstractRNG, ::Type{ComplexF64})
    return complex(randn(rng), randn(rng))
end

function random_entry(rng::AbstractRNG, ::Type{Float64})
    return randn(rng)
end

# Independent Uniform(lo, hi) draws for the real and imaginary parts.
function random_uniform_complex_entry(rng::AbstractRNG; lo::Real = 0.0, hi::Real = 1.0)
    lo_f = Float64(lo)
    width = Float64(hi - lo)
    return complex(lo_f + width * rand(rng), lo_f + width * rand(rng))
end

function random_state(
        rng::AbstractRNG,
        g::NamedGraph;
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
        element_type::Type = ComplexF64,
    )
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(physical_dim, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(bond_dim, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[reverse_edge(e)] = ind
    end

    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        local_inds = Index[sitedict[v]]
        for vn in neighbors(g, v)
            push!(local_inds, edgeinds[NamedEdge(v => vn)])
        end
        tensor = ITensor(element_type, local_inds)
        for iv in eachindval(tensor)
            tensor[iv...] = random_entry(rng, element_type)
        end
        tensors[v] = tensor
    end

    return TensorNetworkState(tensors, Dict(v => Index[sitedict[v]] for v in vs), g)
end

# Random complex PEPS: each entry is complex(re ~ Uniform(lo, hi), im ~ Uniform(lo, hi))
# with independent real/imag draws. The double-layer norm network is built from
# these ket tensors via `traced_norm_factors`.
function random_uniform_complex_state(
        rng::AbstractRNG,
        g::NamedGraph;
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
        lo::Real = 0.0,
        hi::Real = 1.0,
    )
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))
    lo <= hi || throw(ArgumentError("require lo <= hi, got lo=$lo hi=$hi"))

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(physical_dim, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(bond_dim, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[reverse_edge(e)] = ind
    end

    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        local_inds = Index[sitedict[v]]
        for vn in neighbors(g, v)
            push!(local_inds, edgeinds[NamedEdge(v => vn)])
        end
        tensor = ITensor(ComplexF64, local_inds)
        for iv in eachindval(tensor)
            tensor[iv...] = random_uniform_complex_entry(rng; lo, hi)
        end
        tensors[v] = tensor
    end

    return TensorNetworkState(tensors, Dict(v => Index[sitedict[v]] for v in vs), g)
end

# Index-only PEPS: same index topology and bond/physical dimensions as
# `random_uniform_complex_state`, but each site tensor uses ITensor *empty*
# storage (no dense χ^d allocation, no random fill). All TNMP operations used by
# the complexity probe (`prime`, `dag`, `commoninds`, `*` with `onehot`) only
# manipulate index metadata, so this is sufficient to derive cavity /
# neighborhood TreeSA contraction orders and their tc/sc. This makes the
# complexity sweep feasible at large χ (e.g. χ = 128), where a dense ket tensor
# (2·χ^4 complex entries per interior site) would be many GB and cannot be
# materialized.
function index_only_double_layer_state(
        g::NamedGraph;
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
    )
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(physical_dim, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(bond_dim, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[reverse_edge(e)] = ind
    end

    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        local_inds = Index[sitedict[v]]
        for vn in neighbors(g, v)
            push!(local_inds, edgeinds[NamedEdge(v => vn)])
        end
        tensors[v] = emptyITensor(ComplexF64, local_inds...)
    end

    return TensorNetworkState(tensors, Dict(v => Index[sitedict[v]] for v in vs), g)
end

# Random PEPS from arXiv:2604.24760: each ket entry is Uniform(-alpha, 1-alpha).
# The double-layer norm network is built from these ket tensors via `traced_norm_factors`.
function random_alpha_state(
        rng::AbstractRNG,
        g::NamedGraph;
        alpha::Real = 0.5,
        physical_dim::Integer = 2,
        bond_dim::Integer = 2,
    )
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(physical_dim, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(bond_dim, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[reverse_edge(e)] = ind
    end

    lo = -Float64(alpha)
    width = 1.0
    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        local_inds = Index[sitedict[v]]
        for vn in neighbors(g, v)
            push!(local_inds, edgeinds[NamedEdge(v => vn)])
        end
        tensor = ITensor(ComplexF64, local_inds)
        for iv in eachindval(tensor)
            tensor[iv...] = complex(lo + width * rand(rng))
        end
        tensors[v] = tensor
    end

    return TensorNetworkState(tensors, Dict(v => Index[sitedict[v]] for v in vs), g)
end

# --- Classical q-state Potts model ------------------------------------------
#
# Ferromagnetic q-state Potts model embedded as a real PEPS so that the
# double-layer norm network reproduces the classical partition function. Each
# spin s_v ∈ {1,…,q} lives on a vertex, the edge factor is
#   W_{a,b} = exp(K · δ_{a,b}),
# and an optional per-state `field` adds a site weight exp(field[s_v]).
#
# The amplitude is built so that |amplitude(s)|² equals the Boltzmann weight
# ∏_edges W · ∏_sites exp(field). Bond dimension equals q: every vertex tensor
# copies its physical spin onto each incident bond leg while applying
# G = √M (with M_{a,b} = exp(K/2 · δ_{a,b})), so contracting the two halves of
# an edge yields M·M = W. As a result `exact_marginal` returns the exact
# single-site Potts marginal and `tnmp_marginal` its local approximation.

# G = M^{1/2} for the q×q matrix M_{a,b} = exp(K/2 · δ_{a,b}) = c·I + 1·1ᵀ with
# c = exp(K/2) − 1 ≥ 0. M has eigenvalue c+q on the uniform vector and c on its
# orthogonal complement, giving the closed-form square root below.
function potts_half_edge_matrix(q::Integer, coupling::Real)
    q >= 1 || throw(ArgumentError("q must be positive"))
    c = exp(Float64(coupling) / 2) - 1
    c >= 0 || throw(ArgumentError("require coupling >= 0 (ferromagnetic), got $coupling"))
    diag = sqrt(c)
    off = (sqrt(c + q) - sqrt(c)) / q
    G = fill(off, q, q)
    for a in 1:q
        G[a, a] += diag
    end
    return G
end

function potts_model_state(
        rng::AbstractRNG,
        g::NamedGraph;
        q::Integer = 3,
        coupling::Real = 1.0,
        field::Union{Nothing, AbstractVector{<:Real}} = nothing,
    )
    q >= 2 || throw(ArgumentError("q must be at least 2"))
    G = potts_half_edge_matrix(q, coupling)
    fld = field === nothing ? zeros(Float64, q) : collect(Float64, field)
    length(fld) == q || throw(ArgumentError("field must have length q=$q"))
    site_weight = exp.(fld ./ 2)  # √exp(field) so |amplitude|² carries exp(field)

    vs = collect(vertices(g))
    sitedict = Dict(v => Index(q, "phys,v=$v") for v in vs)

    edgeinds = Dict{Any, Index}()
    for e in edges(g)
        ind = Index(q, "bond,$(src(e))-$(dst(e))")
        edgeinds[e] = ind
        edgeinds[reverse_edge(e)] = ind
    end

    tensors = Dict{eltype(vs), ITensor}()
    for v in vs
        bond_inds = Index[edgeinds[NamedEdge(v => vn)] for vn in neighbors(g, v)]
        local_inds = Index[sitedict[v]; bond_inds...]
        tensor = ITensor(Float64, local_inds)
        for iv in eachindval(tensor)
            s = iv[1].second
            amp = site_weight[s]
            for k in 2:length(iv)
                amp *= G[s, iv[k].second]
            end
            tensor[iv...] = amp
        end
        tensors[v] = tensor
    end

    return TensorNetworkState(tensors, Dict(v => Index[sitedict[v]] for v in vs), g)
end

# --- Perturbed product / pair-factor PEPS -----------------------------------
#
# Tensor builders live in `double_layer_ising.jl` (fully-frustrated Ising,
# spin glass, weak circuit, TFIM imaginary-time, ...).

include(joinpath(@__DIR__, "double_layer_ising.jl"))

function build_frustrated_copy_noise_tensors(
        rng::AbstractRNG,
        g::NamedGraph;
        eps::Real,
        physical_dim::Integer = 2,
        bond_dim::Integer = 8,
    )
    physical_dim > 0 || throw(ArgumentError("physical_dim must be positive"))
    bond_dim > 0 || throw(ArgumentError("bond_dim must be positive"))
    tensors = Dict{eltype(vertices(g)), ITensor}()
    siteinds_dict = Dict{eltype(vertices(g)), Vector{Index}}()
    for v in vertices(g)
        site, local_tensors = frustrated_copy_noise_site_tensor(
            rng, g, v;
            eps = eps,
            physical_dim = physical_dim,
            bond_dim = bond_dim,
        )
        tensors[v] = site
        siteinds_dict[v] = local_tensors
    end
    return tensors, siteinds_dict
end

function frustrated_copy_noise_state(
        rng::AbstractRNG,
        g::NamedGraph;
        eps::Real,
        physical_dim::Integer = 2,
        bond_dim::Integer = 8,
    )
    tensors, siteinds_dict = build_frustrated_copy_noise_tensors(
        rng, g;
        eps = eps,
        physical_dim = physical_dim,
        bond_dim = bond_dim,
    )
    return TensorNetworkState(tensors, siteinds_dict, g)
end

function weak_entangled_biased_circuit_state(
        rng::AbstractRNG,
        g::NamedGraph;
        theta::Real = 0.4,
        phi::Real = 0.15,
        depth::Integer = 2,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    tensors, siteinds_dict = weak_entangled_biased_circuit_tensors(
        rng,
        g;
        theta,
        phi,
        depth,
        physical_dim,
        bond_dim,
    )
    return TensorNetworkState(tensors, siteinds_dict, g)
end

function tfim_imaginary_time_state(
        rng::AbstractRNG,
        g::NamedGraph;
        tau::Real = 0.05,
        coupling_j::Real = 1.0,
        field_h::Real = 0.8,
        steps::Integer = 5,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    tensors, siteinds_dict = tfim_imaginary_time_tensors(
        rng,
        g;
        tau,
        coupling_j,
        field_h,
        steps,
        physical_dim,
        bond_dim,
    )
    return TensorNetworkState(tensors, siteinds_dict, g)
end

function spin_glass_pair_factor_state(
        rng::AbstractRNG,
        g::NamedGraph;
        beta::Real = 0.8,
        bias::Real = 0.2,
        disorder_seed::Integer = 7,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    tensors, siteinds_dict = spin_glass_pair_factor_tensors(
        rng,
        g;
        beta,
        bias,
        disorder_seed,
        physical_dim,
        bond_dim,
    )
    return TensorNetworkState(tensors, siteinds_dict, g)
end

function fully_frustrated_pair_factor_state(
        rng::AbstractRNG,
        g::NamedGraph;
        K::Real = 1.0,
        field::Real = 0.2,
        physical_dim::Integer = 2,
        bond_dim::Integer = 4,
    )
    tensors, siteinds_dict = fully_frustrated_pair_factor_tensors(
        rng,
        g;
        K,
        field,
        physical_dim,
        bond_dim,
    )
    return TensorNetworkState(tensors, siteinds_dict, g)
end
