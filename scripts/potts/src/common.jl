# Shared, dependency-free helpers for the q-state Potts 3-method comparison.
#
# This file uses ONLY Base Julia so it can be `include`d from three different
# package environments (CATN, GenericMessagePassing, TensorNetworkQuantumSimulator),
# guaranteeing all three methods build the *exact same* classical Potts model.
#
# Model: q-state Potts on an open-boundary L x L square lattice with per-edge
# (possibly antiferromagnetic / frustrated) couplings.
#   * spin s_v in {1,...,q} on every site v = (x, y), x,y in 1:L
#   * edge weight  B_e[a,b] = exp(K * J_e) if a == b, else 1   (J_e in {+1,-1})
#   * site weight  exp(field[s_v])
# Partition function Z = sum_{configs} prod_edges B_e * prod_sites exp(field).
#
# Couplings (`--couplings`):
#   * "ferro"      : J_e = +1 on every edge (ferromagnetic).
#   * "frustrated" : fully-frustrated square-lattice pattern (mirrors
#     `double_layer_ising.jl:fully_frustrated_square_couplings`): x-direction
#     edges are FM (J=+1), y-direction edges are AF (J=-1) on odd columns, so
#     every elementary plaquette carries exactly one AF + three FM bonds
#     (product J = -1 -> genuine frustration for every q).
#
# Single-layer tensor network encoding (one tensor per site, bond dim q): every
# edge carries the SAME shared B_e, split *asymmetrically* across the two
# endpoints so all tensors stay real and non-negative even for AF bonds. For an
# edge oriented c -> n (n the +x/+y neighbour), site c carries the full bond
# matrix B_e on that leg while site n carries the identity (delta); contracting
# the shared leg gives sum_b B_e[s_c, b] * delta[s_n, b] = B_e[s_c, s_n]. Each
# site tensor is  T[s, b_1,...,b_d] = exp(field[s]) * prod_k M_k[s, b_k]  with
# M_k = B_e (outgoing leg) or identity (incoming leg).

# --- CLI parsing -----------------------------------------------------------

function parse_opt(args::Vector{String}, key::AbstractString, default)
    flag = "--$key"
    for i in eachindex(args)
        if args[i] == flag && i < length(args)
            return args[i + 1]
        elseif startswith(args[i], "$flag=")
            return args[i][(length(flag) + 2):end]
        end
    end
    return default === nothing ? nothing : string(default)
end

parse_int_opt(args, key, default) = (v = parse_opt(args, key, default); v === nothing ? nothing : parse(Int, v))
parse_float_opt(args, key, default) = (v = parse_opt(args, key, default); v === nothing ? nothing : parse(Float64, v))

# --- Model parameters ------------------------------------------------------

struct PottsParams
    L::Int
    q::Int
    coupling::Float64
    field::Vector{Float64}
    center::Tuple{Int,Int}
    couplings::String   # "ferro" or "frustrated"
end

# Deterministic symmetry-breaking field so every method gets an identical,
# non-trivial single-site marginal. Override with --field "a,b,c".
function default_field(q::Int)
    q == 1 && return [0.0]
    return [0.1 * (1 - 2 * (s - 1) / (q - 1)) for s in 1:q]
end

grid_center(L::Int) = ((L + 1) ÷ 2, (L + 1) ÷ 2)

function parse_potts_params(args::Vector{String} = ARGS)
    L = parse_int_opt(args, "L", 10)
    q = parse_int_opt(args, "q", 3)
    coupling = parse_float_opt(args, "coupling", 0.3)
    field_str = parse_opt(args, "field", nothing)
    field = field_str === nothing ? default_field(q) : parse.(Float64, split(field_str, ","))
    length(field) == q || error("field length $(length(field)) != q=$q")
    cx = parse_int_opt(args, "cx", grid_center(L)[1])
    cy = parse_int_opt(args, "cy", grid_center(L)[2])
    couplings = parse_opt(args, "couplings", "ferro")
    couplings in ("ferro", "frustrated") ||
        error("--couplings must be ferro or frustrated, got $couplings")
    return PottsParams(L, q, coupling, field, (cx, cy), couplings)
end

# --- Lattice ---------------------------------------------------------------

inbounds(c::Tuple{Int,Int}, L::Int) = 1 <= c[1] <= L && 1 <= c[2] <= L

all_sites(L::Int) = [(x, y) for x in 1:L for y in 1:L]

function site_neighbors(c::Tuple{Int,Int}, L::Int)
    (x, y) = c
    ns = Tuple{Int,Int}[]
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
        n = (x + d[1], y + d[2])
        inbounds(n, L) && push!(ns, n)
    end
    return ns
end

# Undirected edges of the open square lattice (each once), as ordered coord pairs.
function lattice_edges(L::Int)
    es = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
    for x in 1:L, y in 1:L
        c = (x, y)
        for n in ((x + 1, y), (x, y + 1))
            inbounds(n, L) && push!(es, (c, n))
        end
    end
    return es
end

# --- Couplings & Potts bond tensors ----------------------------------------

# Sign J_e in {+1,-1} of the edge between site `c` and its neighbour `n`.
# `c` is taken to be the lower (B-carrying) endpoint, i.e. n = c + x_hat / y_hat.
function coupling_sign(c::Tuple{Int,Int}, n::Tuple{Int,Int}, mode::AbstractString)
    mode == "ferro" && return 1
    if mode == "frustrated"
        if c[2] == n[2]          # same y  -> edge along x  : ferromagnetic
            return 1
        else                     # same x  -> edge along y  : AF on odd columns
            return isodd(c[1]) ? -1 : 1
        end
    end
    error("unknown couplings mode: $mode")
end

# B_e[a,b] = exp(K*J) if a==b, else 1.
function bond_matrix(q::Int, K::Float64, J::Int)
    B = ones(Float64, q, q)
    d = exp(K * J)
    for a in 1:q
        B[a, a] = d
    end
    return B
end

function identity_matrix(q::Int)
    M = zeros(Float64, q, q)
    for a in 1:q
        M[a, a] = 1.0
    end
    return M
end

# Whether the edge (w, nb) is oriented outgoing from `w` (w is the lower / B side).
is_outgoing(w::Tuple{Int,Int}, nb::Tuple{Int,Int}) =
    nb == (w[1] + 1, w[2]) || nb == (w[1], w[2] + 1)

# Per-leg q×q matrix for the leg of site `w` towards neighbour `nb`:
# the full bond matrix B_e on the outgoing (B) side, identity on the incoming
# (delta) side. The same shared B_e is thus used by all three methods.
function leg_matrix(w::Tuple{Int,Int}, nb::Tuple{Int,Int}, q::Int, K::Float64, mode::AbstractString)
    if is_outgoing(w, nb)
        return bond_matrix(q, K, coupling_sign(w, nb, mode))
    else
        return identity_matrix(q)
    end
end

# Per-leg matrices for site `w`, aligned with the neighbour order `nbrs`.
function site_leg_matrices(w::Tuple{Int,Int}, nbrs, q::Int, K::Float64, mode::AbstractString)
    return Matrix{Float64}[leg_matrix(w, nb, q, K, mode) for nb in nbrs]
end

# --- Size-only placeholder array -------------------------------------------
# Carries shape (and element type) but allocates no data. Used by the
# complexity-only sweep so that large-q site tensors (q^(degree+1), e.g. q^5 at
# an interior site) never get materialised: TreeSA contraction-order complexity
# depends solely on the index sizes, never on the tensor values, and
# `tnbp_precompute` reads only `size`/`eltype` (it never contracts these).
struct SizeArray{T,N} <: AbstractArray{T,N}
    sz::NTuple{N,Int}
end
SizeArray{T}(sz::NTuple{N,Int}) where {T,N} = SizeArray{T,N}(sz)
Base.size(a::SizeArray) = a.sz
Base.eltype(::Type{<:SizeArray{T}}) where {T} = T
Base.getindex(::SizeArray{T}, ::Vararg{Int}) where {T} = zero(T)

# --- Result IO (plain key=value text, environment-agnostic) ----------------

_fmt(v::AbstractVector) = join(v, ",")
_fmt(v::Tuple) = join(v, ",")
_fmt(v) = string(v)

function write_result(path::AbstractString; kwargs...)
    mkpath(dirname(path))
    open(path, "w") do io
        for (k, v) in kwargs
            println(io, "$(k)=$(_fmt(v))")
        end
    end
    return path
end

function read_result(path::AbstractString)
    d = Dict{String,String}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        kv = split(line, "=", limit = 2)
        length(kv) == 2 && (d[kv[1]] = kv[2])
    end
    return d
end

parse_vec(s::AbstractString) = parse.(Float64, split(s, ","))
