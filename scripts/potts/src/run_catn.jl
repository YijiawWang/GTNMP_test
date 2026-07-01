#!/usr/bin/env julia
# Method 1/3: CATN contraction of the single-layer classical q-state Potts model.
#
# The single-site marginal P(s) at the lattice center is obtained by pinning the
# center spin to each value s0 in 1:q and contracting the (closed) network to get
# Z_{s0}; then P(s0) = Z_{s0} / sum_s0 Z_{s0}. Because the network is closed,
# every bond label appears exactly twice, so CATN ingests the genuine 2D lattice
# (interior degree-4 vertices / loops).
#
# Run with:
#   julia --project=../../../env_catn run_catn.jl [--L 10 --q 3 --coupling 0.3 ...]

using CATN: TensorNetwork, contraction!

include(joinpath(@__DIR__, "common.jl"))

# A_v[b_1,...,b_d] = sum_s exp(field[s]) * prod_k Ms[k][s, b_k]   (physical leg summed)
function site_tensor_summed(Ms::Vector{Matrix{Float64}}, field::Vector{Float64})
    q = length(field)
    d = length(Ms)
    A = zeros(Float64, ntuple(_ -> q, d))
    for s in 1:q
        w = exp(field[s])
        for idx in CartesianIndices(A)
            t = w
            for k in 1:d
                t *= Ms[k][s, idx[k]]
            end
            A[idx] += t
        end
    end
    return A
end

# Center site pinned to spin s0: only the s0 term is kept.
function site_tensor_pinned(Ms::Vector{Matrix{Float64}}, field::Vector{Float64}, s0::Int)
    q = length(field)
    d = length(Ms)
    A = zeros(Float64, ntuple(_ -> q, d))
    w = exp(field[s0])
    for idx in CartesianIndices(A)
        t = w
        for k in 1:d
            t *= Ms[k][s0, idx[k]]
        end
        A[idx] = t
    end
    return A
end

# Build the closed CATN network (tensors, ixs) with the center pinned to s0.
function build_network(p::PottsParams, edgeid::Dict, s0::Int)
    tensors = Array{Float64}[]
    ixs = Vector{Int}[]
    for v in all_sites(p.L)
        nbrs = site_neighbors(v, p.L)
        Ms = site_leg_matrices(v, nbrs, p.q, p.coupling, p.couplings)
        lbls = Int[edgeid[minmax(v, n)] for n in nbrs]
        A = v == p.center ? site_tensor_pinned(Ms, p.field, s0) :
            site_tensor_summed(Ms, p.field)
        push!(tensors, A)
        push!(ixs, lbls)
    end
    return tensors, ixs
end

function catn_logZ(tensors, ixs; Dmax::Int, chi::Int)
    tn = TensorNetwork(tensors, ixs; Dmax = Dmax, chi = chi, select = 1, compress = true)
    lnZ, err, psi = contraction!(tn)
    return lnZ, err, psi
end

function main()
    args = ARGS
    p = parse_potts_params(args)
    Dmax = parse_int_opt(args, "catn-dmax", 64)
    chi = parse_int_opt(args, "catn-chi", 64)
    out = parse_opt(args, "out", joinpath(@__DIR__, "..", "results", "potts_catn.txt"))

    println("[catn] L=$(p.L) q=$(p.q) coupling=$(p.coupling) couplings=$(p.couplings) " *
            "field=$(p.field) center=$(p.center) Dmax=$Dmax chi=$chi")
    flush(stdout)

    edgeid = Dict{Any,Int}()
    for (i, e) in enumerate(lattice_edges(p.L))
        edgeid[minmax(e[1], e[2])] = i
    end

    lnZs = Float64[]
    psis = Float64[]
    errs = Float64[]
    t0 = time()
    for s0 in 1:p.q
        tensors, ixs = build_network(p, edgeid, s0)
        lnZ, err, psi = catn_logZ(tensors, ixs; Dmax = Dmax, chi = chi)
        push!(lnZs, real(lnZ))
        push!(psis, real(psi))
        push!(errs, real(err))
        println("[catn]  pin s=$s0 -> lnZ=$(real(lnZ)) psi=$(real(psi)) trunc_err=$(real(err))")
        flush(stdout)
    end
    elapsed = time() - t0

    all(x -> isapprox(x, 1.0; atol = 1e-6), psis) ||
        @warn "CATN returned non-trivial phases (expected +1 for real nonneg tensors)" psis

    m = maximum(lnZs)
    w = exp.(lnZs .- m) .* psis
    p_marg = w ./ sum(w)
    logZ = m + log(sum(exp.(lnZs .- m) .* psis))

    println("[catn] marginal = $p_marg")
    println("[catn] logZ = $logZ  elapsed = $(round(elapsed; digits=3)) s")

    write_result(out;
        method = "catn",
        L = p.L, q = p.q, coupling = p.coupling, couplings = p.couplings,
        field = p.field, center = p.center,
        marginal = p_marg, logZ = logZ,
        catn_dmax = Dmax, catn_chi = chi,
        trunc_err = maximum(errs), elapsed = round(elapsed; digits = 4),
    )
    println("[catn] saved -> $out")
end

main()
