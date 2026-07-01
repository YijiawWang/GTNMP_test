#!/usr/bin/env julia
# Method 3/3: single-layer Belief Propagation (BP) for the classical q-state
# Potts model, using TensorNetworkQuantumSimulator's BeliefPropagationCache.
#
# The model is built as a single-layer ITensor TensorNetwork (one tensor per
# site, bond dim q, physical leg summed out). BP is run to convergence, then the
# center single-site marginal P(s) is read off by contracting the center site's
# *physical-leg* tensor with the converged incoming (cavity) messages. This is BP
# = TNMP with a trivial (single-site) region, so it is the natural baseline for
# the 3x3 TNMP run.
#
# Run with:
#   julia --project=TNMP_test run_bp.jl [--L 10 --q 3 --coupling 0.3 ...]

using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using ITensors: ITensors, Index, ITensor, onehot, order
using Dictionaries: Dictionary
using NamedGraphs: src, dst

include(joinpath(@__DIR__, "common.jl"))

ITensors.disable_warn_order()

# A_v = sum_s exp(field[s]) * prod_k Ms[k][s, :]   (physical leg summed out).
# Ms[k] is the q×q leg matrix paired with bond_inds[k].
function build_summed_tensor(field::Vector{Float64}, Ms::Vector{Matrix{Float64}}, bond_inds)
    q = length(field)
    A = ITensor(Float64, bond_inds...)
    for s in 1:q
        term = ITensor(exp(field[s]))
        for (k, bi) in enumerate(bond_inds)
            vvec = ITensor(Float64, bi)
            for a in 1:q
                vvec[bi => a] = Ms[k][s, a]
            end
            term = term * vvec
        end
        A += term
    end
    return A
end

# Center tensor keeping the physical leg open:
#   Tc[s, b...] = exp(field[s]) * prod_k Ms[k][s, b_k]
function build_phys_tensor(field::Vector{Float64}, Ms::Vector{Matrix{Float64}}, bond_inds, phys::Index)
    q = length(field)
    Tc = ITensor(Float64, phys, bond_inds...)
    for s in 1:q
        term = ITensor(exp(field[s]))
        for (k, bi) in enumerate(bond_inds)
            vvec = ITensor(Float64, bi)
            for a in 1:q
                vvec[bi => a] = Ms[k][s, a]
            end
            term = term * vvec
        end
        Tc += term * onehot(phys => s)
    end
    return Tc
end

function update_bp_with_info(tn; maxiter::Integer, tolerance::Real)
    cache = BeliefPropagationCache(tn)
    alg = TNQS.set_default_kwargs(
        ITensors.Algorithm("bp"; maxiter = Int(maxiter), tolerance = Float64(tolerance)),
        cache,
    )
    updated = copy(cache)
    TNQS.invalidate_contraction_sequences!(updated)

    final_diff = Inf
    iterations = 0
    converged = false
    edge_sequence = alg.kwargs.edge_sequence
    for it in 1:Int(maxiter)
        diff = Ref(0.0)
        TNQS.update_iteration!(
            alg, updated, edge_sequence; update_diff! = diff,
        )
        final_diff = diff[] / length(edge_sequence)
        iterations = it
        if final_diff <= Float64(tolerance)
            converged = true
            break
        end
    end

    TNQS.invalidate_contraction_sequences!(updated)
    return updated, (; converged, iterations, final_diff)
end

function main()
    args = ARGS
    p = parse_potts_params(args)
    maxiter = parse_int_opt(args, "bp-max-iter", 200)
    tol = parse_float_opt(args, "bp-tol", 1e-10)
    out = parse_opt(args, "out", joinpath(@__DIR__, "..", "results", "potts_bp.txt"))

    println("[bp] L=$(p.L) q=$(p.q) coupling=$(p.coupling) couplings=$(p.couplings) " *
            "field=$(p.field) center=$(p.center) max_iter=$maxiter tol=$tol")
    flush(stdout)

    g = named_grid((p.L, p.L))
    vs = collect(vertices(g))

    bondind = Dict{Any,Index}()
    for e in edges(g)
        ind = Index(p.q, "bond")
        bondind[(src(e), dst(e))] = ind
        bondind[(dst(e), src(e))] = ind
    end
    getbond(u, v) = bondind[(u, v)]

    tlist = ITensor[]
    for v in vs
        nbrs = neighbors(g, v)
        binds = Index[getbond(v, n) for n in nbrs]
        Ms = site_leg_matrices(Tuple(v), Tuple.(nbrs), p.q, p.coupling, p.couplings)
        push!(tlist, build_summed_tensor(p.field, Ms, binds))
    end
    tensors = Dictionary(vs, tlist)
    tn = TensorNetwork(tensors, g)

    t0 = time()
    bpc, bp_info = update_bp_with_info(tn; maxiter = maxiter, tolerance = tol)

    # Center marginal: contract the physical-leg center tensor with the converged
    # incoming cavity messages.
    cnbrs = neighbors(g, p.center)
    cbinds = Index[getbond(p.center, n) for n in cnbrs]
    cMs = site_leg_matrices(Tuple(p.center), Tuple.(cnbrs), p.q, p.coupling, p.couplings)
    phys = Index(p.q, "phys")
    Tc = build_phys_tensor(p.field, cMs, cbinds, phys)

    incoming = TNQS.incoming_messages(bpc, p.center)
    marg_it = Tc
    for m in incoming
        marg_it = marg_it * m
    end
    order(marg_it) == 1 ||
        error("center marginal tensor has order $(order(marg_it)) (expected 1)")
    pmarg = Float64[real(marg_it[phys => s]) for s in 1:p.q]
    pmarg = max.(pmarg, 0.0)
    pmarg ./= sum(pmarg)
    elapsed = time() - t0

    println("[bp] marginal = $pmarg")
    println("[bp] converged=$(bp_info.converged) iters=$(bp_info.iterations) " *
            "final_diff=$(bp_info.final_diff)")
    println("[bp] elapsed = $(round(elapsed; digits=3)) s")

    write_result(out;
        method = "bp",
        L = p.L, q = p.q, coupling = p.coupling, couplings = p.couplings,
        field = p.field, center = p.center,
        marginal = pmarg,
        bp_max_iter = maxiter, bp_tol = tol,
        converged = bp_info.converged,
        iters = bp_info.iterations,
        final_diff = bp_info.final_diff,
        elapsed = round(elapsed; digits = 4),
    )
    println("[bp] saved -> $out")
end

main()
