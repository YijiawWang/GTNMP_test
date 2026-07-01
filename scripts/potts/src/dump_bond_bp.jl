#!/usr/bin/env julia
using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using ITensors: ITensors, Index, ITensor
using Dictionaries: Dictionary
using NamedGraphs: src, dst, NamedEdge, edges, neighbors, vertices
using NamedGraphs.NamedGraphGenerators: named_grid

include(joinpath(@__DIR__, "common.jl"))
ITensors.disable_warn_order()

function build_summed_tensor(field, Ms, bond_inds)
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

function update_bp_with_info(tn; maxiter, tolerance)
    cache = BeliefPropagationCache(tn)
    alg = TNQS.set_default_kwargs(
        ITensors.Algorithm("bp"; maxiter=Int(maxiter), tolerance=Float64(tolerance)),
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
        TNQS.update_iteration!(alg, updated, edge_sequence; update_diff! = diff)
        final_diff = diff[] / length(edge_sequence)
        iterations = it
        final_diff <= Float64(tolerance) && (converged = true; break)
    end
    TNQS.invalidate_contraction_sequences!(updated)
    return updated, (; converged, iterations, final_diff)
end

function itensor_to_vec(t::ITensor)
    i = only(ITensors.inds(t))
    return Float64[real(t[i => s]) for s in 1:ITensors.dim(i)]
end

function main()
    p = parse_potts_params(ARGS)
    bx = parse_int_opt(ARGS, "bx", 3)
    by = parse_int_opt(ARGS, "by", 3)
    ax = parse_int_opt(ARGS, "ax", 4)
    ay = parse_int_opt(ARGS, "ay", 3)
    out = parse_opt(ARGS, "out", joinpath(@__DIR__, "..", "results", "bond_bp.txt"))
    max_iter = parse_int_opt(ARGS, "max-iter", 200)
    tol = parse_float_opt(ARGS, "tol", 1e-8)
    u, v = (bx, by), (ax, ay)

    g = named_grid((p.L, p.L))
    vs = collect(vertices(g))
    bondind = Dict{Any,Index}()
    for e in edges(g)
        ind = Index(p.q, "bond")
        bondind[(src(e), dst(e))] = ind
        bondind[(dst(e), src(e))] = ind
    end
    getbond(a, b) = bondind[(a, b)]
    tlist = ITensor[]
    for vtx in vs
        nbrs = neighbors(g, vtx)
        binds = Index[getbond(vtx, n) for n in nbrs]
        Ms = site_leg_matrices(Tuple(vtx), Tuple.(nbrs), p.q, p.coupling, p.couplings)
        push!(tlist, build_summed_tensor(p.field, Ms, binds))
    end
    tn = TensorNetwork(Dictionary(vs, tlist), g)
    bpc, info = update_bp_with_info(tn; maxiter=max_iter, tolerance=tol)
    m_uv = itensor_to_vec(TNQS.message(bpc, NamedEdge(u => v))); m_uv ./= sum(m_uv)
    m_vu = itensor_to_vec(TNQS.message(bpc, NamedEdge(v => u))); m_vu ./= sum(m_vu)

    open(out, "w") do io
        println(io, "method=bp")
        println(io, "L=$(p.L)")
        println(io, "q=$(p.q)")
        println(io, "coupling=$(p.coupling)")
        println(io, "couplings=$(p.couplings)")
        println(io, "bond=$(u[1]),$(u[2]),$(v[1]),$(v[2])")
        println(io, "converged=$(info.converged)")
        println(io, "iters=$(info.iterations)")
        println(io, "final_diff=$(info.final_diff)")
        println(io, "direct from=$(u) to=$(v) message=$(join(m_uv, ","))")
        println(io, "direct from=$(v) to=$(u) message=$(join(m_vu, ","))")
    end
    println("saved BP bond messages -> $out")
end

main()
