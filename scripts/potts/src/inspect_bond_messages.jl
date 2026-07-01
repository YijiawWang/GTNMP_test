#!/usr/bin/env julia
# Compare TNMP (rank-2 / GMP) and BP messages on a chosen bond for the
# frustrated Potts model. Reports all TNMP messages (w -> region) whose
# boundary node w is the bond variable, and BP directed edge messages.

using GenericMessagePassing
const GMP = GenericMessagePassing
using GenericMessagePassing: FactorGraph, TNBPConfig
using OMEinsum: EinCode, getixsv, getiyv
using OMEinsum.OMEinsumContractionOrders: IncidenceList

using TensorNetworkQuantumSimulator
const TNQS = TensorNetworkQuantumSimulator
using ITensors: ITensors, Index, ITensor, onehot, order
using Dictionaries: Dictionary
using NamedGraphs: src, dst, NamedEdge, reverse, edges, neighbors, vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Printf

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "run_tnmp.jl"))

ITensors.disable_warn_order()

# Minimal BP helpers (avoid including run_bp.jl which calls main() at load time).
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
        TNQS.update_iteration!(alg, updated, edge_sequence; update_diff! = diff)
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

function bond_key(a::Tuple{Int,Int}, b::Tuple{Int,Int})
    return minmax(a, b)
end

function find_bond_label(label_meaning, idict, a, b)
    key = bond_key(a, b)
    for (i, m) in label_meaning
        m[1] === :bond && m[2] == key && return idict[i], i
    end
    error("bond $key not found")
end

function run_tnmp_with_messages(p::PottsParams; region_L=3, max_iter=200, tol=1e-8)
    R = (region_L - 1) ÷ 2
    code, tensors, label_meaning, factor_coord, phys_label = build_potts_code(p)
    icode, idict = GMP.intcode(code)
    ixs = getixsv(icode)
    iy = getiyv(icode)
    hyper = IncidenceList(Dict([i => ix for (i, ix) in enumerate(ixs)]); openedges = iy)
    fg = FactorGraph(hyper)
    nvars = fg.num_vars
    ntot = nvars + length(tensors)
    lab2int = Dict(idict[i] => i for i in 1:nvars)

    var_meaning(u) = label_meaning[idict[u]]
    function member(u::Int, S)
        if u <= nvars
            m = var_meaning(u)
            if m[1] === :phys
                return m[2] in S
            else
                a, b = m[2]
                return (a in S) || (b in S)
            end
        else
            return factor_coord[u - nvars] in S
        end
    end
    function region_of(u::Int)
        m = var_meaning(u)
        return m[1] === :phys ? site_block(m[2], p.L, R) :
               bond_block(m[2][1], m[2][2], p.L, R)
    end

    neibs = Dict{Int,Vector{Int}}()
    boundaries = Dict{Int,Vector{Int}}()
    for v in 1:nvars
        S = region_of(v)
        nb = Int[u for u in 1:ntot if member(u, S)]
        neibs[v] = nb
        boundaries[v] = GMP.open_boundaries(fg, nb)
    end

    cfg = TNBPConfig(max_iter = max_iter, error = tol, damping = 0.0,
        random_order = false, verbose = false)
    messages, eins, ptensors, mars_eins, mars_tensors =
        GMP.tnbp_precompute(fg, icode, tensors, neibs, boundaries, cfg.optimizer)

    finalerr = Inf
    iters = 0
    for i in 1:max_iter
        finalerr = GMP.tnbp_update!(messages, eins, ptensors, cfg)
        iters = i
        finalerr < tol && break
    end

    return (; messages, label_meaning, idict, lab2int, var_meaning, region_of,
        neibs, boundaries, converged = finalerr < tol, iters, finalerr)
end

function describe_node(u, var_meaning)
    m = var_meaning(u)
    if m[1] === :phys
        return "phys$(m[2])"
    elseif m[1] === :bond
        return "bond$(m[2][1])-$(m[2][2])"
    else
        return string(m)
    end
end

function vec_str(v::AbstractVector)
    return "[" * join([@sprintf("%.6f", x) for x in v], ", ") * "]"
end

function max_diff(a, b)
    la, lb = length(a), length(b)
    la == lb || error("length mismatch $la vs $lb")
    return maximum(abs.(collect(a) .- collect(b)))
end

function itensor_to_vec(t::ITensor)
    is = ITensors.inds(t)
    length(is) == 1 || error("expected rank-1 ITensor, got order $(length(is))")
    i = only(is)
    return Float64[real(t[i => s]) for s in 1:dim(i)]
end

function main()
    args = ARGS
    p = parse_potts_params(args)
    bx = parse_int_opt(args, "bx", 3)
    by = parse_int_opt(args, "by", 3)
    ax = parse_int_opt(args, "ax", 4)
    ay = parse_int_opt(args, "ay", 3)
    bond_a = (bx, by)
    bond_b = (ax, ay)
    max_iter = parse_int_opt(args, "max-iter", 200)
    tol = parse_float_opt(args, "tol", 1e-8)

    println("=== frustrated Potts bond-message inspection ===")
    println("L=$(p.L) q=$(p.q) K=$(p.coupling) couplings=$(p.couplings)")
    println("target bond: $bond_a - $bond_b")
    flush(stdout)

    tnmp = run_tnmp_with_messages(p; max_iter = max_iter, tol = tol)
    bond_int, bond_label = find_bond_label(tnmp.label_meaning, tnmp.idict, bond_a, bond_b)
    println("\nTNMP bond variable: int=$bond_int label=$bond_label")
    println("TNMP converged=$(tnmp.converged) iters=$(tnmp.iters) finalerr=$(tnmp.finalerr)")

    # All messages whose boundary node w is this bond variable: (bond -> region)
    incoming = Tuple{Int,Int}[]
    for ((w, v), msg) in tnmp.messages
        w == bond_int && push!(incoming, (w, v))
    end
    sort!(incoming, by = x -> x[2])

    println("\n--- TNMP messages (bond -> region), $(length(incoming)) total ---")
    tnmp_vecs = Dict{Int,Vector{Float64}}()
    for (w, v) in incoming
        msg = collect(Float64, tnmp.messages[(w, v)])
        msg ./= sum(msg)
        tnmp_vecs[v] = msg
        reg = describe_node(v, tnmp.var_meaning)
        println("  bond -> region $reg (v=$v): $(vec_str(msg))")
    end

    if length(incoming) >= 2
        vs = [v for (_, v) in incoming]
        println("\n--- pairwise TNMP consistency (max |diff|) ---")
        for i in 1:length(vs), j in (i + 1):length(vs)
            vi, vj = vs[i], vs[j]
            d = max_diff(tnmp_vecs[vi], tnmp_vecs[vj])
            ri, rj = describe_node(vi, tnmp.var_meaning), describe_node(vj, tnmp.var_meaning)
            println("  $(ri) vs $(rj): max_diff=$d")
        end
    end

    # BP
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
    bpc, bp_info = update_bp_with_info(tn; maxiter = max_iter, tolerance = tol)
    println("\nBP converged=$(bp_info.converged) iters=$(bp_info.iterations) final_diff=$(bp_info.final_diff)")

    # Directed messages on this undirected bond (both orientations)
    u, v = bond_a, bond_b
    e_uv = NamedEdge(u => v)
    e_vu = NamedEdge(v => u)
    m_uv = itensor_to_vec(TNQS.message(bpc, e_uv))
    m_vu = itensor_to_vec(TNQS.message(bpc, e_vu))
    m_uv ./= sum(m_uv)
    m_vu ./= sum(m_vu)
    println("\n--- BP directed edge messages ---")
    println("  BP $u -> $v: $(vec_str(m_uv))")
    println("  BP $v -> $u: $(vec_str(m_vu))")

    # Compare TNMP bond->region with BP messages.
    # BP message u->v flows into site u from neighbor v (cavity from v side).
    # For bond (3,3)-(4,3): compare with BP (3,3)<-(4,3) i.e. (4,3)->(3,3).
    println("\n--- TNMP vs BP (max |diff|) ---")
    for (w, rv) in incoming
        reg = describe_node(rv, tnmp.var_meaning)
        d1 = max_diff(tnmp_vecs[rv], m_vu)  # bond->region vs BP bond_b->bond_a
        d2 = max_diff(tnmp_vecs[rv], m_uv)  # bond->region vs BP bond_a->bond_b
        println("  TNMP->$reg vs BP $bond_b->$bond_a: max_diff=$d1")
        println("  TNMP->$reg vs BP $bond_a->$bond_b: max_diff=$d2")
    end

    println("\n--- regions containing bond endpoints ---")
    for u in 1:length(tnmp.idict)
        S = tnmp.region_of(u)
        if bond_a in S || bond_b in S
            reg = describe_node(u, tnmp.var_meaning)
            has_msg = any((w, v) == (bond_int, u) for (w, v) in keys(tnmp.messages))
            println("  region $reg: bond_on_open_boundary=$(bond_int in tnmp.boundaries[u]) " *
                    "incoming_from_bond=$has_msg")
        end
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
