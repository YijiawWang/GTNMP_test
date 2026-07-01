#!/usr/bin/env julia
# Method 2/3: single-layer Tensor Network Message Passing (TNMP) for the
# classical q-state Potts model, using GenericMessagePassing's TNMP engine
# (`tnbp_precompute` / `tnbp_update!`) but with CUSTOM neighborhoods so that the
# neighborhood of every node is exactly the 3x3 sub-lattice centered on that node
# (and the L x (L-1) block for a bond), matching the established region_vertices
# convention. The center site's marginal is read off the (open) physical leg.
#
# Run with:
#   julia --project=../../../env_gmp run_tnmp.jl [--L 10 --q 3 --coupling 0.3 ...]

using GenericMessagePassing
const GMP = GenericMessagePassing
using GenericMessagePassing: FactorGraph, TNBPConfig
using OMEinsum: EinCode, getixsv, getiyv
using OMEinsum.OMEinsumContractionOrders: IncidenceList

include(joinpath(@__DIR__, "common.jl"))

# Site tensor with an explicit (open) physical leg:
#   T[s, b_1,...,b_d] = exp(field[s]) * prod_k Ms[k][s, b_k]
function site_tensor_phys(Ms::Vector{Matrix{Float64}}, field::Vector{Float64})
    q = length(field)
    d = length(Ms)
    T = zeros(Float64, ntuple(_ -> q, d + 1))
    bsize = ntuple(_ -> q, d)
    for s in 1:q
        w = exp(field[s])
        for bidx in CartesianIndices(bsize)
            t = w
            for k in 1:d
                t *= Ms[k][s, bidx[k]]
            end
            T[s, Tuple(bidx)...] = t
        end
    end
    return T
end

# 3x3-style region builders (region size = 2R+1; default R=1 -> 3x3 / 3x2).
function site_block(c::Tuple{Int,Int}, L::Int, R::Int)
    S = Set{Tuple{Int,Int}}()
    for x in (c[1] - R):(c[1] + R), y in (c[2] - R):(c[2] + R)
        (1 <= x <= L && 1 <= y <= L) && push!(S, (x, y))
    end
    return S
end

function bond_block(a::Tuple{Int,Int}, b::Tuple{Int,Int}, L::Int, R::Int)
    (x1, y1) = a
    (x2, y2) = b
    S = Set{Tuple{Int,Int}}()
    if x1 == x2  # vertical bond: span endpoints along y, widen +-R along x
        ya, yb = minmax(y1, y2)
        xlo, xhi = x1 - R, x1 + R
        ylo, yhi = ya - (R - 1), yb + (R - 1)
    else        # horizontal bond
        xa, xb = minmax(x1, x2)
        xlo, xhi = xa - (R - 1), xb + (R - 1)
        ylo, yhi = y1 - R, y1 + R
    end
    for x in xlo:xhi, y in ylo:yhi
        (1 <= x <= L && 1 <= y <= L) && push!(S, (x, y))
    end
    return S
end

# Build the single-layer (code, tensors) with open physical legs.
# `sizes_only=true` substitutes zero-allocation `SizeArray` placeholders for the
# site tensors (same shapes/labels) so that large-q complexity sweeps avoid
# materialising q^(degree+1) arrays; the index structure handed to
# `tnbp_precompute` is identical.
function build_potts_code(p::PottsParams; sizes_only::Bool = false)
    label_meaning = Dict{Int,Any}()
    counter = Ref(0)
    newlabel() = (counter[] += 1; counter[])

    phys_label = Dict{Tuple{Int,Int},Int}()
    for v in all_sites(p.L)
        l = newlabel()
        phys_label[v] = l
        label_meaning[l] = (:phys, v)
    end
    edge_label = Dict{Any,Int}()
    for e in lattice_edges(p.L)
        key = minmax(e[1], e[2])
        l = newlabel()
        edge_label[key] = l
        label_meaning[l] = (:bond, (e[1], e[2]))
    end

    sites = all_sites(p.L)
    tensors = sizes_only ? SizeArray{Float64}[] : Array{Float64}[]
    ixs = Vector{Int}[]
    factor_coord = Tuple{Int,Int}[]
    for v in sites
        nbrs = site_neighbors(v, p.L)
        if sizes_only
            T = SizeArray{Float64}(ntuple(_ -> p.q, length(nbrs) + 1))
        else
            Ms = site_leg_matrices(v, nbrs, p.q, p.coupling, p.couplings)
            T = site_tensor_phys(Ms, p.field)
        end
        labs = Int[phys_label[v]]
        for n in nbrs
            push!(labs, edge_label[minmax(v, n)])
        end
        push!(tensors, T)
        push!(ixs, labs)
        push!(factor_coord, v)
    end
    iy = Int[phys_label[v] for v in sites]
    code = EinCode(ixs, iy)
    return code, tensors, label_meaning, factor_coord, phys_label
end

# Build the factor graph + the CUSTOM 3x3 (2R+1) neighborhoods/boundaries that
# `tnbp_precompute` consumes. This is the single source of truth for the regions:
# both the real marginal run (`tnmp_center_marginal`) and the complexity sweep
# call it, so the sweep measures the *exact* sub-TNs the algorithm contracts.
function build_potts_regions(code, tensors, label_meaning, factor_coord, L, R)
    icode, idict = GMP.intcode(code)
    ixs = getixsv(icode)
    iy = getiyv(icode)
    hyper = IncidenceList(Dict([i => ix for (i, ix) in enumerate(ixs)]); openedges = iy)
    fg = FactorGraph(hyper)
    nvars = fg.num_vars
    ntot = nvars + length(tensors)  # variables (indices) + factors (site tensors)
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
        return m[1] === :phys ? site_block(m[2], L, R) :
               bond_block(m[2][1], m[2][2], L, R)
    end

    neibs = Dict{Int,Vector{Int}}()
    boundaries = Dict{Int,Vector{Int}}()
    for v in 1:nvars
        S = region_of(v)
        nb = Int[u for u in 1:ntot if member(u, S)]
        neibs[v] = nb
        boundaries[v] = GMP.open_boundaries(fg, nb)
    end
    return (; fg, icode, idict, ixs, nvars, ntot, lab2int, var_meaning, neibs, boundaries)
end

function tnmp_center_marginal(code, tensors, label_meaning, factor_coord, L, R,
        center_phys_label; max_iter, tol, damping, random_order, verbose)
    reg = build_potts_regions(code, tensors, label_meaning, factor_coord, L, R)
    fg, icode, lab2int, nvars, ntot = reg.fg, reg.icode, reg.lab2int, reg.nvars, reg.ntot

    cfg = TNBPConfig(max_iter = max_iter, error = tol, damping = damping,
        random_order = random_order, verbose = verbose)
    messages, eins, ptensors, mars_eins, mars_tensors =
        GMP.tnbp_precompute(fg, icode, tensors, reg.neibs, reg.boundaries, cfg.optimizer)

    iters = 0
    finalerr = Inf
    for i in 1:max_iter
        finalerr = GMP.tnbp_update!(messages, eins, ptensors, cfg)
        iters = i
        verbose && (println("[tnmp] iter $i err=$finalerr"); flush(stdout))
        finalerr < tol && break
    end

    cint = lab2int[center_phys_label]
    t = mars_eins[cint](mars_tensors[cint]...)
    pmarg = collect(Float64, t ./ sum(t))
    return pmarg, (; converged = finalerr < tol, iters = iters, finalerr = finalerr,
        nvars = nvars, nfactors = ntot - nvars)
end

function main()
    args = ARGS
    p = parse_potts_params(args)
    region_L = parse_int_opt(args, "region-L", 3)
    isodd(region_L) || error("region-L must be odd")
    R = (region_L - 1) ÷ 2
    max_iter = parse_int_opt(args, "tnmp-max-iter", 200)
    tol = parse_float_opt(args, "tnmp-tol", 1e-8)
    damping = parse_float_opt(args, "tnmp-damping", 0.0)
    random_order = parse_int_opt(args, "tnmp-random-order", 0) != 0
    verbose = parse_int_opt(args, "verbose", 1) != 0
    out = parse_opt(args, "out", joinpath(@__DIR__, "..", "results", "potts_tnmp.txt"))

    println("[tnmp] L=$(p.L) q=$(p.q) coupling=$(p.coupling) couplings=$(p.couplings) " *
            "field=$(p.field) center=$(p.center) region_L=$region_L max_iter=$max_iter tol=$tol")
    flush(stdout)

    code, tensors, label_meaning, factor_coord, phys_label = build_potts_code(p)

    t0 = time()
    pmarg, info = tnmp_center_marginal(code, tensors, label_meaning, factor_coord,
        p.L, R, phys_label[p.center];
        max_iter = max_iter, tol = tol, damping = damping,
        random_order = random_order, verbose = verbose)
    elapsed = time() - t0

    println("[tnmp] marginal = $pmarg")
    println("[tnmp] converged=$(info.converged) iters=$(info.iters) " *
            "finalerr=$(info.finalerr) vars=$(info.nvars) factors=$(info.nfactors)")
    println("[tnmp] elapsed = $(round(elapsed; digits=3)) s")

    write_result(out;
        method = "tnmp",
        L = p.L, q = p.q, coupling = p.coupling, couplings = p.couplings,
        field = p.field, center = p.center,
        marginal = pmarg, region_L = region_L,
        converged = info.converged, iters = info.iters,
        finalerr = info.finalerr, elapsed = round(elapsed; digits = 4),
    )
    println("[tnmp] saved -> $out")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
