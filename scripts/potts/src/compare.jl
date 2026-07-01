#!/usr/bin/env julia
# Read the three method result files and print the marginals + L1 distances.
# Uses CATN as the (high-bond-dimension) reference.
#
#   julia compare.jl results/potts_catn.txt results/potts_tnmp.txt results/potts_bp.txt

include(joinpath(@__DIR__, "common.jl"))

function main()
    files = isempty(ARGS) ? [
        joinpath(@__DIR__, "..", "results", "potts_catn.txt"),
        joinpath(@__DIR__, "..", "results", "potts_tnmp.txt"),
        joinpath(@__DIR__, "..", "results", "potts_bp.txt"),
    ] : ARGS

    results = Dict{String,Dict{String,String}}()
    for f in files
        isfile(f) || (println("(missing: $f)"); continue)
        d = read_result(f)
        results[get(d, "method", f)] = d
    end

    println("="^70)
    println("q-state Potts model: single-site marginal P(s) at the lattice center")
    ref_d = get(results, "catn", get(results, "tnmp", get(results, "bp", nothing)))
    if ref_d !== nothing
        d = ref_d
        println("  model: L=$(d["L"]) q=$(d["q"]) coupling=$(d["coupling"]) " *
                "couplings=$(get(d,"couplings","ferro")) " *
                "field=[$(d["field"])] center=($(d["center"]))")
    end
    println("-"^70)
    for m in ("catn", "tnmp", "bp")
        haskey(results, m) || continue
        d = results[m]
        v = parse_vec(d["marginal"])
        extra = m == "catn" ? "logZ=$(get(d,"logZ","?")) Dmax=$(get(d,"catn_dmax","?"))" :
                m == "tnmp" ? "converged=$(get(d,"converged","?")) iters=$(get(d,"iters","?")) region_L=$(get(d,"region_L","?"))" :
                "iters<=$(get(d,"bp_max_iter","?"))"
        println(rpad(uppercase(m), 6), " marginal = ",
            "[" * join(round.(v; digits = 6), ", ") * "]",
            "   elapsed=$(get(d,"elapsed","?"))s   $extra")
    end
    println("-"^70)

    l1(a, b) = sum(abs.(a .- b))
    if haskey(results, "catn")
        ref = parse_vec(results["catn"]["marginal"])
        for m in ("tnmp", "bp")
            haskey(results, m) || continue
            println("  L1( $(uppercase(m)) , CATN ) = ",
                round(l1(parse_vec(results[m]["marginal"]), ref); digits = 8))
        end
    end
    if haskey(results, "tnmp") && haskey(results, "bp")
        println("  L1( TNMP , BP )          = ",
            round(l1(parse_vec(results["tnmp"]["marginal"]), parse_vec(results["bp"]["marginal"])); digits = 8))
    end
    println("="^70)
end

main()
