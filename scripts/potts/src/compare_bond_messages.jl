#!/usr/bin/env julia
using Printf

function parse_vec(s)
    return parse.(Float64, split(s, ","))
end

function vec_str(v)
    "[" * join([@sprintf("%.6f", x) for x in v], ", ") * "]"
end

function max_diff(a, b)
    return maximum(abs.(a .- b))
end

function read_kv(path)
    d = Dict{String,String}()
    rows = Vector{Tuple{String,Bool,Vector{Float64}}}()
    bond = nothing
    for line in eachline(path)
        if startswith(line, "incoming ")
            m = match(r"^incoming region=(.+) open_boundary=(true|false) message=(.+)$", line)
            m === nothing && error("bad incoming line: $line")
            reg = m.captures[1]
            onb = m.captures[2] == "true"
            msg = parse_vec(m.captures[3])
            push!(rows, (reg, onb, msg))
            continue
        end
        kv = split(line, "=", limit=2)
        length(kv) < 2 && continue
        k, v = kv[1], kv[2]
        if k == "bond"
            parts = parse.(Int, split(v, ","))
            bond = (parts[1], parts[2]), (parts[3], parts[4])
        else
            d[k] = v
        end
    end
    return d, bond, rows
end

function read_bp(path)
    d = Dict{String,String}()
    bond = nothing
    m_uv = m_vu = Float64[]
    u = v = (0, 0)
    for line in eachline(path)
        kv = split(line, "=", limit=2)
        length(kv) < 2 && continue
        k, val = kv[1], kv[2]
        if k == "bond"
            parts = parse.(Int, split(val, ","))
            bond = (parts[1], parts[2]), (parts[3], parts[4])
        elseif startswith(k, "direct")
            m = match(r"^direct from=\((\d+), (\d+)\) to=\((\d+), (\d+)\) message=(.+)$", line)
            m === nothing && error("bad direct line: $line")
            from = (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
            to = (parse(Int, m.captures[3]), parse(Int, m.captures[4]))
            msg = parse_vec(m.captures[5])
            if from == bond[1] && to == bond[2]
                m_uv = msg
            else
                m_vu = msg
            end
        else
            d[k] = val
        end
    end
    bond === nothing && error("bond not found in $path")
    return d, bond, bond[1], bond[2], m_uv, m_vu
end

function main()
    tnmp_path, bp_path = ARGS[1], ARGS[2]
    td, bond, incoming = read_kv(tnmp_path)
    bd, _, u, v, m_uv, m_vu = read_bp(bp_path)
    u, v = bond[1], bond[2]

    println("=== bond message comparison ===")
    println("bond: $(bond[1]) - $(bond[2])")
    println("K=$(td["coupling"]) couplings=$(td["couplings"]) q=$(td["q"]) L=$(td["L"])")
    println("TNMP converged=$(td["converged"]) iters=$(td["iters"]) err=$(td["finalerr"])")
    println("BP   converged=$(bd["converged"]) iters=$(bd["iters"]) diff=$(bd["final_diff"])")

    println("\n--- TNMP: bond -> region ($(length(incoming)) messages) ---")
    for (reg, onb, msg) in incoming
        println("  -> $reg (open_bnd=$onb): $(vec_str(msg))")
    end

    println("\n--- BP directed messages ---")
    println("  $u -> $v: $(vec_str(m_uv))")
    println("  $v -> $u: $(vec_str(m_vu))")

    if length(incoming) >= 2
        println("\n--- TNMP pairwise consistency ---")
        for i in 1:length(incoming), j in (i+1):length(incoming)
            ri, rj = incoming[i][1], incoming[j][1]
            d = max_diff(incoming[i][3], incoming[j][3])
            println("  $ri vs $rj: max_diff=$d")
        end
    end

    println("\n--- TNMP vs BP ---")
    for (reg, _, msg) in incoming
        d1 = max_diff(msg, m_vu)
        d2 = max_diff(msg, m_uv)
        println("  TNMP->$reg vs BP $v->$u: max_diff=$d1")
        println("  TNMP->$reg vs BP $u->$v: max_diff=$d2")
    end
end

main()
