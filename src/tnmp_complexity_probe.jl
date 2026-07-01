# Optional TreeSA complexity recorder for TNMP contractions.
#
# This file deliberately does not construct any TNMP sub-network. The main
# algorithm still builds cavity and neighborhood tensor lists in `tnmp.jl`; the
# probe only records tc/sc when TreeSA is already asked to optimize a contraction
# sequence for that exact tensor list.

using Serialization: serialize

struct TNMPComplexityProbe
    rows::Vector{NamedTuple}
    lock::ReentrantLock
end

TNMPComplexityProbe() = TNMPComplexityProbe(NamedTuple[], ReentrantLock())

function max_index_log2(ixs, size_dict)
    isempty(ixs) && return 0.0
    return maximum(ixs) do ix
        isempty(ix) ? 0.0 : sum(ind -> log2(size_dict[ind]), ix)
    end
end

index_log2(ixs, size_dict) =
    isempty(ixs) ? 0.0 : sum(ind -> log2(size_dict[ind]), ixs)

function contraction_kind(key)
    if key isa Tuple && !isempty(key)
        first(key) === :message && return "cavity"
        first(key) === :marginal && return "neighborhood"
    end
    return "unknown"
end

function contraction_metrics(code, size_dict, optcode, n_tensors::Integer)
    cc = contraction_complexity(optcode, size_dict)
    return (;
        n_tensors = Int(n_tensors),
        n_open_legs = length(code.iy),
        max_input_log2 = max_index_log2(code.ixs, size_dict),
        output_log2 = index_log2(code.iy, size_dict),
        sc = cc.sc,
        tc = cc.tc,
    )
end

function one_tensor_metrics(tensors::Vector{<:ITensor})
    code, size_dict = to_eincode(tensors)
    return (;
        n_tensors = length(tensors),
        n_open_legs = length(code.iy),
        max_input_log2 = max_index_log2(code.ixs, size_dict),
        output_log2 = index_log2(code.iy, size_dict),
        sc = 0.0,
        tc = 0.0,
    )
end

# TreeSA used to optimize every cavity/neighborhood contraction. `sc_target`
# defaults to TreeSA's own default (20) but can be overridden via the
# `TNMP_TREESA_SC_TARGET` environment variable so complexity sweeps can probe
# how the optimizer trades tc against the space-complexity target.
function _default_treesa()
    raw = get(ENV, "TNMP_TREESA_SC_TARGET", "")
    isempty(raw) && return TreeSA()
    return TreeSA(; score = ScoreFunction(; sc_target = parse(Float64, raw)))
end

function contraction_plan(tensors::Vector{<:ITensor}; collect_complexity::Bool = false)
    length(tensors) == 1 &&
        return (; sequence = 1, complexity = collect_complexity ? one_tensor_metrics(tensors) : nothing)

    code, size_dict = to_eincode(tensors)
    optcode = optimize_code(code, size_dict, _default_treesa())
    metrics = collect_complexity ? contraction_metrics(code, size_dict, optcode, length(tensors)) : nothing
    return (; sequence = nested_to_sequence(optcode), complexity = metrics)
end

record_complexity!(::Nothing, key, tensors, metrics) = nothing

function record_complexity!(probe::TNMPComplexityProbe, key, tensors, metrics)
    metrics === nothing && return nothing
    row = (;
        kind = contraction_kind(key),
        key = string(key),
        metrics...,
    )
    lock(probe.lock) do
        push!(probe.rows, row)
    end
    return nothing
end

function complexity_rows(probe::TNMPComplexityProbe)
    lock(probe.lock) do
        return copy(probe.rows)
    end
end

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function save_complexity_probe(path::AbstractString, probe::TNMPComplexityProbe)
    rows = complexity_rows(probe)
    mkpath(dirname(path))

    if lowercase(splitext(path)[2]) == ".jls"
        open(path, "w") do io
            serialize(io, rows)
        end
        return path
    end

    columns = (:kind, :key, :n_tensors, :n_open_legs, :max_input_log2, :output_log2, :sc, :tc)
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            println(io, join((csv_escape(getproperty(row, c)) for c in columns), ","))
        end
    end
    return path
end
