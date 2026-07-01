# Contraction planning and exact-marginal helpers.
#
# The exact contractions here are used by TNMP, the rank-1 TNMP variant, and
# the real-double-layer benchmark code.

function contraction_sequence(tensors::Vector{<:ITensor})
    return contraction_plan(tensors).sequence
end

# Space complexity (sc) of contracting `tensors`: log2 of the number of
# elements in the largest *intermediate* tensor along a TreeSA-optimised order.
# The TreeSA settings match `docs/figures/contraction_sc.jl` so the numbers can
# be compared directly against `docs/figures/contraction_sc_results.md`.
function contraction_sc(
        tensors::Vector{<:ITensor};
        ntrials::Integer = 20,
        niters::Integer = 60,
        βs = 1.0:1.0:18.0,
    )
    length(tensors) <= 1 && return 0.0
    code, size_dict = to_eincode(tensors)
    optcode = optimize_code(code, size_dict, TreeSA(; ntrials = ntrials, niters = niters, βs = βs))
    return contraction_complexity(optcode, size_dict).sc
end

function contract_all(tensors::Vector{<:ITensor}; sequence = nothing)
    isempty(tensors) && return ITensor(1.0)
    length(tensors) == 1 && return only(tensors)
    ITensors.disable_warn_order()
    seq = something(sequence, contraction_sequence(tensors))
    return ITensors.contract(tensors; sequence = seq)
end

function scalar_weight(tensors::Vector{<:ITensor}; sequence = nothing)
    z = contract_all(tensors; sequence = sequence)[]
    w = real(z)
    if w < 0 && abs(w) < 1e-12
        return 0.0
    end
    return w < 0 ? abs(w) : w
end

function exact_marginal(psi::TensorNetworkState, target)
    verts = collect(vertices(graph(psi)))
    d = dim(only(siteinds(psi, target)))
    weights = [scalar_weight(marginal_factors(psi, verts, target, state)) for state in 1:d]
    return normalize_weights(weights)
end

# Compute TreeSA outside the lock so independent keys can optimize in parallel.
function _ensure_contraction_sequence!(
        sequences::Dict{Any, Any},
        seq_lock::ReentrantLock,
        key,
        tensors::Vector{<:ITensor},
        ;
        complexity_probe = nothing,
    )
    if haskey(sequences, key)
        return sequences[key]
    end
    plan = contraction_plan(tensors; collect_complexity = complexity_probe !== nothing)
    lock(seq_lock) do
        if haskey(sequences, key)
            return sequences[key]
        end
        sequences[key] = plan.sequence
        record_complexity!(complexity_probe, key, tensors, plan.complexity)
        return plan.sequence
    end
end

function prewarm_contraction_sequences!(
        sequences::Dict{Any, Any},
        seq_lock::ReentrantLock,
        specs::Vector{Tuple{Any, Vector{ITensor}}},
        ;
        nthreads::Union{Nothing, Integer} = nothing,
        progress_label::AbstractString = "",
        complexity_probe = nothing,
    )
    isempty(specs) && return 0
    nt = message_passing_nthreads(nthreads)
    n = length(specs)
    if nt <= 1
        for (key, tensors) in specs
            _ensure_contraction_sequence!(sequences, seq_lock, key, tensors; complexity_probe)
        end
    else
        Threads.@threads for i in 1:n
            key, tensors = specs[i]
            _ensure_contraction_sequence!(sequences, seq_lock, key, tensors; complexity_probe)
        end
    end
    if !isempty(progress_label)
        println("[$progress_label] pre-warmed $n contraction sequence(s)")
        flush(stdout)
    end
    return n
end
