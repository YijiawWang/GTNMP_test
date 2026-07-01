module TNMPTest

using OMEinsumContractionOrders:
    EinCode, NestedEinsum, SlicedEinsum, TreeSA, ScoreFunction, contraction_complexity, optimize_code
using ITensors:
    ITensor,
    Index,
    Algorithm,
    commoninds,
    dag,
    delta,
    dim,
    eachindval,
    inds,
    norm,
    onehot,
    prime,
    replaceinds
using Base.Threads
using LinearAlgebra: dot
using NamedGraphs: NamedEdge, NamedGraph, add_edge!, dst, edges, neighbors, src, vertices
using Random: AbstractRNG, rand
import ITensors

# `nthreads <= 0` or `nothing` -> use all Julia threads (`Threads.nthreads()`).
message_passing_nthreads(nthreads::Union{Nothing, Integer} = nothing) =
    nthreads === nothing || Int(nthreads) <= 0 ? Threads.nthreads() : Int(nthreads)

# Model-agnostic solver API. Concrete state constructors and neighborhood
# definitions live at the model level under `examples/` (see
# `examples/state_models.jl` and `examples/neighborhoods.jl`).
export TNMPCache,
    TNMPComplexityProbe,
    complexity_rows,
    contraction_sc,
    exact_marginal,
    graph,
    incoming_boundary_edges,
    subdivision_graph,
    marginal_tensors,
    message_tensors,
    prewarm_rank2_tnmp_sequences!,
    save_complexity_probe,
    run_message_passing!,
    siteinds,
    tnmp_marginal

include("tensor_network_state.jl")
include("graph_regions.jl")
include("tnmp_contraction_base.jl")
include("tnmp_complexity_probe.jl")
include("tnmp_contraction.jl")
include("tnmp_cache.jl")
include("tnmp_messages.jl")
include("tnmp_marginal.jl")

end
