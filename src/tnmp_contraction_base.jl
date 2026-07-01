# Contraction-prep helpers shared by TNMP, rank-1 TNMP, and complexity probes.
#
# This file only translates ITensor lists into the OMEinsum/ITensors
# contraction representation. Higher-level contraction planning and caching
# live in the files that include this one.

function to_eincode(tensors::Vector{<:ITensor})
    ixs = [Any[ind for ind in inds(tensor)] for tensor in tensors]
    size_dict = Dict{Any, Int}()
    counts = Dict{Any, Int}()
    for ix in ixs, ind in ix
        size_dict[ind] = dim(ind)
        counts[ind] = get(counts, ind, 0) + 1
    end
    iy = Any[ind for (ind, c) in counts if c == 1]
    return EinCode(ixs, iy), size_dict
end

# Convert an OMEinsumContractionOrders tree (leaves = 1-based tensor indices)
# into the nested-Vector format that ITensors `contract(...; sequence)` expects.
nested_to_sequence(ne::NestedEinsum) =
    ne.tensorindex >= 1 ? ne.tensorindex : Any[nested_to_sequence(c) for c in ne.args]
nested_to_sequence(se::SlicedEinsum) = nested_to_sequence(se.eins)
