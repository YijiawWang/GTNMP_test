# Core, model-agnostic graph helpers for TNMP.
#
# `subdivision_graph` and `incoming_boundary_edges` only depend on the abstract
# graph structure, so they live in the solver. The model-dependent *choice of
# neighborhood* (the L*L grid window, the first-order graph neighborhood, the
# kagome / 3d windows, ...) lives with the models in `examples/neighborhoods.jl`
# and is supplied to `TNMPCache` via the `region_fn` keyword.

function subdivision_graph(g::NamedGraph)
    canonical = Dict{Tuple{Any, Any}, NamedEdge}()
    for e in edges(g)
        canonical[(src(e), dst(e))] = e
        canonical[(dst(e), src(e))] = e
    end

    nodes = Any[]
    for v in vertices(g)
        push!(nodes, (:site, v))
    end
    for e in edges(g)
        push!(nodes, (:bond, e))
    end

    gp = NamedGraph(nodes)
    for e in edges(g)
        add_edge!(gp, (:site, src(e)), (:bond, e))
        add_edge!(gp, (:site, dst(e)), (:bond, e))
    end
    return gp, canonical
end

function incoming_boundary_edges(g::NamedGraph, region)
    region_set = Set(region)
    bedges = NamedEdge[]
    for e in edges(g)
        u, v = src(e), dst(e)
        u_inside = u in region_set
        v_inside = v in region_set
        u_inside == v_inside && continue
        push!(bedges, u_inside ? NamedEdge(v => u) : NamedEdge(u => v))
    end
    return bedges
end
