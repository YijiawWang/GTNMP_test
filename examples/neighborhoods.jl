# Model-dependent neighborhood definitions for TNMP.
#
# A neighborhood (region) function has the signature `region_fn(gp, g, node)`:
#   * `gp`   : the subdivision graph (site + bond nodes)
#   * `g`    : the original state graph
#   * `node` : a subdivision-graph node, either `(:site, v)` or `(:bond, e)`
# and returns the original-graph vertices forming the neighborhood of `node`.
#
# These live at the model level (not in the `TNMPTest` solver) so that new
# lattices (kagome, 3d, ...) can define their own windows and pass them to
# `TNMPCache(psi, L; region_fn = ...)`. Convenience builders (`grid_region_fn`,
# `first_order_region_fn`) wrap the raw functions into the `region_fn` shape.

using NamedGraphs: NamedGraph, neighbors, src, dst, vertices

# Sub-lattice neighborhood on the (pre-subdivision) grid whose vertices are
# `(x, y)`. A site gets the L*L block centered on it; a bond gets the
# L*(L-1) block centered on the bond (L sites across the bond, L-1 sites along
# it). L must be odd.
function region_vertices(gp::NamedGraph, center_node, L::Integer)
    isodd(L) || throw(ArgumentError("L must be odd"))
    r = (L - 1) ÷ 2
    if first(center_node) === :site
        (cx, cy) = last(center_node)
        xlo, xhi, ylo, yhi = cx - r, cx + r, cy - r, cy + r
    else
        e = last(center_node)
        (x1, y1), (x2, y2) = src(e), dst(e)
        if x1 == x2
            ya, yb = minmax(y1, y2)
            xlo, xhi = x1 - r, x1 + r
            ylo, yhi = ya - (r - 1), yb + (r - 1)
        else
            xa, xb = minmax(x1, x2)
            xlo, xhi = xa - (r - 1), xb + (r - 1)
            ylo, yhi = y1 - r, y1 + r
        end
    end
    region = Any[]
    for node in vertices(gp)
        first(node) === :site || continue
        (x, y) = last(node)
        if xlo <= x <= xhi && ylo <= y <= yhi
            push!(region, last(node))
        end
    end
    return region
end

# Graph-based neighborhood that needs no grid coordinates, so it works on any
# graph (e.g. a tree): a node's own original-graph vertices plus every
# first-order (graph-distance-1) neighbor. For a site node `(:site, v)` this is
# `{v} ∪ neighbors(g, v)`; for a bond node `(:bond, e)` it is the union of the
# first-order neighborhoods of the two endpoints of `e`.
function first_order_region(g::NamedGraph, center_node)
    seeds = first(center_node) === :site ? Any[last(center_node)] :
        Any[src(last(center_node)), dst(last(center_node))]
    region = Any[]
    seen = Set{Any}()
    for s in seeds
        for u in (s, neighbors(g, s)...)
            if !(u in seen)
                push!(seen, u)
                push!(region, u)
            end
        end
    end
    return region
end

# `region_fn` builders in the shape `TNMPCache` expects.
grid_region_fn(L::Integer) = (gp, g, node) -> region_vertices(gp, node, L)
first_order_region_fn() = (gp, g, node) -> first_order_region(g, node)
