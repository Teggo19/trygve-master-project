struct Road{RealType <: AbstractFloat}
    id::Int
    length::RealType
    v_max::RealType
    sigma::RealType
    N::Int
    dx::RealType
end

struct Intersection
    id::Int
    n_incoming::Int
    n_outgoing::Int
    incoming::Array{Road}
    outgoing::Array{Road}
end

struct TrafficProblem
    roads::Array{Road}
    intersections::Array{Intersection}
end
