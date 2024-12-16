using CUDA

struct Road{RoadType <: AbstractFloat}
    id::Int32
    length::RoadType
    v_max::RoadType
    sigma::RoadType
    N::Int
    dx::RoadType
end

struct Intersection{Intersectiontype <: AbstractFloat}
    id::Int
    n_incoming::Int
    n_outgoing::Int
    incoming::Array{Road}
    outgoing::Array{Road}
    alpha::CuArray{Intersectiontype}
    P::CuArray{Intersectiontype}
end

struct TrafficProblem
    roads::Array{Road}
    intersections::Array{Intersection}
end
