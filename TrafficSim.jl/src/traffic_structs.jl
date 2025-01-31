using CUDA

struct Road{RealType, VelocityType}
    id::Int
    length::RealType
    v_max::VelocityType
    sigma::RealType
    N::Int
    dx::RealType
    incoming_flux::Function
end

struct Intersection{Intersectiontype}
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
    velocityType
end

TrafficProblem(roads, intersections) = TrafficProblem(roads, intersections, eltype(roads[1].v_max))