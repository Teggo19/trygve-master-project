

struct Road
    id::Int
    length::Float32
    v_max::Float32
    sigma::Float32
    N::Int
    dx::Float32
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
