

struct Road
    id::Int
    length::Int
    v_max::Int
    sigma::Float16
end

struct Intersection
    id::Int
    n_incoming::Int
    n_outgoing::Int
    incoming::Array{Int}
    outgoing::Array{Int}
end
