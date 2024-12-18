using ForwardDiff

function f(x)
    return x[1]^2 + x[2]^2 + x[3]^2
end

x = [1.0, 2.0, 3.0]

ForwardDiff.gradient(f, x)

include("../src/TrafficSim.jl")
using .TrafficSim

N = 10
x = range(0, 1, N)
T = 3
dx = Float32(1.0/N)

x1 = x[1]
x2 = x[end]
U_01 = zeros(N)
for i in 1:N
    U_01[i] = 2*exp(-1/(1-(x[i]*2- (x1+x2)/(x2-x1))^2))
end


function total_flux(v)
    #v1 = ForwardDiff.value(v[1])
    #v2 = ForwardDiff.value(v[2])

    road_1 = TrafficSim.Road{Float32, eltype(v)}(1, 100.f0, v[1], 0.f5, N, dx)
    road_2 = TrafficSim.Road{eltype(v)}(2, 100.f0, v[2], 0.f5, N, dx)
    intersection = TrafficSim.Intersection{Float32}(1, 1, 1, [road_1], [road_2], [1.0], [1.0])
    trafficProblem = TrafficSim.TrafficProblem([road_1, road_2], [intersection])
    U_0 = [U_01, zeros(N)]
    rho  = TrafficSim.traffic_solve(trafficProblem, T, U_0)

    return sum(rho[1]) + sum(rho[2])
end

total_flux([400.f0, 200.f0])

ForwardDiff.gradient(total_flux, [10.f0, 80.f0])