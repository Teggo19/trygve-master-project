include("../src/TrafficSim.jl")
using .TrafficSim

# A bug happens when N=100, not sure why, also not support for N>512 at the moment
N = 100

road_1 = TrafficSim.Road{Float32}(1, 100, 20, 0.5, N, 1/N)
road_2 = TrafficSim.Road{Float32}(2, 100, 30, 0.5, N, 1/N)
road_3 = TrafficSim.Road{Float32}(3, 100, 20, 0.5, N, 1/N)
road_4 = TrafficSim.Road{Float32}(4, 100, 20, 0.5, N, 1/N)


intersection = TrafficSim.Intersection(1, 1, 1, [road_1], [road_2])
intersection2 = TrafficSim.Intersection(2, 1, 1, [road_2], [road_3])
intersection3 = TrafficSim.Intersection(3, 1, 1, [road_3], [road_4])

trafficProblem = TrafficSim.TrafficProblem([road_1, road_2, road_3, road_4], [intersection, intersection2, intersection3])

x = range(0, 1, N)

x1 = x[1]
x2 = x[end]
U_01 = zeros(N)
for i in 1:N
    U_01[i] = exp(-1/(1-(x[i]*2- (x1+x2)/(x2-x1))^2))
end


U_0 = [U_01, zeros(N), U_01, U_01]


T = 4
# time it
rho  = TrafficSim.traffic_solve(trafficProblem, T, U_0)

TrafficSim.plot_traffic(trafficProblem, rho, x)

using CUDA
CUDA.@profile rho  = TrafficSim.traffic_solve(trafficProblem, T, U_0)

using DelimitedFiles

# writedlm("rho.csv", rho, ',')

