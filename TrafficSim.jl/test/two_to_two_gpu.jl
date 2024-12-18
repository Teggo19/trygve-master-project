include("../src/TrafficSim.jl")
using .TrafficSim


N = 512

road_1 = TrafficSim.Road{Float32}(1, 100, 20, 0.5, N, 1/N)
road_2 = TrafficSim.Road{Float32}(2, 100, 20, 0.5, N, 1/N)
road_3 = TrafficSim.Road{Float32}(3, 100, 20, 0.5, N, 1/N)
road_4 = TrafficSim.Road{Float32}(4, 100, 20, 0.5, N, 1/N)


intersection = TrafficSim.Intersection{Float32}(1, 2, 2, [road_1, road_2], [road_3, road_4], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5])
#intersection2 = TrafficSim.Intersection(2, 1, 1, [road_2], [road_3])
#intersection3 = TrafficSim.Intersection(3, 1, 1, [road_3], [road_4])

trafficProblem = TrafficSim.TrafficProblem([road_1, road_2, road_3, road_4], [intersection])
#trafficProblem = TrafficSim.TrafficProblem([road_1, road_2, road_3, road_4], [intersection, intersection2, intersection3])

x = range(0, 1, N)

x1 = x[1]
x2 = x[end]
U_01 = zeros(N)
for i in 1:N
    U_01[i] = exp(-1/(1-(x[i]*2- (x1+x2)/(x2-x1))^2))
end


U_0 = [U_01, U_01, zeros(N), zeros(N)]


T = 4
# time it
rho  = TrafficSim.traffic_solve(trafficProblem, T, U_0)

@time rho  = TrafficSim.traffic_solve(trafficProblem, T, U_0)
TrafficSim.plot_traffic(trafficProblem, rho, x)

using CUDA
#CUDA.@profile rho  = TrafficSim.traffic_solve(trafficProblem, T, U_0)

using DelimitedFiles

writedlm("rho_2_to_2.csv", rho, ',')