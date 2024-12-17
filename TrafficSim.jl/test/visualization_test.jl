include("../src/TrafficSim.jl")
using .TrafficSim


N = 100

road_1 = TrafficSim.Road{Float32}(1, 100, 20, 0.5, N, 1/N)
road_2 = TrafficSim.Road{Float32}(2, 100, 10, 0.5, N, 1/N)
road_3 = TrafficSim.Road{Float32}(3, 100, 20, 0.5, N, 1/N)
road_4 = TrafficSim.Road{Float32}(4, 100, 20, 0.5, N, 1/N)
road_5 = TrafficSim.Road{Float32}(5, 100, 20, 0.5, N, 1/N)
road_6 = TrafficSim.Road{Float32}(6, 100, 20, 0.5, N, 1/N)


intersection = TrafficSim.Intersection{Float32}(1, 1, 1, [road_1], [road_2], [1.0], [1.0])
intersection2 = TrafficSim.Intersection{Float32}(2, 1, 1, [road_2], [road_3], [1.0], [1.0])
intersection3 = TrafficSim.Intersection{Float32}(3, 1, 1, [road_3], [road_4], [1.0], [1.0])

intersection = TrafficSim.Intersection{Float32}(1, 2, 2, [road_1, road_2], [road_3, road_4], [0.8, 0.2, 0.8, 0.2], [0.5, 0.5])
trafficProblem = TrafficSim.TrafficProblem([road_1, road_2], [intersection])
trafficProblem = TrafficSim.TrafficProblem([road_1, road_2, road_3, road_4], [intersection, intersection2, intersection3])

x = range(0, 1, N)

x1 = x[1]
x2 = x[end]
U_01 = zeros(N)
for i in 1:N
    U_01[i] = 2*exp(-1/(1-(x[i]*2- (x1+x2)/(x2-x1))^2))
end


U_0 = [U_01, U_01, U_01, U_01]

T = 10
dt = 0.5
coordinates = [[[0, 0], [100, 0]], [[100, 0], [200, 0]], [[200, 0], [300, 0]], [[300, 0], [400, 0]]]
coordinates = [[[0, 100], [100, 100]], [[100, 200], [100, 100]], [[100, 100], [200, 100]], [[100, 100], [100, 0]]]
TrafficSim.make_traffic_visualization(trafficProblem, coordinates, dt, T, U_0)