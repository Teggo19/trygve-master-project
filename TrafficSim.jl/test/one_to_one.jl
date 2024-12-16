include("../src/TrafficSim.jl")
using .TrafficSim

# A bug happens when N=100, not sure why, also not support for N>512 at the moment
N = 100

road_1 = TrafficSim.Road{Float32}(1, 100, 20, 0.5, N, 1/N)
road_2 = TrafficSim.Road{Float32}(2, 100, 20, 0.5, N, 1/N)

intersection = TrafficSim.Intersection(1, 1, 1, [road_1], [road_2])

trafficProblem = TrafficSim.TrafficProblem([road_1, road_2], [intersection])

x = range(0, 1, N)

x1 = x[1]
x2 = x[end]
U_01 = zeros(N)
for i in 1:N
    U_01[i] = exp(-1/(1-(x[i]*2- (x1+x2)/(x2-x1))^2))
end


U_0 = [U_01, zeros(N)]


T = 3

rho  = TrafficSim.traffic_solve(trafficProblem, T, U_0)


using GLMakie

fig = Figure()

ax = Axis(fig[1, 1])
colors = [:blue, :red, :green, :yellow, :purple, :orange, :black, :cyan]
for i in 1:length(trafficProblem.roads)
    j = (i-1)*N + 1
    lines!(ax, x, rho[j:j+N-1], color = colors[i], label = "Road $i")
end

axislegend(ax)
fig