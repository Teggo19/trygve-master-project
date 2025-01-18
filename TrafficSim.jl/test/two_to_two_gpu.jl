include("../src/TrafficSim.jl")
using .TrafficSim
using Plots

function test_method(T, N, device_string)
    road_1 = TrafficSim.Road{Float64, Float32}(1, 100., 20.f0, 0.5, N, 1/N)
    road_2 = TrafficSim.Road{Float64, Float32}(2, 100., 20.f0, 0.5, N, 1/N)
    road_3 = TrafficSim.Road{Float64, Float32}(3, 100., 20.f0, 0.5, N, 1/N)
    road_4 = TrafficSim.Road{Float64, Float32}(4, 100., 20.f0, 0.5, N, 1/N)


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

    rho, n_time_steps = TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, device_string, true)

    result, time, allocs, mem = @timed TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, device_string, true)
    return rho, time, n_time_steps
end

T = 4

M = 4

N_vals = [10^i for i in 1:M]

device_string = "gpu"
gpu_times = zeros(length(N_vals))
gpu_n_time_steps = zeros(length(N_vals))

for i in 1:M
    N = N_vals[i]
    rho, time, n_time_steps = test_method(T, N, device_string)
    gpu_times[i] = time
    gpu_n_time_steps[i] = time/n_time_steps
    println("gpu: N = $N, time = $(gpu_times[i]), n_time_steps = $n_time_steps")
end

device_string = "cpu"
cpu_times = zeros(length(N_vals))
cpu_n_time_steps = zeros(length(N_vals))

for i in 1:M
    N = N_vals[i]
    rho, time, n_time_steps = test_method(T, N, device_string)
    cpu_times[i] = time
    cpu_n_time_steps[i] = time/n_time_steps
    println("cpu: N = $N, time = $time, n_time_steps = $n_time_steps")
end

python_vals = [0.06630373001098633, 0.1772904396057129, 1.32804274559021, 24.16429328918457]

p_1 = plot(N_vals, gpu_times, label="GPU", xaxis=:log, yaxis=:log, title="GPU vs CPU total time")
plot!(p_1, N_vals, cpu_times, label="CPU", xaxis=:log, yaxis=:log)
plot!(p_1, N_vals, python_vals, label="Python", xaxis=:log, yaxis=:log)
# add legends to the above plot
p_2 = plot(N_vals, gpu_n_time_steps, label="GPU", xaxis=:log, yaxis=:log, title="GPU vs CPU time per time step")
plot!(p_2, N_vals, cpu_n_time_steps, label="CPU", xaxis=:log, yaxis=:log)
# add legends to the above plot




