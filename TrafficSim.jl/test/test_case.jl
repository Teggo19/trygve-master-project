include("../src/TrafficSim.jl")
using .TrafficSim
using Plots
using Interpolations

M = 3
N = 100

trafficProblem, _ = TrafficSim.make_test_trafficProblem(M, N)
coordinates = TrafficSim.test_case_coordinates(M)


dt = 1
T = 200

function def_u0(x)
    return 0.4f0
end
x = range(0, 1, N)

u_0 = [def_u0(x) for x in x]

U_0 = [u_0 for i in 1:M*(M+1)*2]

TrafficSim.make_traffic_visualization(trafficProblem, coordinates, dt, T, U_0)
rho_1 = TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, "gpu", false)
rho_2 = TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, "cpu", false)
@time rho = TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, "gpu", false);
rho_1 == rho_2
# save rho to a file
open("test_case_M=3_N=1000.txt", "w") do io
    for i in 1:length(rho)
        println(io, rho[i])
    end
end

#read rho from a file
rho_read = [[0.0f0 for i in 1:N] for j in 1:M*(M+1)*2]
open("test_case_M=3_N=1000.txt") do io
    i = 1
    for line in eachline(io)
        rho_read[i] = eval(Meta.parse(line))
        i += 1
    end
end


interpolations = [linear_interpolation(x , rho_read[i]) for i in 1:M*(M+1)*2]

N_new = 1000
x_new = range(0, 1, N_new)

rho_new = [interpolations[i](x_new) for i in 1:M*(M+1)*2]
# make a dot plot
plot(x_new, rho_new[1], label = "interpolated", seriestype=:scatter)
plot!(x, rho_read[1], label= "original", seriestype=:scatter)

function test_case_M3(N, device_string)
    M = 3
    trafficProblem, U_0 = TrafficSim.make_test_trafficProblem(3, N)
    T = 200/(N)

    
    
    rho = TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, device_string)
    time, n_time_steps = TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, device_string, true)

    println("device_string = $device_string, N = $N, time = $time, n_time_steps = $n_time_steps")
    return time/n_time_steps 

end

function test_case(N, M, device_string)
    trafficProblem, U_0 = TrafficSim.make_test_trafficProblem(M, N)
    T = 200/(N)

    
    rho = TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, device_string)
    time, n_time_steps = TrafficSim.traffic_solve_ka(trafficProblem, T, U_0, device_string, true)

    println("device_string = $device_string, N = $N, M = $M, time = $time, n_time_steps = $n_time_steps")
    return time/n_time_steps 

end

N_vals = [10^i for i in 1:7]

gpu_times_1 = [test_case_M3(N, "gpu") for N in N_vals]

cpu_times_1 = [test_case_M3(N, "cpu") for N in N_vals]

python_times_1 = [0.0146576, 0.011741, 0.0118774, 0.01394056, 0.03967956, 0.19785327, 7.4786]

plot(N_vals, gpu_times_1, label = "GPU", xaxis=:log, yaxis=:log, xlabel="Number of grids per road", ylabel="Time per step (s)", color=:blue)# title = "3x3 grid with varying number of cells per road")
# add scatter plot for eveay value
scatter!(N_vals, gpu_times_1, primary=false, xaxis=:log, yaxis=:log, color=:blue)
plot!(N_vals, cpu_times_1, label = "CPU", xaxis=:log, yaxis=:log, color=:green)
scatter!(N_vals, cpu_times_1, primary=false, xaxis=:log, yaxis=:log, color=:green)
plot!(N_vals, python_times_1, label = "Python", xaxis=:log, yaxis=:log, color=:red)
scatter!(N_vals, python_times_1, primary=false, xaxis=:log, yaxis=:log, color=:red)
plot!(legend=:topleft)
# save plot as a png file
savefig("cells_per_road_2.png")

N = 10000
M_vals = 4:2:40
gpu_times_2 = [test_case(N, M, "gpu") for M in M_vals]

cpu_times_2 = [test_case(N, M, "cpu") for M in M_vals]
python_times_2 = [0.021393174216860815, 0.039512645630609425, 0.06755207833789643, 0.10304125150044759, 0.14736841973804293, 0.2002872966584705, 0.2564094974881127, 0.32325189454214914, 0.39804472242082867, 0.47872541064307805, 0.5692330428532192, 0.671597923551287, 0.7764394396827334, 0.8886502129690987, 1.013495922088623, 1.1416500977107458, 1.2827952929905482, 1.430901220866612, 1.5840960684276761]
plot(M_vals, gpu_times_2, label = "GPU", xlabel="Size of the traffic grid", ylabel="Time per step (s)", color=:blue)# title = "10000 cells per road with varying grid size", yaxis=:log)
scatter!(M_vals, gpu_times_2, primary=false, color=:blue, yaxis=:log)
plot!(M_vals, cpu_times_2, label = "CPU", color=:green, yaxis=:log)
scatter!(M_vals, cpu_times_2, primary=false, color=:green, yaxis=:log)
plot!(M_vals, python_times_2, label = "Python", color=:red, yaxis=:log)
scatter!(M_vals, python_times_2, primary=false, color=:red, yaxis=:log)
plot!(legend=:topleft)

savefig("grid_size_2.png")

using Printf
function print_latex_table(N_vals, gpu_times, cpu_times, python_times)
    println("\\begin{tabular}{|c|c|c|c|}")
    println("\\hline")
    println("N & GPU & CPU & Python \\\\")
    println("\\hline")
    for i in 1:length(N_vals)
        # print the values in scientific notation with 4 decimal places
        println("\$10^$i\$ & $(@sprintf("%.5f", gpu_times[i])) & $(@sprintf("%.5f", cpu_times[i])) & $(@sprintf("%.5f", python_times[i])) \\\\")
    end
    println("\\hline")
    println("\\end{tabular}")
end

print_latex_table(N_vals, gpu_times_1, cpu_times_1, python_times_1)