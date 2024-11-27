include("../src/TrafficSim.jl")
using .TrafficSim


road_length = 300
v_max = 60

function f(rho)
    return v_max/road_length*rho*(1-rho)
end

function J(rho)
    return v_max/road_length*(1-2*rho)
end


function solve_one_to_one_junction(intersection, dx, dt, U_l0, U_r0, T)
    
end


T = 20
N = 100

x = range(0, 1, N)
u_0 = [TrafficSim.bump(x[1:50]); x[51:end].*0]

dx = x[2] - x[1]
dt = dx*0.9*road_length/v_max

U_friedrichs = TrafficSim.lax_friedrichs(f, u_0, dx, dt, T, true)
U_high_res = TrafficSim.high_res_torjei(f, J, u_0, dx, dt, T)

t = range(0, T, length=size(U_friedrichs, 1))

plot_2ds(x, t, [U_friedrichs, U_high_res], ["Friedrichs", "High res"])