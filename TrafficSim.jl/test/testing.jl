using LinearAlgebra
# using Plots
# using GLMakie

include("../src/TrafficSim.jl")
using .TrafficSim

a = 1

function test_func(x)
    if x <= 0.1
        return 1
    else
        return 0
    end
end

x = range(0, 1, 100)
u_0 = [TrafficSim.bump(x[1:50]); TrafficSim.square(x[51:end])]
# u_0 = test_func.(x)
N = length(x)

dx = x[2] - x[1]
dt = dx*0.9/a
T = 3

f(x) = a*x^2/2
J(x) = a*x

U_friedrichs = TrafficSim.lax_friedrichs(f, u_0, dx, dt, T, true)
# U_upwind = central_diff(f, u_0, dx, dt, T)
U_wendroff = TrafficSim.lax_wendroff(f, u_0, dx, dt, T, true)
U_upwind = TrafficSim.upwind(f, J, u_0, dx, dt, T, true)
U_high_res = TrafficSim.high_res_torjei(f, J, u_0, dx, dt, T)

t = range(0, T, length=size(U_friedrichs, 1))
# surface(x, t, U_friedrichs, title="Lax-Friedrichs")
# surface(x, t, U_upwind, title="Central Upwind")
# surface(x, t, U_wendroff, title="Lax-Wendroff")

#plot_2d(x, t, U_friedrichs)
#plot_2d(x, t, U_upwind)
#plot_2d(x, t, U_wendroff)
TrafficSim.plot_2ds(x, t, [U_friedrichs, U_wendroff, U_upwind, U_high_res], ["Lax-Friedrichs", "Lax-Wendroff", "Upwind", "High res"])