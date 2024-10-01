using LinearAlgebra
# using Plots
# using GLMakie

include("plot_helper.jl")
using .PlotHelper

include("scalar_solvers.jl")
using .ScalarSolvers


a = 1

x = range(0, 2*pi, 100)
u_0 = sin.(x)
N = length(x)

dx = 0.1
dt = 0.01
T = 20

f(x) = a*x
J(x) = a

U_friedrichs = lax_friedrichs(f, u_0, dx, dt, T)
U_upwind = central_diff(f, u_0, dx, dt, T)
U_wendroff = lax_wendroff(f, J, u_0, dx, dt, T)
U_high_res = high_res_torjei(f, J, u_0, dx, dt, T)

t = range(0, T, length=size(U_friedrichs, 1))
# surface(x, t, U_friedrichs, title="Lax-Friedrichs")
# surface(x, t, U_upwind, title="Central Upwind")
# surface(x, t, U_wendroff, title="Lax-Wendroff")

#plot_2d(x, t, U_friedrichs)
#plot_2d(x, t, U_upwind)
#plot_2d(x, t, U_wendroff)
plot_2ds(x, t, [U_friedrichs, U_upwind, U_wendroff, U_high_res], ["Lax-Friedrichs", "Central diff", "Lax-Wendroff", "High res"])