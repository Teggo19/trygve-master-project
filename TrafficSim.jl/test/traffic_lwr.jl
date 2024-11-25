include("plot_helper.jl")
using .PlotHelper

include("scalar_solvers.jl")
using .ScalarSolvers

include("scalar_test_functions.jl")
using .ScalarTestFunctions


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
u_0 = [bump(x[1:50]); x[51:end].*0]

dx = x[2] - x[1]
dt = dx*0.9*road_length/v_max

U_friedrichs = lax_friedrichs(f, u_0, dx, dt, T, true)

t = range(0, T, length=size(U_friedrichs, 1))

plot_2d(x, t, U_friedrichs)