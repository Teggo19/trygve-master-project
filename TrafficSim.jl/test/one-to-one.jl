include("traffic_structs.jl")
using .TrafficStructs

include("scalar_test_functions.jl")
using .ScalarTestFunctions

include("plot_helper.jl")
using .PlotHelper

function f(rho, v_max, road_length)
    return v_max/road_length*rho*(1-rho)
end

function find_f(v_max, road_length)
    function f(rho)
        return v_max/road_length*rho*(1-rho)
    end
    return f
end

function J(rho, v_max, road_length)
    return v_max/road_length*(1-2*rho)
end

function D(rho, f, sigma)
    if rho <= sigma
        return f(rho)
    else
        return f(sigma)
    end
end

function S(rho, f, sigma)
    if rho <= sigma
        return f(sigma)
    else
        return f(rho)
    end
end

function solve_one_to_one(Ul0, Ur0, road_l, road_r, dt, dx, T, flux_type)
    """
    Must be same dx for both
    """
    N_l = length(Ul0)
    N_r = length(Ur0)
    M = ceil(Int, T/dt)
    u_l = zeros(M+1, N_l)
    u_r = zeros(M+1, N_r)

    u_l[1, :] = Ul0
    u_r[1, :] = Ur0

    f_l = find_f(road_l.v_max, road_l.length)
    f_r = find_f(road_r.v_max, road_r.length)
    J = 0

    function f_lr(rho_l, rho_r)
        return min(D(rho_l, f_l, road_l.sigma), S(rho_r, f_r, road_r.sigma))
    end

    flux = flux_type(dx, dt)

    for i in 2:M+1
        for j in 1:N_l+N_r
            F = f_lr(u_l[i-1, N_l], u_r[i-1, 1])
            if j == 1
                # Can input some incoming traffic stream here
                u_l[i, j] = u_l[i-1, j]
            elseif j == N_l+N_r
                # Should add something here to let the traffic exit
                u_r[i, j-N_l] = u_r[i, j-N_l]
            elseif j < N_l
                u_l[i, j] = u_l[i-1, j] - dt/dx*(flux(f_l, J, u_l[i-1, j], u_l[i-1, j+1])- flux(f_l, J, u_l[i-1, j-1], u_l[i-1, j]))
            elseif j == N_l
                u_l[i, j] = u_l[i-1, j] - dt/dx*(F - flux(f_l, J, u_l[i-1, j-1], u_l[i-1, j]))
            elseif j == N_l+1
                u_r[i, j-N_l] = u_r[i-1, j-N_l] - dt/dx*(flux(f_r, J, u_r[i-1, j-N_l], u_r[i-1, j+1-N_l]) - F)
            else
                u_r[i, j-N_l] = u_r[i-1, j-N_l] - dt/dx*(flux(f_r, J, u_r[i-1, j-N_l], u_r[i-1, j+1-N_l])- flux(f_r, J, u_r[i-1, j-1-N_l], u_r[i-1, j-N_l]))
            end
        end
    end

    return u_l, u_r

end


function lax_friedrichs_flux(dx, dt)
    function flux(f, J, u1, u2)
        return 0.5*dx/dt*(u1-u2) + 0.5*(f(u1) + f(u2))
    end
    return flux
end

road_l = Road(1, 100, 10, 0.5)
road_r = Road(1, 100, 1, 0.5)

T = 20

x = range(0, 1, 100)

ul0 = bump(x)
ur0 = zeros(100)

dx = x[2] - x[1]
dt = dx*0.9*road_l.length/road_l.v_max

u_l, u_r = solve_one_to_one(ul0, ur0, road_l, road_r, dt, dx, T, lax_friedrichs_flux)

t = range(0, T, length=size(u_l, 1))

plot_2ds(x, t, [u_l, u_r], ["Left", "Right"])