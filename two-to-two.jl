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

function solve_one_to_one(Ul10, Ul20, Ur10, Ur20, road_l1, road_l2, road_r1, road_r2, dt, dx, T, alpha, P, flux_type)
    """
    Must be same dx for all
    """
    N_l1 = length(Ul10)
    N_l2 = length(Ul20)
    N_r1 = length(Ur0)
    N_r2 = length(Ur0)
    M = ceil(Int, T/dt)
    u_l1 = zeros(M+1, N_l1)
    u_l2 = zeros(M+1, N_l2)
    u_r1 = zeros(M+1, N_r1)
    u_r2 = zeros(M+1, N_r2)

    u_l1[1, :] = Ul10
    u_l2[1, :] = Ul20
    u_r1[1, :] = Ur10
    u_r2[1, :] = Ur20

    f_l1 = find_f(road_l1.v_max, road_l1.length)
    f_l2 = find_f(road_l2.v_max, road_l2.length)
    f_r1 = find_f(road_r1.v_max, road_r1.length)
    f_r2 = find_f(road_r2.v_max, road_r2.length)
    J = 0


    flux = flux_type(dx, dt)

    for i in 2:M+1

        F11 = min(alpha[1, 1]*D())
        F = f_lr(u_l[i-1, N_l], u_r[i-1, 1])
        for j in 1:N_l1
            
            if j == 1
                # Can input some incoming traffic stream here
                u_l1[i, j] = u_l1[i-1, j]
            elseif j < N_l1
                u_l1[i, j] = u_l1[i-1, j] - dt/dx*(flux(f_l1, J, u_l1[i-1, j], u_l1[i-1, j+1])- flux(f_l1, J, u_l1[i-1, j-1], u_l1[i-1, j]))
            elseif j == N_l1
                u_l1[i, j] = u_l1[i-1, j] - dt/dx*(Fl1 - flux(f_l1, J, u_l1[i-1, j-1], u_l1[i-1, j]))
            end
        end
        for j in 1:N_l2
            
            if j == 1
                # Can input some incoming traffic stream here
                u_l2[i, j] = u_l2[i-1, j]
            elseif j < N_l2
                u_l2[i, j] = u_l2[i-1, j] - dt/dx*(flux(f_l2, J, u_l2[i-1, j], u_l2[i-1, j+1])- flux(f_l2, J, u_l2[i-1, j-1], u_l2[i-1, j]))
            elseif j == N_l2
                u_l2[i, j] = u_l2[i-1, j] - dt/dx*(Fl2 - flux(f_l2, J, u_l2[i-1, j-1], u_l2[i-1, j]))
            end
        end
        for j in 1:N_r1
            
            if j == N_r1
                # Can input some incoming traffic stream here
                u_r1[i, j] = u_r1[i-1, j]
            elseif j < N_r1
                u_r1[i, j] = u_r1[i-1, j] - dt/dx*(flux(f_r1, J, u_r1[i-1, j], u_r1[i-1, j+1])- flux(f_r1, J, u_r1[i-1, j-1], u_r1[i-1, j]))
            elseif j == 1
                u_r1[i, j] = u_r1[i-1, j] - dt/dx*(flux(f_r1, J, u_r1[i-1, j], u_r1[i-1, j+1]) - Fr1)
            end
        end
        for j in 1:N_r2
            
            if j == N_r2
                # Can input some incoming traffic stream here
                u_r2[i, j] = u_r2[i-1, j]
            elseif j < N_r2
                u_r2[i, j] = u_r2[i-1, j] - dt/dx*(flux(f_r2, J, u_r2[i-1, j], u_r2[i-1, j+1])- flux(f_r2, J, u_r2[i-1, j-1], u_r2[i-1, j]))
            elseif j == 1
                u_r2[i, j] = u_r2[i-1, j] - dt/dx*(flux(f_r2, J, u_r2[i-1, j], u_r2[i-1, j+1]) - Fr2)
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