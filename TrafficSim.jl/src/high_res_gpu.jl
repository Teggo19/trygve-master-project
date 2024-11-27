
using CUDA

using LinearAlgebra

using BenchmarkTools

# use the TrafficSim.jl module
include("../src/TrafficSim.jl")
using .TrafficSim

function step_solver(u, u_next, f, J, dx, dt, flux)
    N = length(u)
    for j in 1:N
        if j == 1
            u_next[j] = u[j] - dt/dx*(flux(f, J, u[j], u[j+1]) - flux(f, J, u[N], u[j]))
        elseif j == N
            u_next[j] = u[j] - dt/dx*(flux(f, J, u[j], u[1]) - flux(f, J, u[j-1], u[j]))
        else
            u_next[j] = u[j] - dt/dx*(flux(f, J, u[j], u[j+1]) - flux(f, J, u[j-1], u[j]))
        end
    end
end


function high_res_torjei_gpu(U_0, dx, dt, T)
    M = ceil(Int, T/dt)
    N = length(U_0)
    
    numthreads = min(N, 512)
    numblocks = ceil(Int, N/numthreads)
  
    u = fill(1.0, (M+1, N))
    u[1, :] = U_0
    U_next = CuArray(U_0)
    U = similar(U_next)
    for i in 2:M+1
        U, U_next = U_next, U
        @cuda threads=numthreads blocks=numblocks high_res_kernel!(U, U_next, dx, dt, N)
        
        #u[i, :] = Array(U_next)
        
    end
    return Array(U_next)
end

function F(u_1, u_2)
    return 0.5*(f(u_1) + f(u_2)) - 0.5*(max(abs(J(u_1)), abs(J(u_2)))*(u_2 - u_1))
end

function high_res_kernel!(u_0, u_1, dx, dt, N)
    i = (blockIdx().x -1)*blockDim().x + threadIdx().x
    if i > N
        return
    end
    if i == 1
        u_1[i] = u_0[i] - 0.5*dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[N], u_0[i]) 
                + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[N], u_0[i])), u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+2]) - F(u_0[i], u_0[i+1]))) 
                - F(u_0[N] - dt/dx*(F(u_0[N], u_0[i+1]) - F(u_0[N-1], u_0[N])) , u_0[i] - dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[N], u_0[i]))))
    elseif i == 2
        u_1[i] = u_0[i] - 0.5*dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i]) 
                + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i])), u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[N]) - F(u_0[i], u_0[i+1]))) 
                - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[i+1]) - F(u_0[N], u_0[i-1])) , u_0[i] - dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i]))))
    elseif i == N
        u_1[i] = u_0[i] - 0.5*dt/dx*(F(u_0[i], u_0[1]) - F(u_0[i-1], u_0[i]) 
                + F(u_0[i] - dt/dx*(F(u_0[i], u_0[1]) - F(u_0[i-1], u_0[i])), u_0[1] - dt/dx*(F(u_0[1], u_0[2]) - F(u_0[i], u_0[1]))) 
                - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[1]) - F(u_0[i-2], u_0[i-1])) , u_0[i] - dt/dx*(F(u_0[i], u_0[1]) - F(u_0[i-1], u_0[i]))))
    elseif i == N-1
        u_1[i] = u_0[i] - 0.5*dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i]) 
                + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i])), u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[1]) - F(u_0[i], u_0[i+1]))) 
                - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[i+1]) - F(u_0[i-2], u_0[i-1])) , u_0[i] - dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i]))))
    else
        u_1[i] = u_0[i] - 0.5*dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i]) 
                + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i])), u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+2]) - F(u_0[i], u_0[i+1]))) 
                - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[i+1]) - F(u_0[i-2], u_0[i-1])) , u_0[i] - dt/dx*(F(u_0[i], u_0[i+1]) - F(u_0[i-1], u_0[i]))))
    end
    
    return
end
"""
function L!(p, l)
    for i in 1:N
        if i == 1
            l[i] = -1/dx*(0.5*(p[i+1] - p[N]) - 0.5*(max(abs(J(p[i])), abs(J(p[i+1])))*(p[i+1] - p[i])) + 
               + 0.5*(max(abs(J(p[i])), abs(J(p[N])))*(p[i] - p[N])))
        elseif i == N
            l[i] = -1/dx*(0.5*(p[1] - p[i-1]) - 0.5*(max(abs(J(p[i])), abs(J(p[1])))*(p[1] - p[i])) + 
               + 0.5*(max(abs(J(p[i])), abs(J(p[i-1])))*(p[i] - p[i-1])))
        else
            l[i] = -1/dx*(0.5*(p[i+1] - p[i-1]) - 0.5*(max(abs(J(p[i])), abs(J(p[i+1])))*(p[i+1] - p[i])) + 
               + 0.5*(max(abs(J(p[i])), abs(J(p[i-1])))*(p[i] - p[i-1])))

        end
    end
    return 
end
"""




const a::Float32 = 1.0

function test_func(x)
    if x <= 0.1
        return 1
    else
        return 0
    end
end
N = 1000
x = range(0, 1, N)
u_0 = [TrafficSim.bump(x[1:50]); TrafficSim.square(x[51:end])]
# u_0 = test_func.(x)
N = length(x)

dx = x[2] - x[1]
dt = dx*0.9/a
T = 3

#f(x) = a*x
#J(x) = a
function f(x)
    res::Float32 = a*x
    return res
end

function J(x)
    res::Float32 = a
    return res
end



CUDA.@profile U_gpu = fv_solver(u_0, dx, dt, T)

@btime U_gpu = fv_solver(u_0, dx, dt, T);

@btime U_cpu = TrafficSim.high_res_torjei(f, J, u_0, dx, dt, T)[331, :];

using Test
@test isapprox(U_gpu, U_cpu, atol=1e-3)

#@btime U_gpu = fv_solver(u_0, dx, dt, T);
#@btime U_friedrichs = lax_friedrichs(f, u_0, dx, dt, T, true);

t = range(0, T, length=size(U_gpu, 1))

plot_2ds(x, t, [U_gpu, U_friedrichs], ["GPU", "Friedrichs"])