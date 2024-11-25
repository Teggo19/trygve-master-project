
using CUDA

using LinearAlgebra

using BenchmarkTools

include("plot_helper.jl")
using .PlotHelper

include("scalar_solvers.jl")
using .ScalarSolvers

include("scalar_test_functions.jl")
using .ScalarTestFunctions

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


function fv_solver(U_0, dx, dt, T)
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
        @cuda threads=numthreads blocks=numblocks lax_friedrichs_kernel!(U, U_next, dx, dt, N)
        
        u[i, :] = Array(U_next)
        
    end
    return u
end

function lax_friedrichs_kernel!(u_0, u_1, dx, dt, N)
    i = (blockIdx().x -1)*blockDim().x + threadIdx().x
    if i > N
        return
    end
    if i == 1
        u_1[i] = 0.5*(u_0[N] + u_0[i+1]) - 0.5*dt/dx*(f(u_0[i+1]) - f(u_0[N]))
    elseif i == N
        u_1[i] = 0.5*(u_0[i-1] + u_0[1]) - 0.5*dt/dx*(f(u_0[1]) - f(u_0[i-1]))
    else
        u_1[i] = 0.5*(u_0[i-1] + u_0[i+1]) - 0.5*dt/dx*(f(u_0[i+1]) - f(u_0[i-1]))
    end
    
    return
end




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
u_0 = [bump(x[1:50]); square(x[51:end])]
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
@btime U_friedrichs = lax_friedrichs(f, u_0, dx, dt, T, true);

t = range(0, T, length=size(U_gpu, 1))

plot_2ds(x, t, [U_gpu, U_friedrichs], ["GPU", "Friedrichs"])