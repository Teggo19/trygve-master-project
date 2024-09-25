# A simple central upwind method for a linear transport equation
# Using periodic boundary conditions
using LinearAlgebra


function central_upwind_lin_transport(a, U_0, dx, dt, T)
    N = length(U_0)
    alpha = a*dt/(2*dx)
    A = Tridiagonal(repeat([alpha], N-1), repeat([1.0], N), repeat([-alpha], N-1))
    A = Matrix(A)
    A[1, N] = alpha
    A[N, 1] = -alpha
    A_inv = inv(A)
    t = dt

    M = ceil(Int, T/dt)
    
    U = fill(1.0, (M+1, N))
    
    U[1, 1:N] = U_0

    for i in 2:M+1
        println("Size of U[i, 1:N]:", size(U[i, 1:N]))
        println("Size of RHS: ", size(transpose(A_inv*U[i-1, 1:N])))
        U[i, 1:N] = transpose(A_inv*U[i-1, 1:N])
    end
    return U
end


a = 1

x = range(0, 2*pi, 100)
u_0 = sin.(x)
N = length(x)

dx = 0.1
dt = 0.01
T = 20

U = central_upwind_lin_transport(a, u_0, dx, dt, T)
U
plot(x, U[end, 1:N])


function central_upwind_burger(U_0, dx, dt, T)
    N = length(U_0)
    M = ceil(Int, T/dt)

    U = fill(1.0, (M+1, N))
    U[1, 1:N] = U_0

    for i in 2:M
        for j in 1:N
            if j == 1
                U[i, j] = U[i-1, j] - dt/(2*dx)*(U[i-1, N]^2 - U[i-1, j+1]^2)
            elseif j == N
                U[i, j] = U[i-1, j] - dt/(2*dx)*(U[i-1, j-1]^2 - U[i-1, 1]^2)
            else
                U[i, j] = U[i-1, j] - dt/(2*dx)*(U[i-1, j-1]^2 - U[i-1, j+1]^2)
            end
        end
    end
    return U
end

V = central_upwind_burger(u_0, dx, dt, T)
plot(x, V[100, 1:N])

function central_upwind(f, U_0, dx, dt, T)
    N = length(U_0)
    M = ceil(Int, T/dt)

    U = fill(1.0, (M+1, N))
    U[1, 1:N] = U_0

    for i in 2:M
        for j in 1:N
        if j == 1
            U[i, j] = U[i-1, j] - dt/(2*dx)*(f(U[i-1, N]) - f(U[i-1, j+1]))
        elseif j == N
            U[i, j] = U[i-1, j] - dt/(2*dx)*(f(U[i-1, j-1]) - f(U[i-1, 1]))
        else
            U[i, j] = U[i-1, j] - dt/(2*dx)*(f(U[i-1, j-1]) - f(U[i-1, j+1]))
        end
        end
    end
    return U
end

f(x) = a*x

W = central_upwind(f, u_0, dx, dt, T)

plot(x, W[2000, 1:N])