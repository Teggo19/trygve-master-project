module ScalarSolvers
    export lax_friedrichs
    export central_diff
    export lax_wendroff
    export high_res_torjei

    function lax_friedrichs(f, u0, dx, dt, T)
        N = length(u0)
        M = ceil(Int, T/dt)
        u = zeros(M+1, N)
        u[1, :] = u0
        for i in 2:M+1
            for j in 1:N
                if j == 1
                    u[i, j] = 0.5*(u[i-1, 2] + u[i-1, N]) - 0.5*dt/dx*(f(u[i-1, 2]) - f(u[i-1, N]))
                elseif j == N
                    u[i, j] = 0.5*(u[i-1, 1] + u[i-1, N-1]) - 0.5*dt/dx*(f(u[i-1, 1]) - f(u[i-1, N-1]))
                else
                    u[i, j] = 0.5*(u[i-1, j+1] + u[i-1, j-1]) - 0.5*dt/dx*(f(u[i-1, j+1]) - f(u[i-1, j-1]))
            
                end
            end
        end
        return u
    end

    function central_diff(f, U_0, dx, dt, T)
        N = length(U_0)
        M = ceil(Int, T/dt)

        U = fill(1.0, (M+1, N))
        U[1, 1:N] = U_0

        for i in 2:M+1
            for j in 1:N
            if j == 1
                U[i, j] = U[i-1, j] + dt/(2*dx)*(f(U[i-1, N]) - f(U[i-1, j+1]))
            elseif j == N
                U[i, j] = U[i-1, j] + dt/(2*dx)*(f(U[i-1, j-1]) - f(U[i-1, 1]))
            else
                U[i, j] = U[i-1, j] + dt/(2*dx)*(f(U[i-1, j-1]) - f(U[i-1, j+1]))
            end
            end
        end
        return U
    end

    function lax_wendroff(f, J, U_0, dx, dt, T)
        N = length(U_0)
        M = ceil(Int, T/dt)

        U = fill(1.0, (M+1, N))
        U[1, 1:N] = U_0
        for i in 2:M+1
            for j in 1:N

                if j==1
                    J_1 = J(0.5*(U[i-1, j] + U[i-1, j+1]))
                    J_2 = J(0.5*(U[i-1, N] + U[i-1, j]))
                    U[i, j] = U[i-1, j] - 0.5*dt/dx*(f(U[i-1, j+1]) - f(U[i-1, N])) + 0.5*(dt/dx)^2*(J_1*(f(U[i-1, j+1])-f(U[i-1, j])) - J_2*(f(U[i-1, j] - f(U[i-1, N]))))
                elseif j==N
                    J_1 = J(0.5*(U[i-1, j] + U[i-1, 1]))
                    J_2 = J(0.5*(U[i-1, j-1] + U[i-1, j]))
                    U[i, j] = U[i-1, j] - 0.5*dt/dx*(f(U[i-1, 1]) - f(U[i-1, j-1])) + 0.5*(dt/dx)^2*(J_1*(f(U[i-1, 1])-f(U[i-1, j])) - J_2*(f(U[i-1, j] - f(U[i-1, j-1]))))
                else
                    J_1 = J(0.5*(U[i-1, j] + U[i-1, j+1]))
                    J_2 = J(0.5*(U[i-1, j-1] + U[i-1, j]))
                    U[i, j] = U[i-1, j] - 0.5*dt/dx*(f(U[i-1, j+1]) - f(U[i-1, j-1])) + 0.5*(dt/dx)^2*(J_1*(f(U[i-1, j+1])-f(U[i-1, j])) - J_2*(f(U[i-1, j] - f(U[i-1, j-1]))))
                end
            end
        end
        return U
    end

    function central_upwind(f, J, U_0, dx, dt, T)
        N = length(U_0)
        M = ceil(Int, T/dt)

        U = fill(1.0, (M+1, N))
        U[1, 1:N] = U_0
        for i in 2:M+1
            for j in 1:N
                if j == 1
                    U[i, j] = U[i-1, j] 
                elseif j == N
                    U[i, j] = U[i-1, j] 
                else
                    U[i, j] = U[i-1, j] 
                end
            end
        end
    end

    function high_res_torjei(f, J, U_0, dx, dt, T)
        N = length(U_0)
        M = ceil(Int, T/dt)

        U = fill(1.0, (M+1, N))
        U[1, 1:N] = U_0
        F_1 = 0.0
        F_2 = 0.0
        for i in 2:M+1
            for j in 1:N
                if j == 1
                    F_1 = (f(U[i-1, N]) + f(U[i-1, j]))/2 - (max(abs(J(U[i-1, N])), abs(J(U[i-1, j]))/2))*(U[i-1, j] - U[i-1, N])
                    F_2 = (f(U[i-1, j]) + f(U[i-1, j+1]))/2 - (max(abs(J(U[i-1, j])), abs(J(U[i-1, j+1]))/2))*(U[i-1, j+1] - U[i-1, j])

                    U[i, j] = U[i-1, j] - dt/dx*(F_2 - F_1)
                elseif j == N
                    F_2 = (f(U[i-1, j]) + f(U[i-1, 1]))/2 - (max(abs(J(U[i-1, j])), abs(J(U[i-1, 1]))/2))*(U[i-1, 1] - U[i-1, j])
                    U[i, j] = U[i-1, j] - dt/dx*(F_2 - F_1)
                else
                    F_2 = (f(U[i-1, j]) + f(U[i-1, j+1]))/2 - (max(abs(J(U[i-1, j])), abs(J(U[i-1, j+1]))/2))*(U[i-1, j+1] - U[i-1, j])
                    U[i, j] = U[i-1, j] - dt/dx*(F_2 - F_1)
                end
                F_1 = F_2
            end
        end
        return U
    end
end
