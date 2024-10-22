module ScalarSolvers
    export lax_friedrichs
    export central_diff
    export lax_wendroff
    export upwind

    function finite_volume_solver(f, J, U_0, dx, dt, T, flux, periodic_boundary)
        N = length(U_0)
        M = ceil(Int, T/dt)
        u = zeros(M+1, N)
        u[1, :] = U_0
        for i in 2:M+1
            for j in 1:N
                if periodic_boundary
                    if j == 1
                        u[i, j] = u[i-1, j] - dt/dx*(flux(f, J, u[i-1, j], u[i-1, j+1]) - flux(f, J, u[i-1, N], u[i-1, j]))
                    elseif j == N
                        u[i, j] = u[i-1, j] - dt/dx*(flux(f, J, u[i-1, j], u[i-1, 1]) - flux(f, J, u[i-1, j-1], u[i-1, j]))
                    else
                        u[i, j] = u[i-1, j] - dt/dx*(flux(f, J, u[i-1, j], u[i-1, j+1]) - flux(f, J, u[i-1, j-1], u[i-1, j]))
                    end
                else
                    if j == 1 || j == N
                        u[i, j] = u[i-1, j]
                    else
                        u[i, j] = u[i-1, j] - dt/dx*(flux(f, J, u[i-1, j], u[i-1, j+1]) - flux(f, J, u[i-1, j-1], u[i-1, j]))
                    end
                end
            end
        end
        return u
    end

    function lax_friedrichs(f, U_0, dx, dt, T, periodic_boundary)
        function flux(f, J, u1, u2)
            return 0.5*dx/dt*(u1-u2) + 0.5*(f(u1) + f(u2))
        end

        return finite_volume_solver(f, 0, U_0, dx, dt, T, flux, periodic_boundary)
    end

    function upwind(f, J, U_0, dx, dt, T, periodic_boundary)
        function flux(f, J, u1, u2)
            if J(u1) >= 0
                return f(u1)
            else
                return f(u2)
            end
        end
        return finite_volume_solver(f, J, U_0, dx, dt, T, flux, periodic_boundary)
    end

    function lax_wendroff(f, U_0, dx, dt, T, periodic_boundary)
        function flux(f, J, u1, u2)
            return f((0.5*(u1 + u2)) - 0.5*dt/dx*(f(u2) - f(u1)))
        end
        return finite_volume_solver(f, 0, U_0, dx, dt, T, flux, periodic_boundary)
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
 
end
