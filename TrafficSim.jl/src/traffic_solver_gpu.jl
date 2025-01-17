using CUDA

include("traffic_structs.jl")



@inline function f(gamma, rho)
    return gamma * rho * (1 - rho)
end

@inline function J(gamma, rho)
    return gamma * (1 - 2*rho)
end

@inline function F(u_1, u_2, gamma)
    return 0.5*(f(gamma, u_1) + f(gamma, u_2)) - 0.5*(max(abs(J(gamma, u_1)), abs(J(gamma, u_2)))*(u_2 - u_1))
end


function find_flux_prime!(rho, gammas, max_dt_arr, N_max, dxs)
    i = threadIdx().x + (blockIdx().x - 1)*N_max
    gamma = gammas[ceil(Int, i/N_max)]
    dx = dxs[ceil(Int, i/N_max)]
    max_dt_arr[i] = dx/abs(J(gamma, rho[i]))
    # J(gamma, rho[i])
    return 
end


function traffic_solve(trafficProblem::TrafficProblem, T, U_0)
    
    N_vals_cpu = [road.N for road in trafficProblem.roads]
    N_max = maximum(N_vals_cpu)
    if N_max > 512
        throw(ArgumentError("N_max > 512 not supported"))
    end
    n_threads = N_max
    n_blocks_tot = length(trafficProblem.roads)

    gammas_cpu = [road.v_max/road.length for road in trafficProblem.roads]
    dxs_cpu = [road.dx for road in trafficProblem.roads]

    # make 2D array to store the density of the roads
    #rho = zeros(Float32, length(trafficProblem.roads)*N_max)
    # rho = CUDA.fill(1.0f0, length(trafficProblem.roads)*N_max)
    rho = CuArray(ones(trafficProblem.velocityType, length(trafficProblem.roads)*N_max))

    for i in 1:length(trafficProblem.roads)
        j = (i-1)*N_max + 1
        rho[j:j+N_vals_cpu[i]-1] = U_0[i]
    end

    rho_1 = similar(rho)

    N_vals = CuArray(N_vals_cpu)
    gammas = CuArray(gammas_cpu)
    dxs = CuArray(dxs_cpu)
    
    t = 0
    while t < T
        dt = T-t
        max_dt_arr = CuArray(ones(trafficProblem.velocityType, length(trafficProblem.roads)*N_max))
        @cuda threads=n_threads blocks=n_blocks_tot find_flux_prime!(rho, gammas, max_dt_arr, N_max, dxs)
        new_dt = CUDA.minimum(max_dt_arr)
        dt = min(dt, new_dt)
        t += dt
        
        #solve the roads
        @cuda threads=n_threads blocks=n_blocks_tot road_solver!(rho, rho_1, N_vals, N_max, gammas, dt, dxs)

        # solve the intersections
        for intersection in trafficProblem.intersections
            # solve the intersection
            intersection_solver!(rho, rho_1, intersection, dt, gammas, N_max, N_vals)
        end
        rho, rho_1 = rho_1, rho
    end
    rho_cpu = collect(rho)
    final_rho = [rho_cpu[(i-1)*N_max+1:(i-1)*N_max+N_vals_cpu[i]] for i in 1:length(trafficProblem.roads)]

    return final_rho
end

function road_solver!(u_0, u_1, N_vals, N_max, gammas, dt, dxs)
    # get the index of the block
    i = threadIdx().x + (blockIdx().x - 1)*N_max
    
    if i > length(u_0) || i < 1
        return
    end
    
    road_index = ceil(Int, i/N_max)

    j = (i-1) % N_max + 1

    gamma = gammas[road_index]
    dx = dxs[road_index]

    update = 0.0f0

    if j > N_vals[road_index]
        u_1[i] = u_0[i]
        return 

    elseif j == 2
        u_1[i] = u_0[i] - 0.5f0 * dt / dx * (F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)
        + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)), 
        u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+2], gamma) - F(u_0[i], u_0[i+1], gamma)), gamma)

        - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[i], gamma) - F(u_0[i-1], u_0[i-1], gamma)), 
        u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)), gamma))
    
    elseif j == 1
        # Should do a check her to see if there is an intersection
        u_1[i] = u_0[i] - 0.5f0 * dt / dx * (F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)
                    + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)), 
                    u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+2], gamma) - F(u_0[i], u_0[i+1], gamma)), gamma)

                    - F(u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)), 
                    u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)), gamma))
    
    elseif j == N_vals[road_index] - 1
        u_1[i] = u_0[i] -0.5f0 * dt / dx * (F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)
                + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)), 
                u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+1], gamma) - F(u_0[i], u_0[i+1], gamma)), gamma)

                - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[i], gamma) - F(u_0[i-2], u_0[i-1], gamma)), 
                u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)), gamma))
    
    elseif j == N_vals[road_index]
        # Should do a check her to see if there is an intersection
        u_1[i] = u_0[i] -0.5f0 * dt / dx * (F(u_0[i], u_0[i], gamma) - F(u_0[i-1], u_0[i], gamma)
                + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i-1], u_0[i], gamma)), 
                u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)), gamma)

                - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[i], gamma) - F(u_0[i-2], u_0[i-1], gamma)), 
                u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i-1], u_0[i], gamma)), gamma))
    
    else 
        #update_rho!(update, u_0[i-2], u_0[i-1], u_0[i], u_0[i+1], u_0[i+2], dt, dx, gamma) 
        #u_1[i] = u_0[i] - update
        
        u_1[i] = u_0[i] -0.5f0 * dt / dx * (F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)
                    + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)), 
                    u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+2], gamma) - F(u_0[i], u_0[i+1], gamma)), gamma)

                    - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[i], gamma) - F(u_0[i-2], u_0[i-1], gamma)), 
                    u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)), gamma))
        
    end
    return
end

function update_rho!(update, u_2, u_1, u, u1, u2, dt, dx, gamma)
    update = 0.5f0 * dt / dx * (F(u, u1, gamma) - F(u_1, u, gamma)
                    + F(u - dt/dx*(F(u, u1, gamma) - F(u_1, u, gamma)), 
                    u1 - dt/dx*(F(u1, u2, gamma) - F(u, u1, gamma)), gamma)

                    - F(u_1 - dt/dx*(F(u_1, u, gamma) - F(u_2, u_1, gamma)), 
                    u - dt/dx*(F(u, u1, gamma) - F(u_1, u, gamma)), gamma))
    return
end

function intersection_solver!(u_0, u_1, intersection, dt, gammas, N_max, N_vals)
    # classify the intersection

    if intersection.n_incoming == 2 && intersection.n_outgoing == 2
        # two to two intersection
        #u_1 = two_to_two!(u_0, intersection, dt)
        @cuda two_to_two!(u_0, u_1, intersection.incoming[1].id, intersection.incoming[2].id, intersection.outgoing[1].id, intersection.outgoing[2].id, 
        intersection.incoming[1].dx, intersection.incoming[2].dx, intersection.outgoing[1].dx, intersection.outgoing[2].dx, gammas, dt, N_max, N_vals,
        intersection.P, intersection.alpha)
    elseif intersection.n_incoming == 1 && intersection.n_outgoing == 1
        # one to one intersection
        @cuda one_to_one!(u_0, u_1, intersection.incoming[1].id, intersection.outgoing[1].id, intersection.incoming[1].dx, intersection.outgoing[1].dx, gammas, dt, N_max, N_vals)
    elseif n_in == 1 && n_out == 2
        # one to two intersection
        # throw not implemented error
        throw(ArgumentError("one to two intersection not implemented"))
        #u_1 = one_to_two!(u_0, intersection, dt)
    elseif n_in == 2 && n_out == 1
        # two to one intersection
        #u_1 = two_to_one!(u_0, intersection, dt)
        throw(ArgumentError("two to one intersection not implemented"))
    end
end

function one_to_one!(u_0, u_1, road_in_id, road_out_id, dx_i, dx_o, gammas, dt, N_max, N_vals)

    j_in = (road_in_id-1)*N_max + N_vals[road_in_id]
    j_out = (road_out_id-1)*N_max +1
    
    in_rho = u_0[j_in]
    out_rho = u_0[j_out]

    sigma = 0.5
    
    D = in_rho <= sigma ? f(gammas[road_in_id], in_rho) : f(gammas[road_in_id], sigma)
    S = out_rho <= sigma ? f(gammas[road_out_id], sigma) : f(gammas[road_out_id], in_rho)
    
    f_intersection = min(D, S)
    
    u_1[j_out] = u_0[j_out] - dt/dx_o*(F(u_0[j_out], u_0[j_out + 1], gammas[road_out_id]) - f_intersection)
    u_1[j_in] = u_0[j_in] - dt/dx_i*(f_intersection - F(u_0[j_in - 1], u_0[j_in], gammas[road_in_id]))
    
    return 
end


function two_to_two!(u_0, u_1, road_in_id1, road_in_id2, road_out_id1, road_out_id2, dx_i1, dx_i2, dx_o1, dx_o2, gammas, dt, N_max, N_vals, P, alpha)



    j_in1 = (road_in_id1-1)*N_max + N_vals[road_in_id1]
    j_in2 = (road_in_id2-1)*N_max + N_vals[road_in_id2]
    j_out1 = (road_out_id1-1)*N_max + 1
    j_out2 = (road_out_id2-1)*N_max + 1

    in_rho1 = u_0[j_in1]
    in_rho2 = u_0[j_in2]
    out_rho1 = u_0[j_out1]
    out_rho2 = u_0[j_out2]

    sigma = 0.5

    D1 = in_rho1 <= sigma ? f(gammas[road_in_id1], in_rho1) : f(gammas[road_in_id1], sigma)
    D2 = in_rho2 <= sigma ? f(gammas[road_in_id2], in_rho2) : f(gammas[road_in_id2], sigma)
    S1 = out_rho1 <= sigma ? f(gammas[road_out_id1], sigma) : f(gammas[road_out_id1], in_rho1)
    S2 = out_rho2 <= sigma ? f(gammas[road_out_id2], sigma) : f(gammas[road_out_id2], in_rho2)

    F11 = min(alpha[1]*D1, max(P[1]*S1, P[1]*S1 - alpha[3]*D2))
    F21 = min(alpha[3]*D2, max((1-P[1])*S1, S1 - alpha[1]*D1))
    Fr1 = F11 + F21
    f_22upper = f(gammas[road_in_id2], sigma) - F11/f(gammas[road_in_id1], sigma) * (f(gammas[road_in_id2], sigma) - 0.f0001) # epsilon = 0.0001
    F12 = min(alpha[2]*D1, max(P[2]*S2, S2 - alpha[2]*D2, S2 - f_22upper))
    F22 = min(f_22upper, alpha[4]*D2, max((1-P[2])*S2, S2 - alpha[2]* D1))
    Fr2 = F12 + F22
    Fl1 = F11 + F12
    Fl2 = F21 + F22


    u_1[j_in1] = u_0[j_in1] - dt/dx_i1*(Fl1 - F(u_0[j_in1 - 1], u_0[j_in1], gammas[road_in_id1]))
    u_1[j_in2] = u_0[j_in2] - dt/dx_i2*(Fl2 - F(u_0[j_in2 - 1], u_0[j_in2], gammas[road_in_id2]))
    u_1[j_out1] = u_0[j_out1] - dt/dx_o1*(F(u_0[j_out1], u_0[j_out1 + 1], gammas[road_out_id1]) - Fr1)
    u_1[j_out2] = u_0[j_out2] - dt/dx_o2*(F(u_0[j_out2], u_0[j_out2 + 1], gammas[road_out_id2]) - Fr2)
    return
end

