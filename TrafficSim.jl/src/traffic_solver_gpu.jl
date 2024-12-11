using CUDA

include("traffic_structs.jl")



@inline function f(gamma::Float32, rho::Float32)::Float32
    return gamma * rho * (1 - rho)
end

@inline function J(gamma::Float32, rho::Float32)::Float32
    return gamma * (1 - 2*rho)
end

@inline function F(u_1::Float32, u_2::Float32, gamma::Float32)::Float32
    return 0.5*(f(gamma, u_1) + f(gamma, u_2)) - 0.5*(max(abs(J(gamma, u_1)), abs(J(gamma, u_2)))*(u_2 - u_1))
end

function traffic_solve(trafficProblem::TrafficProblem, T, dt::Float32, U_0)
    # Add potential timestamps to be exported with the final solution
    timestamps = []
    
    n_threads = 512
    n_blocks_tot = length(trafficProblem.roads)

    N_vals = [road.N for road in trafficProblem.roads]
    N_max = maximum(N_vals)

    N_max = max(N_max, n_threads)

    #n_blocks = cumsum([ceil(Int, N/n_threads) for N in N_vals])
    #n_blocks_tot = n_blocks[end]

    gammas = [road.v_max/road.length for road in trafficProblem.roads]
    dxs = [road.dx for road in trafficProblem.roads]

    M = ceil(Int, T/dt)
    # make 2D array to store the density of the roads
    #rho = zeros(Float32, length(trafficProblem.roads)*N_max)
    rho = CUDA.fill(1.0f0, length(trafficProblem.roads)*N_max)

    rho_total = zeros(Real, M+1, length(trafficProblem.roads), N_max)

    for road in trafficProblem.roads
        road_index = road.id
        rho_total[1, road_index, 1:road.N] = U_0[road_index]
    end

    for i in 1:length(trafficProblem.roads)
        j = (i-1)*N_max + 1
        rho[j:j+N_vals[i]-1] = U_0[i]
    end

    rho_1 = similar(rho)

    N_vals = CuArray(N_vals)
    gammas = CuArray(gammas)
    dxs = CuArray(dxs)
    

    for i in 2:M+1
        # solve the traffic problem on the roads independently
        
        #solve the roads
        @cuda threads=n_threads blocks=n_blocks_tot road_solver!(rho, rho_1, N_vals, N_max, gammas, dt, dxs)
        
        #CUDA.@sync
        

        # solve the intersections
        for intersection in trafficProblem.intersections
            # solve the intersection
            intersection_solver!(rho, rho_1, intersection.incoming, intersection.outgoing, dt, gammas, N_max, N_vals)
        end
        #CUDA.@sync
        rho, rho_1 = rho_1, rho
        for road in trafficProblem.roads
            road_index = road.id
            rho_cpu = Array(rho)
            rho_total[i, road_index, 1:road.N] = rho_cpu[N_max*(road_index-1)+1:N_max*(road_index-1)+road.N]
            println("i : ", i, " road_index : ", road_index, " rho : ", rho_total[i, road_index, 1:road.N])
        end
    end
    

    rho_cpu = collect(rho)
    return rho_cpu, rho_total
end

function road_solver!(u_0::CuDeviceVector{Float32}, u_1::CuDeviceVector{Float32}, N_vals::CuDeviceVector{Int}, N_max::Int, gammas::CuDeviceVector{Float32}, dt::Float32, dxs::CuDeviceVector{Float32})
    # get the index of the block
    #i = threadIdx().x + (blockIdx().x - 1)*N_max
    i = 1
    if i > length(u_0) || i < 1
        return
    end
    
    road_index = ceil(Int, i/N_max)

    j = (i-1) % N_max + 1

    gamma = gammas[road_index]
    dx = dxs[road_index]


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
        u_1[i] = u_0[i] - 0.5f0 * dt / dx * (F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)
                    + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)), 
                    u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+2], gamma) - F(u_0[i], u_0[i+1], gamma)), gamma)

                    - F(u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)), 
                    u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)), gamma))
    
    elseif j == N_vals[road_index] - 1
        u_1[i] = u_0[i] - 0.5f0 * dt / dx * (F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)
                    + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)), 
                    u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+1], gamma) - F(u_0[i], u_0[i+1], gamma)), gamma)

                    - F(u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)), 
                    u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i], u_0[i], gamma)), gamma))
    
    elseif j == N_vals[road_index]
        u_1[i] = u_0[i] - 0.5f0 * dt / dx * (F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)
                    + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)), 
                    u_0[i+1] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)), gamma)

                    - F(u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)), 
                    u_0[i] - dt/dx*(F(u_0[i], u_0[i], gamma) - F(u_0[i], u_0[i], gamma)), gamma))
    
    else 
        
        u_1[i] = u_0[i] - 0.5f0 * dt / dx * (F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)
                    + F(u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)), 
                    u_0[i+1] - dt/dx*(F(u_0[i+1], u_0[i+2], gamma) - F(u_0[i], u_0[i+1], gamma)), gamma)

                    - F(u_0[i-1] - dt/dx*(F(u_0[i-1], u_0[i], gamma) - F(u_0[i-2], u_0[i-1], gamma)), 
                    u_0[i] - dt/dx*(F(u_0[i], u_0[i+1], gamma) - F(u_0[i-1], u_0[i], gamma)), gamma))
        
    end
    return
end

function intersection_solver!(u_0, u_1, roads_incoming, roads_outgoing, dt, gammas, N_max, N_vals)
    # classify the intersection
    n_in = length(roads_incoming)
    n_out = length(roads_outgoing)

    if n_in == 2 && n_out == 2
        # two to two intersection
        #u_1 = two_to_two!(u_0, intersection, dt)
        throw(ArgumentError("two to two intersection not implemented"))
    elseif n_in == 1 && n_out == 1
        # one to one intersection
        @cuda one_to_one(u_0, u_1, roads_incoming[1].id, roads_outgoing[1].id, roads_incoming[1].dx, roads_outgoing[1].dx, gammas, dt, N_max, N_vals)
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

function one_to_one(u_0::CuDeviceVector{Float32}, u_1::CuDeviceVector{Float32}, road_in_id::Int, road_out_id::Int, dx_i::Float32, dx_o::Float32, gammas::CuDeviceVector{Float32}, dt::Float32, N_max::Int, N_vals::CuDeviceVector{Int})
    j_in = (road_in_id-1)*N_max + N_vals[road_in_id] - 1
    j_out = (road_out_id-1)*N_max 
    
    in_rho = u_0[j_in]
    out_rho = u_0[j_out]

    sigma::Float32 = 0.5
    
    D = f(gammas[road_out_id], in_rho)
    

    
    D = in_rho <= sigma ? f(gammas[road_in_id], in_rho) : f(gammas[road_in_id], sigma)
    S = out_rho <= sigma ? f(gammas[road_out_id], sigma) : f(gammas[road_out_id], in_rho)
    
    f_intersection = min(D, S)
    
    u_1[j_out] = u_0[j_out] - dt/dx_o*(F(u_0[j_out], u_0[j_out + 1], gammas[road_out_id]) - f_intersection)
    u_1[j_in] = u_0[j_in] - dt/dx_i*(f_intersection - F(u_0[j_in - 1], u_0[j_in], gammas[road_in_id]))
    
    return 
end




N = 20

road_1 = Road(1, 100, 10, 0.5, N, 100/N)
road_2 = Road(2, 100, 5, 0.5, N, 100/N)

intersection = Intersection(1, 1, 1, [road_1], [road_2])

trafficProblem = TrafficProblem([road_1, road_2], [intersection])

x = range(0, 100, N)

include("scalar_test_functions.jl")

U_0 = [bump(x), zeros(N)]


T = 13
dt::Float32 = 0.1

rho, rho_total = traffic_solve(trafficProblem, T, dt, U_0)

rho_1 = rho_total[:, 1, 1:N]
rho_2 = rho_total[:, 2, 1:N]

print(size(rho_1))
t = range(0, T, length = size(rho_1)[1])

include("plot_helper.jl")

plot_2ds(x, t, [rho_1, rho_2], ["Road 1", "Road 2"])


# plot the solution using GLMakie
"""
using GLMakie

fig = Figure()

ax = Axis(fig[1, 1])

for i in 1:length(trafficProblem.roads)
    j = (i-1)*512 + 1
    lines!(ax, x, rho[j:j+N-1], color = :blue)
end
fig
"""
