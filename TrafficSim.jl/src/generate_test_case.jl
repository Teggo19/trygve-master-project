include("traffic_structs.jl")
using Interpolations

function main_incoming_flux(t)
    # random number between 0.5 and 0.7
    base = 0.65f0
    # random number between 0.1 and .3
    amplitude = 0.1f0
    # random number between 1 and 10
    frequency = 1.f0
    return base + amplitude*sin(2*pi*frequency*t/100)
end

function side_incoming_flux(t)
    base = 0.25f0 
    amplitude = 0.05f0
    frequency = 9.f0
    return base + amplitude*sin(2*pi*frequency*t/100)
end

function zero_flux(t)
    return 0.f0
end


function make_test_trafficProblem(M, N)
    main_road_vmax = 10.f0
    side_road_vmax = 6.f0

    P = [0.7f0, 0.5f0]
    alpha = [0.8f0, 0.2f0, 0.4f0, 0.6f0]

    # make an empty array to store 2*M*(M+1) roads
    roads = Array{TrafficSim.Road}(undef, 2*M*(M+1))

    intersections = Array{TrafficSim.Intersection}(undef, M^2)

    for i in 1:(2*M*(M+1))
        if i <= M*(M+1)
            if i < M+1 && i%2 == 1
                roads[i] = TrafficSim.Road{Float64, Float32}(i, 100., main_road_vmax, 0.5, N, 1/N, main_incoming_flux)
            elseif i > M^2 && (i-M^2)%2 == 0
                roads[i] = TrafficSim.Road{Float64, Float32}(i, 100., main_road_vmax, 0.5, N, 1/N, main_incoming_flux)
            else
                roads[i] = TrafficSim.Road{Float64, Float32}(i, 100., main_road_vmax, 0.5, N, 1/N, zero_flux)
            end
        else
            if i <= M^2 + 2*M && (i-M*(M+1))%2 == 0
                roads[i] = TrafficSim.Road{Float64, Float32}(i, 30., side_road_vmax, 0.5, N, 1/N, side_incoming_flux)
            elseif i > M*(2*M+1) && (i-M*(2*M+1))%2 == 1
                roads[i] = TrafficSim.Road{Float64, Float32}(i, 30., side_road_vmax, 0.5, N, 1/N, side_incoming_flux)
            else
                roads[i] = TrafficSim.Road{Float64, Float32}(i, 30., side_road_vmax, 0.5, N, 1/N, zero_flux)
            end

        end
    end

    for i in 1:M^2
        y_coor = floor(Int, (i-1)/M)+1
        x_coor = (i-1) % M + 1

        if x_coor%2==0
            incoming_road_1 = roads[(y_coor)*M+x_coor]
            outgoing_road_1 = roads[(y_coor-1)*M + x_coor]
        else
            incoming_road_1 = roads[(y_coor-1)*M + x_coor]
            outgoing_road_1 = roads[(y_coor)*M + x_coor]
        end
        if y_coor%2 == 1
            incoming_road_2 = roads[M^2+M + y_coor + x_coor*M]
            outgoing_road_2 = roads[M^2+M + y_coor + (x_coor-1)*M]
        else
            incoming_road_2 = roads[M^2+M + y_coor + (x_coor-1)*M]
            outgoing_road_2 = roads[M^2+M + y_coor + (x_coor)*M ]
        end
        intersections[i] = TrafficSim.Intersection{Float32}(i, 2, 2, [incoming_road_1, incoming_road_2], [outgoing_road_1, outgoing_road_2], alpha, P)
    end

    trafficProblem = TrafficSim.TrafficProblem(roads, intersections)
    #trafficProblem = TrafficSim.TrafficProblem(roads, [])

    rho_read = [[0.0f0 for i in 1:100] for j in 1:24]
    open("src/test_case_M=3_N=1000.txt") do io
        i = 1
        for line in eachline(io)
            rho_read[i] = eval(Meta.parse(line))
            i += 1
        end
    end
    x_1000 = range(0, 1, 100)
    
    interpolations = [linear_interpolation(x_1000 , rho_read[i]) for i in 1:24]

    x = range(0, 1, N)
    U_0 = [zeros(Float32, N) for i in 1:M*(M+1)*2]
    for i in 1:M*(M+1)*2
        if i <= M*(M+1)
            U_0[i] = [interpolations[(i-1)%12+1](x) for x in x]
        else
            U_0[i] = [interpolations[(i-1)%12+12+1](x) for x in x]
        end
    end

    return trafficProblem, U_0
end


function test_case_coordinates(M)
    # make a 3D array of integers of dimension (2M*(M+1), 2 , 2)
    coordinates = ones(Int, 2*M*(M+1), 2, 2)
    for i in 1:(2*M*(M+1))
        if i<=M*(M+1)
            x_id = (i-1)%M + 1
            y_id = floor(Int, (i-1)/M)+1
            coordinates[i,1,1]=x_id
            coordinates[i,2,1]=x_id
            if x_id%2==0
                coordinates[i,1,2]= y_id + 1
                coordinates[i,2,2] = y_id
            else
                coordinates[i,1,2]= y_id
                coordinates[i,2,2]=y_id+1
            end
        else
            x_id = floor(Int, (i-1-(M*(M+1)))/M)
            y_id = (i-1-M*(M+1))%M + 2
            coordinates[i,1,2]= y_id
            coordinates[i,2,2] = y_id
            if y_id%2==0
                coordinates[i,1,1]= x_id+1
                coordinates[i,2,1]= x_id
            else
                coordinates[i,1,1]= x_id
                coordinates[i,2,1]= x_id+1
            end
        end
    end
    return coordinates
end
