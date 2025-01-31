using GLMakie
using Colors

include("traffic_structs.jl")
include("traffic_solver_gpu.jl")

function make_traffic_visualization(trafficProblem, coordinates, dt, T, U_0)
    """
    Create a visualization of the traffic problem

    coordinates: a 3D array where the first index is the road number, the second index is the start and end coordinates of the road, 
    and the third index is the x and y coordinates of the start and end of the road
    """
    # Calculate all the necessary values
    M = ceil(Int, T/dt)
    n_roads = length(trafficProblem.roads)
    N_vals = [road.N for road in trafficProblem.roads]
    N_max = maximum(N_vals)
    U = zeros(Float32, M+1, n_roads, N_max)

    for i in 1:n_roads
        U[1, i, 1:N_vals[i]] = U_0[i]
    end
    t = 0
    j = 2
    while t < T-1e-6
        if t + dt > T
            dt = T - t
        end
        t += dt
        U_1 = traffic_solve_ka(trafficProblem, dt, U_0, "gpu", false)
        for i in 1:n_roads
            
            U[j, i, 1:N_vals[i]] = U_1[i]
        end
        U_0 = U_1
        j+=1
    end 
    
    

    # Create the figure
    fig = Figure()
    # Hide the axis

    ax = Axis(fig[1, 1])# , visible = false)
    hidedecorations!(ax)  # hides ticks, grid and lables
    hidespines!(ax)  # hide the frame

    

    # Create the slider
    sl_t = Slider(fig[2, 1], range = 0:dt:T, startvalue = 0)
    # Add a label showing the time next to the slider
    label = Label(fig[2, 2], text = "Time: 0")
    on(sl_t.value) do t_val
        label.text = "Time: $t_val"
    end

    # Create the lines where each road i starts at coordinates[i, 1] and ends at coordinates[i, 2]
    # Make the lines have as many segments as the road has cells N_vals[i]
    segments = []
    for i in 1:n_roads
        x1 = coordinates[i,1,1]
        y1 = coordinates[i,1,2]
        x2 = coordinates[i,2,1]
        y2 = coordinates[i,2,2]
        dx = (x2 - x1) / N_vals[i]
        dy = (y2 - y1) / N_vals[i]
        x = zeros(Float32, N_vals[i]+1)
        y = zeros(Float32, N_vals[i]+1)
        x[1], x[end] = x1, x2
        y[1], y[end] = y1, y2
        for j in 2:N_vals[i]
            x[j] = x1 + (j+0.5)*dx
            y[j] = y1 + (j+0.5)*dy
        end

        #x = [(x1 +(j+0.5)*dx) for j in 1:(N_vals[i]-1)]
        #y = [(y1 +(j+0.5)*dy) for j in 1:(N_vals[i]-1)]
        
        # Adding background lines to see the road
        
        #lines!(ax, [Point2f(x1, y1), Point2f(x2, y2)], color = :black, linewidth = 5)

        density = U[1, i, 1:N_vals[i]]
        colors = [RGBA(d, 1-d, 0, 1) for d in density]
        for j in 1:N_vals[i]
            push!(segments, linesegments!(ax, [Point2f(x[j], y[j]), Point2f(x[j+1], y[j+1])], color = colors[j], linewidth = 10))
        end
    end

    on(sl_t.value) do t_val
        show_visuals(t_val)
    end

    function show_visuals(t_val)
        j = Int(round(t_val / dt)) + 1
        for i in 1:n_roads
            density = U[j, i, 1:N_vals[i]]
            colors = [RGBA(d, 1-d, 0, 1) for d in density]
            for k in 1:N_vals[i]
                segments[(i-1)*N_vals[i] + k].color = colors[k]
            end
        end
    end

    # add button that plays the simulation from the start
    button = Button(fig, label="Play")
    simulation_speed = T/10
    on(button.clicks) do n
        sl_t.value = 0
        @async for i in 1:M
            sleep(dt/simulation_speed)
            show_visuals(i*dt)
            set_close_to!(sl_t, i*dt)
        end
        
    end

    fig

end


