
#=
fig = Figure()

ax = Axis(fig[1, 1])

sl_t = Slider(fig[2, 1], range = 0:0.01:10, startvalue = 0)

arr = range(0, 2*pi, 100)
y = lift(sl_t.value) do t
    sin.(arr .+ t)
end


limits!(ax, 0, 10, 0, 10)

lines!(arr, y)

fig
=#

using GLMakie

export plot_2d
export plot_2ds

function plot_2d(xs, ts, u)
    fig = Figure()
    ax = Axis(fig[1, 1])
    sl_t = Slider(fig[2, 1], range = ts, startvalue = 0)

    limits!(ax, xs[1], xs[end], -2, 2)
    y = lift(sl_t.value) do t
        u[findfirst(isequal(t), ts), :]
    end
    
    lines!(xs, y)
    fig
end

function plot_2ds(xs, ts, u_arr, labels)
    fig = Figure()
    ax = Axis(fig[1, 1])
    sl_t = Slider(fig[2, 1], range = ts, startvalue = 0)

    limits!(ax, xs[1], xs[end], -2, 2)
    for (i, u) in enumerate(u_arr)
        y = lift(sl_t.value) do t
            u[findfirst(isequal(t), ts), :]
        end
        lines!(xs, y, label=labels[i])
    end
    axislegend(ax)
    fig
end
