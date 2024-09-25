using GLMakie

fig = Figure()

ax = Axis(fig[1, 1])

sl_x = Slider(fig[2, 1], range = 0:0.01:10, startvalue = 3)
sl_y = Slider(fig[1, 2], range = 0:0.01:10, horizontal = false, startvalue = 6)

point = lift(sl_x.value, sl_y.value) do x, y
    Point2f(x, y)
end

scatter(point, color = :red, markersize = 20)

limits!(ax, 0, 10, 0, 10)
arr = range(0, 2*pi, 100)
length(arr)
# plot the sine of each element of the array
y = sin.(arr)
y_2 = sin.(arr .+ pi/2)
lines!(arr, y)

fig