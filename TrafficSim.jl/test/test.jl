using Plots
using LinearAlgebra
using .PlotHelper

arr = range(0, 2*pi, 100)
length(arr)
# plot the sine of each element of the array
y = sin.(arr)
y_2 = sin.(arr .+ pi/2)
plot(arr, [y, y_2], label=["sin(x)" "sin(x + pi/2)"])

a = [2 2]
b = [-1 0; 0 1]
mul!(a, b) -> c
b*transpose(a)

append!(b, a)

ceil(Int, 10/0.03)
(1:5)[begin]

A = Tridiagonal([-1, -1], [1, 1, 1], [-1, -1])
A_mat = Matrix(A)
A_mat[1, 1:3]

x = range(-2, stop=2, length=100)
y = range(-2, stop=2, length=100)
f(x,y) = x*sin(y)
z = f.(x', y)

println(findfirst(isequal(0), x))

surface(x, y, z)