using KernelAbstractions
using CUDA

@kernel function test_kernel(a, b, c)
    i = @index(Global)
    c[i] = a[i] + b[i]
end

dev = CPU()
a = rand(Float32, 10)
b = rand(Float32, 10)
c = similar(a)

kernel = test_kernel(dev, 256)

kernel(a, b, c, ndrange=length(a))
