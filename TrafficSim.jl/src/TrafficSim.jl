module TrafficSim

    include("plot_helper.jl")
    include("scalar_solvers.jl")
    include("scalar_test_functions.jl")
    include("traffic_structs.jl")
    include("traffic_solver_gpu.jl")
    include("traffic_visualization.jl")
    include("traffic_solver_kernelabstraction.jl")
    include("generate_test_case.jl")
end