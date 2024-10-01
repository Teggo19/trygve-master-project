module ScalarTestFunctions
    export square
    export triangle
    export bump
    export step_func

    function square(x)
        N = length(x)
        u = zeros(N)

        for i in div(N, 3):div(2N, 3)
            u[i] = 1
        end
        return u

    end

    function triangle(x)
        N = length(x)
        u = zeros(N)

        for i in div(N, 3):div(2N, 3)
            u[i] = 1 - abs(x[i] - (x[div(N, 3)] + x[div(2N, 3)])/2)
        end
        return u
    end

    function bump(x)

        N = length(x)
        u = zeros(N)

        x_1 = x[div(N, 3)]
        x_2 = x[div(2N, 3)]

        for i in div(N, 3)+1:div(2N, 3)
            u[i] = exp(1)*exp(-1/(1 - ((x[i]*2 - (x_1 + x_2))/(x_2-x_1))^2))
        end
        print(u[div(N, 3)])
        
        return u
    end

    function step_func(x)
        N = length(x)
        u = zeros(N)

        for i in div(N, 2):N
            u[i] = 1
        end
        return u
    end


end