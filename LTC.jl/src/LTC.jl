module LTC
using SymbolicRegression

function qloss(x, y, action)
    # xx = x[CartesianIndex.(1:length(action), action + 1)]
    # yy = y[CartesianIndex.(1:length(action), action + 1)]
    print(x)
    print(y)

    # return sum
    index = Int32.(action .+ 1)
    return @. sum((x[index] - y[index])^2)
end

function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    # print(options)
    return sum((prediction .- dataset.y) .^ 2) / dataset.n
end

function f()
    # @info 43
    # @warn  "This is a warning message."
    # @show "test"

    return 45
end


end # module LTC
