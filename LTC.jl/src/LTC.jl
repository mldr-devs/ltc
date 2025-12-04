module LTC
using SymbolicRegression



function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    # print(options)
    return sum((prediction .- dataset.y) .^ 2) / dataset.n
end


end # module LTC
