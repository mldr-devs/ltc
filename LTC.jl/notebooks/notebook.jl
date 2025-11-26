### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 94d01d6c-1177-4b83-b070-6c5f56162e6f
begin
	
	ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
	ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, "..","..",".venv","bin","python")  # optional
	
	import Pkg
	
	Pkg.activate(joinpath(@__DIR__, ".."))  # activate LTC project
	Pkg.instantiate()
	# Pkg.build("PythonCall")
	
	import LTC
	using PythonCall
	using Revise
	
	
	ltcsym = pyimport("ltc.symbolic")
	print("LTC symbolic imported:", ltcsym)
	
	jax = pyimport("jax")
	jnp = pyimport("jax.numpy")
	jeden=jnp.ones((3,3))
	print("JAX jnp.ones:",jeden)
	
end

# ╔═╡ 77667d4b-874c-4de4-a5eb-bd7a4ddad8aa
jnp.ones(4).sum()

# ╔═╡ 3cbcafbb-97f6-4d8c-b7dc-f85212697ef1


# ╔═╡ a55e89a4-addf-43a8-a84d-ea8b2a098d8f
LTC.my_loss

# ╔═╡ 7d773505-f248-4316-9eb2-b29c7b3bb01c
import SymbolicRegression: SRRegressor,MultitargetSRRegressor , Dataset, eval_tree_array

# ╔═╡ fbb93a24-9b52-44d5-9f0e-f16314c4a178
import MLJ: machine, fit!, predict, report

# ╔═╡ 9ddad752-315b-47db-af00-2cf7f42feaee
begin
	 # Dataset with two named features:
	 X = (a=rand(500), b=rand(500))
	
	 # and one target:
	 y = @. 2 * cos(X.a * 23.5) - X.b^2
	 # y = [y';y']'
	
	 # with some noise:
	 y = y .+ randn(500) .* 1e-3
	
	 model = SRRegressor(
		 niterations=50,
		 binary_operators=[+, -, *],
		 unary_operators=[cos],
		 # elementwise_loss=LTC.qloss
	 )
end

# ╔═╡ e3a12c8d-3fcc-4640-8f1e-db2a08fb48de
ones_like(x)=ones(eltype(x), size(x)...)

# ╔═╡ 1b22fab6-c920-47a3-922a-4916fc190fc3
w=ones_like(y)[:,1]

# ╔═╡ f7be3b67-754f-4f45-9387-c3f83909d0d0
begin
	
	 mach = machine(model, X, y, w)
	
	 fit!(mach)
end

# ╔═╡ b8598a79-6e67-4980-946d-284df6455c18
begin
	function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
	    prediction, flag = eval_tree_array(tree, dataset.X, options)
	    if !flag
	        return L(Inf)
	    end
	    print(options)
	    return sum((prediction .- dataset.y) .^ 2) / dataset.n
	end
	Y = [y';y']'
	model2 = MultitargetSRRegressor(
		niterations=10,
		binary_operators=[+, -, *],
		unary_operators=[cos],
		loss_function=my_loss
	)
	ww = ones(Float64, size(y)...)
	mach2 = machine(model2, X, Y,ww[:,1])
	
	fit!(mach2)
end

# ╔═╡ 4dae4a63-deb8-4a93-94d9-e564a24cc46f
begin
	function f()
		Y = [y';y']'
		w = ones(Float64, size(y)...)
		LTC.qloss(Y,Y,w)
	end
	f()
end

# ╔═╡ 35292c16-bc42-4999-91e4-21819d7a3e2c


# ╔═╡ Cell order:
# ╠═94d01d6c-1177-4b83-b070-6c5f56162e6f
# ╠═77667d4b-874c-4de4-a5eb-bd7a4ddad8aa
# ╠═3cbcafbb-97f6-4d8c-b7dc-f85212697ef1
# ╠═a55e89a4-addf-43a8-a84d-ea8b2a098d8f
# ╠═7d773505-f248-4316-9eb2-b29c7b3bb01c
# ╠═fbb93a24-9b52-44d5-9f0e-f16314c4a178
# ╠═9ddad752-315b-47db-af00-2cf7f42feaee
# ╠═e3a12c8d-3fcc-4640-8f1e-db2a08fb48de
# ╠═1b22fab6-c920-47a3-922a-4916fc190fc3
# ╠═f7be3b67-754f-4f45-9387-c3f83909d0d0
# ╠═b8598a79-6e67-4980-946d-284df6455c18
# ╠═4dae4a63-deb8-4a93-94d9-e564a24cc46f
# ╠═35292c16-bc42-4999-91e4-21819d7a3e2c
