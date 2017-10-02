##########################
# FunctionCell Interface #	
##########################
"""
	ica([;kwargs])

Constructs an untrained cell that when piped data inside, calculates the ICA
projection matrix of the input data.

# Keyword arguments (same as in `MultivariateStats`)
  * `alg` must be `:fastica` (as of June 2017 at least) 
  * `fun` is the approx neg-entripy functor. `icafun(:tanh)`, `icafun(:tanh,a)`, `icafun(:gaus)` (default `icafun(:tanh)`)
  * `do_whiten` whether to perform pre-whitening (default `true)`
  * `maxiter` maximum number of iterations (default `100`)
  * `tol` tolerable change of W at convergence (default `1e-6`)
  * `mean` can be 0 e.g. data already centralized, `nothing` e.g. function will compute mean, or a pre-computed vector (default `nothing`)
  * `winit` initial guess of `W` (default `zeros(0,0)`)
  * `verbose` whether to display iteration information (default `false`)

Read the `MultivariateStats.jl` documentation for more information.  
"""
ica(k::Int; kwargs...) = FunctionCell(ica, (k,), ModelProperties(), kwtitle("ICA", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	ica(x, [;kwargs])

Trains a function cell that when receiving data into will compute the idependent components of
the observations based on the projection matrix W extracted from `x`.
"""
# Training
ica(x::T where T<:CellData, k::Int; kwargs...) = ica(getx!(x), k;kwargs...)
ica(x::T where T<:AbstractVector, k::Int; kwargs...) = ica(mat(x, LearnBase.ObsDim.Constant{2}()), k; kwargs...)
ica(x::T where T<:AbstractMatrix, k::Int; kwargs...) = begin
	
	icadata = fit(MultivariateStats.ICA, getobs(x), k; kwargs...)

	# Build model properties 
	modelprops = ModelProperties(size(icadata.W,1), 					# Wᵀx = y so size(x,1) == size(Wᵀ,2) == size(W,1)
		 		     size(icadata.W,2)						# also, size(y e.g. out,1) == size(Wᵀ,1) == size(W,w)
	)
	
	# Returned trained cell
	FunctionCell(ica, Model(icadata, modelprops), kwtitle("ICA",kwargs))	 
end



# Execution
ica(x::T where T<:CellData, model::Model{<:MultivariateStats.ICA}) =
	datacell(ica(getx!(x), model), gety(x)) 	
ica(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.ICA}) =
	ica(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
ica(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.ICA}) =
	MultivariateStats.transform(model.data, getobs(x))
