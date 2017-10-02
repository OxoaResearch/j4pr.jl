##########################
# FunctionCell Interface #	
##########################
"""
	whiten([;kwargs])

Constructs an untrained cell that when piped data inside, calculates a transform `W`
of the input data `X` so that `Wᵀcov(X)W=I`.

# Keyword arguments (same as in `MultivariateStats`)
  * `regcoef::Real` is the ratio of variaces preserved in the principal subspace 
  * `mean` is the mean vector, can be `nothing`, `0` or precomputed mean vector (default `nothing`)

Read the `MultivariateStats.jl` documentation for more information.  
"""
whiten(;kwargs...) = FunctionCell(whiten, (), ModelProperties(), kwtitle("Whitening", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	whiten(x, [;kwargs])

Trains a function cell that when receiving data into will whiten the input
the observations based on the projection matrix W extracted from `x`.
"""
# Training
whiten(x::T where T<:CellData; kwargs...) = whiten(getx!(x); kwargs...)
whiten(x::T where T<:AbstractVector; kwargs...) = whiten(mat(x, LearnBase.ObsDim.Constant{2}()); kwargs...)
whiten(x::T where T<:AbstractMatrix; kwargs...) = begin
	
	whdata = fit(MultivariateStats.Whitening, getobs(x); kwargs...)

	# Build model properties 
	modelprops = ModelProperties(nvars(x), nvars(x))
	
	# Returned trained cell
	FunctionCell(whiten, Model(whdata, modelprops), kwtitle("Whitening",kwargs))	 
end



# Execution
whiten(x::T where T<:CellData, model::Model{<:MultivariateStats.Whitening}) =
	datacell(whiten(getx!(x), model), gety(x)) 	
whiten(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.Whitening}) =
	whiten(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
whiten(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.Whitening}) =
	MultivariateStats.transform(model.data, getobs(x))
