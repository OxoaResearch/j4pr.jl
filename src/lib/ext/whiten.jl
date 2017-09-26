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
whiten(;kwargs...) = FunctionCell(whiten, (), Dict(), kwtitle("Whitening", kwargs); kwargs...) 



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
	modelprops = Dict("size_in" => nvars(x),
		   	  "size_out" => nvars(x)
	)
	
	# Returned trained cell
	FunctionCell(whiten, Model(whdata), modelprops, kwtitle("Whitening",kwargs))	 
end



# Execution
whiten(x::T where T<:CellData, model::Model{<:MultivariateStats.Whitening}, modelprops::Dict) = datacell(whiten(getx!(x), model, modelprops), gety(x)) 	
whiten(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.Whitening}, modelprops::Dict) = whiten(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
whiten(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.Whitening}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	MultivariateStats.transform(model.data, getobs(x))
end
