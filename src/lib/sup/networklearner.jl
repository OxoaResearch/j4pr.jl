##########################
# FunctionCell Interface #
##########################
"""
	networklearner(args,[;kwargs])

Constructs an untrained cell that when piped data inside, returns a network learner trained
function cell based on the input data and labels.

# Arguments
  * `
  
# Keyword arguments
  * 

Read the `NetworkLearning.jl` documentation for more information.
"""
networklearner(args...; kwargs...) = FunctionCell(networklearner, args, ModelProperties(), "Network Learner"; kwargs...) 



############################
# DataCell/Array Interface #
############################
"""
	networklearner(args, [;kwargs])

Trains a network learner model that using the data `x`.
"""
# Training
networklearner(x::T where T<:CellDataL; kwargs...) = 
	networklearner((getx!(x), gety(x)); kwargs...)
networklearner(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector; kwargs...) =
	networklearner((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]); kwargs...)
networklearner(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[networklearner] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)
	
	# Train model
	nldata = NetworkLearning.fit(getobs(x[1]), yenc; kwargs...)

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(networklearner, Model(nldata, modelprops), "Network Learner", kwargs)) 

end



# Execution
networklearner(x::T where T<:CellData, model::Model{<:NetworkLearning.AbstractNetworkLearner}) = 
	datacell(networklearner(getx!(x), model), gety(x)) 	

networklearner(x::T where T<:AbstractVector, model::Model{<:NetworkLearning.AbstractNetworkLearner}) = 
	networklearner(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	

networklearner(x::T where T<:AbstractMatrix, model::Model{<:NetworkLearning.AbstractNetworkLearner}) =
	NetworkLearning.transform(model.data, getobs(x))
