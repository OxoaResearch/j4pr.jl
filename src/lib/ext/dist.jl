##########################
# FunctionCell Interface #	
##########################
"""
	dist(d=Distances.Euclidean())

Constructs an untrained function cell using a `d::Distances.PreMetric` function cell. `d` defaults to `Distances.Euclidean()` if not specified.

For more information on the distances allowed, see ?Distances.
"""
dist(d::Distances.PreMetric=Distances.Euclidean()) = FunctionCell(dist, (d,), ModelProperties(), "Distance ($d))") 



############################
# DataCell/Array Interface #	
############################
"""
	dist(x, d=Distances.Euclidean())

Trains the function cell using a `d::Distances.PreMetric` function and storing the `x` matrix
for future distance calculations.
"""
# Training
dist(x::T where T<:CellData, d::Distances.PreMetric=Distances.Euclidean()) = dist(getx!(x), d)
dist(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractArray, d::Distances.PreMetric=Distances.Euclidean()) = dist(x[1], d)
dist(x::T where T<:AbstractVector, d::Distances.PreMetric=Distances.Euclidean()) = dist(mat(x, LearnBase.ObsDim.Constant{2}()), d)
dist(x::T where T<:AbstractMatrix, d::Distances.PreMetric=Distances.Euclidean()) = begin
	
	# Build name
	namedist = "Distance ($d)"
	
	# Build model properties
	modelprops = ModelProperties(nvars(x),nobs(x))
	
	# Returned trained cell
	FunctionCell(dist, Model((d,getobs(x)), modelprops), namedist)	 
end



# Execution
dist(x::T where T<:CellData, model::Model{<:Tuple{<:Distances.PreMetric,Matrix}}) =
	datacell(dist(getx!(x), model), gety(x)) 	
dist(x::T where T<:AbstractVector, model::Model{<:Tuple{<:Distances.PreMetric,Matrix}}) =
	dist(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
dist(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:Distances.PreMetric,Matrix}}) =
	Distances.pairwise(model.data[1], model.data[2], getobs(x))



