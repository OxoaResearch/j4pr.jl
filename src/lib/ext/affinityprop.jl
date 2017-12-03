##########################
# FunctionCell Interface #	
##########################
"""
	affinityprop(fs [;kwargs])

Constructs an untrained cell that when piped data inside, clusters the data using the
affinity propagation algorithm; `fs` is a function that is used to calculate the
similarity matrices during training as well as when assigning new data to existing clusters.
By default, `fs(x,y) = 1 - pairwise(CosineDist(),x,y)`

# Keyword arguments (same as in `Clustering`)
  * `maxiter` maximum number of iterations (default `200`)
  * `tol` tolerable change in objective at convergence (default `1.0e-6`)
  * `damp` dampening coefficient; values should be in [0.0, 1.0] (default `0.5`); larger values indicate
slower and probably more stable updates
  * `display` Level of information; can be `:none`, `:final`, `:iter` (default `:none`)

!!! note 
	The implementation differes from the one in `Clustering.jl` by allowing the user to 
	input directly data and not a similarity matrix to the algorithm.

Read the `Clustering.jl` documentation for more information.  
"""
affinityprop(fs::Function=(x,y)->one(eltype(x)) .- Distances.pairwise(Distances.CosineDist(),x,y); kwargs...) = 
	FunctionCell(affinityprop, (fs,), ModelProperties(), kwtitle("Affinity propagation", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	affinityprop(x, fs::Function=(x,y)->1-Distances.pairwise(CosineDist(),x,y) [;kwargs])

Performs affinity propagation clustering on the dataset `x` using `fs` to calculate the self-similarity matrix; 
When piping data to the resulting clustering (e.g. trained cell), the similarities between each input observation 
and all exemplars are returned. One can apply `targets(indmax, ...)` to obtain the corresponding exemplar 
index (e.g. cluster) for each input observation.
"""
# Training
affinityprop(x::T where T, fs::Function=(x,y)->1-Distances.pairwise(Distances.CosineDist(),x,y); kwargs...) = 
	affinityprop(getx!(x), fs; kwargs...)
affinityprop(x::T where T<:AbstractVector, fs::Function=(x,y)->1-Distances.pairwise(Distances.CosineDist(),x,y); kwargs...) = 
	affinityprop(mat(x, LearnBase.ObsDim.Constant{2}()), fs; kwargs...)
affinityprop(x::T where T<:AbstractMatrix, fs::Function=(x,y)->1-Distances.pairwise(Distances.CosineDist(),x,y); kwargs...) = 	
begin	
	# Perform clustering 
	affpdata = Clustering.affinityprop(fs(x,x); kwargs...) 
	
	# Build model properties 
	modelprops = ModelProperties(nvars(x),length(affpdata.exemplars))
	
	FunctionCell(affinityprop, Model((fs,x[:,sort(affpdata.exemplars)],affpdata), modelprops), kwtitle("Affinity propagation", kwargs));	 
end



# Execution
affinityprop(x::T where T<:CellData, model::Model{<:Tuple{<:Function, <:AbstractArray, <:Clustering.AffinityPropResult}}) =
	datacell(affinityprop(getx!(x), model), gety(x)) 	
affinityprop(x::T where T<:AbstractVector, model::Model{<:Tuple{<:Function, <:AbstractArray, <:Clustering.AffinityPropResult}}) =
	affinityprop(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
affinityprop(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:Function, <:AbstractArray, <:Clustering.AffinityPropResult}}) = 
	model.data[1](model.data[2], x) #fs(exemplars,x)	
