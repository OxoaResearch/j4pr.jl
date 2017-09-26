##########################
# FunctionCell Interface #	
##########################
"""
	kmedoids(k, d=Distances.Euclidean() [;kwargs])

Constructs an untrained cell that when piped data inside, clusters the data using the
k-medoids algorithm where `k` is the desired number of clusters and `d` is the distance
that will be used to calculate distances from new samples to the clustering generated
from the untrained function cell.

# Keyword arguments (same as in `Clustering`)
  * `init` initialization algorithm; can be `:rand` for random, `:kmpp` for K-medoids++ or `:kmcen` for choosing samples with highest centrality (default `:kmpp`)
or a vector of size 'k' providing the indexes of the initial seeds.
  * `maxiter` maximum number of iterations (default `100`)
  * `tol` tolerable change in objective at convergence (default `1.0e-6`)
  * `display` Level of information; can be `:none`, `:final`, `:iter` (default `:none`)

Read the `Clustering.jl` documentation for more information.  
"""
kmedoids(k::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = FunctionCell(kmedoids, (k,d), Dict(), kwtitle("K-medoids ($k clusters)", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	kmedoids(x, k, d=Distances.Euclidean() [;kwargs])

Performs k-medoids clustering on the dataset `x` (square cost matrix) using `k` number of clusters and returns the clustering; when piping
data to the resulting clustering (e.g. trained cell), the distances from input data to the medoids are 
returned, using `d` as distance. 
"""
# Training
kmedoids(x::T where T<:CellData, k::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = kmedoids(getx!(x), k, d; kwargs...)
kmedoids(x::T where T<:AbstractVector, k::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = 
	error("[kmedoids] Vectors are not supported as input data. Try using `v |> dist(v) |> kmedoids(n)`") 
kmedoids(x::T where T<:AbstractMatrix, k::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = begin
	
	# Perform clustering 
	kmeddata = Clustering.kmedoids(x, k; kwargs...) 
	
	@assert length(kmeddata.medoids) == k "[kmedoids] The number of medoids (cluster centers) differes from the desired one k=$k."

	# Build model properties 
	modelprops = Dict("size_in" => length(kmeddata.acosts),
		   	  "size_out" => k	 
	)
	
	FunctionCell(kmedoids, Model((d, x[:,sort(kmeddata.medoids)], kmeddata)), modelprops, kwtitle("K-medoids ($k clusters)", kwargs) );	 
end



# Execution
kmedoids(x::T where T<:CellData, model::Model{<:Tuple{<:Distances.PreMetric, <:AbstractArray, <:Clustering.KmedoidsResult}}, modelprops::Dict) = datacell(kmedoids(getx!(x), model, modelprops), gety(x)) 	
kmedoids(x::T where T<:AbstractVector, model::Model{<:Tuple{<:Distances.PreMetric, <:AbstractArray, <:Clustering.KmedoidsResult}}, modelprops::Dict) = kmedoids(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
kmedoids(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:Distances.PreMetric, <:AbstractArray, <:Clustering.KmedoidsResult}}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	# Return distances from each input sample to the medoids 
	Distances.pairwise(model.data[1], model.data[2], x)
end





##########################
# FunctionCell Interface #	
##########################
"""
	kmedoids!(medoids, d=Distances.Euclidean() [;kwargs])

Constructs an untrained cell that when piped data inside, clusters the data using the
k-medoids algorithm where `medoids` contains the initial seed observation indices 
(will be updated inplace) and `d` is the distance that will be used to calculate distances 
from new samples to the clustering generated from the untrained function cell.

# Keyword arguments (same as in `Clustering`)
  * `maxiter` maximum number of iterations (default `100`)
  * `tol` tolerable change in objective at convergence (default `1.0e-6`)
  * `display` Level of information; can be `:none`, `:final`, `:iter` (default `:none`)
"""
kmedoids!(medoids::Vector{Int}, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = 
	FunctionCell(kmedoids!, (medoids,d), Dict(), kwtitle("K-medoids! ($(length(medoids)) clusters)", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	kmedoids!(x, medoids, d=Distances.Euclidean() [;kwargs])

Performs k-medoids clustering on the dataset `x` using `medoids` as initial seed observation indices 
returning a clustering; when piping data to the resulting clustering (e.g. trained cell), 
the distances from input data to the new medoids are returned, using `d` as distance.
"""
# Training
kmedoids!(x::T where T<:CellData, medoids::Vector{Int}, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = kmedoids!(getx!(x), medoids, d; kwargs...)
kmedoids!(x::T where T<:AbstractVector, medoids::Vector{Int}, d::Distances.PreMetric=Distances.Euclidean() ; kwargs...) = 
	error("[kmedoids!] Vectors are not supported as input data. Try using `v |> dist(v) |> kmedoids!(medoids)`") 
kmedoids!(x::T where T<:AbstractMatrix, medoids::Vector{Int}, d::Distances.PreMetric=Distances.Euclidean() ; kwargs...) = begin
	
	# Perform clustering 
	kmeddata = Clustering.kmedoids!(x, medoids; kwargs...) 
	
	# Build model properties 
	modelprops = Dict("size_in" => length(kmeddata.acosts),
		   	  "size_out" => length(kmeddata.medoids) 
	)
	
	# Return a trained cell that uses `kmedoids` and not `kmedoids!` as execution function 
	# since the only difference is in training (e.g. clustering) and not in
	# assigning new data to clusters.
	FunctionCell(kmedoids, Model((d, x[:,sort(kmeddata.medoids)], kmeddata)), modelprops, kwtitle("K-medoids ($(length(medoids)) clusters)", kwargs) );	 
end
