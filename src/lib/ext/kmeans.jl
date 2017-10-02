##########################
# FunctionCell Interface #	
##########################
"""
	kmeans(k, d=Distances.Euclidean() [;kwargs])

Constructs an untrained cell that when piped data inside, clusters the data using the
k-means algorithm where `k` is the desired number of clusters and `d` is the distance
that will be used to calculate distances from new samples to the clustering generated
from the untrained function cell.

# Keyword arguments (same as in `Clustering`)
  * `init` initialization algorithm; can be `:rand` for random, `:kmpp` for K-medoids++ or `:kmcen` for choosing samples with highest centrality
  or a vector of size 'k' providing the indexes of the initial seeds (default `:kmpp`)
  * `maxiter` maximum number of iterations (default `100`)
  * `tol` tolerable change in objective at convergence (default `1.0e-6`)
  * `weights` can be `nothing` for unit weight or a vector of length `n` for individual sample weights (default `nothing`)
  * `display` Level of information; can be `:none`, `:final`, `:iter` (default `:none`)

Read the `Clustering.jl` documentation for more information.  
"""
kmeans(k::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = FunctionCell(kmeans, (k,d), ModelProperties(), kwtitle("K-means ($k clusters)", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	kmeans(x, k, d=Distances.Euclidean() [;kwargs])

Performs k-means clustering on the dataset `x` using `k` number of clusters and returns the clustering; when piping
data to the resulting clustering (e.g. trained cell), the distances from input data to the center clusters are 
returned, using `d` as distance.
"""
# Training
kmeans(x::T where T<:CellData, k::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = kmeans(getx!(x), k, d; kwargs...)
kmeans(x::T where T<:AbstractVector, k::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = kmeans(mat(x, LearnBase.ObsDim.Constant{2}()), k, d; kwargs...)
kmeans(x::T where T<:AbstractMatrix, k::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = begin
	
	# Perform clustering 
	kmeansdata = Clustering.kmeans(x, k; kwargs...) 
	
	@assert size(kmeansdata.centers,2) == k "[kmeans] The number of resulting clusters differes from the desired one k=$k."

	# Build model properties 
	modelprops = ModelProperties(size(kmeansdata.centers,1),k) # output dimension is 'k' i.e. the number of cluster centers is 'k'
	
	FunctionCell(kmeans, Model((d,kmeansdata), modelprops), kwtitle("K-means ($k clusters)", kwargs) );	 
end



# Execution
kmeans(x::T where T<:CellData, model::Model{<:Tuple{<:Distances.PreMetric, <:Clustering.KmeansResult}}) =
	datacell(kmeans(getx!(x), model), gety(x)) 	
kmeans(x::T where T<:AbstractVector, model::Model{<:Tuple{<:Distances.PreMetric, <:Clustering.KmeansResult}}) =
	kmeans(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
kmeans(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:Distances.PreMetric, <:Clustering.KmeansResult}}) =
	# Return distances from each input sample to clustering centers 
	Distances.pairwise(model.data[1], model.data[2].centers, x) # d(centers,x)





##########################
# FunctionCell Interface #	
##########################
"""
	kmeans!(centers, d=Distances.Euclidean() [;kwargs])

Constructs an untrained cell that when piped data inside, clusters the data using the
k-means algorithm where `centers` contains the initial cluster centers (will be updated 
inplace) and `d` is the distance that will be used to calculate distances from new samples 
to the clustering generated from the untrained function cell.

# Keyword arguments (same as in `Clustering`)
  * `maxiter` maximum number of iterations (default `100`)
  * `tol` tolerable change in objective at convergence (default `1.0e-6`)
  * `weights` can be `nothing` for unit weight or a vector of length `n` for individual sample weights (default `nothing`)
  * `display` Level of information; can be `:none`, `:final`, `:iter` (default `:none`)

Read the `Clustering.jl` documentation for more information.  
"""
kmeans!(centers::T where T<:AbstractMatrix, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = 
	FunctionCell(kmeans!, (centers,d), ModelProperties(), kwtitle("K-means! ($(size(centers,2)) clusters)", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	kmeans!(x, centers, d=Distances.Euclidean() [;kwargs])

Performs k-means clustering on the dataset `x` using `centers` as initial cluster centers 
and returns the clustering; when piping data to the resulting clustering (e.g. trained cell), 
the distances from input data to the new cluster centers are returned, using `d` as distance.
"""
# Training
kmeans!(x::T where T<:CellData, centers::S where S<:AbstractMatrix, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = kmeans!(getx!(x), centers, d; kwargs...)
kmeans!(x::T where T<:AbstractVector, centers::S where S<:AbstractMatrix, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = kmeans!(mat(x, LearnBase.ObsDim.Constant{2}()), centers, d; kwargs...)
kmeans!(x::T where T<:AbstractMatrix, centers::S where S<:AbstractMatrix, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = begin
	
	# Perform clustering 
	kmeansdata = Clustering.kmeans!(x, centers; kwargs...) 
	
	# Build model properties 
	modelprops = ModelProperties(size(kmeansdata.centers,1), size(kmeansdata.centers,2), nothing, Dict("eval_distance" => d))
	
	# Return a trained cell that uses `kmeans` and not `kmeans!` as execution function 
	# since the only difference is in training (e.g. clustering) and not in
	# assigning new data to clusters.
	FunctionCell(kmeans, Model((d,kmeansdata), modelprops), kwtitle("K-means ($(size(kmeansdata.centers,2)) clusters)", kwargs) );	 
end
