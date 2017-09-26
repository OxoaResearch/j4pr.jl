###########################
# Function Cell Interface #	
###########################
"""
	pca([;kwargs])

Constructs an untrained cell that when piped data inside, calculates the PCA
projection matrix of the input data. We assume that `d` is the dimesionality
of the input dataset, `n` the number of samples

# Keyword arguments (same as in `MultivariateStats`)
  * `method` is the PCA method used `:auto`, `:cov`, `:svd` (default `:auto` e.g. use `:cov` when the number of dimensions
  smaller than the number of samples, `svd` otherwise)
  * `maxoutdim` is the number of output dimensions, default `<number of variables>`
  * `pratio` is the ratio of variaces preserved in the principal subspace (default `0.99`)
  * `mean` is the mean vector, can be `nothing`, `0` or precomputed mean vector (default `nothing`)

Read the `MultivariateStats.jl` documentation for more information.  
"""
pca(;kwargs...) = FunctionCell(pca, (), Dict(), kwtitle("PCA", kwargs);kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	pca(x, [;kwargs])

Trains a function cell that when piped data into will compute the principal components of
the observations based on the projection matrix extracted from `x`.
"""
# Training
pca(x::T where T<:CellData; kwargs...) = pca(getx!(x);kwargs...)
pca(x::T where T<:AbstractVector; kwargs...) = pca(mat(x, LearnBase.ObsDim.Constant{2}());kwargs...)
pca(x::T where T<:AbstractMatrix; kwargs...)  = begin
	
	ppadata = fit(MultivariateStats.PCA, getobs(x); kwargs...)

	# Build model properties
	modelprops = Dict("size_in" => size(ppadata.proj,1), 					# Size of the input data
		   	  "size_out" => size(ppadata.proj,2),
	)
	
	# Returned trained cell
	FunctionCell(pca, Model(ppadata), modelprops, kwtitle("PCA",kwargs))	 
end



# Execution
pca(x::T where T<:CellData, model::Model{<:MultivariateStats.PCA}, modelprops::Dict) = datacell(pca(getx!(x), model, modelprops), gety(x)) 	
pca(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.PCA}, modelprops::Dict) = pca(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
pca(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.PCA}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	MultivariateStats.transform(model.data, getobs(x))
end



"""
	pcar(M::Cell), where `M=pca(x)`

Trains a cell based on a previously trained PCA function cell. When piped data into, 
the returned output will be the reconstruction of the observations in the original
space. Obviously, the input data (e.g. data to be reconstructed) is expected to be
principal components extracted with `M`.

Read the `MultivariateStats.jl` documentation for more information.  

# Examples
```
julia> P= rand(5,1000) |> pca(maxoutdim=4)
PCA: maxoutdim=4, 5 -> 4, trained

julia> PR = pcar(P)
PCA: reconstruct, 4 -> 5, trained

julia> ([1.0 2 3 4 5]' |> P ) |> pcar(P)
5×1 Array{Float64,2}:
 0.305402
 1.16768 
 2.25037 
 5.23595 
 1.77198 
```
"""
# Training for the reconstruction tranform (fixed cell)
pcar(x::T where T<:CellFunT{<:Model{<:MultivariateStats.PCA}}) = begin
	
	# Build model properties
	modelprops = Dict("size_in" => gety(x)["size_out"],
		   	  "size_out"=> gety(x)["size_in"]
	)
	FunctionCell(pcar, getx(x), modelprops,"PCA: reconstruct")
end



# Execution
pcar(x::T where T<:CellData, model::Model{<:MultivariateStats.PCA}, modelprops::Dict) = datacell(pcar(getx!(x), model, modelprops), gety(x)) 	
pcar(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.PCA}, modelprops::Dict) = pcar(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
pcar(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.PCA}, modelprops::Dict)::Matrix{Float64} = MultivariateStats.reconstruct(model.data, getobs(x))
