###########################
# Function Cell Interface #	
###########################
"""
	kpca([;kwargs])

Constructs an untrained cell that when piped data inside, calculates the Kernel-PCA
projection matrix of the input data. We assume that `d` is the dimesionality
of the input dataset, `n` the number of samples

# Keyword arguments (same as in `MultivariateStats`)
  * `kernel` the kernel; a function that accepts two vector arguments `x` and `y`, and returns a scalar value (default `(x,y)->x'y`)
  * `solver` is the solver. Can be `:eig` uses `eigfact`, `:eigs` uses `eigs` (default `:eig`, for sparse data, `:eigs` is always used)
  smaller than the number of samples, `svd` otherwise)
  * `maxoutdim` is the number of output dimensions, default `min(<number of dimensions>,<number of samples>)`
  * `inverse` specified whether to perform calculation for inverse transform for non-precomputed kernelsi (default `false`)
  * `β` is a hyperparameter of the ridge regression that learns the inverse transform, when inverse is true (default `1.0`)
  * `tol` convergence tolerance for eigs solver (default `0.0`)
  * `maxiter` maximum number of iterations for eigs solver (default `300`)

# Kernels
List of the commonly used kernels:
    function 				description
    `(x,y)->x'y` 			 `Linear`
    `(x,y)->(x'y+c)^d` 			 `Polynomial`
    `(x,y)->exp(-γ*norm(x-y)^2.0)` 	 `Radial basis function (RBF)`

Read the `MultivariateStats.jl` documentation for more information.
"""
kpca(;kwargs...) = FunctionCell(kpca, (), ModelProperties(), kwtitle("Kernel-PCA", kwargs);kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	kpca(x, [;kwargs])

Trains a function cell that when piped data into will compute the principal components of
the observations based on the .
"""
# Training
kpca(x::T where T<:CellData; kwargs...) = kpca(getx!(x);kwargs...)
kpca(x::T where T<:AbstractVector; kwargs...) = kpca(mat(x, LearnBase.ObsDim.Constant{2}());kwargs...)
kpca(x::T where T<:AbstractMatrix; kwargs...)  = begin
	
	kpcadata = fit(MultivariateStats.KernelPCA, getobs(x); kwargs...)

	# Build model properties
	modelprops = ModelProperties(MultivariateStats.indim(kpcadata), 
				     MultivariateStats.outdim(kpcadata))
	
	# Returned trained cell
	FunctionCell(kpca, Model(kpcadata, modelprops), kwtitle("Kernel-PCA",kwargs))	 
end



# Execution
kpca(x::T where T<:CellData, model::Model{<:MultivariateStats.KernelPCA}) =
	datacell(kpca(getx!(x), model), gety(x)) 	
kpca(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.KernelPCA}) =
	kpca(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
kpca(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.KernelPCA}) =
	MultivariateStats.transform(model.data, getobs(x))





"""
	kpcar(M::Cell), where `M=kpca(x)`

Trains a cell based on a previously trained Kernel-PCA function cell. When piped data into, 
the returned output will be the reconstruction of the observations in the original
space. Obviously, the input data (e.g. data to be reconstructed) is expected to be
principal components extracted with `M`.

Read the `MultivariateStats.jl` documentation for more information.  
```
"""
# Training for the reconstruction tranform (fixed cell)
kpcar(x::T where T<:CellFunT{<:Model{<:MultivariateStats.KernelPCA}}) = begin
	
	# Build model properties
	modelprops = ModelProperties(getx(x).properties.odim, getx(x).properties.idim)

	FunctionCell(kpcar, Model(getx(x).data, modelprops),"Kernel-PCA: reconstruct")
end



# Execution
kpcar(x::T where T<:CellData, model::Model{<:MultivariateStats.KernelPCA}) = datacell(kpcar(getx!(x), model), gety(x)) 	
kpcar(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.KernelPCA}) = kpcar(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
kpcar(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.KernelPCA})::Matrix{Float64} = MultivariateStats.reconstruct(model.data, getobs(x))
