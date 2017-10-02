###########################
# Function Cell Interface #	
###########################
"""
	fa([;kwargs])

Constructs an untrained cell that when piped data inside, trains a 
Factor Analysis transform. We assume that `d` is the dimesionality
of the input dataset, `n` the number of samples

# Keyword arguments (same as in `MultivariateStats`)
* `method` is the FA method used `:em` and `:cm` (default `:cm`)
* `maxoutdim` is the number of output dimensions, default `d-1`
* `mean` is the mean vector, can be `nothing`, `0` or precomputed mean vector (default `nothing`)
* `tol` is the convergence tolerance (default `1.0e-6`)
* `tot` is the maximum number of iterations (default `1000`)
* `η` is the variabce low bound (default `1.0e-6`)

Read the `MultivariateStats.jl` documentation for more information.  
"""
fa(;kwargs...) = FunctionCell(fa, (), ModelProperties(), kwtitle("Factor Analysis", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	fa(x, [;kwargs])

Trains a function cell that when piped data into will compute the principal components of
the observations based on the projection matrix extracted from `x`.
"""
# Training
fa(x::T where T<:CellData; kwargs...) = fa(getx!(x); kwargs...)
fa(x::T where T<:AbstractVector; kwargs...) = fa(mat(x, LearnBase.ObsDim.Constant{2}()); kwargs...)
fa(x::T where T<:AbstractMatrix; kwargs...)  = begin
	
	fadata = fit(MultivariateStats.FactorAnalysis, getobs(x); kwargs...)

	# Build model properties
	modelprops = ModelProperties(length(fadata.mean),size(fadata.W,2))
	
	# Returned trained cell
	FunctionCell(fa, Model(fadata, modelprops), kwtitle("Factor Analysis",kwargs))	 
end



# Execution
fa(x::T where T<:CellData, model::Model{<:MultivariateStats.FactorAnalysis}) =
	datacell(fa(getx!(x), model), gety(x)) 	
fa(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.FactorAnalysis}) =
	fa(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
fa(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.FactorAnalysis}) = 
	MultivariateStats.transform(model.data, getobs(x))



"""
	far(M::Cell), where `M=fa(x)`

Trains a cell based on a previously trained Factor Analysis function cell. When piped data into, 
the returned output will be the reconstruction of the observations in the original
space.

Read the `MultivariateStats.jl` documentation for more information.  

# Examples
```
julia> using j4pr;

julia> F = randn(5,1000) |> fa(maxoutdim=2)
Factor Analysis: maxoutdim=2, 5->2, trained

julia> FR = far(F)
Factor Analysis: reconstruct, 2->5, trained

julia> ([1.0 2 3 4 5]' |> P) |> PR
5×1 Array{Float64,2}:
 0.45788
 2.09689
 3.04864
 3.9806 
 4.998
```
"""
# Training for the reconstruction tranform (fixed cell)
far(x::T where T<:CellFunT{<:Model{<:MultivariateStats.FactorAnalysis}}) = begin
	
	# Build model properties
	modelprops = ModelProperties(getx!(x).properties.odim, getx!(x).properties.idim)
	
	FunctionCell(far, Model(getx(x).data, modelprops), "Factor Analysis: reconstruct")
end



# Execution
far(x::T where T<:CellData, model::Model{<:MultivariateStats.FactorAnalysis}) =
	datacell(far(getx!(x), model), gety(x)) 	
far(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.FactorAnalysis}) =
	far(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
far(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.FactorAnalysis})::Matrix{Float64} =
	MultivariateStats.reconstruct(model.data, getobs(x))
