##########################
# FunctionCell Interface #	
##########################
"""
	ppca([;kwargs])

Constructs an untrained cell that when piped data inside, calculates the 
probabilistic PCA projection matrix of the input data. We assume that `d` 
is the dimesionality of the input dataset, `n` the number of samples

# Keyword arguments (same as in `Distributions`)
  * `method` is the core algorithm used: `:ml`, `:em` and `:bayes` (default `:ml`)
  * `maxoutdim` is the number of output dimensions (default `<number of variables>-1`)
  * `mean` is the mean vector; can be `nothing`, `0` or precomputed mean (default `nothing`)
  * `tol` convergence tolerance (default `1.0e-6`)
  * `tot` maximum number of iterations (default `1000`)

Read the `Distributions.jl` documentation for more information.  
"""
ppca(;kwargs...) = FunctionCell(ppca, (), Dict(), kwtitle("PPCA", kwargs);kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	ppca(x, [;kwargs])

Trains a function cell that when piped data into will compute the principal components of
the observations based on the projection matrix extracted from `x`.
"""
# Training
ppca(x::T where T<:CellData; kwargs...) = ppca(getx!(x);kwargs...)
ppca(x::T where T<:AbstractVector; kwargs...) = ppca(mat(x, LearnBase.ObsDim.Constant{2}());kwargs...)
ppca(x::T where T<:AbstractMatrix; kwargs...)  = begin
		
	ppcadata = fit(MultivariateStats.PPCA, getobs(x); kwargs...)

	# Build model properties
	modelprops = Dict("size_in" => size(ppcadata.W,1), 					# Size of the input data
		   	  "size_out" => size(ppcadata.W,2),
	)
	
	# Returned trained cell
	FunctionCell(ppca, Model(ppcadata), modelprops, kwtitle("PPCA",kwargs))	 
end



# Execution
ppca(x::T where T<:CellData, model::Model{<:MultivariateStats.PPCA}, modelprops::Dict) = datacell(ppca(getx!(x), model, modelprops), gety(x)) 	
ppca(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.PPCA}, modelprops::Dict) = ppca(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
ppca(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.PPCA}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	MultivariateStats.transform(model.data, getobs(x))
end



"""
	ppcar(M::Cell), where `M=ppca(x)`

Trains a cell based on a previously trained PPCA function cell. When piped data into, 
the returned output will be the reconstruction of the observations in the original
space. Obviously, the input data (e.g. data to be reconstructed) is expected to be
principal components extracted with `M`.

Read the `Distributions.jl` documentation for more information.  
"""
# Training for the reconstruction tranform (fixed cell)
ppcar(x::T where T<:CellFunT{<:Model{<:MultivariateStats.PPCA}}) = begin
	
	# Build model properties
	modelprops = Dict("size_in" => gety(x)["size_out"],
		   	  "size_out"=> gety(x)["size_in"]
	)
	FunctionCell(ppcar, getx(x), modelprops, "PPCA: reconstruct")
end



# Execution
ppcar(x::T where T<:CellData, model::Model{<:MultivariateStats.PPCA}, modelprops::Dict) = datacell(ppcar(getx!(x), model, modelprops), gety(x)) 	
ppcar(x::T where T<:AbstractVector, model::Model{<:MultivariateStats.PPCA}, modelprops::Dict) = ppcar(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
ppcar(x::T where T<:AbstractMatrix, model::Model{<:MultivariateStats.PPCA}, modelprops::Dict)::Matrix{Float64} = MultivariateStats.reconstruct(model.data, getobs(x))
