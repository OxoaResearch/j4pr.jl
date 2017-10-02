##########################
# FunctionCell Interface #	
##########################
"""
	lda(distances=true [;kwargs])

Constructs an untrained cell that when piped data inside, returns a LDA
model based on the input data and labels. The argument `distances` indicates
whether execution of the model will result in distances (from test observations
to the class-wise mean vectors of the transformed reference space) or the transformed
space of the test objects (distances can be subsequently calculated using this data
and the model).

# Keyword arguments (same as in `MultivariateStats`)
  * `method` can be `:gevd` e.g. generalized eigenvalue decomp or `:whiten` e.g. whiten first (default `:gevd`)
  * `outdim` output dimension e.g. of the transformed space (default `min(<number of variables>, <number_of_classes>-1)`
  * `regcoef` regularization coefficient, improves numerical stability (default `1e-6`)

Read the `MultivariateStats.jl` documentation for more information.  
"""
lda(distances::Bool=true; kwargs...) = FunctionCell(lda, (distances,), ModelProperties(), kwtitle(distances ? "LDA (distance)" : "LDA (transform)", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	lda(x, distances=true [;kwargs])

Trains a LDA model that when executed will compute either the distances from each input
observation to the mean vectors of `x` in the transformed space (distances = true)
or alternatively, the LDA transform of the input observations. 

"""
# Training
lda(x::T where T<:CellDataL, distances::Bool=true; kwargs...) = lda((getx!(x), gety(x)), distances; kwargs...)
lda(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, distances::Bool=true; kwargs...) = lda((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), distances; kwargs...)
lda(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, distances::Bool=true; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[lda] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)
	
	# Train model
	ldadata = fit(MultivariateStats.MulticlassLDA, length(enc.label), getobs(x[1]), yenc; kwargs...)

	# Build model properties 
	modelprops = ModelProperties(ldadata.stats.dim, distances ? length(enc.label) : size(ldadata.proj,2), enc) 
	
	FunctionCell(lda, Model((distances,ldadata), modelprops), kwtitle(distances ? "LDA (distance)" : "LDA (transform)", kwargs))	 
end



# Execution
lda(x::T where T<:CellData, model::Model{<:Tuple{<:Bool,<:MultivariateStats.MulticlassLDA}}) =
	datacell(lda(getx!(x), model), gety(x)) 	
lda(x::T where T<:AbstractVector, model::Model{<:Tuple{<:Bool,<:MultivariateStats.MulticlassLDA}}) =
	lda(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
lda(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:Bool,<:MultivariateStats.MulticlassLDA}}) = begin
	
	if model.data[1] == true 
		# Return distances between the mean class vectors and each sample in x 
		Distances.pairwise(Distances.Euclidean(), model.data[2].pmeans, MultivariateStats.transform(model.data[2], getobs(x)))
	else
		# Return the transformed observations   
		MultivariateStats.transform(model.data[2], getobs(x))
	end
end





##########################
# FunctionCell Interface #	
##########################
"""
	ldasub(distances=true [;kwargs])

Constructs an untrained cell that when piped data inside, returns a LDA subspace
model based on the input data and labels. The argument `distances` indicates
whether execution of the model will result in distances (from test observations
to the class-wise mean vectors of the transformed reference space) or the transformed
space of the test objects (distances can be subsequently calculated using this data
and the model).

# Keyword arguments (same as in `MultivariateStats`)
  * `normalize` regularization coefficient, improves numerical stability (default `false`)

Read the `MultivariateStats.jl` documentation for more information.  
"""
ldasub(distances::Bool=true; kwargs...) = FunctionCell(ldasub, (distances,), ModelProperties(), 
					 kwtitle(distances ? "LDA-Subspace (distance)" : "LDA-Subspace (transform)", kwargs); 
					 kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	ldasub(x, distances=true [;kwargs])

Trains a LDA subspace model that when executed will compute either the distances from each input
observation to the mean vectors of `x` in the transformed space (distances = true)
or alternatively, the LDA transform of the input observations. 

"""
# Training
ldasub(x::T where T<:CellDataL, distances::Bool=true; kwargs...) = ldasub((getx!(x), gety(x)), distances; kwargs...)
ldasub(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, distances::Bool=true; kwargs...) = ldasub((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), distances; kwargs...)
ldasub(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, distances::Bool=true; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2])
	
	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)
	
	# Train model
	ldadata = fit(MultivariateStats.SubspaceLDA, getobs(x[1]), yenc, length(enc.label); kwargs...)

	# Build model properties 
	modelprops = ModelProperties(size(ldadata.cmeans,1), distances ? length(enc.label) : size(ldadata.projLDA,2), enc)
	
	FunctionCell(ldasub, Model((distances,ldadata), modelprops), kwtitle(distances ? "LDA-Subspace (distance)" : "LDA-subspace (transform)", kwargs))	 
end



# Execution
ldasub(x::T where T<:CellData, model::Model{<:Tuple{<:Bool,<:MultivariateStats.SubspaceLDA}}) =
	datacell(ldasub(getx!(x), model), gety(x)) 	
ldasub(x::T where T<:AbstractVector, model::Model{<:Tuple{<:Bool,<:MultivariateStats.SubspaceLDA}}) =
	ldasub(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
ldasub(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:Bool,<:MultivariateStats.SubspaceLDA}}) = begin
	
	if model.data[1] == true 
		# Return distances between the mean class vectors and each sample in x 
		Distances.pairwise(Distances.Euclidean(), 
		     MultivariateStats.transform(model.data[2], model.data[2].cmeans), 
		     MultivariateStats.transform(model.data[2], getobs(x))
		)
	else
		# Return the transformed observations   
		MultivariateStats.transform(model.data[2], getobs(x))
	end
end
