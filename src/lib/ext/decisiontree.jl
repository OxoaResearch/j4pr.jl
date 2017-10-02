##########################
# FunctionCell Interface #	
##########################
"""
	tree(nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0  [;kwargs])

Generates a function cell that when piped data, trains a decision tree classifier using the DecisionTree.jl interface.

# Arguments (most from `DecisionTree`)
  * `nsubfeatures` number of features to consider for node thresholding (default 0 e.g. all)
  * `maxdepth` maximum tree depth (default -1 e.g. grow fully)
  * `prune_purity` fraction in [0.0,1.0] of combined purity for leaves to be merged (-1.0 indicates no pruning)

# Keyword arguments (from `DecisionTree`)
  * `rng::AbstractRNG=Base.GLOBAL_RNG` is the random number generator

Read the `DecisionTree.jl` documentation for more information.  
"""
tree(nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0; kwargs...) = 
	FunctionCell(tree, (nsubfeatures, maxdepth, prune_purity), ModelProperties(), 
	      kwtitle("Decision Tree (classifier)", kwargs);  kwargs...)



############################
# DataCell/Array Interface #	
############################
"""
	tree(x, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0  [;kwargs])

Generates a trained function cell by training a decision tree classifier using the DecisionTree.jl interface. 
"""
# Training
tree(x::T where T<:CellDataL, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0; kwargs...) = 
	tree((getx!(x), gety(x)), nsubfeatures, maxdepth, prune_purity; kwargs...)

tree(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0; kwargs...) = 
	tree((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), nsubfeatures, maxdepth, prune_purity; kwargs...)

tree(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[tree] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)
	
	# Train model
	treedata = DecisionTree.build_tree(yenc, getobs(x[1])', nsubfeatures, maxdepth; kwargs...) 

	# Prune if necessary
	if prune_purity >=0
		treedata = DecisionTree.prune_tree(treedata, prune_purity)
	end

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(tree, Model(treedata, modelprops), kwtitle("Decision Tree (classifier)", kwargs))	 
end



# Execution
tree(x::T where T<:CellData, model::Model{<:DecisionTree.LeafOrNode}) =
	datacell(tree(getx!(x), model), gety(x)) 	
tree(x::T where T<:AbstractVector, model::Model{<:DecisionTree.LeafOrNode}) =
	tree(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
tree(x::T where T<:AbstractMatrix, model::Model{<:DecisionTree.LeafOrNode}) =
	# Always return probabilities
	DecisionTree.apply_tree_proba(model.data, getobs(x)', collect(1:model.properties.odim))'::Matrix{Float64} # labels are integer-coded, just use [1,2, ..., C]





##########################
# FunctionCell Interface #	
##########################
"""
	treer(maxlabels::Int=5, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0  [;kwargs])

Generates a function cell that when piped data, trains a decision tree regressor using the DecisionTree.jl interface.

# Arguments (most from `DecisionTree`)
  * `maxlabels` number of samples used in each leaf to construct estimates (default 5)
  * `nsubfeatures` number of features to consider for node thresholdingi (default 0 e.g. all)
  * `maxdepth` maximum tree depth (default -1 e.g. grow fully)
  * `prune_purity` fraction in [0.0,1.0] of combined purity for leaves to be merged (-1.0 indicates no pruning)

# Keyword arguments (from `DecisionTree`)
  * `rng::AbstractRNG=Base.GLOBAL_RNG` is the random number generator

!!! note
	The effects of `prune_purity` in the case of regression are not quite clear. Should work well for very
	low values e.g. leaves containing many distinct values are combined.

Read the `DecisionTree.jl` documentation for more information.  
"""
treer(maxlabels::Int = 5, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0; kwargs...) = 
	FunctionCell(treer, (maxlabels, nsubfeatures, maxdepth, prune_purity), ModelProperties(), 
	      kwtitle("Decision Tree (regressor)", kwargs);  kwargs...)



############################
# DataCell/Array Interface #	
############################
"""
	treer(x, maxlabels::Int=5, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0  [;kwargs])

Generates a trained function cell by training a decision tree regressor using the DecisionTree.jl interface. 
"""
# Training
treer(x::T where T<:CellDataL, maxlabels::Int=5, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0; kwargs...) = 
	treer((getx!(x), gety(x)), maxlabels, nsubfeatures, maxdepth, prune_purity; kwargs...)

treer(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, maxlabels::Int=5, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0; kwargs...) = 
	treer((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), maxlabels, nsubfeatures, maxdepth, prune_purity; kwargs...)

treer(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, maxlabels::Int=5, nsubfeatures::Int=0, maxdepth::Int=-1, prune_purity::Real=-1.0; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[treer] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Convert targets to vector of Float first 
	yenc = Vector{typeof(1.0)}(vec(x[2])) 
	
	# Train model
	treedata = DecisionTree.build_tree(yenc, getobs(x[1])', maxlabels, nsubfeatures, maxdepth; kwargs...) 

	# Prune if necessary
	if prune_purity >=0
		treedata = DecisionTree.prune_tree(treedata, prune_purity)
	end

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]),1)
	
	FunctionCell(treer, Model(treedata, modelprops), kwtitle("Decision Tree (regressor)", kwargs))	 
end



# Execution
treer(x::T where T<:CellData, model::Model{<:DecisionTree.LeafOrNode}) =
	datacell(treer(getx!(x), model), gety(x)) 	
treer(x::T where T<:AbstractVector, model::Model{<:DecisionTree.LeafOrNode}) =
	treer(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
treer(x::T where T<:AbstractMatrix, model::Model{<:DecisionTree.LeafOrNode})::Matrix{Float64} =
	Matrix(DecisionTree.apply_tree(model.data, getobs(x)')')





##########################
# FunctionCell Interface #	
##########################
"""
	randomforest(ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1 [;kwargs])

Generates a function cell that when piped data, trains a random forest classifier using the DecisionTree.jl interface.
The trees are not pruned.

# Arguments (from `DecisionTree`)
  * `ntrees` number of trees (default 50)
  * `nsubfeatures` number of features to consider for node thresholding (default 0 e.g. all)
  * `partialsampling` is the random fraction in [0.0,1.0] of training samples to be used for each tree (default 0.7)
  * `maxdepth` maximum tree depth (default -1 e.g. grow fully)

# Keyword arguments (from `DecisionTree`)
  * `rng::AbstractRNG=Base.GLOBAL_RNG` is the random number generator

Read the `DecisionTree.jl` documentation for more information.  
"""
randomforest(ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1; kwargs...) = 
	FunctionCell(randomforest, (ntrees, nsubfeatures, partialsampling, maxdepth), ModelProperties(), 
	      	     kwtitle("Random Forest (classifier, $ntrees trees)", kwargs);  kwargs...)



############################
# DataCell/Array Interface #	
############################
"""
	randomforest(x, ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1 [;kwargs])

Generates a trained function cell by training a random forest classifier using the DecisionTree.jl interface. 
"""
# Training
randomforest(x::T where T<:CellDataL, ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1; kwargs...) = 
	randomforest((getx!(x), gety(x)), ntrees, nsubfeatures, partialsampling, maxdepth; kwargs...)

randomforest(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1; kwargs...) = 
	randomforest((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), ntrees, nsubfeatures, partialsampling, maxdepth; kwargs...)

randomforest(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1; kwargs...) = 
begin	
	@assert nobs(x[1]) == nobs(x[2]) "[randomforest] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)

	# Train model
	rforestdata = DecisionTree.build_forest(yenc, getobs(x[1])', nsubfeatures, ntrees, partialsampling, maxdepth; kwargs...) 

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc) 
	
	FunctionCell(randomforest, Model(rforestdata, modelprops), kwtitle("Random Forest (classifier, $ntrees trees)", kwargs))	 
end



# Execution
randomforest(x::T where T<:CellData, model::Model{<:DecisionTree.Ensemble}) =
	datacell(randomforest(getx!(x), model), gety(x)) 	
randomforest(x::T where T<:AbstractVector, model::Model{<:DecisionTree.Ensemble}) =
	randomforest(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
randomforest(x::T where T<:AbstractMatrix, model::Model{<:DecisionTree.Ensemble})::Matrix{Float64} =
	DecisionTree.apply_forest_proba(model.data, getobs(x)', collect(1:model.properties.odim))' # labels are integer-coded, just use [1,2, ..., C]





##########################
# FunctionCell Interface #	
##########################
"""
	randomforestr(ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1, maxlabels::Int=5 [;kwargs])

Generates a function cell that when piped data, trains a random forest regressor using the DecisionTree.jl interface.
The trees are not pruned.

# Arguments (from `DecisionTree`)
  * `ntrees` number of trees (default 50)
  * `nsubfeatures` number of features to consider for node thresholding (default 0 e.g. all)
  * `partialsampling` is the random fraction in [0.0,1.0] of training samples to be used for each tree (default 0.7)
  * `maxdepth` maximum tree depth (default -1 e.g. grow fully)
  * `maxlabels` number of samples used in each leaf to construct estimates (default 5)

# Keyword arguments (from `DecisionTree`)
  * `rng::AbstractRNG=Base.GLOBAL_RNG` is the random number generator

Read the `DecisionTree.jl` documentation for more information.  
"""
randomforestr(ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1, maxlabels::Int=5; kwargs...) = 
	FunctionCell(randomforestr, (ntrees, nsubfeatures, partialsampling, maxdepth, maxlabels), ModelProperties(), 
	             kwtitle("Random Forest (regressor, $ntrees trees)", kwargs);  kwargs...)



############################
# DataCell/Array Interface #	
############################
"""
	randomforestr(x, ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1, maxlabels::Int=5 [;kwargs])

Generates a trained function cell by training a random forest regressor using the DecisionTree.jl interface. 
"""
# Training
randomforestr(x::T where T<:CellDataL, ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1, maxlabels::Int=5; kwargs...) = 
	randomforestr((getx!(x), gety(x)), ntrees, nsubfeatures, partialsampling, maxdepth, maxlabels; kwargs...)

randomforestr(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1, maxlabels::Int=5; kwargs...) = 
	randomforestr((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), ntrees, nsubfeatures, partialsampling, maxdepth, maxlabels; kwargs...)

randomforestr(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, ntrees::Int=50, nsubfeatures::Int=0, partialsampling::Float64=0.7, maxdepth::Int=-1, maxlabels::Int=5; kwargs...) = 
begin	
	@assert nobs(x[1]) == nobs(x[2]) "[randomforestr] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Convert targets to vector of Float first 
	yenc = Vector{typeof(1.0)}(vec(x[2])) 

	# Train model
	rforestdata = DecisionTree.build_forest(yenc, getobs(x[1])', nsubfeatures, ntrees, maxlabels, partialsampling, maxdepth; kwargs...) 

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]),1)
	
	FunctionCell(randomforestr, Model(rforestdata, modelprops), kwtitle("Random Forest (regressor, $ntrees trees)", kwargs))	 
end



# Execution
randomforestr(x::T where T<:CellData, model::Model{<:DecisionTree.Ensemble}) =
	datacell(randomforestr(getx!(x), model), gety(x)) 	
randomforestr(x::T where T<:AbstractVector, model::Model{<:DecisionTree.Ensemble}) =
	randomforestr(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
randomforestr(x::T where T<:AbstractMatrix, model::Model{<:DecisionTree.Ensemble}) =
	Matrix(DecisionTree.apply_forest(model.data, getobs(x)')')





##########################
# FunctionCell Interface #	
##########################
"""
	adaboostump(niterations::Int=10 [;kwargs])

Generates a function cell that when piped data, trains an Adaboosted-stump classifier using the DecisionTree.jl interface.

# Arguments (most from `DecisionTree`)
  * `niterations` number of iterations e.g ensemble size (default 10)

# Keyword arguments (from `DecisionTree`)
  * `rng::AbstractRNG=Base.GLOBAL_RNG` is the random number generator

Read the `DecisionTree.jl` documentation for more information.  
"""
adaboostump(niterations::Int=10; kwargs...) = 
	FunctionCell(adaboostump, (niterations,), ModelProperties(), 
	      kwtitle("Adaboost Stumps (classifier, $niterations stumps)", kwargs);  kwargs...)



############################
# DataCell/Array Interface #	
############################
"""
	adaboostump(x, niterations::Int=10 [;kwargs])

Generates a trained function cell by training an Adaboosted-stump classifier using the DecisionTree.jl interface. 
"""
# Training
adaboostump(x::T where T<:CellDataL, niterations::Int=10; kwargs...) = adaboostump((getx!(x), gety(x)), niterations; kwargs...)

adaboostump(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, niterations::Int=10; kwargs...) = 
	adaboostump((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), niterations; kwargs...)

adaboostump(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, niterations::Int=10; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[adaboostump] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)
	
	# Train model
	stumpdata = DecisionTree.build_adaboost_stumps(yenc, getobs(x[1])', niterations; kwargs...) 

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(adaboostump, Model(stumpdata, modelprops), kwtitle("Adaboost Stumps (classifier, $niterations stumps)", kwargs));  	 
end



# Execution
adaboostump(x::T where T<:CellData, model::Model{<:Tuple{DecisionTree.Ensemble,Vector}}) =
	datacell(adaboostump(getx!(x), model), gety(x)) 	
adaboostump(x::T where T<:AbstractVector, model::Model{<:Tuple{DecisionTree.Ensemble,Vector}}) =
	adaboostump(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
adaboostump(x::T where T<:AbstractMatrix, model::Model{<:Tuple{DecisionTree.Ensemble,Vector}}) =
	DecisionTree.apply_adaboost_stumps_proba(model.data[1], model.data[2], 
					  getobs(x)', collect(1:model.properties.odim))'::Matrix{Float64}
