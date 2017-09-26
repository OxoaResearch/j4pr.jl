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
	FunctionCell(tree, (nsubfeatures, maxdepth, prune_purity), Dict(), kwtitle("Decision Tree (classifier)", kwargs);  kwargs...)



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

	# Transform labels first (this makes the labels usable even if they are floats)
	yu = sort(unique(x[2]))
	yenc = Vector{Int}(ohenc_integer(x[2],yu)) # encode to Int labels based on position in the sorted vector of unique labels 

	# Train model
	treedata = DecisionTree.build_tree(yenc, getobs(x[1])', nsubfeatures, maxdepth; kwargs...) 

	# Prune if necessary
	if prune_purity >=0
		treedata = DecisionTree.prune_tree(treedata, prune_purity)
	end

	# Build model properties 
	modelprops = Dict("size_in" => nvars(x[1]),
		   	  "size_out" => length(yu),
			  "labels" => yu 
	)
	
	FunctionCell(tree, Model(treedata), modelprops, kwtitle("Decision Tree (classifier)", kwargs))	 
end



# Execution
tree(x::T where T<:CellData, model::Model{<:DecisionTree.LeafOrNode}, modelprops::Dict) = datacell(tree(getx!(x), model, modelprops), gety(x)) 	
tree(x::T where T<:AbstractVector, model::Model{<:DecisionTree.LeafOrNode}, modelprops::Dict) = tree(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
tree(x::T where T<:AbstractMatrix, model::Model{<:DecisionTree.LeafOrNode}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	# Always return probabilities
	DecisionTree.apply_tree_proba(model.data, getobs(x)', collect(1:modelprops["size_out"]))'::Matrix{Float64} # labels are integer-coded, just use [1,2, ..., C]
end





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
	FunctionCell(treer, (maxlabels, nsubfeatures, maxdepth, prune_purity), Dict(), kwtitle("Decision Tree (regressor)", kwargs);  kwargs...)



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
	modelprops = Dict("size_in" => nvars(x[1]),
		   	  "size_out" => 1
	)
	
	FunctionCell(treer, Model(treedata), modelprops, kwtitle("Decision Tree (regressor)", kwargs))	 
end



# Execution
treer(x::T where T<:CellData, model::Model{<:DecisionTree.LeafOrNode}, modelprops::Dict) = datacell(treer(getx!(x), model, modelprops), gety(x)) 	
treer(x::T where T<:AbstractVector, model::Model{<:DecisionTree.LeafOrNode}, modelprops::Dict) = treer(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
treer(x::T where T<:AbstractMatrix, model::Model{<:DecisionTree.LeafOrNode}, modelprops::Dict)::Matrix{Float64} = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	Matrix(DecisionTree.apply_tree(model.data, getobs(x)')')
end





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
	FunctionCell(randomforest, (ntrees, nsubfeatures, partialsampling, maxdepth), Dict(), 
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

	# Transform labels first (this makes the labels usable even if they are floats)
	yu = sort(unique(x[2]))
	yenc = Vector{Int}(ohenc_integer(x[2],yu)) # encode to Int labels based on position in the sorted vector of unique labels 

	# Train model
	rforestdata = DecisionTree.build_forest(yenc, getobs(x[1])', nsubfeatures, ntrees, partialsampling, maxdepth; kwargs...) 

	# Build model properties 
	modelprops = Dict("size_in" => nvars(x[1]),
		   	  "size_out" => length(yu),
			  "labels" => yu 
	)
	
	FunctionCell(randomforest, Model(rforestdata), modelprops, kwtitle("Random Forest (classifier, $ntrees trees)", kwargs))	 
end



# Execution
randomforest(x::T where T<:CellData, model::Model{<:DecisionTree.Ensemble}, modelprops::Dict) = datacell(randomforest(getx!(x), model, modelprops), gety(x)) 	
randomforest(x::T where T<:AbstractVector, model::Model{<:DecisionTree.Ensemble}, modelprops::Dict) = randomforest(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
randomforest(x::T where T<:AbstractMatrix, model::Model{<:DecisionTree.Ensemble}, modelprops::Dict)::Matrix{Float64} = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	# Always return probabilities
	DecisionTree.apply_forest_proba(model.data, getobs(x)', collect(1:modelprops["size_out"]))' # labels are integer-coded, just use [1,2, ..., C]
end





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
	FunctionCell(randomforestr, (ntrees, nsubfeatures, partialsampling, maxdepth, maxlabels), Dict(), 
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
	modelprops = Dict("size_in" => nvars(x[1]),
		   	  "size_out" => 1,
	)
	
	FunctionCell(randomforestr, Model(rforestdata), modelprops, kwtitle("Random Forest (regressor, $ntrees trees)", kwargs))	 
end



# Execution
randomforestr(x::T where T<:CellData, model::Model{<:DecisionTree.Ensemble}, modelprops::Dict) = datacell(randomforestr(getx!(x), model, modelprops), gety(x)) 	
randomforestr(x::T where T<:AbstractVector, model::Model{<:DecisionTree.Ensemble}, modelprops::Dict) = randomforestr(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
randomforestr(x::T where T<:AbstractMatrix, model::Model{<:DecisionTree.Ensemble}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	Matrix(DecisionTree.apply_forest(model.data, getobs(x)')')
end





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
adaboostump(niterations::Int=10; kwargs...) = FunctionCell(adaboostump, (niterations,), Dict(), 
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

	# Transform labels first (this makes the labels usable even if they are floats)
	yu = sort(unique(x[2]))
	yenc = Vector{Int}(ohenc_integer(x[2],yu)) # encode to Int labels based on position in the sorted vector of unique labels 

	# Train model
	stumpdata = DecisionTree.build_adaboost_stumps(yenc, getobs(x[1])', niterations; kwargs...) 

	# Build model properties 
	modelprops = Dict("size_in" => nvars(x[1]),
		   	  "size_out" => length(yu),
			  "labels" => yu 
	)
	
	FunctionCell(adaboostump, Model(stumpdata), modelprops, kwtitle("Adaboost Stumps (classifier, $niterations stumps)", kwargs));  	 
end



# Execution
adaboostump(x::T where T<:CellData, model::Model{<:Tuple{DecisionTree.Ensemble,Vector}}, modelprops::Dict) = datacell(adaboostump(getx!(x), model, modelprops), gety(x)) 	
adaboostump(x::T where T<:AbstractVector, model::Model{<:Tuple{DecisionTree.Ensemble,Vector}}, modelprops::Dict) = adaboostump(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
adaboostump(x::T where T<:AbstractMatrix, model::Model{<:Tuple{DecisionTree.Ensemble,Vector}}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	# Always return probabilities
	DecisionTree.apply_adaboost_stumps_proba(model.data[1], model.data[2], getobs(x)', collect(1:modelprops["size_out"]))'::Matrix{Float64}
end
