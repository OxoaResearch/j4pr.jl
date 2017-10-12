# Random sub-space ensemble module
module RandomSubspace

	using j4pr.ClassifierCombiner
	using MLDataUtils: getobs, datasubset, ObsDim
	export SubspaceEnsemble, randomsubspace_train, randomsubspace_exec

	"""
	Random subspace ensemble object (it describes the ensemble). 
	
	# Fields
	 * `idx::Vector{Vector{Int}}` is a vector of variable index vectors, 1 vector/ensemble member
	 * `f_train::Function` is the training function
	 * `f_exec::Function` is the execution function
	 * `members::Vector` is a vector containing the trained classifiers 
	 * `combiner::ClassifierCombiner.AbstractCombiner` the means by which ensemble outputs are combined. Non-trainable combiners should be used. If a trainable
	    combiner is to be used, it has to be trained outside the ensemble
	* `parallel_train::Bool` specifies whether to parallelize training or not 
	* `parallel_execution::Bool` specifies whether to parallelize execution or not 
	
	"""
	struct SubspaceEnsemble{F, T<:Vector{S} where S, U<:AbstractCombiner}
		idx::Vector{Vector{Int}}	# vector of vectors of variable indices
		f_train::Function		# training function
		f_exec::F			# execution function
		members::T			# ensemble members (trained)
		combiner::U			# combiner (only non-trainable combiners supported)
		parallel_train::Bool		# whether to parallelize ensemble training or not
		parallel_execution::Bool	# whether to parallelize ensemble execution or not
	end
	
	# Aliases
	const SubspaceEnsembleCell{F<:Void, T, U} = SubspaceEnsemble{F,T,U}		# no execution function, it is dynamically generated in training
	const SubspaceEnsembleClassic{F<:Function, T, U} = SubspaceEnsemble{F,T,U} 	# the execution function is specified

	# Printers
	Base.show(io::IO, m::SubspaceEnsembleCell) = print(io, "Sub-space ensemble, $(length(m.idx)) members, Cell version")
	Base.show(io::IO, m::SubspaceEnsembleClassic) = print(io, "Sub-space ensemble, $(length(m.idx)) members")



	"""
		subspace_indices(n, L, M, replace)
	
	Function that generates a `Vector{Vector{Int}` containing the `L` random subsets of variable indices for a given dataset of `n` variables using `M` variables.
	If `replace` is `true`, the same variable can appear in multiple subsets and the number of subsets has no upper bound.
	"""
	function subspace_indices(n::Int, L::Int, M::Int, replace::Bool)
		@assert n > 1 "The dataset has to have more than 1 variable."
		@assert L > 0 "The random subset ensemble has to have at least 1 member."
		@assert M > 0 "The number of random variables has to be at least 1. "	
		
		# Generate V based on n,L,M 
		if replace
			V=Vector{Vector{Int}}(L)
			for i =1:L
				V[i] = randperm(n)[1:M]
			end
			return V
		else
			@assert L*M <=n "Cannot split $n variables to and ensemble of $L elements with $M distinct variables each."
			ro = randperm(n) # random order
			V=Vector{Vector{Int}}(L)
			k=1
			for i=1:M:L*M
				V[k] = ro[i:i+M-1]
				k+=1
			end
			return V
		end
	end
	
	
	
	"""
		randomsubspace_train(X, y, L, M, f_train, combiner, replace; parallel_train=true, parallel_execution=false)

	Trains a random subspace classifier. The function runs `f_train` on subspaces of `X` and returns `RandomSubspace` object. 

	# Arguments
	 * `X::AbstractMatrix` is the training data; columns are assumed to be samples, lines variables
	 * `y::AvstractVector` are the labels
	 * `L::Int` is the ensemble size
	 * `M::Int` is the number of variables for each ensemble member
	 * `f_train::Function` is the training function for the ensemble members
	 * `f_exec::Union{Function,Void}` is the execution function for the ensemble members
	 * `combiner::ClassifierCombiner.AbstractCombiner` is the combiner
	 * `replace::Bool` specifies whether the variable subsets can contain (`true`) of not (`false`) the same variable. If `true`, any number of subsets can be generated; if 
	 `false` a finite maximum number can be generated but all subsets will be distinct
	
	# Keyword arguments
	 * `parallel_train::Bool=false` specifies whether to paralellize or not the training. Generally a good ideea except for lazy classifiers (e.g. kNN).
	 * `parallel_execution::Bool=false` specifies whether to paralellize or not the execution. Unless the execution is time consuming (e.g. Parzen classifier), it should be left to `false`.
	
	It is important to note that the training method calls internally `x->f_train((x,y))` where `x` are the subsets of `X` constructed using `L`, `M` and the total number of variables of `X`.
	Considering a generic training function `g` desired to be used in the ensemble, of the following forms:
	 * `g(X, y, arg1, arg2)`, one should provide as argument something as `f_train = (x)->g_train(x[1],y[2], arg1, arg2) # uses labels` 
	 * `g(X, arg1, arg2)`, one should provide as argument something as `f_train = (x)->g_train(x[1],arg1, arg2) # ignores labels` 
	 * `g(X, arg1, arg2)` where `X==(data,labels)`, one should provide as argument something as `f_train = (x)->g_train(x, arg1, arg2) # MLDataUtils container case`
	 * `g(args...)::CellFunU`, one can simply provide `g.f` (i.e. the training function of the untrained function cell) since it already contains the training arguments.

	 During execution, `f_exec(ensemble.members[i], Xᵢ)` will be called for all subspace subsets `Xᵢ` of `X` corresponding to the members of the ensemble	
	"""
	function randomsubspace_train(X::T where T<:AbstractMatrix, y::S where S<:AbstractVector, L::Int, M::Int, f_train::Function, f_exec::Union{Function,Void},
			       		combiner::S where S<:AbstractCombiner, replace::Bool; 
					parallel_train::Bool=true, parallel_execution::Bool=false)
		
		# Calculate the number of variables
		n = size(X,1)
		
		# Calculate subspace indices
		idx = subspace_indices(n, L, M, replace)

		# Return data subsets (Vector of datasubsets of X) and V
		if parallel_train
			members = pmap(x->f_train((x,y)), (getobs(datasubset(X,i,ObsDim.First())) for i in idx) )
		else
			members = map(x->f_train((x,y)), (getobs(datasubset(X,i,ObsDim.First())) for i in idx) )
		end
		
		return SubspaceEnsemble(idx, f_train, f_exec, members, combiner, parallel_train, parallel_execution)
	end



	"""
		randomsubspace_exec(ensemble, X)
	
	Executes the random subspace ensemble. 

	# Arguments
	 * `ensemble::RandomSubspace.SubspaceEnsemble` the subspace ensemble
	 * `X::AbstractMatrix` is the data; columns are assumed to be samples, lines variables
	"""
	function randomsubspace_exec(ensemble::SubspaceEnsembleClassic, X::T where T<:AbstractMatrix) 
		if ensemble.parallel_execution
			out = @sync @parallel (vcat) for i in eachindex(ensemble.members)
				ensemble.f_exec(ensemble.members[i], getobs(datasubset(X, ensemble.idx[i], ObsDim.First())))
			end
		else
			out = [ensemble.f_exec(ensemble.members[i], getobs(datasubset(X, ensemble.idx[i], ObsDim.First()))) for i in eachindex(ensemble.members)]

		end
		
		return combiner_exec(ensemble.combiner, out)
	end
	
	# j4pr case: no execution function, the ensemble members contain it already ('f_exec' is not present at all)
	function randomsubspace_exec(ensemble::SubspaceEnsembleCell, X::T where T<:AbstractMatrix ) 
		if ensemble.parallel_execution
			out = @sync @parallel (vcat) for i in eachindex(ensemble.members)
				ensemble.members[i]( getobs(datasubset(X, ensemble.idx[i], ObsDim.First())) ) 
			end
		else
			out = [ensemble.members[i](getobs(datasubset(X, ensemble.idx[i],ObsDim.First()))) for i in eachindex(ensemble.members)]
		end

		return combiner_exec(ensemble.combiner, out)
	end
end


##########################
# FunctionCell Interface #
##########################
"""
	randomsubspace(f, L, M, combiner, replace; parallel_train=true, parallel_execution=false)

Creates an untrained random subspace ensemble.

# Arguments
 * `f::CellFunU` is a untrained cell corresponding to the classifer/transform that the ensemble will contain
 * `L::Int` is the ensemble size
 * `M::Int` is the number of variables for each ensemble member
 * `f_train::Function` is the training function
 * `combiner::ClassifierCombiner.AbstractCombiner` is the combiner
 * `replace::Bool` specifies whether the variable subsets can contain (`true`) or not (`false`) the same variable. If `true`, any number of subsets can be generated; if 
 `false` a finite maximum number can be generated but all subsets will be distinct (default: `true`)

# Keyword arguments
 * `parallel_train::Bool=true` specifies whether to paralellize or not the training.
 * `parallel_execution::Bool=false` specifies whether to paralellize or not the execution.

For more information:
	[1] T.K. Ho "The Random Subspace Method for Constructing Decision Forests" 1998 IEEE Transactions on Pattern Analysis and Machine Intelligence 20(8): 832-844.
 	[2] L. Kuncheva et al. "Random Subspace Ensembles for fMRI Classification" IEEE Transactions on Medical Imaging 29 (2): 531-542. 	
	[3] L. Kuncheva "Combining Pattern Classifiers 2'nd Ed." 2014, ISBN 978-1-118-31523-1

Try `?j4pr.RandomSubspace.randomsubspace_train` and `?j4pr.RandomSubspace.randomsubspace_exec` and for more details.

# Examples
```
julia> using j4pr

julia> D=DataGenerator.iris() # Get the iris dataset
Iris Dataset, 150 obs, 4 vars, 1 target(s)/obs, 3 distinct values: "virginica"(50),"setosa"(50),"versicolor"(50)

julia> w=j4pr.knn(5,smooth=:ml) # 5-NN classifier, maximum likelihood posterior smoothing
5-NN classifier: smooth=ml, no I/O size information, untrained

julia> we=randomsubspace(w, 5, 2,j4pr.ClassifierCombiner.GeneralizedMeanCombiner(5,3,1.0)) # ensemble of 5 members, 2 variables; the combiner uses 3 classes
Sub-space ensemble (5-NN classifier: smooth=ml, 5 members × 2 vars), no I/O size information, untrained

julia> we_trained = D |>we # train ensemble
Sub-space ensemble (5-NN classifier: smooth=ml, 5 members × 2 vars), 4->3, trained

julia> +D |> we_trained # run ensemble on training data
3×150 Array{Float64,2}:
 1.0          1.0          1.0          1.0          1.0          1.0          1.0          …  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16
 2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16     2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  0.12       
 2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16     1.0          1.0          1.0          1.0          1.0          1.0          0.88   
```
"""
randomsubspace(f::F where F<:CellFunU, L::Int, M::Int, 
	       combiner::ClassifierCombiner.AbstractCombiner=ClassifierCombiner.NoCombiner(), 
	       replace::Bool=true; parallel_train::Bool=true, parallel_execution::Bool=false) = 
	FunctionCell(randomsubspace, (f, L, M, combiner, replace), ModelProperties(), "Sub-space ensemble ($(f.tinfo), $L members × $M vars)";
			parallel_train=parallel_train, parallel_execution=parallel_execution) # untrained function cell



############################
# DataCell/Array Interface #
############################
"""
	randomsubspace(x, f, L, M, combiner, replace; parallel_train=true, parallel_execution=false)

Trains a random subspace ensemble using the data `x`.
"""
# Training
randomsubspace(x::T where T<:CellDataL, f::F where F<:CellFunU, L::Int, M::Int, combiner::ClassifierCombiner.AbstractCombiner=ClassifierCombiner.NoCombiner(), 
	       replace::Bool=true; parallel_train::Bool=true, parallel_execution::Bool=false) = 
	randomsubspace((getx!(x), gety(x)), f, L, M, combiner, replace; parallel_train=parallel_train, parallel_execution=parallel_execution)

randomsubspace(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, f::F where F<:CellFunU, L::Int, M::Int, 
	combiner::ClassifierCombiner.AbstractCombiner=ClassifierCombiner.NoCombiner(), replace::Bool=true; parallel_train::Bool=true, parallel_execution::Bool=false) = 
	randomsubspace((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), f, L, M, combiner, replace; parallel_train=parallel_train, parallel_execution=parallel_execution)

randomsubspace(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, f::F where F<:CellFunU, L::Int, M::Int, 
	       combiner::ClassifierCombiner.AbstractCombiner=ClassifierCombiner.NoCombiner(), replace::Bool=true; 
	       parallel_train::Bool=true, parallel_execution::Bool=false) = 
begin

	@assert nobs(x[1]) == nobs(x[2]) "[randomsubspace] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)

	# Train model
	ensembledata = RandomSubspace.randomsubspace_train(x[1], yenc, L, M, f.f, nothing, combiner, replace; 
						    parallel_train=parallel_train, parallel_execution=parallel_execution)
	
	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(randomsubspace, Model(ensembledata, modelprops), 
	      "Sub-space ensemble ($(f.tinfo), $L members × $M vars)" ) # trained function cell
end


# Execution
randomsubspace(x::T where T<:CellData, model::Model{<:RandomSubspace.SubspaceEnsemble}) =
	datacell(randomsubspace(getx!(x), model), gety(x)) 	
randomsubspace(x::T where T<:AbstractVector, model::Model{<:RandomSubspace.SubspaceEnsemble}) = 
	randomsubspace(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
randomsubspace(x::T where T<:AbstractMatrix, model::Model{<:RandomSubspace.SubspaceEnsemble}) =
	RandomSubspace.randomsubspace_exec(model.data,x)
