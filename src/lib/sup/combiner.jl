module ClassifierCombiner

	using StatsBase: countmap
	using j4pr: countapp, countappw
	
	export AbstractCombiner, NoCombiner, LabelCombiner, ContinuousCombiner, VoteCombiner, WeightedVoteCombiner,
	       NaiveBayesCombiner, GeneralizedMeanCombiner, WeightedMeanCombiner, MedianCombiner, ProductCombiner,
	       combiner_train, combiner_exec

	abstract type AbstractCombiner end
	abstract type LabelCombiner <: AbstractCombiner end
	abstract type ContinuousCombiner <: AbstractCombiner end
	
	"""
	Combiner that does not combine, fake.
	"""
	struct NoCombiner <: AbstractCombiner end
	
	"""
	Majority voting label combiner. 
	
	# Fields
	 * `L::Int` is the expected ensemble size on whose output it operates
	"""
	struct VoteCombiner <: LabelCombiner
		L::Int 						# Ensemble size
		C::Int						# Number of classes	
	end

	"""
	Weighted majority label combiner. 
	
	# Fields
	 * `L::Int` is the expected ensemble size on whose output it operates
	 * `weights::Vector{Float64}` weight vector; its length should be equal to the field `L`
	"""
	struct WeightedVoteCombiner <: LabelCombiner
		L::Int 						# Ensemble size
		C::Int						# Number of classes	
		weights::Vector{Float64}			# Ensemble members weights
	end
	
	WeightedVoteCombiner(L::Int) = WeightedVoteCombiner(L,fill(1.0,L))
	
	"""
	Naive Bayes label combiner. 
	
	# Fields
	 * `L::Int` is the expected ensemble size on whose output it operates
	 * `labels::Vector{S}` is the training labels vector
	 * `labelcount::Vector{Float64}` is the unique label count vector (used to calculate priors)
	 * `CM::Array{Float64,3}` is the confusion matrix array; one confusion matrix for each ensemble member, hence it is required that `size(CM,3)==L`
	"""
	struct NaiveBayesCombiner{S} <: LabelCombiner
		L::Int 						# Ensemble size
		labels::Vector{S}				# List of labels (should be in some order)
		labelcount::Vector{Float64}			# Counts corresponding to labels (in the same order)
		CM::Array{Float64,3}				# Confusion matrices for each classifier:
								#  - CM(k,s,i) indicates ensemble member 'i', true class 'k' and estimate 's';
								#  - k and s are indices in the sorted vector of unique labels.
	end							
	
	"""
	Generalized mean combiner. 
	
	# Fields
	 * `L::Int` is the expected ensemble size on whose output it operates
	 * `C::Int` is the number of classes 
	 * `α::Float64` is the parameter of the generalized mean
	"""
	struct GeneralizedMeanCombiner <: ContinuousCombiner
		L::Int 						# Ensemble size
		C::Int						# Number of classes	
		α::Float64					# Generalized mean parameter
	end

	"""
	Weighted generalized mean combiner. 
	
	# Fields
	 * `L::Int` is the expected ensemble size on whose output it operates
	 * `C::Int` is the number of classes 
	 * `α::Float64` is the parameter of the generalized mean
	 * `weights::Vector{Float64}` weight vector; its length should be equal to the field `L`
	"""
	struct WeightedMeanCombiner <: ContinuousCombiner
		L::Int 						# Ensemble size
		C::Int						# Number of classes	
		α::Float64					# Generalized mean parameter
		weights::Vector{Float64}			# Ensemble members weights
	end
	
	"""
	Median combiner. 
	
	# Fields
	 * `L::Int` is the expected ensemble size on whose output it operates
	 * `C::Int` is the number of classes 
	"""
	struct MedianCombiner <: ContinuousCombiner
		L::Int 						# Ensemble size
		C::Int						# Number of classes	
	end

	
	"""
	Median combiner. 
	
	# Fields
	 * `L::Int` is the expected ensemble size on whose output it operates
	 * `C::Int` is the number of classes 
	"""
	struct ProductCombiner <: ContinuousCombiner
		L::Int 						# Ensemble size
		C::Int						# Number of classes	
	end

	"""
		decision_profile_labels(x)
	
	Transforms the input `x` into a matrix; `x` can be a `Vector{Vector{T} where T}` in which case the result is a `Matrix{T}` where each element of `x` becomes a row in the output matrix.
	"""
	function decision_profile_labels(x::T where T<:AbstractVector{<:AbstractVector{S}}) where S
		# Check that all dimensions of inner vectors match
		@assert all(i == j for i in length.(x), j in length.(x)) "[Classifier Combiner] The dimensions of the vectors to be combined must match."
		
		n = length(x[1])	# Number of samples
		m = length(x)		# Size of ensemble
		out = Matrix{S}(m,n)	# Output
		
		for (i,v) in enumerate(x)
			out[i,:] = v
		end
		
		return out
	end
	
	# Desired form already, do nothing
	function decision_profile_labels(x::T where T<:AbstractMatrix{S}) where S 
		x
	end



	"""
		decision_profile_continuous(x [,C])
	
	Transforms the input `x` into an `Array{T,3}`; `x` can be `Vector{Vector{T} where T}`, `Vector{Matrix{T} where T}` or `Matrix{T}`; `C` is the number of classes and can be either
	inferred from the sizes of `x` as:
	 * `1` if the input is a vector of vectors 
	 * `size(x[1],1)` (the number of rows of the first matrix), if the input is a vector of matrices; ofcourse, the number of rows has to be the same for all matrices 
	 * if the input is a `Matrix{T}`, `C` has to be explicitly specified as the ensemble outputs are already stacked together as rows of the matrix.
	
	So, the first dimension of the output is the number of classes, the second the number of samples and the third, the number of ensemble memebers.
	"""
	function decision_profile_continuous(x::T where T<:AbstractVector{<:AbstractVector{S}}, C::Int=1) where S # C is ignored.
		# Check that all dimensions of inner vectors match
		@assert all(i == j for i in length.(x), j in length.(x)) "[Classifier Combiner] The dimensions of vectors to be combined must match."
		
		n = length(x[1])	# Number of samples
		m = length(x)		# Size of ensemble
		out = Array{S,3}(1,n,m)	# Output
		
		for (i,v) in enumerate(x)
			out[1,:,i] = v
		end
		
		return out
	end	
	
	function decision_profile_continuous(x::T where T<:AbstractVector{<:AbstractMatrix{S}}, C::Int=size(x[1],1)) where S # C is ignored here as well
		# Check that all dimensions of inner vectors match
		@assert all(i == j for i in size.(x), j in size.(x)) "[Classifier Combiner] The dimensions of matrices to be combined must match."
		
		n = size(x[1],2)	# Number of samples
		m = length(x)		# Size of ensemble
		out = Array{S,3}(size(x[1],1),n,m) # Output
		
		for (i,v) in enumerate(x)
			out[:,:,i] = v
		end
		
		return out
	end	

	function decision_profile_continuous(x::T where T<:AbstractMatrix{S}, C::Int) where S
		# Check that all dimensions of inner vectors match
		@assert mod(size(x,1),C) == 0 "[Classifier Combiner] The first dimension of the input matrix should be a multiple of the number of classes."
		
		n = size(x,2)		# Number of samples
		m = div(size(x,1),C)	# Size of ensemble
		out = Array{S,3}(C,n,m)	# Output
		
		for (i,ki) in enumerate(1:C:size(x,1))
			out[:,:,i] = x[ki:ki+C-1,:]
		end
		
		return out
	end	


	"""
		combiner_train(predictions, labels, combiner_type)

	Trains a label combiner based on multiple ensemble outputs and returns a combiner object of `combiner_type`.

	# Arguments
	 * `predictions` are the estimated labels or continuous outputs; can be either a `Vector{Vector}` or `Matrix` for label outputs or, a `Vector{Matrix}` or `Matrix` for continuous outputs
	 * `labels` is a `Vector` containing the training labels
	 * `combiner_type` is the type of the resulting combiner and must be `<:ClassifierCombiner.AbstractCombiner`
	 
	 It is assumed that the input size is consistent (e.g. if a `Vector{Vector}` the lengths of the inner vectors are equal), and that
	 the type of the elements of the input matches the element type of the training labels.
	"""
	# Function for training the vote combiners
	function combiner_train(predictions::T where T<:AbstractArray, labels::S where S<:AbstractVector, ::Type{NoCombiner})
		NoCombiner()
	end
	
	function combiner_train(predictions::T where T<:AbstractArray, labels::S where S<:AbstractVector, ::Type{VoteCombiner}) 
		_ = decision_profile_labels(predictions) # fake, just to check internal consistency of the input
		VoteCombiner(length(predictions), length(unique(labels))) 
	end

	function combiner_train(predictions::T where T<:AbstractArray, labels::S where S<:AbstractVector, ::Type{WeightedVoteCombiner}) 
		ip = decision_profile_labels(predictions)
		m = size(ip,1)		# Size of ensemble
		n = size(ip,2)		# Number of samples
		w = zeros(m) 		# Weights
		c = length(unique(labels)) #number of classes
		
		@assert n==length(labels) "[Classifier Combiner] The size of the training labels must match the number of output samples."
		
		@inbounds @fastmath for i in 1:m
			p = sum(isequal.(ip[i,:], labels))/n # individual ensemble member accuracy
			w[i] = log((p+eps())/(1-p+eps()))
		end
		return WeightedVoteCombiner(m, c, w)
	end

	function combiner_train(predictions::T where T<:AbstractArray, labels::S where S<:AbstractVector{V}, ::Type{NaiveBayesCombiner}) where V
		ip = decision_profile_labels(predictions)
		m = size(ip,1)						# Size of ensemble
		n = size(ip,2)						# Number of samples
		ulabels::Vector{V} = sort(unique(labels))		# Sorted unique labels
		labelcount::Vector{Float64} = countapp(labels, ulabels)	# Label count
		C::Int = length(ulabels)				# Number of classes
		CM::Array{Float64,3} = zeros(Float64,C,C,m)		# Confusion matrices (for all ensemble members)

		@assert n==length(labels) "[Classifier Combiner] The size of the training labels must match the number of output samples."
		
		# Build confusion matrices 
		@inbounds @fastmath for i in 1:m 	# for each ensemble memeber
			for s in 1:C			# for each "estimated" class
				for k in 1:C		# for each "true" class 
					CM[k,s,i] = 1.0/(labelcount[k]+1.0) * (sum( isequal.(labels, ulabels[k]) .& isequal.(ip[i,:], ulabels[s]) ) + 1.0/C)
				end
			end
		end
		
		return NaiveBayesCombiner(m, ulabels, labelcount, CM)
	end

	

	"""
		combiner_exec(combiner, predictions)

	Combines the predictions from multiple ensemble members using a combiner returning the combined predictions

	# Arguments
	 * `combiner` is an combiner object of type `<:ClassifierCombiner.AbstractCombiner` used to combine `predictions`
	 * `predictions` are the estimated labels or continuous outputs; can be either a `Vector{Vector}` or `Matrix` for label outputs or, a `Vector{Matrix}` or `Matrix` for continuous outputs
	 
	 It is assumed that the input size is consistent (e.g. if a `Vector{Vector}` the lengths of the inner vectors are equal), that
	 the type of the elements of the input matches the element type of the training labels and that the first dimension of `predictions` matched the learned ensemble size, `combiner.L`.
	"""
	# Functions for the execution of the combiners
	function combiner_exec(x::NoCombiner, predictions::T where T<:AbstractArray)
		predictions
	end
	
	function combiner_exec(x::VoteCombiner, predictions::T where T<:AbstractArray)
		p = decision_profile_labels(predictions)
		@assert size(p,1) == x.L "[Classifier combiner] Mismatch between expected and actual ensemble results size" 
		
		out = Vector{eltype(p)}(size(p,2))	# Output
		pu = unique(p)				# Unique predicted labels
		@inbounds for i in 1:size(p,2)
			@fastmath out[i] = pu[findmax(countapp(view(p,:,i), pu))[2]]
		end
		
		return out
	end

	function combiner_exec(x::WeightedVoteCombiner, predictions::T where T<:AbstractArray)
		p = decision_profile_labels(predictions)
		m = size(p,1)				# Ensemble size
		n = size(p,2)				# Number of samples
		@assert m == x.L == length(x.weights) "[Classifier combiner] Mismatch between expected and actual ensemble results size or, weight vector size" 
		
		out = Vector{eltype(p)}(n)		# Output
		pu = unique(p)				# Unique predicted labels
		@inbounds for i in 1:n
			@fastmath out[i] = pu[findmax(countappw(view(p,:,i), pu, x.weights, -Inf))[2]]
		end
		
		return out
	end
	
	function combiner_exec(x::S where S<:NaiveBayesCombiner{B}, predictions::T where T<:AbstractArray) where B
		p = decision_profile_labels(predictions)
		@assert size(p,1) == x.L "[Classifier combiner] Mismatch between expected and actual ensemble results size" 
		
		n = size(p,2)				# Number of samples
		classes::Vector{B} = x.labels		# Ordered observed classes (in training)
		C = length(classes)			# Number of classes
		priors::Vector{Float64} = x.labelcount./sum(x.labelcount) # Vector of class priors
		out = Vector{B}(n)			# Output
		
		# Function that searches fast a value through a vector of unique values
		function _fastsearch_(v::T,uv::Vector{T})::Int where T
			i = 0
			@inbounds for (i,vi) in enumerate(uv)
				if isequal(vi,v) 
					return i
				end
			end
			return i
		end
		
		tmp = Vector{Float64}(C)		# Temporary class-wise priors
		@inbounds for j in 1:n			# for each sample,	
			tmp = priors			# initialize posterior
			for k in 1:C			# for each "target" class
				for i in 1:size(x.CM,3)	# loop through confusion matrices
 					# Search for the estimated class (from each ensemble member) in the unique class vector;
					# use its position to retrieve the corresponding likelihood from the learned confusion matrices 
					tmp[k] *= x.CM[k,_fastsearch_(p[i,j],classes)[1],i]
				end
			end
			out[j] = classes[findmax(tmp)[2]]
		end	
		
		return out
	end

	function combiner_exec(x::GeneralizedMeanCombiner, predictions::T where T<:AbstractArray)
		p = decision_profile_continuous(predictions, x.C)
		@assert size(p,1) == x.C "[Classifier combiner] Mismatch between expected and input number of classes" 
		@assert size(p,3) == x.L "[Classifier combiner] Mismatch between expected and input ensemble size" 

		alpha::Float64=x.α	
		return (squeeze(mean((p+eps()).^alpha,3),3)).^(1/alpha)
	end

	function combiner_exec(x::WeightedMeanCombiner, predictions::T where T<:AbstractArray)
		p = decision_profile_continuous(predictions, x.C)
		@assert size(p,1) == x.C "[Classifier combiner] Mismatch between expected and input number of classes" 
		@assert size(p,3) == x.L "[Classifier combiner] Mismatch between expected and input ensemble size" 
		@assert size(p,3) == length(x.weights) "[Classifier combiner] Mismatch between ensemble size and weights length" 
		alpha::Float64=x.α	
		return mean(((p[:,:,i]*x.weights[i]+eps()).^alpha for i in 1:length(x.weights))).^(1/alpha)
	end
	
	function combiner_exec(x::MedianCombiner, predictions::T where T<:AbstractArray)
		p = decision_profile_continuous(predictions, x.C)
		@assert size(p,1) == x.C "[Classifier combiner] Mismatch between expected and input number of classes" 
		@assert size(p,3) == x.L "[Classifier combiner] Mismatch between expected and input ensemble size" 
		
		return Matrix{eltype(p)}(squeeze(median(p,3),3)) # median returns Float64, make sure original type is preserved
	end
	
	function combiner_exec(x::ProductCombiner, predictions::T where T<:AbstractArray)
		p = decision_profile_continuous(predictions, x.C)
		@assert size(p,1) == x.C "[Classifier combiner] Mismatch between expected and input number of classes" 
		@assert size(p,3) == x.L "[Classifier combiner] Mismatch between expected and input ensemble size" 
		
		return squeeze(prod(p,3),3)
	end
end





########################################
# FunctionCell and DataCell Interfaces #	
########################################

## Label combiners

"""
	votecombiner(L::Int [,C::Int])

Majority vote label combiner. `L` is the size of the upstream ensemble. The number of classes `C` is most of the time not necessary are defaults to `-1`. 
"""
votecombiner(L::Int, C::Int=-1) = FunctionCell(genericcombiner, Model(ClassifierCombiner.VoteCombiner(L,C), ModelProperties(L,1)), "Vote combiner") 



"""
	wvotecombiner(L::Int)

Generates an untrained function cell that when piped data into, trains a weighted vote label combiner. `L` is the size of the upstream ensemble.
"""
wvotecombiner(L::Int) = FunctionCell(wvotecombiner, (L,), ModelProperties(L,1), "Weighted-Vote combiner" ) # untrained function cell

"""
	wvotecombiner(x,L)

Trains a weighted vote label combiner using the data `x`, considering an upstream ensemble size `L`.
"""
# Training
wvotecombiner(x::T where T<:CellDataL, L::Int) = wvotecombiner((getx!(x), gety(x)), L)
wvotecombiner(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, L::Int) = wvotecombiner((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), L)
wvotecombiner(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, L::Int) = begin

	@assert nobs(x[1]) == nobs(x[2]) "[wvotecombiner] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Train combiner
	wvcombiner = ClassifierCombiner.combiner_train(getobs(x[1]), getobs(x[2]), ClassifierCombiner.WeightedVoteCombiner)
	
	# Check that the specified ensemble number is equal to the determined one
	@assert L==wvcombiner.L "[wvotecombiner] Expected ensemble dimension is $L and the one determined from input data is $(wvcombiner.L)"

	FunctionCell(genericcombiner, Model(wvcombiner, ModelProperties(L,1)), "Weighted-Vode combiner") 
end



"""
	naivebayescombiner(L::Int)

Generates an untrained function cell that when piped data into, trains a Naive Bayes label combiner. `L` is the size of the upstream ensemble.
"""
naivebayescombiner(L::Int) = FunctionCell(naivebayescombiner, (L,), ModelProperties(L,1), "Naive-Bayes combiner" ) # untrained function cell

"""
	naivebayescombiner(x,L)

Trains a Naive Bayes label combiner using the data `x`, considering an upstream ensemble size `L`.
"""
# Training
naivebayescombiner(x::T where T<:CellDataL, L::Int) = naivebayescombiner((getx!(x), gety(x)), L)
naivebayescombiner(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, L::Int) = naivebayescombiner((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), L)
naivebayescombiner(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, L::Int) = begin

	@assert nobs(x[1]) == nobs(x[2]) "[naivebayescombiner] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Train combiner
	nbcombiner = ClassifierCombiner.combiner_train(getobs(x[1]), getobs(x[2]), ClassifierCombiner.NaiveBayesCombiner)
	
	# Check that the specified ensemble number is equal to the determined one
	@assert L==nbcombiner.L "[naivebayescombiner] Expected ensemble dimension is $L and the one determined from input data is $(nbcombiner.L)"

	FunctionCell(genericcombiner, Model(nbcombiner, ModelProperties(L,1)), "Naive-Bayes combiner") 
end



# Continuous output combiners

"""
	meancombiner(L::Int, C::Int [;α=1.0])

Trains a mean continuous output combiner. `L` is the size of the upstream ensemble, `C` is the expected number of classes and `α` is the parameter of the generalized mean formula.
"""
meancombiner(L::Int, C::Int;α::Float64=1.0) = 
	FunctionCell(genericcombiner, Model(ClassifierCombiner.GeneralizedMeanCombiner(L,C,α), ModelProperties(L*C,C)), 
	      kwtitle("Generalized mean combiner",((:α,α),))) 



"""
	wmeancombiner(L::Int, C::Int, weights::Vector{Float64} [;α=1.0])

Trains a weighted mean continuous output combiner. `L` is the size of the upstream ensemble, `C` is the expected number of classes, `weights` are the 
individual weights of the ensemble memebers and `α` is the parameter of the generalized mean formula.
"""
wmeancombiner(L::Int, C::Int, weights::Vector{Float64};α::Float64=1.0) = 
	FunctionCell(genericcombiner, Model(ClassifierCombiner.WeightedMeanCombiner(L,C,α,weights), ModelProperties(L*C,C)), 
	      kwtitle("Weighted mean combiner",((:α,α),))) 



"""
	productcombiner(L::Int, C::Int)

Trains a product continuous output combiner. `L` is the size of the upstream ensemble, `C` is the expected number of classes.
"""
productcombiner(L::Int, C::Int) = FunctionCell(genericcombiner, Model(ClassifierCombiner.ProductCombiner(L,C), ModelProperties(L*C,C)), "Product combiner") 



"""
	mediancombiner(L::Int, C::Int)

Trains a median continuous output combiner. `L` is the size of the upstream ensemble, `C` is the expected number of classes.
"""
mediancombiner(L::Int, C::Int) = FunctionCell(genericcombiner, Model(ClassifierCombiner.MedianCombiner(L,C), ModelProperties(L*C,C)), "Median combiner") 



# Execution methods for all combiners ;)
genericcombiner(x::T where T<:CellData, model::Model{<:ClassifierCombiner.AbstractCombiner}) = 
	datacell(genericcombiner(getx!(x), model), gety(x)) 	
genericcombiner(x::T where T<:AbstractVector, model::Model{<:ClassifierCombiner.AbstractCombiner}) = 
	genericcombiner(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
genericcombiner(x::T where T<:AbstractMatrix, model::Model{<:ClassifierCombiner.AbstractCombiner}) = 
	mat(ClassifierCombiner.combiner_exec(model.data, getobs(x)), LearnBase.ObsDim.Constant{2}())	
