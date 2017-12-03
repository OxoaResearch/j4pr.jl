# AdaBoost ensemble module
module AdaBoost

	using j4pr.ClassifierCombiner
	using j4pr: countapp, ohenc_integer
	using Distances: colwise, Euclidean 
	using MLDataPattern: nobs, getobs, datasubset, undersample, targets 
	using StatsBase: sample!,weights
	using StatsFuns: softmax!
	using UnicodePlots: lineplot
	export AdaBoostEnsemble, adaboost_train, adaboost_exec
	
	abstract type BoostType end
	struct AdaBoostM1 <: BoostType end
	struct AdaBoostM2 <: BoostType end
	
	"""
	AdaBoost ensemble object (it describes the ensemble). 
	
	# Fields
	 * `weights::Vector{Float64}` is a vector weights associated to each ensemble member
	 * `f_train::Function` is the training function
	 * `f_exec::Function` is the execution function
	 * `members::Vector` is a vector containing the trained classifiers
	 * `boost_type::AdaBoost.BoostType` indicates the type of AdaBoost approach used
	 * `combiner::ClassifierCombiner.AbstractCombiner` the means by which ensemble outputs are combined
	"""
	struct AdaBoostEnsemble{F, T<:Vector{S} where S, B<:BoostType, U<:AbstractCombiner}
		weights::Vector{Float64}	# vector of ensemble weights
		f_train::Function		# training function
		f_exec::F			# execution function
		members::T			# ensemble members (trained)
		boost_type::B			# the type of boosting  
		combiner::U			# combiner
	end

	# Aliases
	const AdaBoostEnsembleM1Cell{F<:Void, T, B<:AdaBoostM1, U} = AdaBoostEnsemble{F,T,B,U}			# M1 boost, no execution function 	
	const AdaBoostEnsembleM1Classic{F<:Function, T, B<:AdaBoostM1, U} = AdaBoostEnsemble{F,T,B,U}		# M1 boost, with execution function
	const AdaBoostEnsembleM2Cell{F<:Void, T, B<:AdaBoostM2, U} = AdaBoostEnsemble{F,T,B,U}			# M2 boost, no execution function 	
	const AdaBoostEnsembleM2Classic{F<:Function, T, B<:AdaBoostM2, U} = AdaBoostEnsemble{F,T,B,U}		# M2 boost, with execution function
	
	# Printers
	Base.show(io::IO, m::AdaBoostEnsembleM1Cell) = print(io, "AdaBoost M1 ensemble, $(length(m.weights)) members, Cell version")
	Base.show(io::IO, m::AdaBoostEnsembleM2Cell) = print(io, "AdaBoost M2 ensemble, $(length(m.weights)) members, Cell version")
	Base.show(io::IO, m::AdaBoostEnsembleM1Classic) = print(io, "AdaBoost M1 ensemble, $(length(m.weights)) members")
	Base.show(io::IO, m::AdaBoostEnsembleM2Classic) = print(io, "AdaBoost M2 ensemble, $(length(m.weights)) members")



	"""
		adaboost_train(X, y, L, f_train, f_exec, boost_type)

	Trains an AdaBoost classifier. The function returns a `Adaboost` object. 

	# Arguments
	 * `X::AbstractMatrix` is the training data; columns are assumed to be samples, lines variables
	 * `y::AvstractVector` are the labels
	 * `L::Int` is the ensemble size
	 * `f_train::Function` is the training function for the ensemble members
	 * `f_exec::Function` is the execution function for the ensemble members	
	 * `boost_type::AdaBoost.BoostType` specifies the training algorithm to use. Supported: `AdaBoostM1` and `AdaBoostM2`; 

	It is important to note that the training method calls internally `x->f_train((X,y))` where `X` is the input data.
	Considering a generic training function `g` desired to be used in the ensemble, of the following forms:
	 * `g(X, y, arg1, arg2)`, one should provide as argument something as `f_train = (x)->g_train(x[1],y[2], arg1, arg2) # uses labels` 
	 * `g(X, arg1, arg2)`, one should provide as argument something as `f_train = (x)->g_train(x[1],arg1, arg2) # ignores labels` 
	 * `g(X, arg1, arg2)` where `X==(data,labels)`, one should provide as argument something as `f_train = (x)->g_train(x, arg1, arg2) # MLDataUtils container case`
	 * `g(args...)::CellFunU`, one can simply provide `g.f` (i.e. the training function of the untrained function cell) since it already contains the training arguments.
	 
	During the training, errors are calculated by calling `f_exec(clf,X)` for a given trained ensemble member `clf`.
	"""
	adaboost_train(X::T where T<:AbstractMatrix, y::S where S<:AbstractVector, L::Int, f_train::Function, f_exec::Union{Function,Void}, ::AdaBoostM1) = 
		adaboost_train_m1(X, y, L, f_train, f_exec)
	adaboost_train(X::T where T<:AbstractMatrix, y::S where S<:AbstractVector, L::Int, f_train::Function, f_exec::Union{Function,Void}, ::AdaBoostM2) = 
		adaboost_train_m2(X, y, L, f_train, f_exec)
		
	function adaboost_train_m1(X::T where T<:AbstractMatrix, y::S where S<:AbstractVector, L::Int, f_train::Function, f_exec::Union{Function,Void})
		
		# Initializations
		n = nobs(X)			# number of observations 
		yu = sort(unique(y))		# unique labels
		C = length(yu)			# number of classes
		w = fill(1/n, n) 		# sample weights
		beta = fill(0.0,L)		# classifier weight

		k = 1
		it = 0
		maxit = max(L, 10_000)
		mismatches = zeros(n)
		members = Vector{typeof(f_train(getobs(undersample((X,y)))))}(L)
		idxs = Vector{Int}(n)								# sampled observation indexes
		idx = collect(1:n)								# observation indexes
		while k <= L
			it += 1
			sample!(idx, weights(w), idxs)						# get sampling indices
		 	
			# Check for missing labels, if any don't go further, resample 
			if isempty(setdiff(yu, unique(y[idxs])))
				
				members[k] = f_train(getobs(datasubset((X,y), idxs)));		# train classifier
				
				# Split execution between the cases where the execution function is implicit (for FunctionCells) or specified (generic)
				if f_exec isa Void
					mismatches = float(yu[targets(indmax, members[k](X))] .!= y)	# find mismatches (FunctionCell classifier)
				else
					mismatches = float(f_exec(members[k],X) .!= y)		# find mismatches 
				end
				
				err = sum(mismatches .* w)					# calculate error for member 'k'	
				if err == 0.0
					fill!(w, 1.0/n)						# re-initialize weights
					k += 1
				elseif err >= (1.0-1/C)
					fill!(w,1.0/n)						# reinitialize weights but do not change iteration
				else
					beta[k] = err/(1-err)					# beta is directly proportional with the error
					w = w .* (beta[k].^(1.0.-mismatches)) 			# if mismatch keep weight, if not, decrease weight 
					w = w./sum(w)						# normalize
					k+=1
				end
			end

			if it == maxit && k<L							# Break if maximum number of iterations reached		
				warn("[adaboost] Maximum number of $maxit iterations reached. Training interrupted.")
				members = members[1:k]
				break
			end
		end	
		return AdaBoostEnsemble(log.(1 ./(beta.+eps())), f_train, f_exec, members, AdaBoostM1(), WeightedVoteCombiner(L, C, log.(1 ./(beta.+eps()))))
	end



	function adaboost_train_m2(X::T where T<:AbstractMatrix, y::S where S<:AbstractVector{V}, L::Int, f_train::Function, f_exec::Union{Function,Void}) where {V}
		
		# Labels need to be a matrix of Floats encoded OneOfK to easily compute distances
		# between their representation and the probabilities return by the classifiers (ensemble members)
		yu::Vector{V} = sort(unique(y))
		yenc::Vector{Int} = Int.(ohenc_integer(y,yu))
		
		# Initializations
		n = nobs(X)			# number of observations 
		C = length(yu)			# number of classes
		beta = fill(0.0,L)		# classifier weight

		k = 1
		it = 0
		maxit = max(L, 10_000)
		members = Vector{typeof(f_train(getobs(undersample((X,y)))))}(L)
		
		# Initialize weight vector (mislabel distribution)
		# (The function does not seem to bring any advantage...)
		function _init_weights_!(w::Vector{Float64}, l::Vector{Int})
			ul::Vector{Int} = unique(l)
			counts::Vector{Float64} = length(l) - countapp(l, ul)
			for i in eachindex(l)
				@inbounds w[i] = 1/(length(l)-counts[find(ul.==l[i])][1])
			end
		end
		w = Vector{Float64}(n)
		fill!(w, 1.0/n)
		#_init_weights_!(w, yenc)
		
		idxs = Vector{Int}(n)								# sampled observation indexes
		idx = collect(1:n)								# observation indexes
		p = zeros(Float64, C, n) 							# ensemble outputs
		
		while k <= L
			it += 1
			sample!(idx, weights(w), idxs)						# get sampling indices

			# Check for missing labels, if any don't go further, resample 
			if isempty(setdiff(yu, unique(yenc[idxs])))
				
				members[k] = f_train(getobs(datasubset((X,yenc), idxs)));	# train classifier

				# Split execution between the cases where the execution function is implicit (for FunctionCells) or specified (generic)
				if f_exec isa Void
					p = members[k](X)
				else
					p = f_exec(members[k],X)				# 'f_exec' must return a matrix of class-wise probabilities sized C x n		 
				end
				
				# Calculate the pseudo-loss 
				err=0.0
				@simd for i in 1:n
					@inbounds err+=w[i]*(1-p[yenc[i],i])
				end
				   
				if err == 0.0
					#_init_weights_!(w,yenc)				# re-initialize weights
					fill!(w, 1.0/n)
					k += 1
				elseif err >= (1-1/C)
					#_init_weights_!(w,yenc)				# reinitialize weights but do not change iteration
					fill!(w, 1.0/n)
				else
					beta[k] = err/(1-err)					# beta is directly proportional with the error
					@simd for i in 1:n
						@inbounds w[i] = w[i]*beta[k]^p[yenc[i],i]
					end
					w = w./sum(w)						# normalize
					k+=1
				end
			end

			if it == maxit && k<L							# Break if maximum number of iterations reached		
				warn("[adaboost] Maximum number of $maxit iterations reached. Training interrupted.")
				members = members[1:k]
				break
			end
		end	
		return AdaBoostEnsemble(log.(1 ./(beta.+eps())), f_train, f_exec, members, AdaBoostM2(), WeightedMeanCombiner(L,C,1.0,log.(1 ./(beta.+eps()))))
	end

	"""
		adaboost_exec(ensemble, X)
	
	Executes the AdaBoost ensemble. 

	# Arguments
	 * `ensemble::AdaBoost.AdaBoostEnsemble` the AdaBoost ensemble
	 * `X::AbstractMatrix` is the data; columns are assumed to be samples, lines variables
	"""
	function adaboost_exec(ensemble::AdaBoostEnsembleM1Classic, X::T where T<:AbstractMatrix)
		out = [ensemble.f_exec(ensemble.members[i], getobs(X)) for i in eachindex(ensemble.members)]
		return combiner_exec(ensemble.combiner, out) 
		
		# If results have to be of matrix form: 
		#return float(MLLabelUtils.convertlabel(MLLabelUtils.LabelEnc.OneOfK(Val{ensemble.combiner.C}), combiner_exec(out, ensemble.combiner))) 
	end
	
	function adaboost_exec(ensemble::AdaBoostEnsembleM2Classic, X::T where T<:AbstractMatrix) 
		out = combiner_exec(ensemble.combiner, 
			 		[ensemble.f_exec(ensemble.members[i], getobs(X)) for i in eachindex(ensemble.members)])
		for j in 1:size(out,2)
			@inbounds softmax!(view(out,:,j))
		end
		return out
	end

	# j4pr case: no execution function, the ensemble members contain it already
	function adaboost_exec(ensemble::AdaBoostEnsembleM1Cell, X::T where T<:AbstractMatrix) 
		out = [targets(indmax, ensemble.members[i](getobs(X))) for i in eachindex(ensemble.members)]  # the 'targets' function is used to get the labels    
		return combiner_exec(ensemble.combiner, out)   

		# If results have to be of matrix form:
		#return float(MLLabelUtils.convertlabel(MLLabelUtils.LabelEnc.OneOfK(Val{ensemble.combiner.C}), combiner_exec(out, ensemble.combiner)))   
	end
	
	function adaboost_exec(ensemble::AdaBoostEnsembleM2Cell, X::T where T<:AbstractMatrix)
		out = combiner_exec(ensemble.combiner, 
		     			[ensemble.members[i](getobs(X)) for i in eachindex(ensemble.members)])
		for j in 1:size(out,2)
			@inbounds softmax!(view(out,:,j))
		end
		return out
	end

end



##########################
# FunctionCell Interface #
##########################
"""
	adaboost(f, L; boost_type=AdaBoostM2())

Creates an untrained AdaBoost ensemble.

# Arguments
 * `f::CellFunU` is a untrained cell corresponding to the classifer/transform that the ensemble will contain
 * `L::Int` is the size of the ensemble

# Keyword arguments
 * `boost_type::AdaBoost.BoostType` specifies whether to use the discrete `AdaBoostM1()` or continuous `AdaBoostM2()` approaches in AdaBoost (default `AdaBoostM1()`)
 
For more information:
	[1] J. Friedman, T. Hastie, R. Tibshirani "Additive Logistic Regression: A statistical view of boosting" 2000 The Annals of Statistics 28(2): 337-407.

Try `?j4pr.AdaBoost.adaboost_train` and `?j4pr.AdaBoost.adaboost_exec` and for more details.

# Examples
```
julia> using j4pr
```
"""
adaboost(f::F where F<:CellFunU, L::Int; boost_type::AdaBoost.BoostType=AdaBoost.AdaBoostM2()) = 
	FunctionCell(adaboost, (f, L), ModelProperties(), "Adaboost ensemble ($(f.tinfo), $L members)"; boost_type=boost_type) # untrained function cell



############################
# DataCell/Array Interface #
############################
"""
	adaboost(x, f, L; boost_type=AdaBoostM2())

Trains an AdaBoost ensemble using the data `x`.
"""
# Training
adaboost(x::T where T<:CellDataL, f::F where F<:CellFunU, L::Int; boost_type::AdaBoost.BoostType=AdaBoost.AdaBoostM2()) =
	adaboost((getx!(x), gety(x)), f, L; boost_type=boost_type)

adaboost(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, f::F where F<:CellFunU, L::Int; boost_type::AdaBoost.BoostType=AdaBoost.AdaBoostM2()) = 
	adaboost((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), f, L; boost_type=boost_type)

adaboost(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, f::F where F<:CellFunU, L::Int; boost_type::AdaBoost.BoostType=AdaBoost.AdaBoostM2()) = 
begin

	@assert nobs(x[1]) == nobs(x[2]) "[adaboost] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)
	
	# Train model
	ensembledata = AdaBoost.adaboost_train(x[1], yenc, L, f.f, nothing, boost_type)
	
	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(adaboost, Model(ensembledata, modelprops), "Adaboost ensemble ($(f.tinfo), $L members)")
end


# Execution
adaboost(x::T where T<:CellData, model::Model{<:AdaBoost.AdaBoostEnsemble}) =
	datacell(adaboost(getx!(x), model), gety(x)) 	
adaboost(x::T where T<:AbstractVector, model::Model{<:AdaBoost.AdaBoostEnsemble}) =
	adaboost(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
adaboost(x::T where T<:AbstractMatrix, model::Model{<:AdaBoost.AdaBoostEnsemble}) =
	AdaBoost.adaboost_exec(model.data,x)
