# Decision stumps 
module DecisionStump 

	using StaticArrays, MLDataPattern
	using StatsBase: entropy, trim, sample
	using Distances: sqeuclidean
	using j4pr: countappw, countappw!, gini, misclassification, linearsplit, densitysplit
	export AbstractStump, StumpClassifier, StumpRegressor, stump_train, stump_exec	

	# Types
	abstract type AbstractStump end

	struct ClassificationStump{N} <: AbstractStump	
		idx::Int			# Index of the threshold variable 
		th::SVector{N}			# Thresholds (N=1 for ordinal/real split, N>1 for nominal split)
		prob::Matrix{Float64}		# Class probabilities corresponding to each class (lines) and threshold (columns)
	
		# Define consistency checks in inner constructor
		function ClassificationStump(idx::Int, th::SVector{N}, prob::Matrix{Float64}) where N
			@assert N == size(prob,2)-1 "The number of thresholds + 1 has to match the number of columns in the probability matrix"
			return new{N}(idx, th, prob)
		end
	end
	
	struct RegressionStump{N} <: AbstractStump	
		idx::Int			# Index of the threshold variable 
		th::SVector{N}			# Thresholds (N=1 for ordinal/real split, N>1 for nominal split)
		coeff::Matrix{Float64}		# Regression coefficient matrix (the number of rows is equivalent to the 
						# order of the polynomial fitted i.e. 1 row, a constant, 2 rows a line etc.)
	
		# Define consistency checks in inner constructor
		function RegressionStump(idx::Int, th::SVector{N}, coeff::Matrix{Float64}) where N
			@assert N == size(coeff,2)-1 "The number of thresholds has to match the number of columns in the regression coefficient matrix"
			return new{N}(idx, th, coeff)
		end
	end

	# Aliases
	const ClassificationStumpOR = ClassificationStump{1}					# stump classifier - ordinal or real variable
	const ClassificationStumpN{N} = ClassificationStump{N} where N 				# stump classifier - nominal variable
	const RegressionStumpOR = RegressionStump{1}						# stump regressor - ordinal or real variable
	const RegressionStumpN = RegressionStump{N} where N					# stump regressor - nominal variable
	
	# Printers
	Base.show(io::IO, m::ClassificationStumpOR) = print(io, "Decision stump, classification, var=$(m.idx), th=$(m.th[1])")
	Base.show(io::IO, m::ClassificationStumpN) = print(io, "Decision stump, classification, var=$(m.idx), $(length(m.th)) thresholds")
	Base.show(io::IO, m::RegressionStumpOR) = print(io, "Decision stump, regression, var=$(m.idx), th=$(m.th[1]), order $(size(m.coeff,2)-1) fit")
	Base.show(io::IO, m::RegressionStumpN) = print(io, "Decision stump, regression, var=$(m.idx), $(length(m.th)) thresholds, order 0 fit")
	


	# Calculate the information gain for ordinal or real variables, for all thresholds provided
	_gain_real_(x::AbstractVector{Float64}, y::AbstractVector{Int}, yu::AbstractVector{Int}, th::Vector{Float64}, fp::Function) = begin
		# Initialize 
		nt = length(th)									# number of thresholds	
		n = length(y)									# number of samples
		Imax::Float64 = -Inf;
		I::Float64=0.0
		pos::Int = -1;
		nleft::Float64 = 0.0
		@inbounds for i = 1:nt
			nleft = sum(x.<=th[i])/n
			I = fp(countappw(y,yu)) - nleft*fp(countappw(view(y,x.<=th[i]),yu)) - (1-nleft)*fp(countappw(view(y,x.>th[i]),yu))
			# Check if the information gain was improved and update
			if I >= Imax
				Imax = I							# update maximum gain
				pos = i 							# update threshold index
			end
		end
		if pos == -1 
			# No split is better	
			return Imax, [th[div(nt/2)]], [countappw(y,yu) countappw(y,yu)]
		else
			# Return split found
			return Imax, th[pos:pos], [countappw(view(y,x.<=th[pos]),yu) countappw(view(y,x.>th[pos]),yu)]
		end
	end

	# Calculate the information gain for nominal variables, for all thresholds provided
	_gain_nominal_(x::AbstractVector{Float64}, y::AbstractVector{Int}, yu::AbstractVector{Int}, th::Vector{Float64}, fp::Function) = begin
		# For nominal splits, N+1 thresholds, if N is the number of distinct values of the variable, the additional one for 'unknown'
		# 'th' is ignored
		ux = unique(x) 									# unique values of the variable (i.e. the thresholds)
		nt = length(ux)									# number of thresholds 
		n = length(y)  									# number of samples
		C = length(yu)									# number of classes
		priors = [sum(yi.==y)/(n+C) for yi in yu]					# priors
		probs = zeros(Float64, C, nt+1)							# preallocate probability matrix (last row for unseen values)
		probs[:,end] = priors
		
		@inbounds for i = 1:nt
			ym = view(y,x.==ux[i])							# the labels for a the leaf ux[i]
			#probs[:,i] = length(ym)/(n+C)*countappw(ym,yu) # with re-normalization to the total number of samples
			probs[:,i] = countappw(ym,yu) 						# count 
		end
		
		# Calculate gain
		Isplit = 0.0
		@inbounds @simd for i in 1:nt
			Isplit += fp(view(probs,:,i))
		end
		I = fp(priors) - Isplit # no actual threshold selection is made
		return I, ux, probs
	end

	# Calculate the fit error for ordinal or real variables, for all thresholds provided
	_regressor_real_(x::AbstractVector{Float64}, y::AbstractVector{Float64}, th::Vector{Float64}, fm::Function, fe::Function) = begin
		# Initialize 
		nt = length(th)									# number of thresholds	
		n = length(y)									# number of samples
		Emin::Float64 = Inf;
		E::Float64=0.0
		pos::Int = -1;
		nleft::Float64 = 0.0
		@inbounds for i = 1:nt
			nleft = sum(x.<=th[i])
			(nleft == 0 || nleft == n) && continue 					# skip theshold if extreme
			E = -fe(y,fm(x,y)[2]) + fe(view(y,x.<=th[i]), fm(view(x,x.<=th[i]),view(y,x.<=th[i]))[2]) + 
			                        fe(view(y,x.>th[i]), fm(view(x,x.>th[i]),view(y,x.>th[i]))[2])
			# Check if the error was improved and update
			if E <= Emin
				Emin = E							# update error difference
				pos = i 							# update threshold index
			end
		end

		if pos == -1
			return Emin, [th[div(nt,2)]], [fm(x,y)[1] fm(x,y)[1]]
		else	
			return Emin, th[pos:pos], [fm(view(x,x.<=th[pos]),view(y,x.<=th[pos]))[1] fm(view(x,x.>th[pos]),view(y,x.>th[pos]))[1]]
		end
	end

	# Calculate the fit error for nominal variables, for all thresholds provided
	_regressor_nominal_(x::AbstractVector{Float64}, y::AbstractVector{Float64}, th::Vector{Float64}, fm::Function, fe::Function) = begin
		# For nominal splits, N+1 thresholds, if N is the number of distinct values of the variable, the additional one for 'unknown'
		# 'th' is ignored
		ux = sort(unique(x))								# unique values of the variable (i.e. the thresholds)
		nt = length(ux)									# number of thresholds 
		n = length(y)  									# number of samples
		deflt = fm(x,y)[1]								# default coefficients if value is not found
		coeff = zeros(Float64, 2, nt+1)							# preallocate coefficient matrix (last row for unseen values)
		coeff[:,end] = deflt 
		
		Esplit = 0.0
		@inbounds @simd for i = 1:nt
			ym = view(y,x.==ux[i])							# select values corresponding to current unique data value
			c, ŷ = fm(fill(ux[i], length(ym)), ym)					# fit model
			coeff[:,i] = c								# get coefficients	
			Esplit += fe(ym,ŷ)							# update split error	
		end
		
		# Calculate the error difference
		E = Esplit - fe(y,fm(x,y)[2]) # no actual threshold selection is made
		return E, ux, coeff
	end



	"""
		stump_train(x,y [;kwargs])

	Trains a decision stump classifier using the data `x` and labels `y`.

	# Arguments
	  * `x::AbstractMatrix{Float64}` the input data; it is assumed that each column of the input matrix represents an observation
	  * `y::AbstractVector{Int}` the labels vector

	# Keyword arguments
	  * `crit::Symbol` is the purity criterion used to assess the split. Supported: `:gini`,`:entropy` and `:misclassification` (default `:gini`)
	  * `nthresh::Int` is the number of thresholds for a ordinal or real variable that will be investigated (default `10`)	
	  * `split::Symbol` describes the variable split used in generating the thresholds. Supported: `:linear`, `:density` (default `:linear`)
	  * `count::Int` is the number of samples in a variable that are discarded at both ends of its value interval in training; see `StatsBase.trim` (default `0`)
	  * `prop::Float64` is the proportion of samples in a variable that are discarded at both ends of its value interval in training; see `StatsBase.trim` (default `0.0`)
	  * `vartypes::Union{Dict{Int=>Symbol}, Symbol}` describes the type of variables from the dataset; if it is a `Symbol`, all variables 
	are assumed to be of the same type. If it is a `Dict{Int=>:Symbol}`, the variable indicated by the key is assumed to be of the type of the value; 
	two values are supported: `:nominal` and any other symbol; if a variable is specified as nominal, the classifier will have `N+1` leaves, where `N` is 
	the number of distinct values seen in training, the last leaf corresponding to unseen values. If the variable is real, only two leaves are present.
	"""
	# Training method (classification)
	function stump_train(x::AbstractMatrix{Float64}, y::AbstractVector{Int}; 
		      		crit::Symbol=:gini, 
		      		vartypes::Union{Dict{Int,Symbol},Symbol}=:real, 
		      		nthresh::Int=10, split::Symbol=:linear, 
				count::Int=0, prop::Float64=0.0)
		m = size(x,1)					# number of variables 
		n = size(x,2)   				# number of observations
		yu = sort(unique(y)) 				# unique classes
		C = length(yu)
		@assert C > 1 "[stump_train] Number of classes has to be larger than 1."
		
		
		# Parse purity criterion 
		if crit == :gini 
			purity_function = gini
		elseif crit == :entropy
			purity_function = entropy
		elseif crit == :misclassification
			purity_function = misclassification
		else
			warn("[stump_train] Unrecognized purity criterion, defaulting to ':gini'.")
			purity_function = gini
		end
		
		# Parse number of thresholds
		nthresh = max(nthresh, 1) # at least one

		# Parse variable split
		if split == :linear
			split_function = linearsplit
		elseif split == :density
			split_function = densitysplit
		else
			warn("[stump_train] Unrecognized variable split, defaulting to ':linear' split.")
			split_function = linearsplit
		end

		# Parse count and prop
		count = clamp(count, 0, div(n,4)) 	# do no remove more than half the values of the variable
		prop = clamp(prop, 0.0, 0.25) 		# do not remove more than half of the values of the variable

		# Parse variable types
		_parse_vartypes_!(gain, vartypes::Symbol) = begin
			for i in 1:length(gain)
				if vartypes == :nominal	
					gain[i] = _gain_nominal_
				else
					gain[i] = _gain_real_
				end
			end
			return gain
		end
		_parse_vartypes_!(gain, vartypes::Dict{Int,Symbol}) = begin
			for i in 1:length(gain)
				if i in keys(vartypes)
					if vartypes[i] == :nominal
						gain[i] = _gain_nominal_
					else
						gain[i] = _gain_real_
					end
				else
					gain[i] = _gain_real_
				end
			end
			return gain
		end

		# Parse the variable specifications to obtain the gain functions corresponding to each
		gain_functions = Vector{Function}(m)
		_parse_vartypes_!(gain_functions, vartypes)
	
		# Loop through variables, looking at the specified variable type
		Imax = -Inf;
		vidx = 0;									# the index of the best variable
		th = -Inf
		prob = Matrix(Float64[])
		@inbounds for i in 1:m
			variable = view(x,i,:)							# variable data
			tv = split_function(variable, nthresh, prop=prop, count=count)		# thresholds to be investivated
			I, t, v = gain_functions[i](variable, y, yu, tv, purity_function) 	# get the information gain, threshold and split for 	
	
			if I > Imax
				vidx = i; th = t; prob = v
			end
	
		end
		return ClassificationStump(vidx, SVector(th...), prob)
	end


	"""
		stump_train(x,y [;kwargs])

	Trains a decision stump regressor using the data `x` and regression targets `y`.

	# Arguments
	  * `x::AbstractMatrix{Float64}` the input data; it is assumed that each column of the input matrix represents an observation
	  * `y::AbstractVector{Float64}` the regression targets vector

	# Keyword arguments
	  * `model::Symbol` is the type of model assumed for the data. Supported: `:mean`,`:median` and `:linear` (default `:mean`); regardless of 
	the model, there are always two coefficients / leaf of the stump, corresponding to the linear and constant terms of a linear model.
	  * `errcrit::Function` A function that returns a measure of the difference between two vectors or scalars (default `Distances.sqeuclidean`)	
	  * `nthresh::Int` is the number of thresholds for a ordinal or real variable that will be investigated (default `10`)	
	  * `split::Symbol` describes the variable split used in generating the thresholds. Supported: `:linear`, `:density` (default `:linear`)
	  * `count::Int` is the number of samples in a variable that are discarded at both ends of its value interval in training; see `StatsBase.trim` (default `0`)
	  * `prop::Int` is the proportion of samples in a variable that are discarded at both ends of its value interval in training; see `StatsBase.trim` (default `0.0`)
	  * `vartypes::Union{Dict{Int=>Symbol}, Symbol}` describes the type of variables from the dataset; if it is a `Symbol`, all variables 
	are assumed to be of the same type. If it is a `Dict{Int=>:Symbol}`, the variable indicated by the key is assumed to be of the type of the value; 
	two values are supported: `:nominal` and any other symbol; if a variable is specified as nominal, the regressor will have `N+1` sets of coefficients, 
	where `N` is the number of distinct variable values seen in training, the last coefficients corresponding to unseen values. 
	If the variable is real, only two sets of coefficients are present.

	!!! Note
	  `model=:linear` is still experimental and it might produce bad results for small values or singular data matrices.
	"""
	# Training method (regression)
	function stump_train(x::AbstractMatrix{Float64}, y::AbstractVector{Float64}; 
				model::Symbol=:mean,
		      		errcrit::Function=sqeuclidean,
		      		vartypes::Union{Dict{Int,Symbol},Symbol}=:real, 
		      		nthresh::Int=10, split::Symbol=:linear, 
				count::Int=0, prop::Float64=0.0)
		m = size(x,1)					# number of variables 
		n = size(x,2)   				# number of observations
		
		# Define regression functions 
		# These have the same signature and same return arguments: 
		#	- a 2 coefficient regression vector
		#       - the fit of that model considering the inut data 
		_mean_ = (x,y::AbstractVector{Float64})->begin
			m = mean(y)
			return [0.0,m], fill(m,length(y))
		end
		
		_median_ = (x,y::AbstractVector{Float64})->begin
			m = median(y)
			return [0.0,m], fill(m,length(y))
		end
		
		_linear_ = (x::AbstractVector{Float64},y::AbstractVector{Float64})->begin
			if length(x)==1
				m=[0.0,y[1]]
			else
				m=[x+1e-12*rand() ones(x)]\y+1e-12*rand() #add just a bit of noise	
			end
			return m, m[1].*x.+m[2] # ŷ
		end
		
		# Parse model type 
		if  model == :mean
			model_function = _mean_
		elseif model == :median
			model_function = _median_ 
		elseif model == :linear
			model_function = _linear_
		else
			warn("[stump_train] Unrecognized model, defaulting to ':mean'.")
			model_function = _mean_
		end
		
		# Parse number of thresholds
		nthresh = max(nthresh, 1) # at least one

		# Parse variable split 
		if split == :linear
			split_function = linearsplit
		elseif split == :density
			split_function = densitysplit
		else
			warn("[stump_train] Unrecognized variable split, defaulting to ':linear' split.")
			split_function = linearsplit
		end

		# Parse count and prop
		count = clamp(count, 0, div(n,4)) 	# do no remove more than half the values of the variable
		prop = clamp(prop, 0.0, 0.25) 		# do not remove more than half of the values of the variable

		# Parse variable types
		_parse_vartypes_!(regressor, vartypes::Symbol) = begin
			for i in 1:length(regressor)
				if vartypes == :nominal	
					regressor[i] = _regressor_nominal_
					if model == :linear 
						warn("[stump_train] A ':linear' regression model is not supported with ':nominal' variables. Changing model to ':mean'.")
						model_function = _mean_
					end
				else
					regressor[i] = _regressor_real_
				end
			end
			return regressor
		end
		_parse_vartypes_!(regressor, vartypes::Dict{Int,Symbol}) = begin
			for i in 1:length(regressor)
				if i in keys(vartypes)
					if vartypes[i] == :nominal
						regressor[i] = _regressor_nominal_
						if model == :linear 
							warn("[stump_train] A ':linear' regression model is not supported with ':nominal' variables. Changing model to ':mean'.")
							model_function = _mean_
						end
					else
						regressor[i] = _regressor_real_
					end
				else
					regressor[i] = _regressor_real_
				end
			end
			return regressor
		end

		# Parse the variable specifications to obtain the corresponding regressor functions
		regressor_functions = Vector{Function}(m)
		_parse_vartypes_!(regressor_functions, vartypes)
	
		# Loop through variables, looking at the specified variable type
		Emin = Inf;
		vidx = 0;									# the index of the best variable
		th = Inf
		coeff = Matrix(Float64[])
		@inbounds for i in 1:m
			variable = view(x,i,:)							# variable data
			tv = split_function(variable, nthresh, prop=prop, count=count)		# thresholds to be investivated
			E, t, c = regressor_functions[i](variable, y, tv, model_function, errcrit) # get the error, threshold and split for 	
	
			if E < Emin
				vidx = i; th = t; coeff = c
			end
	
		end
		return RegressionStump(vidx, SVector(th...), coeff)
	end



	# Execution methods (classification)
	stump_exec(m::ClassificationStumpOR, x::Matrix{Float64})=begin 
		out = zeros(Float64, size(m.prob,1), size(x,2))
		@inbounds for j = 1:size(x,2)	
			# Check input value for the variable: take left side of prob if smaller, right if larger
			out[:,j] = ifelse(x[m.idx,j] <= m.th[1], m.prob[:,1], m.prob[:,2])
		end
		return out
	end
	
	stump_exec(m::ClassificationStumpN, x::Matrix{Float64})=begin 
		out = zeros(Float64, size(m.prob,1), size(x,2))
		@inbounds for j = 1:size(x,2)	
			# Search for input value of the variable in thresholds:
			# take corresponding prob if found, default if not
			#out[:,j] = ifelse(x[m.idx,j] in m.th, m.prob[:,indexin(x[m.idx,j:j],m.th)], m.prob[:,end])
			if x[m.idx,j] in m.th
				out[:,j] = m.prob[:,indexin(x[m.idx,j:j],m.th)]
			else
				out[:,j] = m.prob[:,end]
			end
		end
		return out
	end
	
	# Execution methods (regression)
	stump_exec(m::RegressionStumpOR, x::Matrix{Float64})=begin 
		out = zeros(Float64, 1, size(x,2))
		@inbounds @simd for j = 1:size(x,2)	
			# Check input value for the variable: take left side coefficients if smaller, right if larger
			out[:,j] = ifelse(x[m.idx,j] <= m.th[1], x[m.idx,j]*m.coeff[1,1]+m.coeff[2,1], # left side 
		     						 x[m.idx,j]*m.coeff[1,2]+m.coeff[2,2]  # right side
		     )
		end
		return out
	end

	stump_exec(m::RegressionStumpN, x::Matrix{Float64})=begin 
		out = zeros(Float64, 1, size(x,2))
		@inbounds @simd  for j = 1:size(x,2)	
			# Search for closest position in thresholds, not an exact match
			pos = findlast(x[m.idx,j].>m.th)
			pos = ifelse(pos==0,1,pos)
			out[:,j] = x[m.idx,j]*m.coeff[1,pos] + m.coeff[2,pos]
		end
		return out
	end

end





##########################
# FunctionCell Interface #
##########################
"""
	stump([;kwargs])

Constructs an untrained cell that when piped data inside, returns a decision stump
classifier trained function cell based on the input data and labels.

# Keyword arguments
  * `crit::Symbol` is the purity criterion used to assess the split. Supported: `:gini`,`:entropy` and `:misclassification` (default `:gini`)
  * `nthresh::Int` is the number of thresholds for a ordinal or real variable that will be investigated (default `10`)	
  * `split::Symbol` describes the variable split used in generating the thresholds. Supported: `:linear`, `:density` (default `:linear`)
  * `count::Int` is the number of samples in a variable that are discarded at both ends of its value interval in training; see `StatsBase.trim` (default `0`)
  * `prop::Int` is the proportion of samples in a variable that are discarded at both ends of its value interval in training; see `StatsBase.trim` (default `0.0`)
  * `vartypes::Union{Dict{Int=>Symbol}, Symbol}` describes the type of variables from the dataset; if it is a `Symbol`, all variables 
are assumed to be of the same type. If it is a `Dict{Int=>:Symbol}`, the variable indicated by the key is assumed to be of the type of the value; 
two values are supported: `:nominal` and any other symbol; if a variable is specified as nominal, the classifier will have `N+1` leaves, where `N` is 
the number of distinct values seen in training, the last leaf corresponding to unseen values. If the variable is real, only two leaves are present.
"""
stump(;kwargs...) = FunctionCell(stump, (), ModelProperties(), kwtitle("Decision stump classifier", kwargs); kwargs...) 



############################
# DataCell/Array Interface #
############################
"""
	stump(x [;kwargs])

Trains a decision stump classifcation model that using the data `x`.
"""
# Training
stump(x::T where T<:CellDataL; kwargs...) = stump((getx!(x), gety(x)); kwargs...)
stump(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector; kwargs...) = 
	stump((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]); kwargs...)
stump(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[stump] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."

	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)
	
	# Train model
	stumpdata = DecisionStump.stump_train(getobs(x[1]), yenc; kwargs...)

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(stump, Model(stumpdata, modelprops), kwtitle("Decision stump classifier, $(length(stumpdata.th)) threshold(s)", kwargs)) 
end



# Execution
stump(x::T where T<:CellData, model::Model{<:DecisionStump.ClassificationStump}) = 
	datacell(stump(getx!(x), model), gety(x)) 	

stump(x::T where T<:AbstractVector, model::Model{<:DecisionStump.ClassificationStump}) = 
	stump(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	

stump(x::T where T<:AbstractMatrix, model::Model{<:DecisionStump.ClassificationStump}) =
	DecisionStump.stump_exec(model.data, getobs(x))





##########################
# FunctionCell Interface #	
##########################
"""
	stumpr([;kwargs])

Constructs an untrained cell that when piped data inside, returns a decision stump 
trained regressor function cell.

# Keyword arguments
  * `model::Symbol` is the type of model assumed for the data. Supported: `:mean`,`:median` and `:linear` (default `:mean`); regardless of 
the model, there are always two coefficients / leaf of the stump, corresponding to the linear and constant terms of a linear model.
  * `errcrit::Function` A function that returns a measure of the difference between two vectors or scalars (default `Distances.sqeuclidean`)	
  * `nthresh::Int` is the number of thresholds for a ordinal or real variable that will be investigated (default `10`)	
  * `split::Symbol` describes the variable split used in generating the thresholds. Supported: `:linear`, `:density` (default `:linear`)
  * `count::Int` is the number of samples in a variable that are discarded at both ends of its value interval in training; see `StatsBase.trim` (default `0`)
  * `prop::Int` is the proportion of samples in a variable that are discarded at both ends of its value interval in training; see `StatsBase.trim` (default `0.0`)
  * `vartypes::Union{Dict{Int=>Symbol}, Symbol}` describes the type of variables from the dataset; if it is a `Symbol`, all variables 
are assumed to be of the same type. If it is a `Dict{Int=>:Symbol}`, the variable indicated by the key is assumed to be of the type of the value; 
two values are supported: `:nominal` and any other symbol; if a variable is specified as nominal, the regressor will have `N+1` sets of coefficients, 
where `N` is the number of distinct variable values seen in training, the last coefficients corresponding to unseen values. 
If the variable is real, only two sets of coefficients are present.

!!! Note
  `model=:linear` is still experimental and it might produce bad results for small values or singular data matrices.
"""
stumpr(;kwargs...) = FunctionCell(stumpr, (), ModelProperties(), kwtitle("Decision stump regressor", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	stumpr(x, [;kwargs])

Trains a decision stump regression model using the data `x`.
"""
# Training
stumpr(x::T where T<:CellDataL; kwargs...) = stumpr((getx!(x), gety(x)); kwargs...)
stumpr(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector; kwargs...) =
	stumpr((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]); kwargs...)
stumpr(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[stumpr] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Train model
	stumpdata = DecisionStump.stump_train(getobs(x[1]), getobs(x[2]); kwargs...)
	
	FunctionCell(stumpr, Model(stumpdata, ModelProperties(nvars(x[1]),1)), 
	      kwtitle("Decision stump regressor", kwargs)) 

end



# Execution
stumpr(x::T where T<:CellData, model::Model{<:DecisionStump.RegressionStump}) = 
	datacell(stumpr(getx!(x), model), gety(x)) 	
stumpr(x::T where T<:AbstractVector, model::Model{<:DecisionStump.RegressionStump}) = 
	stumpr(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
stumpr(x::T where T<:AbstractMatrix, model::Model{<:DecisionStump.RegressionStump}) = 
	DecisionStump.stump_exec(model.data, getobs(x))
