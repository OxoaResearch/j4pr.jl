module ROC

	using ROCAnalysis, MLLabelUtils
	using StatsBase: sample
	
	export AbstractOP, ComplexOP, SimpleOP, findop, changeop!, simpleop, 
		AbstractPerfMetric, TPr, FPr, TNr, FNr
	
	abstract type AbstractOP end

	"""	
	Operating point object with additional information.
	"""
	mutable struct ComplexOP <: AbstractOP
		targetclass::Int		# index of the target class
		weights::Matrix{Float64}	# weight matrix (class weights present only on the diagonal)
		rocdata::Matrix{Float64}	# Three column data matrix with the columns: θ i.e. threshold, FPr and FNr
		pos::Int			# position of chosen treshold in r.θ
		
		function ComplexOP(targetclass::Int, weights::Matrix{Float64}, rocdata::Matrix{Float64}, pos::Int)
			@assert isdiag(weights) "Weight matrix for operating point objects has to be diagonal."
			@assert pos>0 && pos<=size(rocdata,1) "Position of threhsold does not match length of the threshold vector"
			@assert targetclass>0 && targetclass<=size(weights,2) "The target class has to be an integer between 0 and $(size(weights,2))"
			return new(targetclass, weights, rocdata, pos)
		end
	end

	ComplexOP(targetclass::Int, weights::T where T<:AbstractVector{<:Real}, rocdata::Matrix{Float64}, pos::Int) = 
		ComplexOP(targetclass, Matrix{Float64}(diagm(weights)), rocdata, pos)	
	
	Base.show(io::IO, op::ComplexOP) = print(io, "Complex operating point object, pos=$(op.pos)/$(size(op.rocdata,1)),"* 
					  		" targetclass=$(op.targetclass), weights=$(diag(op.weights))")
	


	"""	
	Operating point object with no additional information.
	"""
	mutable struct SimpleOP <: AbstractOP
		weights::Matrix{Float64}	# weight matrix (class weights present only on the diagonal)
	end
	
	SimpleOP(w::T where T<:AbstractVector{<:Real}) = SimpleOP(Matrix{Float64}(diagm(w)))	
	SimpleOP(cop::ComplexOP) = SimpleOP(cop.weights)	
	
	Base.show(io::IO, op::SimpleOP) = print(io, "Simple Operating point object, weights=$(diag(op.weights))")



	# Define performance metric types 
	abstract type AbstractPerfMetric end
	struct TPr <: AbstractPerfMetric end 	# true positive rate (sensitivity)
	struct TNr <: AbstractPerfMetric end 	# true negative rate (specificity)
	struct FPr <: AbstractPerfMetric end 	# false positive rate (fall-out)
	struct FNr <: AbstractPerfMetric end 	# false negative rate


	Base.show(io::IO, apm::TPr) = print(io, "TPr")
	Base.show(io::IO, apm::TNr) = print(io, "TNr")
	Base.show(io::IO, apm::FPr) = print(io, "FPr")
	Base.show(io::IO, apm::FNr) = print(io, "FNr")

	
	
	# Define threshold search functions
	_θ_search_(rocdata::AbstractMatrix{Float64}, pm::TPr, threshold::Float64) = begin 
		pos = findlast(view(rocdata,:,3).<=1-threshold) # 1-rocdata[:,3].>=threshold 
		pos = ifelse(pos==0, 1, pos)
		return rocdata[pos,1], pos
	end
	
	_θ_search_(rocdata::AbstractMatrix{Float64}, pm::FNr, threshold::Float64) = begin
		pos = findlast(view(rocdata,:,3).<=threshold) 
		return rocdata[pos,1], pos
	end

	_θ_search_(rocdata::AbstractMatrix{Float64}, pm::FPr, threshold::Float64) = begin
		pos = findlast(view(rocdata,:,2).>=threshold)
		pos = ifelse(pos==0, 1, pos)
		return rocdata[pos,1], pos
	end
	
	_θ_search_(rocdata::AbstractMatrix{Float64}, pm::TNr, threshold::Float64) = begin 
		pos = findlast(rocdata[:,2].>=1-threshold) # 1-rocdata[:,2].<=threshold 
		return rocdata[pos,1], pos
	end


	# Define functions that generate the 'rocdata' i.e. threhsholds, false positive and false negative rates
	_get_rocdata_(x::AbstractMatrix{T}, y::AbstractVector{S}, yu::AbstractVector{S}, targetclass::S, maxops::Int=100) where {T,S} = begin 
		idx = findin(yu, [targetclass])[1] # find index corresponding to the targetclass 
		n = size(x,2)

		# Define maximum number of operating points 
		# and sample thresholds if necessary
		positives = view(x, idx, y.==targetclass)
		upositives::Vector{T} = sort(unique(positives))
		if length(upositives) + 2 > maxops 
			upositives = sort(sample(upositives, maxops-2, replace=false))
		end
		
		@show length(upositives)
		# Pre-allocate output and calculate false positives, false negatives for the data
		rocdata = zeros(Float64, maxops, 3)
		rocdata[1,:] = [0.0, 1.0, 0.0]						# minimum threshold case
		rocdata[maxops,:] = [1.0, 0.0, 1.0]					# maximum threshold case
		nP = length(positives)
		nN = n-nP
		
		fp::Float64 = 0.0
		fn::Float64 = 0.0
		@inbounds for (i,θ) in enumerate(upositives)
			fp = 0.0
			fn = 0.0
			# Count false positives and false negatives for the current threshold
			@simd for j in 1:n
				fp = fp + ifelse( (y[j]!=targetclass) && (x[idx,j]*(1.0-θ)/θ >= 1.0-x[idx,j]), 1.0, 0.0) 
				fn = fn + ifelse( (y[j]==targetclass) && (x[idx,j]*(1.0-θ)/θ < 1.0-x[idx,j]), 1.0, 0.0) 
			end
			rocdata[i+1,1] = upositives[i]
			rocdata[i+1,2] = fp/nP
			rocdata[i+1,3] = fn/nN
		end
		return idx, rocdata 
	end	
	
	_get_rocdata_ra_(x::AbstractMatrix, y::AbstractVector{S}, yu::AbstractVector{S}, targetclass::S, maxops::Int=100) where S = begin
		# Preprocess the matrix of probabilities (transform from probability to log-likelihood ration scores)
		idx = findin(yu, [targetclass])[1] 					# find index corresponding to the targetclass 
		notidx = setdiff(1:length(yu), idx)     				# find indexes corresponding to the other classes 	
		positives = 2 * x[idx, y.==targetclass] - 1 				# process probabilities for ROCAnalysis.jl
		negatives = 2* x[idx, y.!=targetclass] - 1   				
	
		# Call ROCAnalysis
		r = ROCAnalysis.roc(positives, negatives)
		
		# If more ops than needed, limit their number
		nops = length(r.θ)+1
		opidx = Vector{Int}(maxops)
		if nops > maxops
			opidx[1] = 1; opidx[end] = nops
			opidx[2:end-1] = sort(sample(2:nops-1, maxops-2, replace=false))
		end
		rocdata = [[(1.0+r.θ)/2.0;1.0][opidx] r.pfa[opidx] r.pmiss[opidx]]
		return idx, rocdata
	end

	
	# Define op generation functions
	findop(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractVector{N}, targetclass::N, 
			pm::AbstractPerfMetric, threshold::Float64, method::Symbol=:j4pr, maxops::Int=100) where N =
		findop(x[1], x[2], targetclass, pm, threshold, method, maxops) 

	findop(x::T where T<:AbstractMatrix, y::S  where S<:AbstractVector{N}, targetclass::N, 
			pm::AbstractPerfMetric, threshold::Float64, method::Symbol=:j4pr, maxops::Int=100) where N = begin
		
		# Checks
		yu = sort(unique(y))
		@assert size(x,1) == length(yu) "The number of unique labels does not match dimensionality of class-wise probabilities"
		@assert size(x,2) == length(y) "The number of labels does not match the number of observations"
		@assert targetclass in yu "Target class does not seem to be present in list of classes" 
		
		
		# Apply appropriate ROC analysis method
		if method == :j4pr
			idx, rocdata = _get_rocdata_(x, y, yu, targetclass, maxops)
		elseif method == :ra
			idx, rocdata = _get_rocdata_ra_(x, y, yu, targetclass, maxops)
		else
			warn("[findop] Unknown method for op selection, using :j4pr.")
			idx, rocdata = _get_rocdata_(x, y, yu, targetclass, maxops)
		end

		# Generate op data and create object
		op = ComplexOP(idx, eye(length(yu)), rocdata, 1)
		
		# Search for best operating point
		changeop!(op, pm, clamp(threshold, 0.0, 1.0))
		
		return op
	end
	

	# Define op changing function
	changeop!(op::ComplexOP, pm::AbstractPerfMetric, threshold::Float64) = begin
			
		# Call threshold search
		θ::Float64, pos::Int = _θ_search_(op.rocdata, pm, threshold) 

		# Update position in op
		op.pos = pos

		# Construct the weights
		op.weights[diagind(op.weights)] = 1.0; 					# update diagonal elements
		op.weights[op.targetclass, op.targetclass] = (1-θ+eps())/(θ+eps()) 	# update target class weights
		nothing;
	end

	changeop!(op::ComplexOP, pos::Int64=1) = begin
			
		# Check position
		@assert pos >= 1 && pos <= size(op.rocdata,1) 
			"[changeop!] OP position has to be an integer between 1 and $(size(op.rocdata,1))."
		
		# Get threshold directly
		θ::Float64 = op.rocdata[pos,1] 
		
		# Update position in op
		op.pos = pos

		# Construct the weights 
		op.weights[diagind(op.weights)] = 1.0; 					# update diagonal elements
		op.weights[op.targetclass, op.targetclass] = (1-θ+eps())/(θ+eps()) 	# update target class weights
		nothing;
	end


	# Conversion function from a complex op to a simple op
	simpleop(op::ComplexOP) = SimpleOP(op)	
	simpleop(weights::T where T<:AbstractVector{<:Real}) = SimpleOP(diagm(float.(weights)))
	simpleop(weights::T where T<:AbstractMatrix{<:Real}) = begin
		@assert isdiag(weights) && (size(weights,1)==size(weights,2)) "[simpleop] Weight matrix has to be square and diagonal"
		SimpleOP(float.(weights))
	end
end



##########################
# FunctionCell Interface #
##########################
"""
	findop(class, pm, val, method, maxops)

Constructs an untrained cell that when piped data inside, returns the operating point 
trying to optimize the performance metric `pm` around the value `val` for class `class`.

# Arguments
  * `class` can be either a specific class i.e. "setosa" or an index in the sorted class list
  * `pm::ROC.AbstractPerfMetric` is the performance metric. Can be `ROC.TPr()`, `ROC.TNr()`, `ROC.FPr()` or `ROC.FNr()`
  * `val::Float64` is the value desired for the metric
  * `method::Symbol` method through which to find the op; available `:j4pr` and `:ra` i.e. `ROCAnalysis.jl` (default `:j4pr`) 
  * `maxops::Int` is the maximum number of operating points to consider (default `100`)

# Examples
```
julia> D = DataGenerator.normal2d(2000);        # generate dataset
       (tr,ts) = splitobs(shuffleobs(D),0.1);   # split in training and testing 10%/90%
       w = knn(5,smooth=:dist);                 # define untrained classifier
       wt = w(tr);                              # train classifier
       out = ts|>wt;                            # calculate probabilities on test data
       vop = findop(1,ROC.FPr(),0.025,:j4pr);   # define an operating point search for the first class
       top = vop(out)                           # search for op where for the first class in FPr=0.025
       pt=wt + top;                             # make pipeline 
       # pass test data through pipe and calculate the confusion matrix 
       +ts |> pt |> x->j4pr.confusionmatrix(targets(indmax,x)-1, Int.(-ts), normalize=true, showmatrix=true)
       # plot ROC
       rocplot(top)

reference labels (columns):
  "0"   "1" 
------------------
0.72   0.025555555555555557   
0.28   0.9744444444444444   
------------------
       ┌────────────────────────────────────────┐ 
     1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡠⠴⠋⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉│ 
       │⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠀⠀⠀⠀⠀⠀⢠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠀⠀⢀⣠⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠀⢸⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
   0.5 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
       └────────────────────────────────────────┘ 
       0                                        1
```
"""
findop(class, pm::ROC.AbstractPerfMetric, val::Float64, method::Symbol=:j4pr, maxops::Int=100) = 
	FunctionCell(findop, (class, pm, clamp(val,0.0,1.0), method, maxops), ModelProperties(), 
	      		"OP finder, on \"$class\", targeting $pm@$val, method=$method, maxops=$maxops") 



############################
# DataCell/Array Interface #
############################

"""
	findop(x, class, pm, val, method, maxops)

Searches for the operating point trying to optimize the performance metric `pm` around the value 
`val` for class `class`. The data `x` is assumed to be either a `Tuple` or a `DataCell` that contain
class probabilities and true labels.
"""
# Training
findop(x::T where T<:CellDataL, class, pm::ROC.AbstractPerfMetric, val::Float64, method::Symbol=:j4pr, maxops::Int=100) = 
	findop((getx!(x), gety(x)), class, pm, val, method, maxops)
findop(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, class, pm::ROC.AbstractPerfMetric, val::Float64, 
       method::Symbol=:j4pr, maxops::Int=100) = findop((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), class, pm, val, method, maxops)
findop(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, class, pm::ROC.AbstractPerfMetric, val::Float64, 
       method::Symbol=:j4pr, maxops::Int=100) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[findop] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Transform labels first
	enc = labelencn(x[2])
	origclass = (class isa Int) ? ind2label(class,enc) : class
	
	# Get op
	op = ROC.findop(x, origclass, pm, clamp(val,0.0,1.0), method, maxops)

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), nvars(x[1]))
	
	FunctionCell(applyop, Model(op, modelprops), "OP, complex, on \"$origclass\"")
end



# Execution (operating point application)
applyop(x::T where T<:CellData, model::Model{<:ROC.AbstractOP}) = 
	datacell(applyop(getx!(x), model), gety(x)) 	

applyop(x::T where T<:AbstractVector, model::Model{<:ROC.AbstractOP}) = 
	applyop(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	

applyop(x::T where T<:AbstractMatrix, model::Model{<:ROC.AbstractOP}) =
	model.data.weights * x



##########################################
# Function interface for OP modification #
##########################################

# Change the op of a trained function cell
changeop!(op::T where T<:CellFunT{<:Model{<:ROC.ComplexOP}}, pm::ROC.AbstractPerfMetric, threshold::Float64) = begin
	ROC.changeop!(getx!(op).data, pm, threshold)
end

changeop!(op::T where T<:CellFunT{<:Model{<:ROC.ComplexOP}}, pos::Int) = begin
	ROC.changeop!(getx!(op).data, pos)
end



# Create a simple op
simpleop(weights::T where T<:AbstractArray{<:Real}) = 
	FunctionCell(applyop, Model(ROC.simpleop(float.(weights)), ModelProperties(length(weights), length(weights))),
	      "OP, simple") 

simpleop(op::T where T<:CellFunT{<:Model{<:ROC.ComplexOP}}) =
	FunctionCell(applyop, Model(ROC.simpleop(getx(op).data), getx(op).properties), 
	      "OP, simple") 
