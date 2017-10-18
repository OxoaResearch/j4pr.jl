module ROC

	using ROCAnalysis, MLLabelUtils
	export AbstractOP, ComplexOP, SimpleOP, findop, changeop!, simpleop, 
		AbstractPerfMetric, TPr, FPr, TNr, FNr

	abstract type AbstractOP end

	"""	
	Operating point object with additional information.
	"""
	mutable struct ComplexOP{T} <: AbstractOP
		targetclass::Int		# index of the target class
		weights::Matrix{Float64}	# weight matrix (class weights present only on the diagonal)
		r::ROCAnalysis.Roc{T}		# ROC thresholds and performance measures (derived from ROCAnalysis)
		pos::Int			# position of chosen treshold in r.θ
		
		function ComplexOP(targetclass::Int, weights::Matrix{Float64}, r::ROCAnalysis.Roc{T}, pos::Int) where T
			@assert isdiag(weights) "Weight matrix for operating point objects has to be diagonal."
			@assert pos >0 && pos <=length(r.θ) "Position of threhsold does not match length of the threshold vector"
			@assert targetclass>0 && targetclass<=size(weights,2) "The target class has to be an integer between 0 and $(size(weights,2))"
			return new{T}(targetclass, weights, r, pos)
		end
	end

	ComplexOP(targetclass::Int, weights::T where T<:AbstractVector{<:Real}, r::ROCAnalysis.Roc, pos::Int) = 
		ComplexOP(targetclass, Matrix{Float64}(diagm(weights)), r, pos)	
	
	Base.show(io::IO, op::ComplexOP) = print(io, "Complex operating point object, pos=$(op.pos)/$(length(op.r.θ)),"* 
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


	
	# ROC generation 
	
	# Define threshold search functions
	_θ_search_(r::ROCAnalysis.Roc, pm::TPr, threshold::Float64) = begin 
		pos = findlast(1-r.pmiss[1:end-1].>=threshold) 
		pos = ifelse(pos==0, 1, pos)
		return r.θ[pos], pos
	end
	
	_θ_search_(r::ROCAnalysis.Roc, pm::FNr, threshold::Float64) = begin
		pos = findlast(r.pmiss[1:end-1].<=threshold)
		return r.θ[pos], pos
	end

	_θ_search_(r::ROCAnalysis.Roc, pm::FPr, threshold::Float64) = begin
		pos = findlast(r.pfa[1:end-1].>=threshold)
		pos = ifelse(pos==0, 1, pos)
		return r.θ[pos], pos
	end
	
	_θ_search_(r::ROCAnalysis.Roc, pm::TNr, threshold::Float64) = begin 
		pos = findlast(1-r.pfa[1:end-1].<=threshold) 
		return r.θ[pos], pos
	end
	

	# Define op generation functions
	findop(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractVector{N}, targetclass::N, 
			pm::AbstractPerfMetric, threshold::Float64) where N =
		findop(x[1], x[2], targetclass, pm, threshold) 

	findop(x::T where T<:AbstractMatrix, y::S  where S<:AbstractVector{N}, targetclass::N, 
			pm::AbstractPerfMetric, threshold::Float64) where N = begin
		# Checks
		yu::Vector{N} = sort(unique(y))
		C = length(yu)
		@assert size(x,1) == length(unique(y)) "Number of unique labels does not match dimensionality of class-wise probabilities"
		@assert targetclass in yu "Target class does not seem to be present on list of classes" 
		
		threshold = clamp(threshold, 0.0, 1.0)
		
		# Preprocess the matrix of probabilities (transform from probability to log-likelihood ration scores)
		tidx = findin(yu, [targetclass])[1] # find index corresponding to the targetclass 
		ntidx = setdiff(1:C, tidx)     
		tvalues = 2 * x[tidx, y.==targetclass] - 1 
		ntvalues = 2* x[tidx, y.!=targetclass] - 1   
		
		# Call ROCAnalysis
		r = ROCAnalysis.roc(tvalues, ntvalues)
		
		# Generate and update op
		op = ComplexOP(tidx, eye(C), r, 1)
		changeop!(op, pm, threshold)
		
		return op
	end
	

	# Define op changing function
	changeop!(op::ComplexOP, pm::AbstractPerfMetric, threshold::Float64) = begin
			
		# Call threshold search
		θ::Float64, pos::Int = _θ_search_(op.r, pm, threshold) 

		# Update position in op
		op.pos = pos

		# Construct the weights
		θ = (θ+1.0)/2.0
		op.weights[diagind(op.weights)] = 1.0/(2*(1-θ+eps())); 		# update diagonal elements
		op.weights[op.targetclass, op.targetclass] = 1.0/(2*θ+eps())	# update target class weights
		nothing;
	end

	changeop!(op::ComplexOP, pos::Int64=1) = begin
			
		# Check position
		@assert pos >= 1 && pos <= length(op.r.θ) "[changeop!] OP position has to be an integer between 1 and $(length(op.r.θ))."
		
		# Get threshold directly
		θ::Float64 = op.r.θ[pos] 
		
		# Update position in op
		op.pos = pos

		# Construct the weights 
		θ = (θ+1.0)/2.0
		op.weights[diagind(op.weights)] = 1.0/(2*(1-θ+eps())); 		# update diagonal elements
		op.weights[op.targetclass, op.targetclass] = 1.0/(2*θ+eps())	# update target class weights
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
	findop(class, pm, val)

Constructs an untrained cell that when piped data inside, returns the operating point 
trying to optimize the performance metric `pm` around the value `val` for class `class`.

# Arguments
  * `class` can be either a specific class i.e. "setosa" or an index in the sorted class list
  * `pm::ROC.AbstractPerfMetric` is the performance metric. Can be `ROC.TPr()`, `ROC.TNr()`, `ROC.FPr()` or `ROC.FNr()`
  * `val::Float64` is the value desired for the metric
"""
findop(class, pm::ROC.AbstractPerfMetric, val::Float64) = 
	FunctionCell(findop, (class, pm, clamp(val,0.0,1.0)), ModelProperties(), "OP finder, on \"$class\", targeting $pm@$val") 



############################
# DataCell/Array Interface #
############################

"""
	findop(x, class, pm, val)

Searches for the operating point trying to optimize the performance metric `pm` around the value 
`val` for class `class`. The data `x` is assumed to be either a `Tuple` or a `DataCell` that contain
class probabilities and true labels.
"""
# Training
findop(x::T where T<:CellDataL, class, pm::ROC.AbstractPerfMetric, val::Float64) = 
	findop((getx!(x), gety(x)), class, pm, val)
findop(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, class, pm::ROC.AbstractPerfMetric, val::Float64) = 
	findop((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), class, pm, val)
findop(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, class, pm::ROC.AbstractPerfMetric, val::Float64) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[findop] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Transform labels first
	enc = labelencn(x[2])
	origclass = (class isa Int) ? ind2label(class,enc) : class
	
	# Get op
	op = ROC.findop(x, origclass, pm, clamp(val,0.0,1.0))

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
