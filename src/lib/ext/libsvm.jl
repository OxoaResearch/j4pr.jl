##########################
# FunctionCell Interface #	
##########################
"""
	libsvm([d] [;kwargs])

Generates a function cell that when piped data, trains a SVM classifier/regressor using the LIBSVM.jl interface.
`d::Distances.PreMetric` is a distance function that can be used to pre-compute the kernel.

# Keyword arguments (from LIBSVM.jl, defaults modified in some cases)
  * `svmtype::Type=LIBSVM.SVC`: Type of SVM to train `SVC` (for C-SVM), `NuSVC`,
    `OneClassSVM`, `EpsilonSVR` or `NuSVR`. Defaults to `OneClassSVM` if
      no labels are provided.
  * `kernel::Kernel.KERNEL=Kernel.Linear`; available: `Linear`, `Polynomial`
    `RadialBasis`, `Sigmoid` or `Precomputed`. If `d` is provided, the default is `Kernel.Precomputed`
  * `degree::Integer=1`: Kernel degree. Used for polynomial kernel
  * `gamma::Float64=1.0/size(X, 1)` : γ for kernels
  * `coef0::Float64=0.0`: parameter for sigmoid and polynomial kernel
  * `cost::Float64=1.0`: cost parameter C of C-SVC, epsilon-SVR, and nu-SVR
  * `nu::Float64=0.5`: parameter nu of nu-SVC, one-class SVM, and nu-SVR
  * `epsilon::Float64=0.1`: epsilon in loss function of epsilon-SVR
  * `tolerance::Float64=0.001`: tolerance of termination criterion
  * `shrinking::Bool=true`: whether to use the shrinking heuristics
  * `probability::Bool=true`: whether to train a SVC or SVR model for probability estimates
  * `weights::Union{Dict{T, Float64}, Void}=nothing`: dictionary of class weights
  * `cachesize::Float64=4000.0`: cache memory size in MB
  * `verbose::Bool=false`: print training output from LIBSVM if true

!!! note 
	For regression (`svmtype=LIBSVM.NuSVR`, or `svmtype=LIBSVM.EpsilonSVR`) `probability` 
	must be set explicitly to `false` otherwise Julia might crash (e.g. unsafe_copy! problem). 	 
	The problem is present in Julia 0.6 and LIBSVM v.0.1.0

Read the `LIBSVM.jl` documentation for more information.  
"""
libsvm(;kwargs...) = FunctionCell(libsvm, (), Dict(), kwtitle("LIBSVM",kwargs); _libsvm_parse_kwargs_(nothing, kwargs)...)
libsvm(d::T where T<:Distances.PreMetric; kwargs...) = FunctionCell(libsvm, (d,), Dict(),  kwtitle("LIBSVM",kwargs); _libsvm_parse_kwargs_(d, kwargs)...) 



############################
# DataCell/Array Interface #	
############################
"""
	libsvm(x, [,d] [;kwargs])

Generates a trained function cell by training a SVM classifier/regressor using the LIBSVM.jl interface. 
`x`::DataCell is the training data and `d::Distances.PreMetric` is a distance function that can be used 
to pre-compute the kernel.
"""
# Training
libsvm(x::T where T<:CellData; kwargs...) = libsvm((getx!(x), gety(x)), nothing; _libsvm_parse_kwargs_(nothing, kwargs)...)

libsvm(x::T where T<:CellData, d::S where S<:Distances.PreMetric; kwargs...) = libsvm((getx!(x), gety(x)), d; _libsvm_parse_kwargs_(d, kwargs)...)

libsvm(x::T where T<:AbstractArray, d::Union{Void, Distances.PreMetric}=nothing; kwargs...) = libsvm((x, nothing), d; _libsvm_parse_kwargs_(d, kwargs)...)

libsvm(x::Tuple{T,S} where T<:AbstractVector where S<:Union{Void,AbstractVector}, d::Union{Void, Distances.PreMetric}=nothing; kwargs...) = 
	libsvm((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), d; _libsvm_parse_kwargs_(d, kwargs)...)

libsvm(x::Tuple{T,S} where T<:AbstractMatrix where S<:Union{Void,AbstractVector}, d::Union{Void, Distances.PreMetric}=nothing; kwargs...) = begin

	newkwargs = _libsvm_parse_kwargs_(d, kwargs)

	# If no labels present, change to OneClassSVM, otherwise check label size 
	if x[2] isa Void
		newkwargs[:svmtype] = LIBSVM.OneClassSVM 
	else
		@assert nobs(x[1]) == nobs(x[2]) "[libsvm] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	end
	
	# Transform labels first if svmtype if SVC or NuSVC
	yu = []; 
	yenc=[];
	clf = true
	if isequal(newkwargs[:svmtype], LIBSVM.SVC) || isequal(newkwargs[:svmtype], LIBSVM.NuSVC)
		yu = sort(unique(x[2]))
		yenc = Vector{Int}(ohenc_integer(x[2],yu)) # encode to Int labels based on position in the sorted vector of unique labels 
	else
		yenc = x[2] # regression or oneclass svm, do not change
		clf = false
	end

	# Train model
	if d isa Void
		# Use internal kernel
		if isequal(newkwargs[:svmtype], LIBSVM.OneClassSVM)
			modeldata = LIBSVM.svmtrain(getobs(x[1]); newkwargs...) 					# one-class SVM
		else
			modeldata = LIBSVM.svmtrain(getobs(x[1]), yenc; newkwargs...)					# classifier/regressor	
		end
	else
		# Use precomputed kernels
		if isequal(newkwargs[:svmtype], LIBSVM.OneClassSVM)
			modeldata = LIBSVM.svmtrain([collect(1:nobs(x[1]))';x[1] |> dist(x[1])]; newkwargs...) 		# one-class SVM
		else
			modeldata = LIBSVM.svmtrain([collect(1:nobs(x[1]))';x[1] |> dist(x[1])], yenc; newkwargs...) 	# classifier/regressor	
		end
	end

	# Keep track of how the labeles were re-ordered
	reorder = indexin(1:length(yu), modeldata.labels) # it is possible to use 1:length(yu) because the unique labels are sorted
							  # and the labels are encoded according to the sorted unique labels (i.e. position)
	# Build model properties 
	modelprops = Dict("size_in" => nvars(x[1]),
		   	  "size_out" => clf ? length(yu) : 1, # the output size is the number of classes (e.g. 1 prob/class) for classification, 1 for regression 
			  "kernel_distance" => d,
			  "labels" => yu,
	)

	# Build title
	title = "LIBSVM: svmtype=$(newkwargs[:svmtype]), kernel=$(newkwargs[:kernel]), probability=$(newkwargs[:probability]), cachesize=$(newkwargs[:cachesize])(MB)" 
	
	# Return trained cell (retain both SVM model and initial training data)
	FunctionCell(libsvm, Model((modeldata, getobs(x[1]), reorder)), modelprops, title)
end



# Execution
libsvm(x::T where T<:CellData, model::Model{<:Tuple}, modelprops::Dict) = datacell(libsvm(getx!(x), model, modelprops), gety(x)) 	
libsvm(x::T where T<:AbstractVector, model::Model{<:Tuple}, modelprops::Dict) = libsvm(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
libsvm(x::T where T<:AbstractMatrix, model::Model{<:Tuple}, modelprops::Dict)::Matrix{Float64} = 
	_libsvm_execute_(model.data[1].SVMtype, model.data[1], x, modelprops["kernel_distance"], model.data[2], model.data[3])  

# Define execution methods (for return type annotations) and coversion of results to matrices
_libsvm_execute_(::Type{LIBSVM.OneClassSVM}, model_data, x, ::Void, xd, reorder=Int[]) = 
	mat(LIBSVM.svmpredict(model_data, x)[1], LearnBase.ObsDim.Constant{2}())

_libsvm_execute_(::Type{LIBSVM.OneClassSVM}, model_data, x, d, xd, reorder=Int[]) = 
	mat(LIBSVM.svmpredict(model_data, [collect(1:nobs(x))'; x |> dist(xd, d)])[1], LearnBase.ObsDim.Constant{2}())

_libsvm_execute_(::Union{Type{LIBSVM.EpsilonSVR}, Type{LIBSVM.NuSVR}}, model_data, x, ::Void, xdm, reorder=Int[]) = 
	mat(LIBSVM.svmpredict(model_data, x)[1], LearnBase.ObsDim.Constant{2}())

_libsvm_execute_(::Union{Type{LIBSVM.EpsilonSVR}, Type{LIBSVM.NuSVR}}, model_data, x, d, xd, reorder=Int[]) = 	
	mat(LIBSVM.svmpredict(model_data, [collect(1:nobs(x))'; x |> dist(xd, d)])[1], LearnBase.ObsDim.Constant{2}())

_libsvm_execute_(::Union{Type{LIBSVM.SVC}, Type{LIBSVM.NuSVC}}, model_data, x, ::Void, xd, reorder::Vector{Int}) = 
	LIBSVM.svmpredict(model_data, x)[2][reorder,:]

_libsvm_execute_(::Union{Type{LIBSVM.SVC}, Type{LIBSVM.NuSVC}}, model_data, x, d, xd, reorder::Vector{Int}) = 	
	LIBSVM.svmpredict(model_data, [collect(1:nobs(x))'; x |> dist(xd, d)])[2][reorder,:]



# A function that generates default keyword arguments
function _libsvm_parse_kwargs_(d, kwargs)
	
	outkwargs = Dict(
	   (:svmtype => LIBSVM.SVC), # SVM type
	   (:kernel => d isa Void ? LIBSVM.Kernel.Linear : LIBSVM.Kernel.Precomputed),
	   (:degree => 1), 
	   #(:gamma => 1.0/size(X,1)),  	
	   (:coef0 => 0.0),
	   (:cost => 1.0),
	   (:nu => 0.5), 
	   (:epsilon => 0.1),	
	   (:tolerance => 0.001),
	   (:shrinking => true),
	   (:probability => true),
	   (:weights => nothing),
	   (:cachesize => 4000.0), 	# cache 4GBs
	   (:verbose => false)
	)

	# Loop through any keyword arguments and modify the defaults
	for (kwarg, argval) in kwargs
		if kwarg == :svmtype 
			# Check if we have regression, if so , make probability=false
			if argval != LIBSVM.SVC && argval != LIBSVM.NuSVC 
				outkwargs[:probability] = false
			end
		end
		
		# Update all default arguments with new values
		if kwarg in keys(outkwargs)
			outkwargs[kwarg] = argval
		else
			push!(outkwargs, kwarg=>argval)
		end
	end

	# Return concatenation of defaults and keyword arguments; any arguments from the 
	# defaults that appear in kwargs explicitly will be modified, as the should be.
	return outkwargs
end
