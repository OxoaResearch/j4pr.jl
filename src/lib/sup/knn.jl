# knn
module kNNClassifier
	import NearestNeighbors, Distances
	export kNNModel, kNN_train, kNN_exec
	
	using j4pr: countapp

	# kNNModel struct {T - type of neighbour, S - type of labels, U - type of distance}
	struct kNNModel{T<:Real, S<:Vector{<:Real}, U<:NearestNeighbors.NNTree, V<:Distances.Metric} 			
		k::T 						# k: T <: Real (if k is Int, do classic kNN, if k is Float, do range search)
		targets::S					# targets: Vector{T}, if T <: Float64, regression , classification otherwise
		treedata::U					# training data: a tree of sorts
		metric::V					# distance: Distances.Metric
		fsmooth::Function 				# smoother function::Function
		priors::Dict{Int,Float64}
	end

	# Aliases
	const kNNModelClassificationNN{T<:Int, S<:Vector{Int}} = kNNModel{T,S}
	const kNNModelClassificationR{T<:Float64, S<:Vector{Int}} = kNNModel{T,S}
	const kNNModelRegressionNN{T<:Int, S<:Vector{Float64}} = kNNModel{T,S}
	const kNNModelRegressionR{T<:Float64, S<:Vector{Float64}} = kNNModel{T,S}

	# Printers
	Base.show(io::IO, m::kNNModelClassificationNN) = print("KNNModel, classification, NN, k=$(m.k)")
	Base.show(io::IO, m::kNNModelClassificationR) = print("KNNModel, classification, range, r=$(m.k)")
	Base.show(io::IO, m::kNNModelRegressionNN) = print("KNNModel, regression, NN, k=$(m.k)")
	Base.show(io::IO, m::kNNModelRegressionR) = print("KNNModel, regression, range, r=$(m.k)")



	# Train method (classification)
	_parse_smooth_clf_(smooth::Symbol)::Function = begin
		# Check for smoothing option 
		if smooth == :none fsmooth = no_smoother!
		elseif smooth == :ml fsmooth = ml_smoother!
		elseif smooth == :laplace fsmooth = laplace_smoother!
		elseif smooth == :mest fsmooth = mest_smoother!
		elseif smooth == :dist fsmooth = dist_smoother!
		else 
			warn("Unrecognized posterior smoothing function, defaulting to :none")
			fsmooth = no_smoother!
		end
		return fsmooth
	end

	_parse_smooth_reg_(smooth::Symbol)::Function = begin
		# Check for smoothing option 
		if smooth == :ml fsmooth = ml_smoother_r!
		elseif smooth == :dist fsmooth = dist_smoother_r!
		else 
			warn("Unrecognized regression smoothing function, defaulting to `:ml`")
			fsmooth = ml_smoother_r!
		end
		return fsmooth
	end



	# Train methods (classification)
	kNN_train(x::Matrix{Float64}, y::Vector{Int}, k::Int; metric::Distances.Metric=Distances.Euclidean(), leafsize::Int=10, smooth::Symbol=:none) = 
		return kNNModel(k, y, NearestNeighbors.KDTree(x, metric, leafsize=leafsize), metric, _parse_smooth_clf_(smooth), Dict(yi=>sum(yi.==y)/length(y) for yi in unique(y)))
	kNN_train(x::Matrix{Float64}, y::Vector{Int}, k::Float64; metric::Distances.Metric=Distances.Euclidean(), leafsize::Int=10, smooth::Symbol=:none) = 
		return kNNModel(k, y, NearestNeighbors.BallTree(x, metric, leafsize=leafsize), metric, _parse_smooth_clf_(smooth), Dict(yi=>sum(yi.==y)/length(y) for yi in unique(y)))

	# Train methods (regression)
	kNN_train(x::Matrix{Float64}, y::Vector{Float64}, k::Int; metric::Distances.Metric=Distances.Euclidean(), leafsize::Int=10, smooth::Symbol=:ml) =
		return kNNModel(k, y, NearestNeighbors.KDTree(x, metric, leafsize=leafsize), metric, _parse_smooth_reg_(smooth), Dict{Int, Float64}())
	kNN_train(x::Matrix{Float64}, y::Vector{Float64}, k::Float64; metric::Distances.Metric=Distances.Euclidean(), leafsize::Int=10, smooth::Symbol=:ml) =
		return kNNModel(k, y, NearestNeighbors.BallTree(x, metric, leafsize=leafsize), metric, _parse_smooth_reg_(smooth), Dict{Int, Float64}())
	
	
	
	# Execution methods (classification)
	kNN_exec(m::kNNModelClassificationNN, x::Matrix{Float64}, classes::Vector{Int})=begin 
		idx, d = NearestNeighbors.knn(m.treedata, x, m.k, true)
		out = zeros(Float64, length(classes), size(x,2))
		@inbounds for j = 1:size(x,2)	
			m.fsmooth(view(out,:,j), m.targets[idx[j]], classes, d[j], m.priors)
		end
		return out
	end
	
	kNN_exec(m::kNNModelClassificationR, x::Matrix{Float64}, classes::Vector{Int})=begin 
		idx = NearestNeighbors.inrange(m.treedata, x, m.k, false)
		out = zeros(Float64, length(classes), size(x,2))
		@inbounds for j = 1:size(x,2)
			d = zeros(Float64, size(idx[j]))
			@inbounds for i = 1: length(d)
				d[i] = Distances.evaluate(m.metric, m.treedata.data[idx[j][i]], x[:,j])
			end
			m.fsmooth(view(out,:,j), m.targets[idx[j]], classes, d, m.priors)
		end
		return out
	end

	# Execution methods (regression)
	kNN_exec(m::kNNModelRegressionNN, x::Matrix{Float64})=begin 
		idx, d = NearestNeighbors.knn(m.treedata, x, m.k, true)
		out = zeros(Float64, 1, size(x,2))
		@inbounds for j = 1:size(x,2)	
			m.fsmooth(view(out,:,j), m.targets[idx[j]], d[j])
		end
		return out
	end
	
	kNN_exec(m::kNNModelRegressionR, x::Matrix{Float64})=begin 
		idx = NearestNeighbors.inrange(m.treedata, x, m.k, false)
		out = zeros(Float64, 1, size(x,2))
		@inbounds for j = 1:size(x,2)
			d = zeros(Float64, length(idx[j]))
			@inbounds for i = 1: length(d)
				d[i] = Distances.evaluate(m.metric, m.treedata.data[idx[j][i]], x[:,j])
			end
			m.fsmooth(view(out,:,j), m.targets[idx[j]], d)
		end
		return out
	end
	
	
	

	# Posterior smoothers (classification); parameters: labels of neighbors, unique labels in dataset and distances
	# All smoothers return a vector yu in length
	no_smoother!(out::T where T<:AbstractVector{Float64}, y::Vector{Int}, yu::Vector{Int}, d=zeros(length(y)), priors=Dict{Int,Float64}()) = begin
		n = countapp(y,yu) 		
		@inbounds out[findmax(n)[2]] = 1.0
		return out
	end
	ml_smoother!(out::T where T<:AbstractVector{Float64}, y::Vector{Int}, yu::Vector{Int}, d=zeros(length(y)), priors=Dict{Int,Float64}()) = begin
		n = countapp(y,yu) 
		@fastmath out[:] = n/length(y)
		return out
	end
	laplace_smoother!(out::T where T<:AbstractVector{Float64}, y::Vector{Int}, yu::Vector{Int}, d=zeros(length(y)), priors=Dict{Int,Float64}()) = begin
		n = countapp(y,yu) 
		@fastmath out[:] = (n+1.0)/(length(y)+length(yu))
		return out
	end

	mest_smoother!(out::T where T<:AbstractVector{Float64}, y::Vector{Int}, yu::Vector{Int}, d=zeros(length(y)), priors=Dict{Int,Float64}()) = begin
		n = countapp(y,yu) 
		p = [priors[i] for i in yu]
		m = 10.0
		@fastmath out[:] = (n+m*p)/(length(y)+m)
		return out
	end

	dist_smoother!(out::T where T<:AbstractVector{Float64}, y::Vector{Int}, yu::Vector{Int}, d::Vector{Float64}, priors=Dict{Int,Float64}()) = begin
		dc = zeros(Float64, size(yu))
		m = length(yu)
		n = length(y)
		for j in 1:m
			for i in 1:n 
				@inbounds if y[i] == yu[j]
					@fastmath dc[j] += 1/(eps()+d[i])
				end
			end
		end
		out[:] = dc / sum(dc)
		return out
	end

	# Posterior smoothers (regression)
	ml_smoother_r!(out::T where T<:AbstractVector{Float64}, y::Vector{Float64}, d=zeros(length(y))) = begin
		@fastmath out[1] = mean(y)
		return out
	end
	dist_smoother_r!(out::T where T<:AbstractVector{Float64}, y::Vector{Float64}, d::Vector{Float64}) = begin
		t1 = @. y*(1/(eps()+d))
		t2 = @. 1/(eps()+d)
		out[1] = sum(t1)/sum(t2)
		return out
	end
end



##########################
# FunctionCell Interface #
##########################
"""
	knn(k=1 [;kwargs])

Constructs an untrained cell that when piped data inside, returns a knn classifier trained
function cell based on the input data and labels.

# Arguments
  * `k::Int` represents the number of neighbours to be considered
  * `k::Float64=1.0` represents a range in which neighboring samples are to be searched (if no neighbours are found, 
  equal class posteriors are generated)
  
# Keyword arguments
  * `metric::Distances.Metric` metric from `Distances.jl` which specifies how distances are calculated between samples (default `Distances.Euclidean()`)
  * `leafsize::Int` Size of the leafs for the tree structures holding the training data (default `10`)
  * `smooth::Symbol` method for smoothing class posteriors; available: `:none`, `:ml`, `:laplace`,`mest (m=10)` and `:dist` (default `:none`)

The posterior smoothing is as such: `:none` return probability `1.0` for the closest class, `:ml` returns class-wise mean probabilities or mean of
neighbouring values for regression, `:laplace` is similar to `:ml` but adds some constants, `:mest` regularizes towards class priors and `:dist` used
the distances from sample to neighbours to obtain the posterior in the case of classification while for regression it weighs neighbor values in inverse 
proportion to their corresponding distance.

For more information on posterior smoothing:
	[1] L. Kuncheva "Combining Pattern Classifiers 2'nd Ed." 2014, ISBN 978-1-118-31523-1

Read the `NearestNeighbours.jl` and `Distances.jl` documentation for more information.
"""
knn(k::Real=1; kwargs...) = FunctionCell(knn, (k,), Dict(), kwtitle(k isa Int ? "$k-NN classifier" : "$k-range-NN classifier", kwargs); kwargs...) 



############################
# DataCell/Array Interface #
############################
"""
	knn(x, k=1 [;kwargs])

Trains a knn classifcation model that using the data `x` and `k` neighbors.
"""
# Training
knn(x::T where T<:CellDataL, k::Real=1; kwargs...) = knn((getx!(x), gety(x)), k; kwargs...)
knn(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, k::Real=1; kwargs...) = knn((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), k; kwargs...)
knn(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, k::Real=1; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[knn] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Transform labels first
	yu = sort(unique(x[2]))
	yenc = Vector{Int}(ohenc_integer(x[2],yu)) # encode to Int labels based on position in the sorted vector of unique labels 

	# Train model
	knndata = kNNClassifier.kNN_train(getobs(x[1]), yenc, k; kwargs...)

	# Build model properties 
	modelprops = Dict("size_in" => nvars(x[1]),
		   	  "size_out" => length(yu),
			  "labels" => yu 
	)
	
	FunctionCell(knn, Model(knndata), modelprops, kwtitle(k isa Int ? "$k-NN classifier" : "$k-range-NN classifier", kwargs)) 

end



# Execution
knn(x::T where T<:CellData, model::Model{<:kNNClassifier.kNNModel}, modelprops::Dict) = datacell(knn(getx!(x), model, modelprops), gety(x)) 	
knn(x::T where T<:AbstractVector, model::Model{<:kNNClassifier.kNNModel}, modelprops::Dict) = knn(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
knn(x::T where T<:AbstractMatrix, model::Model{<:kNNClassifier.kNNModel}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	# Return the transformed observations   
	kNNClassifier.kNN_exec(model.data, getobs(x), collect(1:modelprops["size_out"]))
end





##########################
# FunctionCell Interface #	
##########################
"""
	knnr(k=1 [;kwargs])

Constructs an untrained cell that when piped data inside, returns a knn trained regressor
function cell based on the input data and labels.

# Arguments
  * `k::Int` represents the number of neighbours to be considered
  * `k::Float64=1.0` represents a range in which neighboring samples are to be searched (if no neighbours are found, 
  equal class posteriors are generated)
  
# Keyword arguments
  * `metric::Distances.Metric` metric from `Distances.jl` which specifies how distances are calculated between samples (default `Distances.Euclidean()`)
  * `leafsize::Int` Size of the leafs for the tree structures holding the training data (default `10`)
  * `smooth::Symbol` method for smoothing class posteriors; available: `:ml`, `:dist` (default `:ml`)
  
Read the `NearestNeighbours.jl` and `Distances.jl` documentation for more information.  
"""
knnr(k::Real=1; kwargs...) = FunctionCell(knnr, (k,), Dict(), kwtitle(k isa Int ? "$k-NN regressor" : "$k-range-NN regressor", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	knnr(x, k=1 [;kwargs])

Trains a knn regression model that using the data `x` and `k` neighbors.
"""
# Training
knnr(x::T where T<:CellDataL, k::Real=1; kwargs...) = knnr((getx!(x), gety(x)), k; kwargs...)
knnr(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, k::Real=1; kwargs...) = knnr((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), k; kwargs...)
knnr(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, k::Real=1; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[knnr] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Train model
	knndata = kNNClassifier.kNN_train(getobs(x[1]), getobs(x[2]), k; kwargs...)

	# Build model properties 
	modelprops = Dict("size_in" => nvars(x[1]),
		   	  "size_out" => 1,
	)
	
	FunctionCell(knnr, Model(knndata), modelprops, kwtitle(k isa Int ? "$k-NN regressor" : "$k-range-NN regressor", kwargs)) 

end



# Execution
knnr(x::T where T<:CellData, model::Model{<:kNNClassifier.kNNModel}, modelprops::Dict) = datacell(knnr(getx!(x), model, modelprops), gety(x)) 	
knnr(x::T where T<:AbstractVector, model::Model{<:kNNClassifier.kNNModel}, modelprops::Dict) = knnr(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
knnr(x::T where T<:AbstractMatrix, model::Model{<:kNNClassifier.kNNModel}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	
	# Return the transformed observations   
	kNNClassifier.kNN_exec(model.data, getobs(x))
end
