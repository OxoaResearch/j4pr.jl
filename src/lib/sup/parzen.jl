# Parzen density estimation and classification
module ParzenClassifier
	import Distances
	using StatsFuns: softmax!
	export ParzenModel, Parzen_train, Parzen_exec
	
	Φ_hat(u::T) where T<:Real = abs(u)<=0.5 ? 1.0 : 0.0
	Φ_linear(u::T) where T<:Real = abs(u)<=0.5 ? 1.0-u : 0.0
	Φ_gaussian(u::T) where T<:Real = 1/sqrt(2*pi)*exp(-u^2/2)
	Φ_exponential(u::T) where T<:Real = u>0.0 ? exp(-u) : 0.0
	Φ_cosine(u::T) where T<:Real = abs(u)<=0.5 ? cos(pi/2*u) : 0.0

	# ParzenModel struct {T - type of neighbour, S - type of labels, U - type of distance}
	struct ParzenModel{S<:Union{Void,Vector{<:Real}}, U<:AbstractArray{<:AbstractFloat}, V<:Distances.Metric} 			
		h::Float64					# h: dimension of the encompasing hypercube
		Φ::Function 					# kernel/parzen window function
		targets::S					# targets: Vector{T<:Real} - classification, Void - density estimation 
		data::U						# training data: Matrix{FloatXX}
		metric::V					# distance: Distances.Metric
		priors::Dict{Int,Float64}			# not used unless specified explicitly (not yet supported)
	end

	# Aliases
	const ParzenModelClassification{S<:Vector{Int}} = ParzenModel{S}
	const ParzenModelDensity{S<:Void} = ParzenModel{S}

	# Printers
	Base.show(io::IO, m::ParzenModelClassification) = print(io, "Parzen model, classification, h=$(m.h)")
	Base.show(io::IO, m::ParzenModelDensity) = print(io, "Parzen model, density estimation, h=$(m.h)")



	# Get kernel function 
	_parse_window_(window::Symbol)::Function = begin
		# Check for the window type
		if window == :hat return Φ_hat
		elseif window == :linear return Φ_linear
		elseif window == :gaussian return Φ_gaussian
		elseif window == :exponential return Φ_exponential
		elseif window == :cosine return Φ_cosine
		else 
			warn("Unrecognized kernel type, defaulting to :hat")
			return Φ_hat
		end
	end
	
	# Train method (classification)
	Parzen_train(x::Matrix{Float64}, y::Vector{<:Real}, h::Float64; window::Symbol=:hat, metric::Distances.Metric=Distances.Cityblock()) = 
		return ParzenModel(h, _parse_window_(window), y, x, metric, Dict(yi=>sum(yi.==y)/length(y) for yi in unique(y)))

	# Train method (density estimation)
	Parzen_train(x::Matrix{Float64}, y::Void, h::Float64; window::Symbol=:hat, metric::Distances.Metric=Distances.Cityblock()) =
		return ParzenModel(h, _parse_window_(window), y, x, metric, Dict{Int, Float64}())
	
	
	
	# Execution methods (classification)
	Parzen_exec(m::ParzenModelClassification, x::Matrix{Float64}, classes::Vector{Int})=begin 
		l = size(m.data,1)
		N = size(m.data,2)
		C = length(classes)
		out = zeros(Float64, size(x,2),C)
		@inbounds for c in 1:C
		#TODO: Improve performance of this bit 
			out[:,c] = 1/(m.h^l) * 1/N * sum(m.Φ.(Distances.pairwise(m.metric, x, m.data[:,m.targets.==classes[c]])/m.h),2) 
		end
		
		# Apply softmax
		for j in 1:size(out,1)
			softmax!(view(out,j,:))
		end
		return out'
	end
	
	Parzen_exec(m::ParzenModelDensity, x::Matrix{Float64}, classes::Vector{Int})=begin # classes is not used, just for method uniformity 
		l = size(m.data,1)
		N = size(m.data,2)
		
		#TODO: Improve performance of this bit 
		return ( 1/(m.h^l) * 1/N * sum(m.Φ.(Distances.pairwise(m.metric, x, m.data)/m.h),2) )'
	end
end



##########################
# FunctionCell Interface #	
##########################
"""
	parzen(h=1.0 [;kwargs])

Constructs an untrained cell that when piped data inside, returns a Parzen window classifier/density estimator trained
function cell based on the input data and labels if the case. 

# Arguments
  * `h`::Float64=1.0 is the shape parameter of the parzen window 
  
# Keyword arguments
  * `window`::Symbol the Parzen window type; available: `:hat`, `:linear`, `:gaussian`, `:exponential` and `:cosine` (default `:hat`)
  * `metric`::Distances.Metric metric from `Distances.jl` which specifies how distances are calculated between samples (default `Distances.Citiblock()`)

For more information:
	[1] S. Theodoridis, K. Koutroumbas, "Pattern Recognition 4'th Ed." 2009, ISBN 978-1-59749-272-0 
	[2] R. Douda, P. Hart, D. Stork, "Pattern Classification 2'nd Ed." 2001, ISBN 978-0-471-05669-0
"""
parzen(h::Float64=1.0; kwargs...) = FunctionCell(parzen, (h,), ModelProperties(), kwtitle("Parzen classifier/density estimator (h=$h)", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	parzen(x, h=1.0 [;kwargs])

Trains a Parzen classifier / density estimator using the data `x` and the window parameter `h`.
"""
# Training (density estimation)
parzen(x::T where T<:CellDataU, h::Float64=1; kwargs...) = parzen(getx!(x), h; kwargs...)
parzen(x::T where T<:AbstractVector, h::Float64=1.0; kwargs...) = parzen(mat(x, LearnBase.ObsDim.Constant{2}()), h; kwargs...)
parzen(x::T where T<:AbstractMatrix, h::Float64=1.0; kwargs...) = begin
	
	@assert nobs(x) > 1 "[parzen] Expected at least 2 samples, got $(nobs(x))."
	
	# Train model
	parzendata = ParzenClassifier.Parzen_train(getobs(x), nothing, h; kwargs...)
	
	FunctionCell(parzen, Model(parzendata, ModelProperties(nvars(x), 1)), kwtitle("Parzen density estimator (h=$h)",kwargs)) 
end



# Training (classification)
parzen(x::T where T<:CellDataL, h::Float64=1; kwargs...) = parzen((getx!(x), gety(x)), h; kwargs...)
parzen(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, h::Float64=1.0; kwargs...) = 
	parzen((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), h; kwargs...)
parzen(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, h::Float64=1.0; kwargs...) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[parzen] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)

	# Train model
	parzendata = ParzenClassifier.Parzen_train(getobs(x[1]), yenc, h; kwargs...)

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(parzen, Model(parzendata, modelprops), kwtitle("Parzen classifier (h=$h)",kwargs)) 

end



# Execution
parzen(x::T where T<:CellData, model::Model{<:ParzenClassifier.ParzenModel}) = 
	datacell(parzen(getx!(x), model), gety(x)) 	
parzen(x::T where T<:AbstractVector, model::Model{<:ParzenClassifier.ParzenModel}) = 
	parzen(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
parzen(x::T where T<:AbstractMatrix, model::Model{<:ParzenClassifier.ParzenModel}) = 
	ParzenClassifier.Parzen_exec(model.data, getobs(x), collect(1:model.properties.odim))
