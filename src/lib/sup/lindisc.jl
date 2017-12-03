# Linear discriminant 
module LinDiscClassifier
	export LinDiscModel, LinDisc_train, LinDisc_exec
	using StatsFuns: softmax!

	# LinDiscModel struct
	struct LinDiscModel 			
		mean::Vector{Vector{Float64}}		# mean vector
		icm::Matrix{Float64}			# inverse covariance matrix covariance matrix	
		priors::Dict{Int,Float64}		# class priors	
		classes::Vector{Int}			# classes 	
		r1::Float64				# diagonal regularization weight
		r2::Float64				# trace regularization weight
	end

	# Printers
	Base.show(io::IO, m::LinDiscModel) = print(io, "Linear Discriminant model, $(length(m.mean)) classes, r1=$(m.r1), r2=$(m.r2)")


	# Train methods (classification)
	LinDisc_train(x::Matrix{Float64}, y::Vector{Int}, r1::Float64=0, r2::Float64=0) = begin
		
		m = size(x,1) 							# number of variables
		classes = sort(unique(y))					# classes
		priors = Dict(yi=>sum(yi.==y)/length(y) for yi in classes)	# priors
		
		mv = [vec(mean(x[:,y.==yi],2)) for yi in classes]		# mean vectors
		
		cv = zeros(m,m)
		@simd for yi in classes # iterate through the classes
			cv += priors[yi]*cov(x[:,y.==yi],2)
		end
		
		cv = (1.0-r1-r2)*cv + r1*Matrix(Diagonal(diag(cv))) + r2*trace(cv)*eye(m)
	
		return LinDiscModel(mv, inv(cv), priors, classes, r1, r2)
	end
	
	
	# Execution methods (classification)
	LinDisc_exec(m::LinDiscModel, x::Matrix{Float64})=begin 
		C = length(m.classes)
		out = zeros(Float64, C, size(x,2))
		@simd for c in 1:C	
		 	# 			Linear term		Constant term log(P(ωᵢ)) - ...
			@inbounds out[c,:] = (m.icm*m.mean[c])'*x .+ log(m.priors[m.classes[c]]).-1/2*m.mean[c]'*m.icm*m.mean[c]
		end
		
		# Apply softmax
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
	lindisc(r1=0.0, r2=0.0)

Constructs an untrained cell that when piped data inside, returns a linear liscriminant classifier trained
function cell based on the input data and labels.

# Arguments
  * `r1`::Float64 represents the diagonal covariance matrix regularization parameter
  * `r2`::Float64 represents the covariance matrix trace regularization parameter

For more information:
	[1] L. Kuncheva "Combining Pattern Classifiers 2'nd Ed." 2014, ISBN 978-1-118-31523-1
"""
lindisc(r1::Float64=0.0, r2::Float64=0.0) = FunctionCell(lindisc, (r1,r2), ModelProperties(), "Linear discriminant classifier: r1=$r1, r2=$r2") 



############################
# DataCell/Array Interface #	
############################
"""
	lindisc(x, r1=0.0, r2=0.0)

Trains a linear discriminant classifcation model that using the data `x` and regularization parameters `r1`, `r2`.
"""
# Training
lindisc(x::T where T<:CellDataL, r1::Float64=0.0, r2::Float64=0.0) = lindisc((getx!(x), gety(x)), r1, r2)
lindisc(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, r1::Float64=0.0, r2::Float64=0.0) = lindisc((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), r1, r2)
lindisc(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, r1::Float64=0.0, r2::Float64=0.0) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[lindisc] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)
	
	# Train model
	lddata = LinDiscClassifier.LinDisc_train(getobs(x[1]), yenc, r1, r2)

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(lindisc, Model(lddata, modelprops), "Linear discriminant classifier: r1=$r1, r2=$r2") 

end



# Execution
lindisc(x::T where T<:CellData, model::Model{<:LinDiscClassifier.LinDiscModel}) = 
	datacell(lindisc(getx!(x), model), gety(x)) 	
lindisc(x::T where T<:AbstractVector, model::Model{<:LinDiscClassifier.LinDiscModel}) = 
	lindisc(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
lindisc(x::T where T<:AbstractMatrix, model::Model{<:LinDiscClassifier.LinDiscModel}) =
	LinDiscClassifier.LinDisc_exec(model.data, getobs(x))
