# Quadratic discriminant 
module QuadDiscClassifier
	export QuadDiscModel, QuadDisc_train, QuadDisc_exec
	using StatsFuns: softmax!

	# QuadDiscModel struct
	struct QuadDiscModel 			
		mean::Vector{Vector{Float64}}		# mean vector
		icm::Array{Float64}			# inverse covariance matrices for each class	
		dcm::Vector{Float64}			# determinants of the covariance matrices
		priors::Dict{Int,Float64}		# class priors	
		classes::Vector{Int}			# classes 	
		r1::Float64				# diagonal regularization weight
		r2::Float64				# trace regularization weight
	end

	# Printers
	Base.show(io::IO, m::QuadDiscModel) = print("Quadratic Discriminant Model, $(length(mean)) classes, r1=$r1, r2=$r2")


	# Train methods (classification)
	QuadDisc_train(x::Matrix{Float64}, y::Vector{Int}, r1::Float64=0, r2::Float64=0) = begin
		
		m = size(x,1) 							# number of variables
		classes = sort(unique(y))					# classes
		C = length(classes)

		priors = Dict(yi=>sum(yi.==y)/length(y) for yi in classes)	# priors
		mv = [vec(mean(x[:,y.==yi],2)) for yi in classes]		# mean vectors
		
		cv = zeros(C,m,m)
		dcm = zeros(C)
		for (i,yi) in enumerate(classes) # iterate through the classes
			
			# Calculate class covariance
			cv[i,:,:] = cov(x[:,y.==yi],2)
			
			# Calculate the log of the determinant of the cov matrix (for the fixed term)
			dcm[i] = -1/2*log(det(cv[i,:,:]))
			
			# Regularize and invert the covariance matrices
			cv[i,:,:] = inv( (1-r1-r2)*cv[i,:,:] + r1*diagm(diag(cv[i,:,:])) + r2*trace(cv[i,:,:])*eye(m) )
			
		end
		
		return QuadDiscModel(mv, cv, dcm, priors, classes, r1, r2)
	end
	
	
	# Execution methods (classification)
	QuadDisc_exec(m::QuadDiscModel, x::Matrix{Float64})=begin 
		
		# Function for calculating the quadratic term: -1/2*xᵀ*inv(Σᵢ)*x (or diag(x'*(-1/2*m.icm[c,:,:])*x)', very slow)  
		xTwx!(out,x,w) = begin
			for i = 1:size(x,2)
				v=view(x,:,i)
				@simd for c = 1:size(w,1)
					@inbounds out[c,i] *=v'*view(w,c,:,:)*v
				end
			end
			return out
		end
		
		# Initializations
		C = length(m.classes)
		out=fill(-0.5, size(m.icm,1), size(x,2))
		xTwx!(out, x, m.icm)
		
		@inbounds @simd for c = 1:C
					# Linear term: inv(Σᵢ)*μᵀ*x			# Constant term log(P(ωᵢ)) - ...
			out[c:c,:] += (m.icm[c,:,:]*m.mean[c])'*x + log(m.priors[m.classes[c]])- 1/2*m.mean[c]'*m.icm[c,:,:]*m.mean[c] + m.dcm[c] 
		end
	
		# Apply softmax
		for j in 1:size(out,2)
			softmax!(view(out,:,j))
		end

		return out
	end
end

##########################
# FunctionCell Interface #	
##########################
"""
	quaddisc(r1=0.0, r2=0.0)

Constructs an untrained cell that when piped data inside, returns a quadratic liscriminant classifier trained
function cell based on the input data and labels.

# Arguments
  * `r1`::Float64 represents the diagonal covariance matrix regularization parameter
  * `r2`::Float64 represents the covariance matrix trace regularization parameter

For more information:
	[1] L. Kuncheva "Combining Pattern Classifiers 2'nd Ed." 2014, ISBN 978-1-118-31523-1
"""
quaddisc(r1::Float64=0.0, r2::Float64=0.0) = FunctionCell(quaddisc, (r1,r2), ModelProperties(), "Quadratic discriminant classifier: r1=$r1, r2=$r2") 



############################
# DataCell/Array Interface #	
############################
"""
	quaddisc(x, r1=0.0, r2=0.0)

Trains a quadratic discriminant classifcation model that using the data `x` and regularization parameters `r1`, `r2`.
"""
# Training
quaddisc(x::T where T<:CellDataL, r1::Float64=0.0, r2::Float64=0.0) = quaddisc((getx!(x), gety(x)), r1, r2)
quaddisc(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractVector, r1::Float64=0.0, r2::Float64=0.0) = quaddisc((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), r1, r2)
quaddisc(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractVector, r1::Float64=0.0, r2::Float64=0.0) = begin
	
	@assert nobs(x[1]) == nobs(x[2]) "[quaddisc] Expected $(nobs(x[1])) labels/values, got $(nobs(x[2]))."
	
	# Transform labels first
	enc = labelencn(x[2])
	yenc = label2ind.(x[2],enc)

	# Train model
	qddata = QuadDiscClassifier.QuadDisc_train(getobs(x[1]), yenc, r1, r2)

	# Build model properties 
	modelprops = ModelProperties(nvars(x[1]), length(enc.label), enc)
	
	FunctionCell(quaddisc, Model(qddata, modelprops), "Quadratic discriminant classifier: r1=$r1, r2=$r2") 

end



# Execution
quaddisc(x::T where T<:CellData, model::Model{<:QuadDiscClassifier.QuadDiscModel}) = 
	datacell(quaddisc(getx!(x), model), gety(x)) 	
quaddisc(x::T where T<:AbstractVector, model::Model{<:QuadDiscClassifier.QuadDiscModel}) = 
	quaddisc(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
quaddisc(x::T where T<:AbstractMatrix, model::Model{<:QuadDiscClassifier.QuadDiscModel}) = 
	QuadDiscClassifier.QuadDisc_exec(model.data, getobs(x))
