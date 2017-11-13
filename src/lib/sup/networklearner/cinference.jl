# Types
abstract type AbstractCollectiveInferer end

struct RelaxationLabelingInferer <: AbstractCollectiveInferer 
	maxiter::Int
	tol::Float64
	tf::Function
	κ::Float64	
	α::Float64
end

struct IterativeClassificationInferer <: AbstractCollectiveInferer 
	maxiter::Int
	tol::Float64
	tf::Function
end

struct GibbsSamplingInferer <: AbstractCollectiveInferer 
	maxiter::Int
	tol::Float64
	tf::Function
	burniter::Int
end



# Show methods
Base.show(io::IO, ci::RelaxationLabelingInferer) = print(io, "Relaxation labeling, maxiter=$(ci.maxiter), tol=$(ci.tol), κ=$(ci.κ), α=$(ci.α)")
Base.show(io::IO, ci::IterativeClassificationInferer) = print(io, "Iterative classification, maxiter=$(ci.maxiter), tol=$(ci.tol)")
Base.show(io::IO, ci::GibbsSamplingInferer) = print(io, "Gibbs sampling, maxiter=$(ci.maxiter), tol=$(ci.tol), burniter=$(ci.burniter)")
Base.show(io::IO, vci::T) where T<:AbstractVector{S} where S<:AbstractCollectiveInferer = 
	print(io, "$(length(vci))-element Vector{$S} ...")



# Transform methods
function transform!(Xo::T, Ci::RelaxationLabelingInferer, Mr::M, fr_exec::E, RL::R, Adj::A, offset::Int, Xr::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{<:AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	# Initializations
	n = nobs(Xo)										# number of observations
	κ = Ci.κ										# a constant between 0 and 1
	α = Ci.α										# decay
	β = κ											# weight of current iteration estimates
	maxiter = Ci.maxiter									# maximum number of iterations
	tol = Ci.tol										# maximum error 
	f_targets = Ci.tf									# function used to obtain targets
	size_out = size(Xo,1)									# ouput size (corresponds to the number of classes)
	Xl = copy(Xo)										# local estimates
	ŷₗ = f_targets(Xo)									# Obtain first the labels corresponding to the local model
	ŷ = ŷₗ											#   and initialize the current estimates
	ŷₒ = similar(ŷ)										#   and the 'previous' iteration estimates

	# Iterate
	Xrᵢ = zeros(size_out,n)									# Initialize temporary storage	
	for it in 1:maxiter
		β = β * α									# Update learning rate
		copy!(ŷₒ, ŷ);									# Update 'previous iteration' estimates 
		
		# Obtain relational dataset for the current iteration
		@inbounds for (i,(RLᵢ,Aᵢ)) in enumerate(zip(RL,Adj))		
		
			# Apply relational learner
			transform!(Xrᵢ, RLᵢ, Aᵢ, Xo, ŷ)

			# Update relational data output
			Xr[offset+(i-1)*size_out+1 : offset+i*size_out,:] = Xrᵢ
		end
		
		# Update estimates
		Xo[:,:] = β.*fr_exec(Mr, Xr) + (1.0-β).*Xo 
		ŷ = f_targets(Xo)

		# Convergence check
		if isequal(ŷ,ŷₒ) || mean(abs.(ŷ-ŷₒ))<=tol
			# println("Convergence reached at iteration $it.")
			break
		else
			# println("Iteration $it: $(sum(ŷ.!= ŷₒ)) estimates changed")
	   	end
		
		# Replace non-converging estimates with local estimates
		if (it == maxiter) && (maxiter != 1)
			_nc = ŷ.!=ŷₒ 		# positions of non-converging estimates
			datasubset(Xo, _nc)[:] = datasubset(Xl, _nc)[:]
		end
	end
	
	return Xo
end

function transform!(Xo::T, Ci::IterativeClassificationInferer, Mr::M, fr_exec::E, RL::R, Adj::A, offset::Int, Xr::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{<:AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	# Initializations
	n = nobs(Xr)				# number of observations 
	ordering = collect(1:n)			# observation estimation order 
	maxiter = Ci.maxiter			# maximum number of iterations
	tol = Ci.tol				# maximum error 
	f_targets = Ci.tf			# function used to obtain targets
	size_out = size(Xo,1)			# ouput size (corresponds to the number of classes)
	Xl = copy(Xo) 				# local estimates	
	ŷₗ = f_targets(Xo)			# Obtain first the labels corresponding to the local model
	ŷ = ŷₗ					#   and initialize the current estimates
	ŷₒ = similar(ŷ)				#   and the 'previous' iteration estimates
	
	# Iterate
	Xrᵢⱼ = zeros(size_out,1)		# Initialize temporary storage	
	for it in 1:maxiter	
		shuffle!(ordering)		# Randomize observation order
		copy!(ŷₒ, ŷ);			# Update 'previous iteration' estimates 

		# Loop over observations and obtain individual estimates
		for j in ordering		
			
			# Get data subsets pertinent to the current observation 
			rⱼ = j:j
			Xrⱼ = datasubset(Xr, rⱼ)
			Xoⱼ = datasubset(Xo, rⱼ)
			ŷⱼ = datasubset(ŷ, rⱼ)

			# Obtain relational data for the current observation
			@inbounds for (i,(RLᵢ,Aᵢ)) in enumerate(zip(RL,Adj))		

				# Apply relational learner
				transform!(Xrᵢⱼ, RLᵢ, Aᵢ, Xo, ŷ, obs=rⱼ)
				
				# Update relational data output for the current sample
				Xrⱼ[offset+(i-1)*size_out+1 : offset+i*size_out,:] = Xrᵢⱼ
			end
		
			# Update estimates
			Xoⱼ[:] = fr_exec(Mr, Xrⱼ) 
			ŷⱼ[:] = f_targets(Xoⱼ)
		end

		# Convergence check
		if isequal(ŷ,ŷₒ) || mean(abs.(ŷ-ŷₒ))<=tol
			# println("Convergence reached at iteration $it.")
			break
		else
			# println("Iteration $it: $(sum(ŷ.!= ŷₒ)) estimates changed")
	   	end

		# Replace non-converging estimates with local estimates
		if (it == maxiter) && (maxiter != 1)
			_nc = ŷ.!=ŷₒ 		# positions of non-converging estimates
			datasubset(Xo, _nc)[:] = datasubset(Xl, _nc)[:]
		end
	end
	
	return Xo
end

function transform!(Xo::T, Ci::GibbsSamplingInferer, Mr::M, fr_exec::E, RL::R, Adj::A, offset::Int, Xr::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{<:AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	# warn("Gibbs sampling not implemented, returning input (local model) estimates.")
	return Xo
end
