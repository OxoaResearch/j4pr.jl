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
function transform!(Xo::T, Mr::M, fr_exec::E, Ci::RelaxationLabelingInferer, RL::R, Adj::A, offset::Int, Xr::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{<:AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	# Initializations
	κ = Ci.κ				# a constant between 0 and 1
	α = Ci.α				# decay
	β = κ					# weight of current iteration estimates
	maxiter = Ci.maxiter			# maximum number of iterations
	tol = Ci.tol				# maximum error 
	f_targets = Ci.tf			# function used to obtain targets
	size_out = size(Xo,1)			# ouput size (corresponds to the number of classes)
	Xl = copy(Xo) 				# local estimates
	
	ŷ = f_targets(Xo)			# Obtain a first estimation of the labels 
	for it in 1:maxiter
		
		β = β * α
		
		# Obtain relational dataset for the current iteration
		for (i,(RLi,Ai)) in enumerate(zip(RL,Adj))		
		
			# Select data subset for relational data output			
			Xs = datasubset(Xr, offset+(i-1)*size_out+1 : offset+i*size_out, ObsDim.Constant{1}())

			# Apply relational learner
			transform!(Xs, RLi, Ai, Xo, ŷ)
		end
		
		# Update estimates
		Xo[:,:] = β.*fr_exec(Mr, Xr) + (1.0-β).*Xo 
		ŷₒ = ŷ; ŷ = f_targets(Xo)

		# Convergence check
		if (sum(ŷ.!= ŷₒ) == 0) || mean(abs.(ŷ-ŷₒ))<=tol
			println("Convergence reached at iteration $it.")
			break
		else
			#println("Iteration $it: $(sum(ŷ.!= ŷₒ)) estimates changed")
	   	end
	end
	
	return Xo
end

function transform!(Xo::T, Mr::M, fr_exec::E, Ci::IterativeClassificationInferer, RL::R, Adj::A, offset::Int, Xr::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	error("Iterative classification not supported.")
	return Xo
end

function transform!(Xo::T, Mr::M, fr_exec::E, Ci::GibbsSamplingInferer, RL::R, Adj::A, offset::Int, Xr::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	println("Gibbs sampling not supported.")
	return Xo
end
