# Types
abstract type AbstractRelationalLearner end

struct WeightedVoteRN <: AbstractRelationalLearner 
	priors::Vector{Float64}
	normalize::Bool
end

struct BayesRN <: AbstractRelationalLearner 
	priors::Vector{Float64}
	normalize::Bool
end

struct ClassDistributionRN{T<:AbstractArray} <: AbstractRelationalLearner
	priors::Vector{Float64}
	normalize::Bool	
	RV::T
end



# Show methods
Base.show(io::IO, rl::WeightedVoteRN) = print(io, "Neighbourhood weighted relational learner, $(length(rl.priors)) priors, normalize=$(rl.normalize)")
Base.show(io::IO, rl::BayesRN) = print(io, "Bayesian relational learner, $(length(rl.priors)) priors, normalize=$(rl.normalize)")
Base.show(io::IO, rl::ClassDistributionRN) = print(io, "Class-distribution relational learner, $(length(rl.priors)) priors, normalize=$(rl.normalize)")
Base.show(io::IO, vrl::T) where T<:AbstractVector{S} where S<:AbstractRelationalLearner = 
	print(io, "$(length(vrl))-element Vector{$S} ...")


# Training methods
fit(::Type{WeightedVoteRN}, args...; priors::Vector{Float64}=Float64[], normalize::Bool=true) = WeightedVoteRN(priors, normalize)

fit(::Type{BayesRN}, args...; priors::Vector{Float64}=Float64[], normalize::Bool=true) = BayesRN(priors, normalize)

#TODO: fit methods for cdRN
fit(::Type{ClassDistributionRN}, args ...; priors::Vector{Float64}=Float64[], normalize::Bool=true) = 
	error("Class-distribution relational learner training not supported yet.")



# Transform methods
function transform!(Xr::T, Rl::WeightedVoteRN, A::AbstractAdjacency, X::S) where {T<:AbstractMatrix, S<:AbstractVector}
	transform!(Xr, Rl, adjacency_matrix(A), X)
end

function transform!(Xr::T, Rl::WeightedVoteRN, A::AbstractAdjacency, X::S) where {T<:AbstractMatrix, S<:AbstractMatrix}	
	transform!(Xr, Rl, adjacency_matrix(A), X)
end

function transform!(Xr::T, Rl::WeightedVoteRN, A::AbstractMatrix, X::S) where {T<:AbstractMatrix, S<:AbstractMatrix}	
	Xr[:] = diagm(Rl.priors)*X*A
	if Rl.normalize
		Xr ./= clamp!(sum(A,1),1.0,Inf)
	end
	return Xr
end



#TODO: transform! methods fo BayesRN, cdRN
