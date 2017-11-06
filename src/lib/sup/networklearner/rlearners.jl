abstract type AbstractRelationalLearner end

struct WeightedVoteRN <: AbstractRelationalLearner end

struct BayesRN <: AbstractRelationalLearner end

struct ClassDistributionRN{T<:AbstractArray} <: AbstractRelationalLearner
	RV::T
end



# Show methods
Base.show(io::IO, rl::WeightedVoteRN) = print(io, "Weighted-vote relation neighbour learner")
Base.show(io::IO, rl::BayesRN) = print(io, "Network-only Bayes learner")
Base.show(io::IO, rl::ClassDistributionRN) = print(io, "Class-distribution relational neighbour learner")



# Constructor functions
wrRN() = WeightedVoteRN()
cdRN(RV::T) where T<:AbstractArray = ClassDistributionRN(RV)
nBC() = BayesRN() 



# Training methods
fit(::Type{WeightedVoteRN}, args...) = WeightedVoteRN()
fit(::Type{BayesRN}, args...) = BayesRN()
# TODO: training for the clas-distribution relational neighbour learner (calculate RV)
fit(::Type{ClassDistributionRN}, args ...) = error("Training for class-distribution relational neighbour leanrer (cdRN) not supported yet!")



# Transform methods
function transform!(Xr::T, Rl::WeightedVoteRN, A::AbstractAdjacency, X::S, priors::U; 
		    normalize::Bool=false) where {T<:AbstractMatrix, S<:AbstractVector, U<:AbstractVector}
	transform!(Xr, Rl, adjacency_matrix(A), X', priors; normalize=normalize)
end

function transform!(Xr::T, Rl::WeightedVoteRN, A::AbstractAdjacency, X::S, priors::U; 
		    normalize::Bool=false) where {T<:AbstractMatrix, S<:AbstractMatrix, U<:AbstractVector}	
	transform!(Xr, Rl, adjacency_matrix(A), X, priors; normalize = normalize)
end

function transform!(Xr::T, Rl::WeightedVoteRN, A::AbstractMatrix, X::S, priors::U; 
		    normalize::Bool=false) where {T<:AbstractMatrix, S<:AbstractMatrix, U<:AbstractVector}	
	Xr[:] = diagm(priors)*X*A
	if normalize
		Xr ./= clamp!(sum(A,1),1.0,Inf)
	end
	return Xr
end



#TODO: transform methods fo BayesRN

#TODO: transform methods fo cdRN

