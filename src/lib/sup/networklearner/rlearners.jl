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

struct ClassDistributionRN <: AbstractRelationalLearner
	priors::Vector{Float64}
	normalize::Bool	
	RV::Matrix{Float64}
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

fit(::Type{ClassDistributionRN}, Ai::AbstractAdjacency, Xl::AbstractMatrix, y::AbstractVector; 
    		priors::Vector{Float64}=Float64[], normalize::Bool=true) = begin
	
	yu = sort(unique(y))
	n = length(yu)
	RV = zeros(n,n) # RV is a matrix where columns correspond to the class vectors of each class;
	@assert length(priors) == n "Expected a prior vector length of $n, got $length(priors)."
	
	tmp = Xl*adjacency_matrix(Ai)
	@inbounds @simd for i in 1:n
		RV[:,i] = mean(view(tmp,:,y.==yu[i]),2)
	end
	return ClassDistributionRN(priors, normalize, RV)
end


# Transform methods
function transform!(Xr::T, Rl::R, Ai::AbstractAdjacency, X::S) where {
		R<:AbstractRelationalLearner, T<:AbstractMatrix, S<:AbstractVector}
	transform!(Xr, Rl, adjacency_matrix(Ai), X')
end

function transform!(Xr::T, Rl::R, Ai::AbstractAdjacency, X::S) where {
		R<:AbstractRelationalLearner, T<:AbstractMatrix, S<:AbstractMatrix}
	transform!(Xr, Rl, adjacency_matrix(Ai), X)
end

function transform!(Xr::T, Rl::WeightedVoteRN, Ai::AbstractMatrix, X::S) where {T<:AbstractMatrix, S<:AbstractMatrix}	
	Xr[:] = diagm(Rl.priors)*X*Ai
	Xr ./= clamp!(sum(Ai,1),1.0,Inf)
	
	if Rl.normalize
		Xr ./= sum(Xr,1)
	end
	return Xr
end

function transform!(Xr::T, Rl::BayesRN, Ai::AbstractMatrix, X::S) where {T<:AbstractMatrix, S<:AbstractMatrix}	
	for i in 1:nobs(X)
            vA = view(Ai,:,i)
            vX = view(X,:,vA.!=0)
            prod!(view(Xr,:,i),vX)
        end
         
	Xr[:] = diagm(Rl.priors)*Xr
	Xr ./= clamp!(sum(Ai,1),1.0,Inf)

	if Rl.normalize
		Xr ./= sum(Xr,1)
	end
	return Xr
end


function transform!(Xr::T, Rl::ClassDistributionRN, Ai::AbstractMatrix, X::S) where {T<:AbstractMatrix, S<:AbstractMatrix}	
	d = Distances.Euclidean()
	Distances.pairwise!(Xr, d, Rl.RV, X*Ai)	
end
