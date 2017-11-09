# Types
abstract type AbstractRelationalLearner end

struct SimpleRN <: AbstractRelationalLearner 
	normalize::Bool
end

struct WeightedRN <: AbstractRelationalLearner 
	normalize::Bool
end

struct BayesRN <: AbstractRelationalLearner 
	priors::Vector{Float64}
	normalize::Bool
	LM::Matrix{Float64}	# likelihood matrix (class-conditional neighbourhood likelihoods)
end

struct ClassDistributionRN <: AbstractRelationalLearner
	normalize::Bool
	RV::Matrix{Float64}
end



# Show methods
Base.show(io::IO, rl::SimpleRN) = print(io, "RN, normalize=$(rl.normalize)")
Base.show(io::IO, rl::WeightedRN) = print(io, "wRN, normalize=$(rl.normalize)")
Base.show(io::IO, rl::BayesRN) = print(io, "bayesRN, normalize=$(rl.normalize), $(length(rl.priors)) classes")
Base.show(io::IO, rl::ClassDistributionRN) = print(io, "cdRN, normalize=$(rl.normalize), $(size(rl.RV,2)) classes")
Base.show(io::IO, vrl::T) where T<:AbstractVector{S} where S<:AbstractRelationalLearner = 
	print(io, "$(length(vrl))-element Vector{$S} ...")


# Training methods (all fit mehods use the same unique signature)
fit(::Type{SimpleRN}, args...; priors::Vector{Float64}=Float64[], normalize::Bool=true) = SimpleRN(normalize)

fit(::Type{WeightedRN}, args...; priors::Vector{Float64}=Float64[], normalize::Bool=true) = WeightedRN(normalize)

fit(::Type{BayesRN}, Ai::AbstractAdjacency, Xl::AbstractMatrix, y::AbstractVector; 
    		priors::Vector{Float64}=Float64[], normalize::Bool=true) = begin
	C = size(Xl,1)
	@assert C == length(priors) "Size of local model estimates is $(C) and prior vector length is $(length(priors))."
	
	# Get for each observation class percentages in neighbourhood
	Am = adjacency_matrix(Ai)
	H = vcat((sum(Am[y.==i,:],1) for i in 1:C)...)./clamp!(sum(Am,1),1.0,Inf) 
	
	# Calculate the means of neighbourhood class percentages for all samples belonging to the same class 
	LM = zeros(C,C)
	@inbounds @simd for c in 1:C
		LM[:,c] = mean(H[:,y.==c],2)
	end
	
	BayesRN(priors, normalize, LM)
end

fit(::Type{ClassDistributionRN}, Ai::AbstractAdjacency, Xl::AbstractMatrix, y::AbstractVector; 
    		priors::Vector{Float64}=Float64[], normalize::Bool=true) = begin
	yu = sort(unique(y))
	n = length(yu)
	RV = zeros(n,n) # RV is a matrix where columns correspond to the class vectors of each class;
	
	# Calculate reference vectors (matrix where each column is a reference vector)
	Am = adjacency_matrix(Ai)
	Xtmp = Xl * adjacency_matrix(Am)
	Xtmp ./= clamp!(sum(Am,1),1.0,Inf)	# normalize to edge weight sum	
	
	@inbounds @simd for i in 1:n
		RV[:,i] = mean(view(Xtmp,:,y.==yu[i]),2)
	end
	
	return ClassDistributionRN(normalize, RV)
end



# Transform methods
function transform!(Xr::T, Rl::R, Ai::AbstractAdjacency, X::S, ŷ::U) where {
		R<:AbstractRelationalLearner, T<:AbstractMatrix, S<:AbstractVector, U<:AbstractVector}
	transform!(Xr, Rl, adjacency_matrix(Ai), X', ŷ)
end

function transform!(Xr::T, Rl::R, Ai::AbstractAdjacency, X::S, ŷ::U) where {
		R<:AbstractRelationalLearner, T<:AbstractMatrix, S<:AbstractMatrix, U<:AbstractVector}
	transform!(Xr, Rl, adjacency_matrix(Ai), X, ŷ)
end

function transform!(Xr::T, Rl::SimpleRN, Am::AbstractMatrix, X::S, ŷ::U) where {
		T<:AbstractMatrix, S<:AbstractMatrix, U<:AbstractVector}	
	for i in 1:size(Xr,1)
		Xr[i,:] = Am * (ŷ.==i)		# summate edge weights for neighbours in class 'i'  
	end
	Xr ./= clamp!(sum(Am,1),1.0,Inf)	# normalize to edge weight sum	
	
	if Rl.normalize				# normalize estimates / observation
		Xr ./= sum(Xr,1)
	end
	return Xr
end

function transform!(Xr::T, Rl::WeightedRN, Am::AbstractMatrix, X::S, ŷ::U) where {
		T<:AbstractMatrix, S<:AbstractMatrix, U<:AbstractVector}	
	Xr[:] = X*Am				# summate edge weighted probabilities of all neighbors
	Xr ./= clamp!(sum(Am,1),1.0,Inf)	# normalize to edge weight sum
	
	if Rl.normalize				# normalize estimates / observation
		Xr ./= sum(Xr,1)
	end
	return Xr
end

function transform!(Xr::T, Rl::BayesRN, Am::AbstractMatrix, X::S, ŷ::U) where {
	 	T<:AbstractMatrix, S<:AbstractMatrix, U<:AbstractVector}	
	#TODO: Improve performance in this bit
	for i in 1:nobs(X)
            	vA = view(Am,i,:)		# get weights for current obs neighbors 
		idx = vA.!=0
		prod!(view(Xr,:,i), Rl.LM[:,ŷ[idx]]) # calculate product of likelihoods of neighbors 
        end
	Xr[:] = diagm(Rl.priors)*Xr
	Xr ./= clamp!(sum(Am,1),1.0,Inf) # unclear if needed

	if Rl.normalize				# normalize estimates / observation
		Xr ./= sum(Xr+eps(),1)
	end
	return Xr
end

function transform!(Xr::T, Rl::ClassDistributionRN, Am::AbstractMatrix, X::S, ŷ::U) where {
	 	T<:AbstractMatrix, S<:AbstractMatrix, U<:AbstractVector}	
	d = Distances.Euclidean()
	Xtmp = X*Am
	Xtmp ./= clamp!(sum(Am,1),1.0,Inf)	# normalize to edge weight sum	
		
	Distances.pairwise!(Xr, d, Rl.RV, Xtmp)	
	
	if Rl.normalize				# normalize estimates / observation
		Xr ./= sum(Xr,1)
	end

	return Xr
end
