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
function transform!(Xo::T, Mr::M, fr_exec::E, Ci::RelaxationLabelingInferer, RL::R, Adj::A, X::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{<:AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	# 

	return Xo
end

function transform!(Xo::T, Mr::M, fr_exec::E, Ci::IterativeClassificationInferer, RL::R, Adj::A, X::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	error("Iterative classification not supported.")
	return Xo
end

function transform!(Xo::T, Mr::M, fr_exec::E, Ci::GibbsSamplingInferer, RL::R, Adj::A, X::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	println("Gibbs sampling not supported.")
	return Xo
end
