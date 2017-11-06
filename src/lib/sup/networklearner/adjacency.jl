#############
# Adjacency #
#############
abstract type AbstractAdjacency end

# Types
mutable struct MatrixAdjacency{T<:AbstractMatrix} <: AbstractAdjacency
	am::T
	function MatrixAdjacency(am::T) where T<:AbstractMatrix
		@assert issymmetric(am) "Adjacency matrix should be symmeric."
		new{T}(am)
	end
end

mutable struct GraphAdjacency{T<:AbstractGraph} <: AbstractAdjacency
	ag::T
end

mutable struct ComputableAdjacency{T,S} <:AbstractAdjacency
	f::T
	data::S
end

mutable struct PartialAdjacency{T<:Function} <: AbstractAdjacency
	f::T
end

mutable struct EmptyAdjacency <: AbstractAdjacency end


# Constructors
adjacency(A::T) where T<:AbstractAdjacency = A
adjacency(A::T, data) where T<:PartialAdjacency = adjacency(A.f(data))
adjacency(am::T) where T<:AbstractMatrix = MatrixAdjacency(am)
adjacency(ag::T) where T<:AbstractGraph = GraphAdjacency(ag)
adjacency(f::T, data::S) where {T,S} = ComputableAdjacency(f,data)
adjacency(t::Tuple{T,S}) where {T,S} = ComputableAdjacency(t[1],t[2])
adjacency(f::T) where T<:Function = PartialAdjacency(f)
adjacency(::Void) = EmptyAdjacency()
adjacency() = EmptyAdjacency()

	

# Show methods
Base.show(io::IO, A::MatrixAdjacency) = print(io, "Matrix adjacency, $(size(A.am,1)) obs")
Base.show(io::IO, A::GraphAdjacency) = print(io, "Graph adjacency, $(nv(A.ag)) obs")
Base.show(io::IO, A::ComputableAdjacency) = print(io, "Computable adjacency")
Base.show(io::IO, A::PartialAdjacency) = print(io, "Partial adjacency, not computable")
Base.show(io::IO, A::EmptyAdjacency) = print(io, "Empty adjacency, not computable")
#Base.show(io::IO, VA::Vector{<:AbstractAdjacency}) = print(io, "$(length(VA))-element of Vector{$(eltype(VA))}")



# Functions to strip the adjacency infomation (used in training of the NetworkLearner)
# The methods return a Partial Adjacency
strip_adjacency(A::MatrixAdjacency{T}) where T <:AbstractMatrix = adjacency(x->T(x))
strip_adjacency(A::GraphAdjacency{T}) where T<:AbstractGraph = adjacency(x->T(x))
strip_adjacency(A::ComputableAdjacency{T,S}) where {T,S} = adjacency(A.f)
strip_adjacency(A::PartialAdjacency) = A
strip_adjacency(A::EmptyAdjacency) = PartialAdjacency(x->adjacency(x))



# Return an adjacency graph from Adjacency types
adjacency_graph(A::MatrixAdjacency{T}) where T <:AbstractMatrix = adjacency_graph(A.am)
adjacency_graph(A::GraphAdjacency{T}) where T = adjacency_graph(A.ag)
adjacency_graph(A::ComputableAdjacency{T,S}) where {T,S} = adjacency_graph(A.f, A.data)
adjacency_graph(A::PartialAdjacency) = error("Insufficient information to obtain an adjacency graph.")
adjacency_graph(A::EmptyAdjacency) = error("Insufficient information to obtain an adjacency graph.")
adjacency_graph(am::T) where T <:AbstractMatrix = Graph(am)
adjacency_graph(am::T) where T <:AbstractMatrix{Float64} = SimpleWeightedGraph(am)
adjacency_graph(ag::T) where T <:AbstractGraph = ag
adjacency_graph(f::T, data::S) where {T,S} = adjacency_graph(f(data))


# Return an adjacency matrix from Adjacency types
adjacency_matrix(A::MatrixAdjacency{T}) where T <:AbstractMatrix = adjacency_matrix(A.am)
adjacency_matrix(A::GraphAdjacency{T}) where T = adjacency_matrix(A.ag)
adjacency_matrix(A::ComputableAdjacency{T,S}) where {T,S} = adjacency_matrix(A.f, A.data)
adjacency_matrix(A::PartialAdjacency) = error("Insufficient information to obtain an adjacency matrix.")
adjacency_matrix(A::EmptyAdjacency) = error("Insufficient information to obtain an adjacency matrix.")
adjacency_matrix(am::T) where T <:AbstractMatrix = am
adjacency_matrix(ag::T) where T <:AbstractGraph = sparse(ag)
adjacency_matrix(ag::T) where T <:AbstractSimpleWeightedGraph = weights(ag)
adjacency_matrix(f::T, data::S) where {T,S} = adjacency_matrix(f(data))
