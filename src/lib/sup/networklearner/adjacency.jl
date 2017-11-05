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

mutable struct IncomputableAdjacency{T} <:AbstractAdjacency
	f::T
end



# Aliases
const FunctionAdjacency{T<:Function} = IncomputableAdjacency{T}
const NoAdjacency{T<:Void} = IncomputableAdjacency{T}

# Constructors
adjacency(am::T) where T <:AbstractMatrix = MatrixAdjacency(am)
adjacency(ag::T) where T <:AbstractGraph = GraphAdjacency(ag)
adjacency(f::T, data::S) where {T,S} = ComputableAdjacency(f,data)
adjacency(f::T) where T = IncomputableAdjacency(f)

# Show methods
Base.show(io::IO, A::MatrixAdjacency) = print(io, "Adjacency, $(size(A.am,1)) obs, matrix")
Base.show(io::IO, A::GraphAdjacency) = print(io, "Adjacency, $(nv(A.ag)) obs, graph")
Base.show(io::IO, A::ComputableAdjacency) = print(io, "Adjacency,  computable")
Base.show(io::IO, A::FunctionAdjacency) = print(io, "Adjacency missing data, not computable")
Base.show(io::IO, A::NoAdjacency) = print(io, "Adjacency missing function and data, not computable")



# Functions to strip the adjacency infomation (used in training of the NetworkLearner)
strip_adjacency(A::MatrixAdjacency{T}) where T <:AbstractMatrix = strip_adjacency(A.am)
strip_adjacency(A::GraphAdjacency{T}) where T = strip_adjacency(A.ag)
strip_adjacency(A::ComputableAdjacency{T,S}) where {T,S} = strip_adjacency(A.f, A.data)
strip_adjacency(A::IncomputableAdjacency) = A
strip_adjacency(am::T) where T <:AbstractMatrix = nothing 
strip_adjacency(ag::T) where T <:AbstractGraph = nothing
strip_adjacency(f::T, data::S) where {T,S} = IncomputableAdjacency(f)



# Return an adjacency graph from Adjacency types
adjacency_graph(A::MatrixAdjacency{T}) where T <:AbstractMatrix = adjacency_graph(A.am)
adjacency_graph(A::GraphAdjacency{T}) where T = adjacency_graph(A.ag)
adjacency_graph(A::ComputableAdjacency{T,S}) where {T,S} = adjacency_graph(A.f, A.data)
adjacency_graph(A::IncomputableAdjacency) = error("Insufficient information to obtain an adjacency graph.")
adjacency_graph(am::T) where T <:AbstractMatrix = Graph(am)
adjacency_graph(am::T) where T <:AbstractMatrix{Float64} = SimpleWeightedGraph(am)
adjacency_graph(ag::T) where T <:AbstractGraph = ag
adjacency_graph(f::T, data::S) where {T,S} = adjacency_graph(f(data))


# Return an adjacency matrix from Adjacency types
adjacency_matrix(A::MatrixAdjacency{T}) where T <:AbstractMatrix = adjacency_matrix(A.am)
adjacency_matrix(A::GraphAdjacency{T}) where T = adjacency_matrix(A.ag)
adjacency_matrix(A::ComputableAdjacency{T,S}) where {T,S} = adjacency_matrix(A.f, A.data)
adjacency_matrix(A::IncomputableAdjacency) = error("Insufficient information to obtain an adjacency matrix.")
adjacency_matrix(am::T) where T <:AbstractMatrix = am
adjacency_matrix(ag::T) where T <:AbstractGraph = sparse(ag)
adjacency_matrix(ag::T) where T <:AbstractSimpleWeightedGraph = weights(ag)
adjacency_matrix(f::T, data::S) where {T,S} = adjacency_matrix(f(data))
