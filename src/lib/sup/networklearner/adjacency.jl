#############
# Adjacency #
#############
abstract type AbstractAdjacency end

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



Base.show(io::IO, A::MatrixAdjacency) = print(io, "Adjacency, $(size(A.am,1)) obs, matrix")
Base.show(io::IO, A::GraphAdjacency) = print(io, "Adjacency, $(nv(A.ag)) obs, graph")
Base.show(io::IO, A::ComputableAdjacency) = print(io, "Adjacency, unknown obs, computable")

obtain_ad_graph(A::MatrixAdjacency{T}) where T <:AbstractMatrix = obtain_ad_graph(A.am)
obtain_ad_graph(A::GraphAdjacency{T}) where T = obtain_ad_graph(A.ag)
obtain_ad_graph(A::ComputableAdjacency{T,S}) where {T,S} = obtain_ad_graph(A.f, A.data)

obtain_ad_graph(am::T) where T <:AbstractMatrix = Graph(am)
obtain_ad_graph(am::T) where T <:AbstractMatrix{Float64} = SimpleWeightedGraph(am)
obtain_ad_graph(ag::T) where T <:AbstractGraph = ag
obtain_ad_graph(f::T, data::S) where {T,S} = obtain_ad_graph(f(data))
