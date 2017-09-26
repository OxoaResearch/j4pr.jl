##########################
# FunctionCell Interface #	
##########################
"""
	mds(p, d=Distances.Euclidean(), [;dowarn=true])

Constructs an fixed cell that when piped data, trasforms the observations
into a lower dimensional space which tries to keep the inter-sample distances as
well as possible. To obstain the dissimilarity matrix, the distance `d` can 
also be provided.

Read the `MultivariateStats.jl` documentation for more information.  
"""
mds(p::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = FunctionCell(mds, (p,d), kwtitle("MDS", kwargs); kwargs...) 



############################
# DataCell/Array Interface #	
############################
"""
	mds(x, p, d=Distances.Euclidean() [;kwargs])

Trasforms the observations in `x` into a lower dimensional space which tries 
to keep the inter-sample distances as well as possible.
"""
mds(x::T where T<:CellData, p::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = datacell(mds(getx!(x), p, d; kwargs...), gety(x))
mds(x::T where T<:AbstractVector, p::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = mds(mat(x, LearnBase.ObsDim.Constant{2}()), p, d; kwargs...)
mds(x::T where T<:AbstractMatrix, p::Int, d::Distances.PreMetric=Distances.Euclidean(); kwargs...) = MultivariateStats.classical_mds(Distances.pairwise(d, x), p; kwargs...)	
