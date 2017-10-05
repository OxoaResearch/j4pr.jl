"""
	countapp(v,u)

Counts how many times each value in `u` appears in `v`. The Function returns a `Vector{Float64}` containing the element count.

# Examples
```
julia> countapp([1,2,3,2,1,2],[2,3,1])
3-element Array{Float64,1}:
 3.0
 1.0
 2.0

julia> countapp([1,2,3,2,1,2],[1,2,4,5,1])
5-element Array{Float64,1}:
 2.0
 3.0
 0.0
 0.0
 2.0
```
"""
countapp(v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}) where V = 
	countapp!(zeros(Float64, length(u)), v, u)

countapp!(out::Vector{Float64}, v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}) where V = begin
	m = length(u)
	@assert length(out) == m "[countapp!] output vector size must be $m"
	
	for i in 1:m				# for after 'u' first: reduce allocations significantly
		out[i] = 0.0 
		@simd for vi in v			
			@inbounds out[i] +=(vi==u[i])
		end
	end
	return out
end

"""
	countappw(v,u [,w,val])

Counts how many times each value in `u` appears in `v`. Each value in `v` can have an associated weight in the vector `w`. The counts
corresponding to a value in `u` not appearing in `v` can receive a cusom value `val` (default `0.0`). 
The function returns a `Vector{Float64}` containing the weighted normalized element count.

# Examples
```
julia> j4pr.countappw([1,2,3,2,1,2],[1,2,4,5,1])
5-element Array{Float64,1}:
 0.333333
 0.5     
 0.0     
 0.0     
 0.333333

julia> j4pr.countappw([1,2,3,2,1,2],[1,2,4,5,1],ones(6),-1.0)
5-element Array{Float64,1}:
 2.0
 3.0
-1.0
-1.0
 2.0
```
"""
countappw(v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}, 
	 w::Vector{Float64}=fill(1/length(v),length(v)), 
	 val::Float64=0.0) where V = 
	countappw!(zeros(Float64, length(u)), v, u, w, val)

countappw!(out::Vector{Float64}, v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}, 
	   w::Vector{Float64}=fill(1/length(v),length(v)), 
	   val::Float64=0.0) where V = 
begin
	m = length(u)
	n = length(v)
	@assert length(out) == m "[countappw!] output vector size must be $m"
	
	mask = falses(m)			# vector that keeps trak of the modified positions in `n`; non-modified elements get val at the end	
	for i in 1:m				# for after 'u' first: reduce allocations significantly
		out[i] = 0.0
		@simd for j in 1:n			
			b=ifelse(v[j]==u[i],true,false)
			@inbounds mask[i] |= b
			@inbounds out[i] += b*w[j]
		end
	end
	out[.!mask] = val
	return out
end


"""
	gini(p)

Compute the gini impurity of an array `p`.
"""
gini(p::AbstractArray{T}) where T<:Real = begin
	s = zero(T)
	for i = 1:length(p)
		s += p[i]^2
	end
	return 1-s
end

"""
	misclassification(p)

Compute  the misclassification impurity of an array `p`.
"""
misclassification(p::AbstractArray{T}) where T<:Real = begin
	m = zero(T)
	for i = 1:length(p)
		if p[i] > m m=p[i] end
	end
	return 1-m
end

"""
	linearsplit(v, n [;count=0, prop=0])

Generate a sorted vector of `n` linearly spaced values starting from the values of an
input vector `v`. The input vector may be trimmed using the parameters `prop` and `count` 
as in `StatsBase.trim`.
"""
linearsplit(v::T where T<:AbstractVector, n::Int; prop::Float64=0.0, count::Int=0) = begin
	vt = trim(v, prop=prop, count=count)		# trim
	n = min(n, length(unique(vt)))			# number of points 	
	L = linspace(minimum(vt),maximum(vt),n+1); 	# generate space
	return collect( (L+L.step.hi/2)[1:end-1] )	# take midpoints
end

"""
	densitysplitv, n [;count=0, prop=0])

Generate a sorted vector of `n` values, spaced according to the value density of an
input vector `v`. The input vector may be trimmed using the parameters `prop` and `count` 
as in `StatsBase.trim`.
"""
densitysplit(v::T where T<:AbstractVector, n::Int; prop::Float64=0.0, count::Int=0) = begin
	if length(v)/length(unique(v)) < n
		v .+= 10.*rand(length(v))*eps()
	end
	vt = trim(v, prop=prop, count=count)		# trim 
	n = min(n, length(unique(vt)))			# number of points
	vt = sort(sample(vt, n, replace=false))		# sort, sample
	vo = Vector{Float64}(n)				# preallocate output
	@inbounds @simd for i in 1:n-1 	
		vo[i] = vt[i] + (vt[i+1]-vt[i])/2	# threshold position
	end
	vo[n] = vt[n]-eps()				# the last point 
	return vo
end
