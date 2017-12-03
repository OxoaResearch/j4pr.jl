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
function countapp(v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}) where V 
	countapp!(zeros(Float64, length(u)), v, u)
end

function countapp!(out::Vector{Float64}, v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}) where V
	m = length(u)
	@assert length(out) == m "[countapp!] output vector size must be $m."
	
	@inbounds for i in 1:m				# for after 'u' first: reduce allocations significantly
		out[i] = 0.0 
		@simd for vi in v			
			out[i] +=(vi==u[i])
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
function countappw(v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}, 
			w::Vector{Float64}=fill(1/length(v),length(v)), val::Float64=0.0) where V 
	countappw!(zeros(Float64, length(u)), v, u, w, val)
end

function countappw!(out::Vector{Float64}, v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}, 
	   		w::Vector{Float64}=fill(1/length(v),length(v)), val::Float64=0.0) where V 
	m = length(u)
	n = length(v)
	@assert length(out) == m "[countappw!] output vector size must be $m."
	
	mask = falses(m)			# vector that keeps trak of the modified positions in `out`; non-modified elements get val at the end	
	@inbounds for i in 1:m			# for after 'u' first: reduce allocations significantly
		out[i] = 0.0
		@simd for j in 1:n			
			b=ifelse(v[j]==u[i],true,false)
			mask[i] |= b
			out[i] += b*w[j]
		end
	end
	out[.!mask] = val
	return out
end



"""
	gini(p)

Compute the gini impurity of an array `p`.
"""
function gini(p::AbstractArray{T}) where T<:Real
	s = zero(T)
	@inbounds @simd for i in 1:length(p)
		s += p[i]^2
	end
	return 1-s
end



"""
	misclassification(p)

Compute  the misclassification impurity of an array `p`.
"""
function misclassification(p::AbstractArray{T}) where T<:Real
	m = zero(T)
	@inbounds for i in 1:length(p)
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
function linearsplit(v::T where T<:AbstractVector, n::Int; prop::Float64=0.0, count::Int=0)
	vt = trim(v, prop=prop, count=count)		# trim
	n = min(n, length(unique(vt)))			# number of points 	
	L = linspace(minimum(vt),maximum(vt),n+1); 	# generate space
	return collect( (L.+L.step.hi/2.0)[1:end-1] )	# take midpoints
end



"""
	densitysplitv, n [;count=0, prop=0])

Generate a sorted vector of `n` values, spaced according to the value density of an
input vector `v`. The input vector may be trimmed using the parameters `prop` and `count` 
as in `StatsBase.trim`.
"""
function densitysplit(v::T where T<:AbstractVector, n::Int; prop::Float64=0.0, count::Int=0)
	if length(v)/length(unique(v)) < n
		v .+= 10*rand(length(v))*eps()
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



"""
	confusionmatrix(predictions, references [;kwargs])

Creates and returns the confusion matrix corresponding to the `predictions` 
and `references` input label vectors. The two vectors must have the same element
type and be of the same size. The returned confusion matrix is a `Matrix{Float64}`.

# Keyword arguments
  * `showmatrix::Bool` specifies whether the matrix should be printed (default `false`)
  * `normalize::Bool` specifies whether the matrix should normalized with respect to the number of labels (default `false`)
  * `positive` is the value of the positive or target class; if its value is `nothing`, the 
  confusion matrix is calculating considering all classes while is a value of the same type
  as the element type of `predictions` and `references` is specified (the value must be present
  in the `references` vector, it will consider that class having the label `true` and 
  all other classes as having the label `false`

# Examples
```
julia> using j4pr

julia> references = ["a","a","c","c","b","b"];

julia> predictions = ["a","b","c","c","a","a"];

julia> confusionmatrix(predictions,references)
3×3 Array{Float64,2}:
 1.0  2.0  0.0
 1.0  0.0  0.0
 0.0  0.0  2.0

julia> confusionmatrix(predictions,references;normalize=true)
3×3 Array{Float64,2}:
 0.5  1.0  0.0
 0.5  0.0  0.0
 0.0  0.0  1.0

julia> confusionmatrix(predictions,references;normalize=true,positive="a")
2×2 Array{Float64,2}:
 0.5  0.5
 0.5  0.5

julia> confusionmatrix(predictions,references;normalize=true,positive="a",showmatrix=true);

reference labels (columns), "a" is "true":
 "true"  "false" 
------------
0.5   0.5   
0.5   0.5   
------------
```
"""
function confusionmatrix(predictions::AbstractArray{T}, references::AbstractArray{T}; 
			showmatrix::Bool=false, normalize::Bool=false, positive=nothing) where T
	@assert length(predictions) == length(references) "[confusionmatrix] The predicted and reference labels should have the same length."
	
	# If positive class is specified, binarize labels	
	_binarize_(predictions,references,::Void) = 
		predictions, references, sort(unique(predictions)), sort(unique(references))
	
	_binarize_(predictions::AbstractArray{T}, references::AbstractArray{T}, positive::T) = begin
		@assert positive in references "[confusionmatrix] $(positive) is not in the reference label vector."
		yb = falses(length(predictions))
		yrb = falses(length(references))
		yb[predictions.==positive] = true
		yrb[references.==positive] = true
		return yb, yrb, sort(unique(yb),rev=true), sort(unique(yrb),rev=true)
	end
	y, yr, yu, yru = _binarize_(predictions, references, positive)


	# Construct confusion matrix
	C = length(yru)
	@assert issubset(yu,yru) "[confusionmatrix] The predicted labels should be a subset of the reference labels."
	cm::Matrix{Float64} = zeros(C,C)
	@inbounds for (j,cr) in enumerate(yru)
		for (i,cp) in enumerate(yru)
			cm[i,j] = sum((yr .== cr) .& (y .== cp))
		end
	end

	# Check for normalization
	if normalize 
		# Loop through the classes and normalize the columns
		# of the confusion matrix with respect to their sum
		# i.e. the sum of each column should be 1.0
		for j in 1:C
			cm[:,j]/=sum(yr .==yru[j])
		end
	end

	# Check if the matrix should be nicely printed or not
	if showmatrix
		println()
		if !(positive isa Void)
			println("reference labels (columns), \"$(positive)\" is \"true\":")
		else
			println("reference labels (columns):")
		end
		lsize=ceil(Int, log10(length(y)))+2
		println(sprint((io::IO,v)->map(x->print(io,lpad(" \"$x\" ",lsize)),v), yru))

		println(repeat("-", (lsize+3)*C))
		for i in 1:size(cm,1)
			for j in 1:size(cm,2)
				print(lpad("$(cm[i,j])   ",lsize))
			end
			println()
		end
		println(repeat("-", (lsize+3)*C))
	end
	return cm
end



# Function that generates integers based on  priors, that sum up to a given number N;
# N random numbers are generated and the cumulative distribution is thresholded so that
# a fraction of the total sum (given by the priors) falls closely to a cardinal in the 
# cumulative distribution
# 
# Input:
# 	N 	- sum of the numbers
# 	priors 	- prior probabilities (must sum up to one)
# Output:
#	out 	- vector of integers that sum up to N and has the same size as the priors vector 
function pintgen(n::Int, priors::Vector{T} where T<:Real)
	if (checkpriors(priors) != true)
		error("[pintgen] Priors must summate to one and be >=0, <=1.")
	end
	m = length(priors)									# Length of the priors vector (and also of output vector)
	out = zeros(m) 										# Initialize output
	v = rand(n) 										# Generate random variable vector
	csv = cumsum(v)
	sv = csv[n]
	csp = cumsum(priors)
	for i = 1 : m
		pos = find(csv .<= csp[i] * sv);						# Find positions where the cumulative sum is smaller
		if isempty(pos)  								# than the total sum multiplied by the prior
			out[i] = 0
		else
			out[i] = maximum(pos)							# Return the last position or 0, depending wether any
		end 										# position has been found or not
		if (i > 1) out[i] = out[i] - sum(out[1:i-1]) end				# Compensate positions for using the cumulative priors
	end	
	return round.(Int,out)
end



# Version of the function that ignores priors and returns whatever the first argument is;
# Basically a wrapper so that both specifying class numbers and total sample size works transparently
function pintgen(n::Vector{Int}, priors::Vector{T} where T<:Real)
	@assert checkpriors(priors) && (size(n) == size(priors)) 
		"[pintgen] Priors are not correct or size mismatch between class sizes and priors."
	return n
end




