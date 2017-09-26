"""
	checktargets(x::T where T<:AbstractArray, y::S where S<:AbstractArray)::S

Function that enforces the targets (e.g. **y** field) of a datacell to be aligned with the size of the data matrix.
"""
function checktargets(x::T where T<:AbstractArray, y::S where S<:AbstractArray)
   
	# If target list is empty, issue error
	@assert !isempty(y)
	
	# If it is not empty, enforce the fact that it should not contain fewer points than the data
	@assert nobs(x) <= nobs(y) "[checktargets] Data size larger than targets size. Resize accordingly." 
   	
	@inline _trunc_(y::T where T<:SubArray{S,1} where S, n::Int) = view(y,1:n)
	@inline _trunc_(y::T where T<:SubArray{S,2} where S, n::Int) = view(y,:,1:n)
	@inline _trunc_(y::T where T<:AbstractArray{S,1} where S, n::Int) = y[1:n]
	@inline _trunc_(y::T where T<:AbstractArray{S,2} where S, n::Int) = y[:,1:n]
	if (nobs(x) < nobs(y))
		warn("[checktargets] Data size size smaller than targets size; taking only the first $(nobs(x)) elements." )
		return _trunc_(y, nobs(x))                                                  	# Truncate label list to data size
	end
	return y
end



# Mask checking functions - takes as input arguments the column indices / mask and the number of columns
checkcolumnmask(x::Vector{T} where T<:Int, N::Integer) = begin
	if all( map( (x)-> x>=0 && x<=1, x ) )							# Check if all integer values are in [0 1] e.g. 0 or 1
		x = Vector{Bool}(x)								# If all values are 0 or 1, convert to boolean array and
		checkcolumnmask(x,N)								# apply booblean checks	
	else											# else, apply integer checks
		@assert length(x) == length(unique(x))						# The mask must contain unique values only
		@assert all(x .> 0)								# No columns should be smaller than 0
		@assert all(x .<= N)	 							# No columns should be larger than the number of columns
		@assert ~isempty(x)								# Mask should not be empty
	end
	xnew = falses(N)
	xnew[x] = true
	return xnew
end

checkcolumnmask(x::Vector{T} where T<:Bool, N::Integer) = begin
	@assert length(x) == N									# Enforce size 
	@assert ~isempty(x)									# Mask should not be empty
	@assert ~isequal(x, falses(size(x))) 							# At least one mask value should be 'true' e.g. to be transformed ...
	return BitVector(x)
end

checkcolumnmask(x::T where T<:BitVector, N::Integer) = begin
	@assert length(x) == N									# Enforce size 
	@assert ~isempty(x)									# Mask should not be empty
	@assert ~isequal(x, T(falses(size(x))) ) 						# At least one mask value should be 'true' e.g. to be transformed ...i
	return x
end



"""
	checkpriors(priors::Vector{T} where T<:Real)

Function that checks the consistency of a priors vector (e.g. all elements between 0 and 1, they sum up to 1)
and returns true of false depending on wether the prior vector is usable or not.
"""
function checkpriors(priors::Vector{T} where T<:Real)
	if (abs(sum(priors) - 1) > 1e-3)							# Check that the priors summate to 1
		return false
	end

	if (any(priors .< 0) || any(priors .> 1))						# Check that all priors are between 0 and 1
		return false
	end
	return true
end

