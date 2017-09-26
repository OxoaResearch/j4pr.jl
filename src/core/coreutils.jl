##############################################################################################################################
# Some core functions needed to make thinks work smoothly :)
##############################################################################################################################

# Data member access functions (useful just in case member names change ...)
# Observation: These functions should be used whenever access to the internals of objects are needed
"""
Get a copy of the `x` field of types derived from `AbstractCell`
"""
getx(ac::T where T<:AbstractCell) = deepcopy(ac.x)		                                        

"""
Reference the `x` field of types derived from `AbstractCell`
"""
getx!(ac::T where T<:AbstractCell) = ac.x

"""
Get a copy of the `y` field of types derived from `AbstractCell`
"""
gety(ac::T where T<:AbstractCell) = deepcopy(ac.y)                                      

"""
Reference the `y` field of types derived from `AbstractCell`
"""
gety!(ac::T where T<:AbstractCell) = ac.y                                            	

"""
Get a copy of the function from the `f` field of `AbstractCell` 
"""
getf(c::T where T<:AbstractCell) = deepcopy(c.f)                                      

"""
Reference the function from the `f` field of `AbstractCell` 
"""
getf!(c::T where T<:AbstractCell) = c.f                                          	



# Number of variables for data vectors, matrices
nvars(x::T where T<:AbstractVector) = 1
nvars(x::T where T<:AbstractMatrix) = size(x,1)

"""
Return the number of variables from a `DataCell` (e.g. the dimensionality of the dataset) 
"""
nvars(x::T where T<:CellData) = nvars(getx!(x))



# Return tuple of data matrix, targets 
"""
Return a Tuple containing the data and targets from a `DataCell` 
"""
strip(x::T where T<:CellDataU) = (getx!(x),)
strip(x::T where T<:CellData) = (getx!(x), gety!(x))


# Size related functions for cell-type objects
size(::T where T<:Void) = 0
size(::T where T<:Void, i::Int) = 0
size(ac::T where T<:AbstractCell) = size(getx!(ac))							# Size of an abstract cells' data
size(ac::T where T<:AbstractCell, i::Int) = size(getx!(ac),i)						# Size along a dimension of an abstract cells' data
ndims(dc::T where T<:CellData) = ndims(getx!(dc))							# Number of dimensions 



# Function needed to properly find NaNs in heterogeneous arrays
isnan(::T where T<:Void) = Bool(false)
isnan(x::T where T<:AbstractString) = Bool(false)
isnan(x::Array{T} where T<:AbstractString) = return BitArray(falses(size(x)))
isnan(x::DataArray{T} where T<:AbstractString) = return BitArray(falses(size(x)))
isnan(x::T where T<:Array) = isnan.(x)

# Function to check for voids
isvoid(::Void) = true
isvoid(x) = x isa Void



# Function to remove NaNs or NAs or both
"""
	uniquenn(v; keep_NA=false, keep_NaN=false)	

Function similar to `unique` that discards `NA`s, `NaN`s or both from the distinct values returned,
according to the values of the parameters `keep_NA` and `keep_NaN`. If the values are `false`, 
these values are discarded.
"""
function uniquenn(v::T where T<:AbstractArray; keep_NA = false, keep_NaN = false)::T
	
	# Keep both NA and NaN
	if (keep_NA && keep_NaN)
		return unique(v)			# Behaviour identical to unique
	
	# Remove NA, keep NaN
	elseif (~keep_NA && keep_NaN)
		uv = unique(v)				# Get unique values (might contain NA, NaN)
		return uv[find(find(.~isna.(uv)))]	# Remove only NAs
	
	# Keep NA, remove NaN
	elseif (keep_NA && ~keep_NaN)
		uv = unique(v)
		NaNpos = find(isnan.(uv))
		deleteat!(uv, NaNpos)

	# Remove both NA, NaN
	elseif (~keep_NA && ~keep_NaN)
		uv = unique(v)				# Get unique values (might contain NA, NaN)
		tmp = uv[find(.~isna.(uv))]		# First remove NAs
		return tmp[find(.~isnan.(tmp))] 	# Second, remove NaNs
	end

end




# Class related functions for DataCells (e.g. class sizes - normalized or not, class names)
"""
Returns a `SortedDict` with the label and associated sample count.
"""
classsizes(c::T where T<:CellDataU) = begin
	return SortedDict(Dict("unlabeled" => nobs(c)))
end

classsizes(c::T where T<:CellDataL) = begin							# Class sizes for DataCells; of the form Dict(clas sname => class size)  
	labels = gety!(c) 
	return SortedDict(countmap(labels))
end

classsizes(c::T where T<:CellDataLL) = begin 							# Class sizes for DataCells with multiple labels; of the form Vector{Dict(...)}
	labels = gety(c)
	return [SortedDict(countmap(labels[i,:])) for i = 1:nvars(gety!(c)) ]
end

"""
Normalized count version of `countmap`.
"""
countmapn(x::T where T<:AbstractVector) = begin
	cm = countmap(x)
	s = eltype(1.0)(sum(values(cm)))
	out = Dict{eltype(x),eltype(1.0)}()
	for k in keys(cm)
		push!(out,k=>eltype(1.0)(cm[k])/s)
	end
	return SortedDict(out)
end

countmapn(::T where T<:Void) = SortedDict(Dict()) 

"""
Returns a `SortedDict` with the label and associated normalized sample count.
"""
nclasssizes(c::T where T<:CellDataU) = SortedDict(Dict("unlabeled"=>1))				# Normalized class sizes for DataCells
	
nclasssizes(c::T where T<:CellDataL) = begin 							# Normalized class sizes for DataCells 
	return countmapn(gety!(c))
end

nclasssizes(c::T where T<:CellDataLL) = begin 							# Normalized class sizes for DataCells with multiple labels
	normcmv = Vector{SortedDict}()
	for i in 1 : size(gety!(c),1)
		push!(normcmv, countmapn(gety!(c)[i,:]))
	end
	return normcmv
end



# Functions that return class names
"""
Return the class labels from a `DataCell`.
"""
classnames(c::T where T<:CellDataU) = []
classnames(c::T where T<:CellDataL) = begin 
	try sort(unique(gety!(c))) 
	catch
		unique(gety!(c))
		
	end
end
classnames(c::T where T<:CellDataLL) = begin
	try	
		[sort(unique(gety!(c)[i,:])) for i = 1:nvars(gety!(c))] 
	catch
	
		[unique(gety!(c)[i,:]) for i = 1:nvars(gety!(c))] 
	end
end


# Functions that return class numbers
"""
Return the number of classes present in a `DataCell`.
"""
nclass(x::T where T<:CellDataU) = 0
nclass(x::T where T<:CellDataL) = size(classnames(x),1)
nclass(x::T where T<:CellDataLL) = [size(i,1) for i = 1:classnames(x)]



# Other various deletion functions
deleteat!(::Void, args...)::Void = nothing

deleteat(x::T where T<:AbstractMatrix{N} where N, rows) = begin	
	m::Int = size(x,1)
	n::Int = size(x,2)
	y::Vector{N} = squeeze(reshape(x,length(x),1),2) 	
	pos=Vector{Int}(length(rows)*n)
	
	k=1
	for i in rows
		@simd for j in 1:n
			@inbounds pos[k]=i+(j-1)*m
			k+=1
		end
	end
	pos = reshape(pos, length(pos))
	y = y[setdiff(1:length(y),pos)]	
	
	return reshape(y, round(Integer,length(y)/n), n)
end		



# Function that counts the layers in an AbstractCell
"""
	countlayers(x::Tuple)

Keeps track of the level of nesting in pipes. 
"""
countlayers(x::T where T<:PTuple{AbstractCell}) = maximum(xi.layer for xi in x)



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



# Small function that builds the titles out of keywork argumens
kwtitle(root::String, kwargs)::String = begin
	title = root 
	if isempty(kwargs) 
		return title
	end
	title *= ": "
	n = length(kwargs)
	for k in 1:length(kwargs)
		title*=string(kwargs[k][1])*"="*string(kwargs[k][2])*(k==n ? "" : " ")
	end
	return title
end


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
countapp(v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}) where V = begin
	m = length(u)
	app = zeros(Float64, m) 			# number of apparitions
	for i in 1:m				# for after 'u' first: reduce allocations significantly
		@simd for vi in v			
			@inbounds app[i] +=(vi==u[i])
		end
	end
	return app
end

"""
	countapp(v,u [,w,val])

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
countappw(v::T where T<:AbstractVector{V}, u::S where S<:AbstractVector{V}, w::Vector{Float64}=fill(1/length(v),length(v)), val::Float64=0.0) where V = begin
	m = length(u)
	n = length(v)
	app = zeros(Float64,m)			# number of apparitions
	mask = falses(m)			# vector that keeps trak of the modified positions in `n`; non-modified elements get val at the end	
	for i in 1:m				# for after 'u' first: reduce allocations significantly
		@simd for j in 1:n			
			b=ifelse(v[j]==u[i],true,false)
			@inbounds mask[i] |= b
			@inbounds app[i] += b*w[j]
		end
	end
	app[.!mask] = val
	return app
end
