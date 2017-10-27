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
isvoid(x) = (x isa Void)



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
		[sort(unique(gety!(c)[i,:])) for i in 1:nvars(gety!(c))] 
	catch
	
		[unique(gety!(c)[i,:]) for i in 1:nvars(gety!(c))] 
	end
end


# Functions that return class numbers
"""
Return the number of classes present in a `DataCell`.
"""
nclass(x::T where T<:CellDataU) = 0
nclass(x::T where T<:CellDataL) = length(classnames(x))
nclass(x::T where T<:CellDataLL) = [length(c) for c in classnames(x)]



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



# Positional argument selectors
idx(args...) = FunctionCell(getindex, args, "Index selector (args=$args)")
idx(f::Function, args...) = FunctionCell((x,args...)->f(getindex(x,args...)), args, "Index selector+function (args=$args)")
