###############################
# Interface for MLDataPattern #
###############################
# Number of objects for data cells
"""
	nobs(data, [obsdim])

Get the number of observations in a `DataCell`.Check out documentation from `MLDataPatterns` for more details.
"""
nobs(::Void, args...) = 0
nobs(::Void, ::LearnBase.ObsDim.Undefined) = 0
nobs(x::T where T<:CellDataVectors, args...) = size(getx!(x),1)
nobs(x::T where T<:CellDataVectors, ::LearnBase.ObsDim.Undefined) = size(getx!(x),1)
nobs(x::T where T<:CellDataMatrices, args...) = size(getx!(x),2)
nobs(x::T where T<:CellDataMatrices, ::LearnBase.ObsDim.Undefined) = size(getx!(x),2)



"""
	getobs(data, [idx], [obsdim])

Get observation from `DataCell`. Check out documentation from `MLDataPatterns` for more details.
"""
getobs(x::T where T<:CellDataVectors, idx) = x[idx]
getobs(x::T where T<:CellDataMatrices, idx) = x[:,idx]
#getobs!{T<:CellData}(buf, x::T, args...) = copy!(buf, datacell(getobs(strip(x), args...))); 



"""
	datasubset(data, [idx], [obsdim])

Returns a lazy subset from a `DataCell`. Check out documentation from `MLDataPatterns` for more details.
"""
#datasubset{T<:CellDataVectors}(x::T, idx) = Cell(view(getx!(x),idx))
#datasubset(x::T where T<:CellData, idx; obsdim=LearnBase.ObsDim.Constant{2}()) = DataCell(datasubset(strip(x), idx; obsdim=obsdim))
datasubset(x::T where T<:CellData, idx::Int) = DataCell(datasubset(strip(x),idx:idx))
datasubset(x::T where T<:CellData, idx::Int, ::LearnBase.ObsDim.Undefined) = DataCell(datasubset(strip(x),idx:idx))
datasubset(x::T where T<:CellData, idx, ::LearnBase.ObsDim.Undefined) = DataCell(datasubset(strip(x), idx))



"""
	varsubset(data,[idx])

Similar to `datasubset` however it returns variable subsets
"""
varsubset(x::T where T<:CellDataVec, idx) = DataCell(view(getx!(x),:,idx))				# for unlabeled vector data, ignore index, labels
varsubset(x::T where T<:CellDataVecVec, idx) = DataCell(view(getx!(x),:,idx), view(gety!(x),:))		# for labeled vector data, ignore index
varsubset(x::T where T<:CellDataVecMat, idx) = DataCell(view(getx!(x),:,idx), view(gety!(x),:,:))	# for labeled vector data, ignore idex
varsubset(x::T where T<:CellDataMat, idx) = DataCell(view(getx!(x),idx,:)) 				# For unlabeled matrix data, ignore labels
varsubset(x::T where T<:CellDataMatVec, idx) = DataCell(view(getx!(x),idx,:), view(gety!(x),:))
varsubset(x::T where T<:CellDataMatMat, idx) = DataCell(view(getx!(x),idx,:), view(gety!(x),:,:))
varsubset(x::T where T<:CellData, idx, ::LearnBase.ObsDim.Undefined) = varsubset(x, idx) 
varsubset(x::T where T<:AbstractArray, idx) = _variable_(x,idx)



# Data access: idx selects variable
_variable_(x::T where T<:AbstractVector, idx) = view(x, :, idx)
_variable_(x::T where T<:AbstractMatrix, idx) = view(x, idx, :)
_variable_(x::T where T<:CellData, idx) = _variable_(getx!(x), idx)

# Data access when specifying empty static arrays (always return nothing)
_variable_(x::T where T<:AbstractArray, ::StaticArrays.SVector{0}) = nothing
_variable_(::AbstractArray{T,2} where T, ::StaticArrays.SArray{Tuple{0},S,1,0} where S) = nothing  

# gettargets and gettarget low-level functions for MLDataPattern integration
LearnBase.gettargets(x::T where T<:CellDataU, idx, obsdim::LearnBase.ObsDimension=LearnBase.ObsDim.Undefined()) = nothing
LearnBase.gettargets(x::T where T<:CellDataL, idx, obsdim::LearnBase.ObsDimension=LearnBase.ObsDim.Undefined()) = gety!(x)[idx]
LearnBase.gettargets(x::T where T<:CellDataLL, idx::Number, obsdim::LearnBase.ObsDimension=LearnBase.ObsDim.Undefined()) = gety!(x)[:,idx:idx]
LearnBase.gettargets(x::T where T<:CellDataLL, idx, obsdim::LearnBase.ObsDimension=LearnBase.ObsDim.Undefined()) = gety!(x)[:,idx]
LearnBase.gettarget(x::T where T<:CellDataU) = nothing
LearnBase.gettarget(f, x::T where T<:CellDataU) = nothing
LearnBase.gettarget(x::T where T<:CellDataL) = gety!(x)[1]
LearnBase.gettarget(f, x::T where T<:CellDataL) = f(gety!(x)[1])
LearnBase.gettarget(x::T where T<:CellDataLL) = gety!(x)
LearnBase.gettarget(f, x::T where T<:CellDataLL) = f(gety!(x))



# Optimization for the targets function
targets(x::T where T<:CellDataU, args...) = nothing
targets(f::Function, x::T where T<:CellDataU, args...) = nothing
targets(f::Function, x::T where T<:CellData, args...) = targets(f, gety!(x), args...)
targets(f::Function, obsdim::LearnBase.ObsDimension=LearnBase.ObsDim.Constant{2}()) = 
	FunctionCell((x)->targets(f, x, obsdim),(), "Target generator ($(string(f)))")

