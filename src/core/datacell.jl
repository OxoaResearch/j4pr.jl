"""
	DataCell{T,S,U}(x::T, y::S, f::U, layer::Int, tinfo::String, oinfo::String)

Constructs the basic object for holding data, the `DataCell`. The constructor is parametric so it can be
easily extended to new data types.

# Main `DataCell` constructors
  * `DataCell(x::AbstractArray, tinfo::String="")` creates an `unlabeled` object.
  * `DataCell(x::AbstractArray, y::AbstractVector, tinfo::String="")` creates a `labeled` object. Useful for classification.
  * `DataCell(x::AbstractArray, y::AbstractMatrix, tinfo::String="")` creates a `multi-labeled` object. Useful for classification/regression

For the creation of data cells, the `datacell` function is quite well suited.
"""
#############
# DataCells # 
#############
struct DataCell{T,S,U} <: AbstractCell{T,S,U}
	x::T                                                                                    # Data
    	y::S                                                                                    # Labels/probabilities etc.
    	f::U                                                                                    # Usually a function
	layer::Int										# Layer
    	tinfo::String										# Title or name
    	oinfo::String                                                                        	# Other information
end

DataCell(x::T where T<:AbstractArray, tinfo::String = "") = DataCell(x, nothing, nothing, 0, tinfo, oinfoglobal)
DataCell(x::T where T<:AbstractArray, y::S where S<:Void, tinfo::String = "") = DataCell(x, nothing, nothing, 0, tinfo, oinfoglobal)
DataCell(x::T where T<:AbstractArray, y::S where S<:AbstractArray{Void}, tinfo::String = "") = DataCell(x, nothing, nothing, 0, tinfo, oinfoglobal)
DataCell(x::T where T<:AbstractArray, y::S where S<:AbstractArray, tinfo::String = "") = DataCell(x, checktargets(x,y), nothing, 0, tinfo, oinfoglobal)



####################
# DataCell aliases #
####################
const DataElement{T<:Union{Number, AbstractString, Symbol}} = Union{T,Nullable{T}}
const FixedElement{T<:Union{Number, AbstractString, Symbol}} = T
const NullableElement{T<:Union{Number, AbstractString, Symbol}} = Nullable{T}
const CellDataVec{T<:AbstractVector} = DataCell{T, Void, Void}
const CellDataVecVec{T<:AbstractVector, S<:AbstractVector} = DataCell{T, S, Void}
const CellDataVecMat{T<:AbstractVector, S<:AbstractMatrix} = DataCell{T, S, Void}
const CellDataMat{T<:AbstractMatrix} = DataCell{T, Void, Void}
const CellDataMatVec{T<:AbstractMatrix, S<:AbstractVector} =  DataCell{T, S, Void}
const CellDataMatMat{T<:AbstractMatrix, S<:AbstractMatrix} =  DataCell{T, S, Void}
const CellDataVectors{T<:AbstractVector, S<:Union{AbstractArray, Void}} = DataCell{T, S, Void}	
const CellDataMatrices{T<:AbstractMatrix, S<: Union{AbstractArray, Void}} = DataCell{T, S, Void}
const CellDataU{T<:AbstractArray} = DataCell{T, Void, Void} 					# Data cell, unlabeled 
const CellDataL{T<:AbstractArray, S<:AbstractVector} = DataCell{T, S, Void}			# Data cell labeled
const CellDataLL{T<:AbstractArray, S<:AbstractMatrix} = DataCell{T, S, Void}			# Data cell multiple labels
const CellData{T<:AbstractArray, S} = DataCell{T, S, Void}
#const TrainableData{U <:Real, T<:AbstractMatrix{U}, S<:AbstractArray} = DataCell{T, S, Void} 	



##############################################################################################################################
# Operators [Data cells]												     #
##############################################################################################################################

# Vertical datacell concatenation (e.g. data concatenation) and auxiliary methods
dcat(x::T where T<:PTuple{<:Void}) = nothing
dcat(x::T where T<:Array{<:Void}) = nothing
dcat(::Void) = nothing 
dcat(::Void,::Void) = nothing
dcat(x,::Void) = dcat(x)
dcat(::Void,x) = dcat(x)
dcat(x::AbstractVector) = getobs(x)
dcat(x::AbstractMatrix) = getobs(x)
dcat(x::Matrix) = x 
dcat(x::Vector) = x
dcat(x::AbstractVector, y::AbstractVector) = vcat(mat(x, ObsDim.Constant{2}()), mat(y,ObsDim.Constant{2}()))
dcat(x::AbstractMatrix, y::AbstractVector) = vcat(dcat(x), mat(y, ObsDim.Constant{2}()))
dcat(x::AbstractVector, y::AbstractMatrix) = vcat(mat(x, ObsDim.Constant{2}()), dcat(y))
dcat(x::AbstractMatrix, y::AbstractMatrix) = vcat(dcat(x), dcat(y))
dcat(x,y,z...) = dcat(dcat(x,y),z...)
dcat(x) = dcat(x...)
vcat(c::T...) where T<:CellData = begin
	@assert all(labx == laby for labx in gety!.(c), laby in gety!.(c)) "[vcat] 'y' fields have to be identical for all DataCells."
	datacell(dcat(getx!.(c)), gety!(c[1]))
end


# Horizontal datacell concatenation (e.g. observation concatenation) and auxiliary methods
ocat(x::T where T<:PTuple{<:Void}) = nothing
ocat(x::T where T<:Array{<:Void}) = nothing
ocat(::Void) = nothing 
ocat(::Void,::Void) = nothing
ocat(x,::Void) = ocat(x)
ocat(::Void,x) = ocat(x)
ocat(x::AbstractVector) = getobs(x)
ocat(x::AbstractMatrix) = getobs(x)
ocat(x::Vector) = x
ocat(x::Matrix) = x
ocat(x::AbstractVector, y::AbstractVector) = vcat(ocat(x), ocat(y))
ocat(x::AbstractMatrix, y::AbstractVector) = hcat(ocat(x), mat(y, ObsDim.Constant{2}()))
ocat(x::AbstractVector, y::AbstractMatrix) = hcat(mat(x, ObsDim.Constant{2}()), ocat(y))
ocat(x::AbstractMatrix, y::AbstractMatrix) = hcat(ocat(x), ocat(y))
ocat(x,y,z...) = ocat(ocat(x,y),z...)
ocat(x) = ocat(x...)
hcat(c::T...) where T<:CellData = datacell(ocat(getx!.(c)), ocat(gety!.(c)))



# Piping operators
|>(x::T where T<:AbstractArray, c::S where S<:CellData) = begin
    	info("[operators] Creating new data cell with Array contents..." )
    	datacell(x)
end

|>(x::T where T<:Tuple, c::S where S<:CellData) = begin 					# Generally used in the case: (data, labels ) |> datacell([])
	if (nobs(c) > 0)
		info("[operators] Creating new data cell appending Tuple contents..." )
		datacell(dcat(x[1],getx!(c)), dcat(x[2], gety!(c)))
	else
		info("[operators] Creating new data cell with Tuple contents..." )
		datacell(x...)
	end
end


|>(x1::T where T<:CellDataU, x2::S where S<:CellDataU) = begin                 			# Generally used to glue together two DataCells
	@assert(nobs(x1) == nobs(x2))
	datacell(dcat(getx!(x1),getx!(x2)))
end

|>(x1::T where T<:CellDataU, x2::S where S<:CellData) = begin                 			# Generally used to glue together two DataCells
	@assert(nobs(x1) == nobs(x2))
	datacell(dcat(getx!(x1),getx!(x2)), gety!(x2))
end

|>(x1::T where T<:CellData, x2::S where S<:CellDataU) = begin                 			# Generally used to glue together two DataCells
	@assert(nobs(x1) == nobs(x2))
	datacell(dcat(getx!(x1),getx!(x2)), gety!(x1))
end

|>(x1::T where T<:CellData, x2::S where S<:CellData) = begin                 			# Generally used to glue together two DataCells
	@assert(nobs(x1) == nobs(x2))
	if isequal(gety!(x1), gety!(x2))	
		datacell(dcat(getx!(x1),getx!(x2)), gety!(x1))
	else
		datacell(dcat(getx!(x1),getx!(x2)), dcat(gety!(x1),gety!(x2)))
	end
end



###########################
# Indexing for data cells #
###########################

# Generic setindex! for data cells
setindex!(c::T where T<:CellData, data, inds...) = setindex!(getx!(c), data, inds...)


# Generic indexing operator, no argument
getindex(c::CellData) = j4pr.datacell(eltype(getx!(c))[])
getindex(c::CellData, ::Colon) = c 
getindex(c::CellData, ::Colon, ::Colon) = c

# Single integer indexing (select sample)
getindex(c::Union{CellDataVec, CellDataVecVec}, i::Int64) = DataCell([getx!(c)[i]], [gety!(c)[i]] )				
getindex(c::CellDataVecMat, i::Int64) = DataCell([getx!(c)[i]], gety!(c)[:,i:i] )				
getindex(c::Union{CellDataMat, CellDataMatVec}, i::Int64) = DataCell(getx!(c)[:,i:i], [gety!(c)[i]] )				
getindex(c::CellDataMatMat, i::Int64) = DataCell(getx!(c)[:,i:i], gety!(c)[:,i:i] )				

# Integer + Colon indexing (select variable) - only for matrix data
getindex(c::CellDataMatrices , i::Int64, ::Colon) = DataCell(getx!(c)[i:i,:], gety!(c) )				

# Colon + Integer indexing (select sample) - only for matrix data
getindex(c::Union{CellDataMat, CellDataMatVec}, ::Colon, i::Int64) = DataCell(getx!(c)[:,i:i], [gety!(c)[i]])  					
getindex(c::CellDataMatMat, ::Colon, i::Int64) = DataCell(getx!(c)[:,i:i], gety!(c)[:,i:i])  					

# Range/Vector indexing (select samples) - only for vector data
getindex(c::Union{CellDataVec, CellDataVecVec}, i::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[i], gety!(c)[i])
getindex(c::CellDataVecMat, i::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[i], gety!(c)[:,i])

# Range/Vector + Colon indexing (select variables) - only for matrix data 
getindex(c::Union{CellDataMat, CellDataMatVec}, i::Union{UnitRange{Int64}, Vector{Int64}}, ::Colon) = DataCell(getx!(c)[i,:], gety!(c)[:]) 
getindex(c::CellDataMatMat, i::Union{UnitRange{Int64}, Vector{Int64}}, ::Colon) = DataCell(getx!(c)[i,:], gety!(c)[:,:]) 

# Colon + Range/Vector indexing (select samples) - only for matrix data
getindex(c::Union{CellDataMat, CellDataMatVec}, ::Colon, i::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[:,i], gety!(c)[i])
getindex(c::CellDataMatMat, ::Colon, i::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[:,i], gety!(c)[:,i])

# Range/Vector + Range/Vector indexing (select samples and variables) - only for matrix data
getindex(c::Union{CellDataMat, CellDataMatVec}, i::Union{UnitRange{Int64}, Vector{Int64}}, j::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[i,j], gety!(c)[j])    
getindex(c::CellDataMatMat, i::Union{UnitRange{Int64}, Vector{Int64}}, j::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[i,j], gety!(c)[:,j]) 

# Getindex for void arguments (needed to get indexes for CellDataU objects)
getindex(c::T where T<:Void, args...) = nothing



######################################
# Iteration interface for data cells #
######################################
start(c::CellData) = 1
next(c::CellData, state) = (datasubset(c, state), state+1)
done(c::CellData, state) = state > nobs(c)
endof(c::CellData) = nobs(c)
eltype(c::CellData) = typeof(c)
length(c::CellData) = nobs(c)



"""
	datacell([x,y,name=""])

Calls the DataCell constructor for creating `data cells`. It is the easyest way of creating such
objects from `AbstractArrays`. Consult `methods(datacell)` for a full list of available methods.
If no arguments are provided, a `FunctionCell` that expects an input compatible with the `DataCell` 
constructor is returned.
"""
# Tier 1 - low level
datacell(x::T where T<:DataElement; name = "") = DataCell([x], name)
datacell(x::T where T<:DataElement, y::S where S<:DataElement; name = "") = DataCell([x], [y], name)
datacell(x::T where T<:DataElement, y::S where S<:AbstractArray; name = "") = DataCell([x], y, name)
datacell(x::T where T<:AbstractArray, y::S where S<:DataElement; name = "") = DataCell(x, [y], name)
datacell(x::T where T<:AbstractArray; name = "") = DataCell(x, name)
datacell(x::T where T<:AbstractArray, ::Void; name = "") =	DataCell(x, name)
datacell(x::T where T<:AbstractArray, y::S where S<:AbstractArray; name = "") = DataCell(x, y, name)
datacell(t :: Tuple{T} where T<:AbstractArray; name = "") = DataCell(t[1], name)
datacell(t :: Tuple{T,S} where {T<:AbstractArray, S<:AbstractArray}; name = "") = DataCell(t[1], t[2], name)
datacell(t :: Tuple{T,S} where {T<:AbstractArray,S}; name = "") = DataCell(t[1], name)

# Tier 2 - j4pr
datacell(x::T where T<:CellData; name = x.tinfo) = DataCell(getx!(x), gety!(x), name)
datacell(x::T where T<:CellData, y::S where S<:CellData; name = "") = datacell(getx!(x), getx!(y), name)

# DataCell creator
datacell() = FunctionCell((args...)->j4pr.DataCell(args...), (), "DataCell creator")

