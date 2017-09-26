"""
	functioncell([x,y,z])

Calls the `FunctionCell` constructor for one, two or three arguments. If no arguments are provided, 
a `FunctionCell` that expects as input a collection with contents compatible with the arguments of
the `FunctionCell` constructor is returned.
"""
functioncell() = FunctionCell((args)->FunctionCell(args...), (), "FunctionCell creator") 	
functioncell(x::T) where T = FunctionCell(x)					# calls to convert methods
functioncell(x::T, y::S) where {T,S}= FunctionCell(x, y)			# Calls any 2 argument parametric constructor 
functioncell(x::T, y::S, z::U) where {T,S,U}= FunctionCell(x, y, z)		# Calls any 3 argument parametric constructor



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
