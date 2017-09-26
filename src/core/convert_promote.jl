##############################################################################################################################
# Conversion and Promotion rules
# Small convention: Convert from type T to type S
##############################################################################################################################

"""
	mat(x::T where T<:AbstractVector, dim=LearnBase.ObsDim.Constant{1})

Function that transforms a vector into a row or column matrix.
"""
mat(x::T where T<:AbstractVector) = mat(getobs(x), LearnBase.ObsDim.Constant{1}())
mat(x::T where T<:AbstractVector, dim::S where S<:LearnBase.ObsDim.Constant{1}) = reshape(getobs(x), size(x,1), 1) 
mat(x::T where T<:AbstractVector, dim::S where S<:LearnBase.ObsDim.Constant{2}) = reshape(getobs(x), 1, size(x,1)) 
mat(x::T where T<:AbstractMatrix, dim::S where S<:LearnBase.ObsDimension) = getobs(x) 
mat(x::T where T<:Vector) = mat(x, LearnBase.ObsDim.Constant{1}())
mat(x::T where T<:Vector, dim::S where S<:LearnBase.ObsDim.Constant{1}) = reshape(x, size(x,1), 1) 
mat(x::T where T<:Vector, dim::S where S<:LearnBase.ObsDim.Constant{2}) = reshape(x, 1, size(x,1)) 
mat(x::T where T<:Matrix, dim::S where S<:LearnBase.ObsDimension) = x 

# Convert from Vectors to Matrices
convert(::Type{S}, x::T where T<:AbstractVector) where S<:AbstractMatrix = mat(x)

# Convert from CellData to AbstractArray
convert(::Type{S}, x::T where T<:DataCell) where S<:AbstractArray = convert(S, getx!(x))

# Convert to DataCell
convert(::Type{S}, x::T where T<:DataElement) where S<:DataCell = DataCell([x], nothing, nothing, 0, "DataCell (convert)", eval(oinfoglobal))
convert(::Type{S}, x::T where T<:AbstractArray) where S<:DataCell = DataCell(x, nothing, nothing, 0, "DataCell (convert)", eval(oinfoglobal))

# Convert to FunctionCell
convert(::Type{S}, x::T where T<:DataElement) where S<:FunctionCell = FunctionCell(nothing, nothing, (args...)->x, (), (), 0, "Constant output=$x", eval(oinfoglobal))
convert(::Type{S}, f::T where T<:Function) where S<:FunctionCell = FunctionCell(f, (), string(f))


# Convert to AbstractCell (default conversions to Cells-like objects)
convert(::Type{S}, x::T where T<:DataElement) where S<:AbstractCell = DataCell([x], nothing, nothing, 0, "DataCell (convert)", eval(oinfoglobal))
convert(::Type{S}, x::T where T<:AbstractArray) where S<:AbstractCell = DataCell(x, "DataCell (convert)")
convert(::Type{S}, f::T where T<:Function) where S<:AbstractCell = FunctionCell(f, (), string(f))
convert(::Type{S}, x::T where T<:Tuple) where S<:AbstractCell = S(x...) 	
