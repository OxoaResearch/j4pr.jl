##############################################################################################################################
# Call functions (used in map/pmap etc ...)
##############################################################################################################################

# DataCell(...) 
(x::T where T<:CellData)(f::S where S<:CellData) = x |> f;		
(x::T where T<:CellData)(f::S where S<:CellFun) = x |> f;		
(x::T where T<:CellData)(p::S where S<:PipeGeneric) = x |> p	

# FunctionCell(...)
(f::T where T<:CellFun)(x::S where S<:CellData) =  x |> f
(f1::T where T<:CellFun)(f2::S where S<:CellFun) = f1 + f2
(f::T where T<:CellFun)(p::S where S<:PipeGeneric) = f + p
(f::T where T<:CellFun)(x::S where S) =  x |> f		

# PipeCell(...)
(p::T where T<:PipeGeneric)(x::S where S<:CellData) = x |> p			
(p::T where T<:PipeGeneric)(f::S where S<:CellFun) = p + f
(p1::T where T<: PipeGeneric)(p2::S where S<:PipeGeneric) = p1 + p2
(p::T where T<:PipeGeneric)(x::S where S) = x |> p					


# Generic calls
(f::T where T<:CellFun)(x::S where S<:AbstractCell) = begin
	try
		getx!(x) |> f
	catch
		warn("[call] Passing generic cell content into function cell failed, will make a pipe.") 
		x+f
	end

end



