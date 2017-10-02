#########################################################################
# Printers for various data types. They are all overloading Base.show() #
#########################################################################

# Function that returns an alias to be printed for any data cell
_show_celltype_(c::CellData) = !isempty(c.tinfo) ? "" : "DataCell"
_show_celltype_(c::T where T<:DataCell{U,Void,Void} where U<:SubArray) = !isempty(c.tinfo) ? "[*]"*c.tinfo : "[*]DataCell"
_show_celltype_(c::T where T<:DataCell{U,V,Void} where U<:SubArray where V<:SubArray) = !isempty(c.tinfo) ? "[*]"*c.tinfo : "[*]DataCell"
_show_celltype_(c::CellFun) = !isempty(c.tinfo) ? "" : "FunctionCell"
_show_celltype_(c::PipeGeneric) = !isempty(c.tinfo) ? "" : "Generic Pipe"
_show_celltype_(c::PipeStacked) = !isempty(c.tinfo) ? "" : "Stacked Pipe"
_show_celltype_(c::PipeParallel) = !isempty(c.tinfo) ? "" : "Parallel Pipe"
_show_celltype_(c::PipeSerial) = !isempty(c.tinfo) ? "" : "Serial Pipe"
_show_celltype_(c::T where T<:AbstractCell) = !isempty(c.tinfo) ? "" : "AbstractCell"



# Function that prints a status based on a defined alias
_show_status_(::AbstractCell) = "<no status>"
_show_status_(::CellDataU) = ""
_show_status_(::CellDataLL) = ""
_show_status_(::CellDataL) = ""
_show_status_(::CellFunF) = "fixed"
_show_status_(::CellFunU) = "untrained"
_show_status_(::CellFunT) = "trained"
_show_status_(::PipeGeneric) = "generic"
_show_status_(::PipeStackedF) = "fixed"
_show_status_(::PipeParallelF) = "fixed"
_show_status_(::PipeSerialF) = "fixed"
_show_status_(::PipeStackedU) = "untrained"
_show_status_(::PipeParallelU) = "untrained"
_show_status_(::PipeSerialU) = "untrained"
_show_status_(::PipeStackedT) = "trained"
_show_status_(::PipeParallelT) = "trained"
_show_status_(::PipeSerialT) = "trained"



# Title printer 
_show_title_(c::T where T<:AbstractCell) = isempty(c.tinfo) ? "" : c.tinfo



# Dimension printers
_show_dims_(c::T where T<:CellFunF) = "varying I/O dimensions"
_show_dims_(c::T where T<:CellFun) = begin

	# Read input and output dimensions
	_getdims_(c::CellFunU) = (gety!(c).idim, gety!(c).odim)
	_getdims_(c::CellFunT) = (getx!(c).properties.idim, getx!(c).properties.odim)
	idim, odim = _getdims_(c)
	
	# Build dimension string
	dimstr = ""
	if idim !=0 
		if odim != 0
			dimstr = string(idim) * "->" * string(odim)  
		else
			dimstr = string(idim) * "->" * "unknown"
		end
	else
		if odim != 0
			dimstr = "unknown" * "->" * string(odim) 
		else
			dimstr="varying I/O dimensions"
		end
	end
	return dimstr
end
_show_dims_(c::T where T<:CellDataU) = string(nobs(c))*" obs, "*string(nvars(c))*" vars, "*"0 target(s)/obs"
_show_dims_(c::T where T<:CellDataL) = string(nobs(c))*" obs, "*string(nvars(c))*" vars, 1 target(s)/obs, "*string(length(classnames(c)))*" distinct values"
_show_dims_(c::T where T<:CellDataLL) = string(nobs(c))*" obs, "*string(nvars(c))*" vars, "*string(length(classnames(c)))*" target(s)/obs"
_show_dims_(c::T where T<:AbstractCell) = "<no dimensions>"



# Printer for unlabeled data blocks
Base.show(io::IO, c::CellDataU) = print(io, _show_celltype_(c), _show_title_(c), ", ", _show_dims_(c))

# Printer for labeled data blocks
Base.show(io::IO, c::CellDataL) = begin
	cm = countmap(gety!(c))                                                                  	
    	cstr = ": "
	nlab = length(keys(cm)) 
	if nlab > 8 
        	cstr ="";
	else
        	for key in keys(cm)
			cstr = cstr * string("\"", key, "\"", "(", Int(cm[key]),"),")                  
        	end
		cstr=chop(cstr)
    	end

	print(io, _show_celltype_(c), _show_title_(c), ", ", _show_dims_(c), cstr)

end

# Printer for data blocks with multiple labels
Base.show(io::IO, c::CellDataLL) = print(io, _show_celltype_(c), _show_title_(c), ", ", _show_dims_(c))

# Printer for fixed/untrained function cells 
Base.show(io::IO, c::CellFun) = print(io, _show_celltype_(c), _show_title_(c), ", ", _show_dims_(c), ", ", _show_status_(c))

# Printer for pipes
Base.show(io::IO, p::PipeGeneric) = print(io, _show_celltype_(p), _show_title_(p), ", ", length(getx!(p)), " element(s)", ", ",p.layer," layer(s)",", ", _show_status_(p))

# Printer for AbstractCells (e.g. everything generic)
Base.show(io::IO, ac::T where T<:AbstractCell) = print(io, _show_celltype_(ac), _show_title_(ac), " [", typeof(getx!(ac)), "]", ", ",ac.layer," layer(s)")

# Low-level printer for tuples containing AbstractCells
Base.show(io::IO, x::T where T<:PTuple{AbstractCell}) = begin
	println("$(length(x))-element Tuple{$(eltype(x))}:")
	for i in x
		print(io,"`- "); println(io, i)
	end
end

Base.show(io::IO, x::Tuple{}) = "()"
