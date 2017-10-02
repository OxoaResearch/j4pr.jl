#################################################################################################
# Functions needed for the manipulation of datacell labels
#
# Desired functionality:
#     - adding labels to existing DataCell
#     - removing labels from existing DataCell
#     - changing some/all labels from existing DataCell
#     - remap labels to something else : e.g. from "apple" to "bear" etc using regexpz
#     - extract column(s) from data and transform them into labels
#
# Observation: All functions return new objects (they do not modify the existing ones)
#################################################################################################


# Remove labels from DataCells
# - Can be done by creating a new datacell(old_datacell.x) or datacell(old_datacell.x, old_datacell.y[:,...])

# Change labels from existing datacell 
# - Extract current labels, modify accordingly and create new datacell re-using the data and new labels.



"""
	addlabels(x, labels)

Creates a new datacell by adding `labels<:AbstractArray` to the existing labels 
of data cell `x`. 
"""
addlabels(x::T where T<:CellDataU, labels::S where S<:AbstractArray) = datacell(x.x, labels, name=x.tinfo)
addlabels(x::T where T<:CellDataL, labels::S where S<:AbstractVector) = datacell(getx!(x), vcat(gety!(x)', checktargets(getx!(x), labels)'), name=x.tinfo)
addlabels(x::T where T<:CellDataL, labels::S where S<:AbstractMatrix) = datacell(getx!(x), vcat(gety!(x)', checktargets(getx!(x), labels)), name=x.tinfo)
addlabels(x::T where T<:CellDataLL, labels::S where S<:AbstractVector) = datacell(getx!(x), vcat(gety!(x), checktargets(getx!(x), labels)'), name=x.tinfo)
addlabels(x::T where T<:CellDataLL, labels::S where S<:AbstractMatrix) = datacell(getx!(x), vcat(gety!(x), checktargets(getx!(x), labels)), name=x.tinfo)


"""
	unlabel(data)

Removes labels (if any) from the data cell and returns a new datacell with the 
data contents only.
"""
unlabel(x::T where T<:CellData) = datacell(getx(x); name = x.tinfo) 




"""
	labelize(x, f [,idx ,remove])
	labelize(x, idx [,remove])
	labelize(x, v [,remove])

Adds labels (also known as 'targets') to `x::DataCell`. If `x` is not provided, returns a fixed `Function cell` that when piped a DataCell into, 
it will add labels/targets and return it.

# Arguments
  * `f` is a `targets`-related function (e.g. `targets(f,...)` ) that it is applied to the existing labels of `x` if `idx` is not specified or to the 
  variables of `x` indicated by `idx`
  * `idx` specifies variables in `x`; can be anything that can be used as a variable index in a `DataCell`
  * `v` a vector that will become the new targets of `x`
  * `remove` defaults to `true` and specifies whether any existing labels/targets are to be kept or, if `idx` is present, whether to remove the 
  variables from `x` from which the new labels/targets were obtained.
"""
labelize(f::T where T<:Function, remove::Bool=true) = FunctionCell(labelize, (f,remove), "Data labeler: f=$f, remove=$remove")
labelize(v::T where T<:AbstractArray, remove::Bool=true) = FunctionCell(labelize, (v,), "Data labeler: preloaded targets")
labelize(idx::T where T, remove::Bool=true) = FunctionCell(labelize, (idx,remove), "Data labeler: idx=$idx remove=$remove")
labelize(f::T where T<:Function, idx::S where S, remove::Bool=true) = FunctionCell(labelize, (f,idx,remove), "Data labeler: f=$f idx=$idx remove=$remove")


labelize(x::T where T<:CellDataU, f::Function, remove::Bool=true) = error("[labelize] Targets or the indices of variables from which to create targets required.") 
labelize(x::T where T<:CellData, f::Function, remove::Bool=true) = begin 
	if remove
		datacell(getx(x), targets(f, gety(x)), name=x.tinfo) 			# replace labels
	else
		datacell(getx(x), dcat(targets(f, gety(x)),gety(x)), name=x.tinfo)	# add to existing labels
	end
end
labelize(x::T where T<:CellData, v::S where S<:AbstractArray, remove::Bool=true) = begin
	if remove
		return datacell( getx(x), getobs(v), name=x.tinfo) 			# replace labels 
	else
		return datacell( getx(x), dcat(getobs(v), gety(x)), name=x.tinfo ) 	# add to existing labels
	end
end
labelize(x::T where T<:CellData, idx::S where S, remove::Bool=true) = labelize(x, identity, idx, remove) 
labelize(x::T where T<:CellData, f::Function, idx::S where S, remove::Bool=true) = begin 
	labels = targets(f, getx!(varsubset(x,idx)))
	if remove
		return datacell( getobs(_variable_(x, setdiff(1:nvars(x),idx))), labels, name=x.tinfo)  
	else
		return datacell( getx(x), labels, name=x.tinfo )
	end
end
