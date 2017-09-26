###########################
# FunctionCell Interface  #	
###########################
"""
	cslice(labels [,idx=1])

Returns a cell that slices data piped to it according
to a vector `labels` provided. If multiple labels are available, 
`idx` selects which one to slice on. The returned structed is a 
`datasubset` of the original data.
# Examples
```
julia> using j4pr

julia> a=datacell([1,2,3,4,5],["a","b","a","b","a"])
DataCell, 5 obs, 1 vars, 2 classes, labeled: "b"(2),"a"(3)

julia> b=a |> cslice(["a"])
[*]DataCell, 3 obs, 1 vars, 1 classes, labeled: "a"(3)

julia> +b
3-element SubArray{Int64,1,Array{Int64,1},Tuple{Array{Int64,1}},false}:
 1
 3
 5

julia> a=datacell([1,2,3],["a" "b" "a"; "c" "c" "d"])
DataCell, 3 obs, 1 vars, 0 classes, multi-labeled: 2 label sets

julia> b=a |> cslice(["d"],2)
[*]DataCell, 1 obs, 1 vars, 0 classes, multi-labeled: 2 label sets

julia> +b
1-element SubArray{Int64,1,Array{Int64,1},Tuple{Array{Int64,1}},false}:
 3
```
"""
cslice(label_list::T where T<:Vector{<:DataElement}, idx::Int=1) = FunctionCell(cslice, (label_list, idx), "Class slicer")



##########################
# DataCell Interface     #	
##########################
"""
	cslice(data, labels [,idx=1])

Slices `data` according to a vector `labels` provided and `idx`. 
"""
cslice(x::T where T<:CellDataU, label_list::T where T<:Vector{<:DataElement}, idx::Int=1) = begin 
	@assert !any(isna.(targets(x)))		
	@assert !any(isnan.(targets(x)))		
	return datasubset(x)
end

cslice(x::T where T<:CellDataL, label_list::T where T<:Vector{<:DataElement}, idx::Int=1) = begin 
	@assert !any(isna.(targets(x)))		
	@assert !any(isnan.(targets(x)))		
	idxs::Vector{Int} = findin(targets(x), label_list)
	return datasubset(x, idxs)
end

cslice(x::T where T<:CellDataLL, label_list::T where T<:Vector{<:DataElement}, idx::Int=1) = begin 
	@assert !any(isna.(targets(x)))		
	@assert !any(isnan.(targets(x)))		
	idxs::Vector{Int} = findin(targets(x)[idx,:], label_list)
	return datasubset(x, idxs)
end
