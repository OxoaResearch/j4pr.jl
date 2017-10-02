###########################
# FunctionCell Interface  #	
###########################
"""
	sample([f,] opts)

Returns a cell that samples data piped to it according to the sampling options specified 
in `opts`. If the dataset is labeled, an additional function `f` can be specified to 
obtain the labels calling `LearnBase.targets(f,data)`. The argument `opts` can be a 
number specifying the number of samples or fraction desired or, a `Dict(class=>samples)` 
if different sampling is to be applied for different classes. In this case, 

* `class` has to have the same type as the labels 
* `samples` is a number. If it is and `Int`, it specifies the exact number of samples
from its corresponding class to sample. If it is a `Float`, a fraction of samples is 
generated.

When data is passed into the sampling cell, a datasubset of the original data and 
corresponding indices in the dataset are returned.

# Examples
```
julia> a = datacell([1 2 3; 4 5 6],["a","a","b"])
DataCell, 3 obs, 2 vars, 1 target(s)/obs, 2 distinct values: "b"(1),"a"(2)

julia> +a
2×3 Array{Int64,2}:
 1  2  3
 4  5  6

julia> a |> sample(7)
([*]DataCell, 7 obs, 2 vars, 1 target(s)/obs, 2 distinct values: "b"(1),"a"(6), [1, 2, 2, 3, 2, 1, 2])

julia> +(ans[1])
2×7 SubArray{Int64,2,Array{Int64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Array{Int64,1}},false}:
 1  2  2  3  2  1  2
 4  5  5  6  5  4  5

julia> a = datacell([1 2 3 4 5; 6 7 8 9 0],["a","a","b","b","b"]);

julia> +a
2×5 Array{Int64,2}:
 1  2  3  4  5
 6  7  8  9  0

julia> b = sample(a,Dict("a"=>4, "b"=>2.3))
([*]DataCell, 11 obs, 2 vars, 1 target(s)/obs, 2 distinct values: "b"(7),"a"(4), [4, 4, 4, 3, 4, 5, 4, 1, 2, 2, 2])

julia> +b[1]
2×11 SubArray{Int64,2,Array{Int64,2},Tuple{Base.Slice{Base.OneTo{Int64}},Array{Int64,1}},false}:
 4  4  4  3  4  5  4  1  2  2  2
 9  9  9  8  9  0  9  6  7  7  7
```
"""
sample(f::Function, opts::S where S<:Dict{T} where T<:DataElement) = FunctionCell(sample, (f,opts), "Labeled data sampler")
sample(opts::S where S<:Dict{T} where T<:DataElement) = sample(identity, opts) 

sample(f::Function, opts::S where S<:Real) = FunctionCell(sample, (f,opts), "Data sampler")
sample(opts::S where S<:Real) = sample(identity, opts) 



############################
# DataCell/Array Interface #	
############################
"""
	sample(data, [f,] opts)

Samples `data` according to the sampling options specified in `opts`. If the dataset is labeled, 
an additional function `f` can be specified to obtain the labels calling `LearnBase.targets(data,f)`.
Returns a `datasubset` and the indices corresponding to the sampled observations.
"""
# Sampling based on number/percentage of samples desired (f is ignored)
sample(x::T where T<:CellData, opts) = begin
	d, idx= sample(strip(x), opts)
	return (datacell(d), idx)
end

sample(x::T where T<:CellData, f::Function, opts::S where S<:Real) = begin 
	d, idx = sample(strip(x), f, opts)
	return (datacell(d), idx)
end


sample(x::T where T<:Union{AbstractArray, Tuple{AbstractArray}, Tuple{<:AbstractArray, <:AbstractArray}}, opts::S where S<:Real) = sample(x, identity, opts) 

sample(x::T where T<:Union{AbstractArray, Tuple{AbstractArray}, Tuple{<:AbstractArray, <:AbstractArray}}, f::Function, opts::S where S<:Real) = begin
	idx = _sample_(opts)(collect(1:nobs(x)))
	return (datasubset(x,idx), idx)
end

# Class-based sampling
sample(x::T where T<:Union{CellDataL, CellDataLL}, opts::Dict) = sample(x, identity, opts)

sample(x::T where T<:Union{CellDataL, CellDataLL}, f::Function, opts::Dict) = begin
	d, idx = sample(strip(x), f, opts)
	return (datacell(d), idx)
end

sample(x::T where T<:Tuple{<:AbstractArray, <:AbstractArray}, f::Function, opts::Dict) = begin
	lm = labelmap(targets(f,x[2]))
	idx = reduce(vcat, (k in keys(opts) ? _sample_(opts[k])(lm[k]) : lm[k] for k in keys(lm)))
	return (datasubset(x, idx), idx) 
end



# Main sampling function: return the elements of a vector in proportion to a integer (exact number of returned samples)
# or float (fraction of original number of samples)
_sample_(m::T where T<:Int) = (x)->(m <= length(x)) ? x[randperm(length(x))[1:m]] : rand(x,m)
_sample_(m::T where T<:AbstractFloat) = 
	(x)->begin
		mi = round(eltype(1), m*length(x))
		n = length(x)
		(mi <= n) ? x[randperm(n)[1:mi]] : rand(x,mi)
	end
