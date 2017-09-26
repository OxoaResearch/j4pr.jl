###########################
# FunctionCell Interface  #	
###########################
"""
	filterdomain!(opts::Dict)

Returns a cell that processes data piped to it according to the domains specified in `opts`. 
The format of `opts` is `Dict(idx=>domain)` where:

  * `idx` can be an `Int`, `Vector{Int}` or `UnitRange` and specifies the variable indices
  * `domain` can be a vector or admissible values, matrix specifying intervals with the 
first column specifying the lower bounds and second column the upper bounds, a single value 
representing the only admissible value or function in which case the function is applied 
to each element of the variable.

Filtered values are replaced by `NaN`,`""` and `:()` depending on the type of the data being
filtered.

# Examples
```
julia> a=[1,2,3,4,5.0]
5-element Array{Float64,1}:
 1.0
 2.0
 3.0
 4.0
 5.0

julia> a |> filterdomain!(Dict(1=>[1.0 2; 4 6.0]))
5-element Array{Float64,1}:
 1.0
 NaN  
 NaN  
 4.0
 5.0

julia> a = rand(["a","b","c"],3,3)
3×3 Array{String,2}:
 "c"  "b"  "a"
 "b"  "a"  "c"
 "a"  "c"  "c"

julia> a |> filterdomain!(Dict(1=>["a","b"], 2=>"a", 3=>x->x*"X"))
3×3 Array{String,2}:
 ""    "b"   "a" 
 ""    "a"   ""  
 "aX"  "cX"  "cX"
```
"""
filterdomain!(opts::T where T<:Dict) = FunctionCell(filterdomain!, (opts,), "Domain filter!")



############################
# DataCell/Array Interface #	
############################
"""
	filterdomain!(data, opts::Dict)

Processes `data` according to the domains specified in `opts`.
"""
filterdomain!(x::T where T<:Union{AbstractArray,CellData}, opts::S where S) = begin
		
	# Construct out-of-bounds values
	_obvals_(::AbstractFloat)=NaN
	_obvals_(::AbstractString)=""
	_obvals_(::Symbol)=:()
	
	# Construct methods for each domain processing option
	_filter_(v::Function) = v
	_filter_(v::Vector{T} where T) = (x)->(x in v) ? x : _obvals_(x)::typeof(x)
 	_filter_(v::Matrix{T} where T) = 	
		(x)->begin
			any( (x >= v[i,1])&(x < v[i,2]) for i in 1:size(v,1) ) && return x
			_obvals_(x)::typeof(x)
		end
	_filter_(v::T where T<:DataElement) = (x)->(x == v) ? x : _obvals_(x)
	
	# Apply the filters to the each variable
	for (idx, method) in opts
		variable = _variable_(x,idx) 			# Assign temp variable
		
		variable[:] = _filter_(method).(variable)	# Apply domain filter
	end
	
	return x
end
