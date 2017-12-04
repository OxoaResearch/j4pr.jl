###########################
# FunctionCell Interface  #	
###########################
"""
	ohenc(opts)

Generates a FunctionCell that, when data is piped into it, trains a one-hot encoder (e.g. stores all
distinct values for a subset or all variables). `opts` can be either a string taking the 
values `"binary"` or `"integer"` depending on the type of data encoding desired 
or a `Dict(idx=>method)` if different methods are to be used for different variables. 
In this case, 

* `idx` can be an `Int`, `Vector{Int}` or `UnitRange` and specifies the variable indices
* `method` is a string that specifies the method. Available methods: `"binary"` and `"integer"`.
Unrecognized options will be ignored. 

#Examples
```
 wu = ohenc("binary")
One-hot encoder (binary), varying I/O dimensions, untrained

julia> wt = ["a","b","c"] |> wu
One-hot encoder (binary), 1->3, trained

julia> (+wt).data
DataStructures.SortedDict{Int64,Tuple{String,Array{String,1}},Base.Order.ForwardOrdering} with 1 entry:
  1 => ("binary", String["a", "b", "c"])

julia> ["a","b","d","c","a","d"] |> wt

3×6 Array{Float64,2}:
 1.0  0.0  NaN  0.0  1.0  NaN
 0.0  1.0  NaN  0.0  0.0  NaN
 0.0  0.0  NaN  1.0  0.0  NaN

julia> a = datacell(["a" "a" "b" "b" "b"; "a" "b" "c" "d" "e"],["a","a","b","b","b"])
DataCell, 5 obs, 2 vars, 1 target(s)/obs, 2 distinct values: "b"(3),"a"(2)

julia> wt = a |> ohenc(Dict(1=>"integer",2=>"binary"))
One-hot encoder (mixed), 2->6, trained

julia> (+wt).data
DataStructures.SortedDict{Int64,Tuple{String,Array{String,1}},Base.Order.ForwardOrdering} with 2 entries:
  1 => ("integer", String["a", "b"])
  2 => ("binary", String["a", "b", "c", "d", "e"])

julia> ["x" "x" "a" "b"; "e" "d" "c" "x"] |> wt
6×4 Array{Float64,2}:
 NaN    NaN    1.0    2.0
   0.0    0.0  0.0  NaN  
   0.0    0.0  0.0  NaN  
   0.0    0.0  1.0  NaN  
   0.0    1.0  0.0  NaN  
   1.0    0.0  0.0  NaN  
```
"""
ohenc(opts::T where T<:AbstractString) = FunctionCell(ohenc, (opts,), ModelProperties(), "One-hot encoder ("*opts*")")
ohenc(opts::T where T<:Dict) = FunctionCell(ohenc, (opts,), ModelProperties(), "One-hot encoder (mixed)")



############################
# DataCell/Array Interface #	
############################
"""
	ohenc(data, opts)

Trains a one-hot encoder (e.g. stores all distinct values for a subset or all variables).
"""
# Training
ohenc(x::T where T<:Union{AbstractArray, CellData}, opts) = begin
	
	# Get dictionary or construct a proper one from original;
	# Inputs: encoding option and total number of variables 
	_sdict_(x::T, m) where T<:String = Dict(i=>x for i in 1:m) 
	_sdict_(x::T, m) where T<:Dict = begin
		out = Dict{Int, String}()
		for (k,v) in x 
			if(k isa Int)
				@assert (k>=1)&&(k<= m) "[ohenc] Index $(k) for a variable out of bounds."
				push!(out, k=>v) 
			elseif (k isa UnitRange{Int} || k isa Vector{Int})
				@assert (minimum(k)>=1)&&(maximum(k)<= m) "[ohenc] Index $(k) for a variable out of bounds."
				push!(out, (ki=>v for ki in k)...) 
			else
				error("[ohenc] Unsupported variable index syntax.")
			end
		end
		return out
	end
	dopts = _sdict_(opts, nvars(x))
	
	# Filter new dictionary for unrecognized entries and calculate size_out
	size_out = nvars(x) - length(keys(dopts)) 		# initialize with number of unprocessed variables  
	for (k,v) in dopts
		if (v == "integer") size_out += 1
		elseif (v == "binary") size_out += length(unique(_variable_(x,k)))
		else 
			delete!(dopts, k)			# non-recognized option, delete processing option from dictionary
			size_out += 1				# variable will still be found in output	
		end
	end
	
	# Determine encoder name (for information only)
	_encodername_(x) = begin 
		if isempty(x) return "One-hot encoder ()"
		elseif (length(unique(x)) == 1) return "One-hot encoder ("*collect(x)[1]*")"
		else return "One-hot encoder (mixed)"
		end
	end
	encodername = _encodername_(values(dopts))
	
	# Get list of variables to process/not to process
	variables_to_process = collect(keys(dopts))
	variables_unprocessed = setdiff(collect(1:nvars(x)), variables_to_process)
	
	# Create model properties
	modelprops = ModelProperties(nvars(x), size_out, nothing,  
			Dict("size_unprocessed" => length(variables_unprocessed),
			"variables_to_process" => variables_to_process, 	# Which variables to process 
			"variables_unprocessed" => variables_unprocessed
			)			  								
	)

	# Train
	modeldata = SortedDict(Dict( idx => (v, unique(_variable_(x,idx))) for (idx,v) in dopts ))	
	
	# Return trained cell	
	FunctionCell(ohenc, Model(modeldata, modelprops), encodername)
end



# Execution
ohenc(x::T where T<:CellData, model::Model) = datacell(ohenc(strip(x), model))

ohenc(x::Tuple{T} where T<:AbstractArray, model::Model) = (ohenc(x[1], model),)

ohenc(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractArray, model::Model) = (ohenc(x[1], model),x[2])

ohenc(x::T where T<:AbstractVector, model::Model) = ohenc(mat(x, ObsDim.Constant{2}()),model)

ohenc(x::T where T<:AbstractMatrix, model::Model) = begin
	
	# Read some inputs
	size_in::Int = model.properties.idim
	size_out::Int = model.properties.odim
	size_unprocessed::Int = model.properties.other["size_unprocessed"]
	variables_unprocessed::Vector{Int} = model.properties.other["variables_unprocessed"]

	# Allocate encoding matrix
	B = fill(NaN, size_out - size_unprocessed, nobs(x))

	# Iterate over processing options and write values
	i = 1 #internal row (e.g. new feature) counter
	for (idx, (encoding, v)) in model.data
		xv = view(x,idx,:)
		if (encoding == "integer")
			ohenc_integer!(_variable_(B,i), xv, v)
			i += 1
		elseif (encoding == "binary")
			ohenc_binary!(_variable_(B,i:i+length(v)-1), xv, v)
			i += length(v)
		else
			# Should not arrive here, unknown options are removed from start
			error("[ohenc] Unrecognized encoding option for variable $idx")
		end
	end

	# Build output from processed (B) and unprocessed data
	if !isempty(variables_unprocessed)
		if !isempty(B)			
			# there are some variable unprocessed
			return dcat(B, _variable_(x,variables_unprocessed))
		else 
			# all variables are unprocessed (B empty)
			return dcat(x)
		end
	else 
		# all variables are processed
		return B
	end
end



# Binary encoder 
function ohenc_binary(x::T where T<:AbstractVector, v::S where S<:AbstractVector, val::Float64=1.0)::Matrix{Float64}
	B = fill(NaN, length(v), length(x))
	ohenc_binary!(B, x, v, val)
	return B
end

function ohenc_binary!(B::AbstractMatrix{Float64}, x::T where T<:AbstractVector, v::S where S<:AbstractVector, val::Float64=1.0)
	vf::Vector{Int} = findin(x, v) 	# indices of samples in x found training values
 
	@inbounds for j in vf
		for i in eachindex(v)
			B[i,j] = ifelse(x[j]==v[i],val,0.0) 
		end
	end
	return B							
end



# Integer encoder
function ohenc_integer(x::T where T<:AbstractVector, v::S where S<:AbstractVector)
	n::Int64 = length(x) 		# number of objects
	B = fill(NaN, n)
	ohenc_integer!(B, x, v)
	return B
end

function ohenc_integer!(B::AbstractVector{Float64}, x::T where T<:AbstractVector, v::S where S<:AbstractVector)
	
	m::Int64 = length(v)		# number of distinct values to search in
	vf::Vector{Int} = findin(x, v) 	# indices of samples in x found training values
	
	l=1 # keeps track of the last location of x found in v 
	
	# Function that searches fast a value through a vector of unique values
	function _fastsearch_(v::T,uv::Vector{T})::Int where T
		@inbounds for (i,vi) in enumerate(uv)
			if isequal(vi,v) 
				return i
			end
		end
		return 0
	end
	
	for i in 1:length(vf)
		if (l < vf[i]) 
			B[l:vf[i]-1]=NaN
		end
		@inbounds B[vf[i]] = (1.0:m)[_fastsearch_(x[vf[i]],v)]
		l = vf[i]+1
	end
	return B
end
