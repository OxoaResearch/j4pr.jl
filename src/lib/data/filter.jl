# Define small function that returns true if a value should be replaced, false otherwise
@inline _isreplaceable_(x)::Bool = DataArrays.isna(x)||isnan(x)
@inline _isreplaceable_(x::Nullable)::Bool = isnull(x)||DataArrays.isna(x.value)||isnan(x.value)



###########################
# FunctionCell Interface  #	
###########################
"""
	filterg([f],[g],opts)

Generic data filtering, returns an untrained filter. When data is piped into it, a train filter is generated.
`f` is function to obtain targets by calling `Learn.Base(data,f)`, defaults to `identity`. `g` is a function 
that must return a `::Bool` and specifies if `true` that a given variable value of an observation is 
to be filtered (e.g. if `g=isnan`, `NaN` values of the variable will be replaced with the results of the 
filtering the variable). `opts` specifies the filtering:

* `"mean"` replaces values by the mean of the variable
* `"c-mean"` replaces values by the class-wise mean of the variable
* `"median"` replaces values by the median of the variable
* `"c-median"` replaces values by the median of the class
* `"majority"` replaces values by the most frequent value of the variable
* `"c-majority"` replaces values by the most frequent value of the variable in the same class
* `::Function` a 2 argument function which is applied on each variable value and its associated label 
* `::Dict` a dictionary indicating the variable (key) and one of the above options (values)

# Examples
```
julia> a=j4pr.datacell([1 NaN 1 2 2; NaN 0 1 2 NaN; NaN NaN 0 0 1])
DataCell, 5 obs, 3 vars, 0 classes, unlabeled

julia> wu = j4pr.filterg(isnan, Dict(1=>"mean", 2=>(a,b)->123, 3=>"median"))
Generic filter (mixed), unknown I/O dims, untrained

julia> wt = a |> wu
Filter (mixed), 3 -> 3, trained

julia> +wt
Dict{Int64,Tuple{Function,Any}} with 3 entries:
  2 => (isnan, #11)
  3 => (isnan, j4pr.#110)
  1 => (isnan, j4pr.#109)

julia> +a
3×5 Array{Float64,2}:
   1.0  NaN    1.0  2.0    2.0
 NaN      0.0  1.0  2.0  NaN  
 NaN    NaN    0.0  0.0    1.0

julia> +(a |> wt)
3×5 Array{Float64,2}:
   1.0  1.5  1.0  2.0    2.0
 123.0  0.0  1.0  2.0  123.0
   0.0  0.0  0.0  0.0    1.0

julia> a=j4pr.datacell([1 2 3 4 5; 6 7 8 9 0.0],["a", "a", "a", "b", "b"])
DataCell, 5 obs, 2 vars, 2 classes, labeled: "b"(2),"a"(3)

julia> wu = j4pr.filterg(x->mod(x,5)>2, "mean")
Generic filter (mean), unknown I/O dims, untrained

julia> wt=a |> wu
Filter (mean), 2 -> 2, trained

julia> +a
2×5 Array{Float64,2}:
 1.0  2.0  3.0  4.0  5.0
 6.0  7.0  8.0  9.0  0.0

julia> +(a |> wt)
2×5 Array{Float64,2}:
 1.0  2.0  2.66667  2.66667  5.0
 6.0  7.0  4.33333  4.33333  0.0
```
"""
filterg(f::Function, g::Function, opts::T where T<:AbstractString) = FunctionCell(filterg, (f,g,opts), Dict(), "Generic filter ("*opts*")")
filterg(f::Function, g::Function, opts::T where T<:Function) = FunctionCell(filterg, (f,g,opts), Dict(), "Generic filter (function)")
filterg(f::Function, g::Function, opts::T where T<:Dict) = FunctionCell(filterg, (f,g,opts), Dict(), "Generic filter (mixed)")
filterg(g::Function, opts) = filterg(identity, g, opts)
filterg(opts) = filterg(identity, _isreplaceable_, opts) 



############################
# DataCell/Array Interface #	
############################
"""
	filterg(data,[f],[g],opts)

Generates a trained filter.
"""
# Training
filterg(x::T where T<:Union{AbstractArray, CellData}, opts) = filterg(x, identity, _isreplaceable_, opts)

filterg(x::T where T<:Union{AbstractArray, CellData}, g::Function, opts) = filterg(x, identity, g, opts)

filterg(x::T where T<:Union{AbstractArray, CellData}, f::Function, g::Function, opts) = begin
	
	# Get dictionary or construct a proper one from original;
	# Inputs: filtering options and total number of variables 
	_sdict_(x::T where T<:String, m) = Dict(i=>x for i in 1:m) 
	_sdict_(x::T where T<:Function, m) = Dict(i=>x for i in 1:m) 
	_sdict_(x::T where T<:Dict, m) = begin
		out = Dict{Int,Any}()
		for (k,v) in x 
			if(k isa Int)
				@assert (k>=1)&&(k<= m) "[filterg] Index $(k) for a variable out of bounds."
				push!(out, k=>v) 
			elseif (k isa UnitRange{Int} || k isa Vector{Int})
				@assert (minimum(k)>=1)&&(maximum(k)<= m) "[filterg] Index $(k) for a variable out of bounds."
				push!(out, (ki=>v for ki in k)...) 
			else
				error("[filterg] Unsupported variable index syntax.")
			end
		end
		return out
	end
	dopts = _sdict_(opts, nvars(x))
	
	# Determine filterg name (for information only)
	_filtername_(x) = begin 
		if isempty(x) return "Generic filter ()"
		elseif (length(unique(x)) == 1) 
			if (collect(x)[1] isa Function) return "Generic filter (function)"
			else return "Generic filter ("*collect(x)[1]*")"
			end
		else return "Generic filter (mixed)"
		end
	end
	filtername = _filtername_(values(dopts))

	# Get targets (nothing for unlabeled data cells and arrays)
	labels = _targets_(f,x)
	@assert !any(isna.(labels))		
	@assert !any(isnan.(labels))		
	@assert (labels isa Vector)||(labels isa Void) "[filterg] Labels need to be a Vector or ::Void." 

	# Get priors
	priors = countmapn(labels)	

	# Create model properties
	modelprops =  Dict("priors" => priors,					# Priors
		    	"size_in" => nvars(x), 					# Size of the input data
			"size_out" => nvars(x), 				# Size of the output data (same as input)
			"variables_to_process" => collect(keys(dopts)),		# Which columns to process (e.g. can be any sort of vector
			"targets_function" => f					# Function used to obtain targets (needed at runtime)
	)

	
	# Define filtering functions (v - train data vector, t - train targets array, x - runtime data, y - runtime label) 
	_mean_(v::T where T<:AbstractVector, ::Void) = begin m = mean(v); return (x,y)->m; end
	_mean_(v::T where T<:AbstractVector, t::S where S<:AbstractArray) = _mean_(v,nothing)
	
	_median_(v::T where T<:AbstractVector, ::Void) = begin m = median(v); (x,y)->m; end
	_median_(v::T where T<:AbstractVector, t::S where S<:AbstractArray) = _median_(v,nothing)
	
	_majority_(v::T where T<:AbstractVector, ::Void) = begin 
			cm = countmap(v)
			_, idx = findmax(values(cm))
			m = collect(keys(cm))[idx]
			return (x,y)->m
		end
	_majority_(v::T where T<:AbstractVector, t::S where S<:AbstractArray) = _majority_(v,nothing)
	
	_cmean_(v::T where T<:AbstractVector, ::Void) = _mean_(v,nothing)
	_cmean_(v::T where T<:AbstractVector, t::S where S<:AbstractVector) = begin
		d = Dict(ti=>mean(v[t.==ti]) for ti in unique(t))
		(x,y) -> d[y]
	end
	_cmedian_(v::T where T<:AbstractVector, ::Void) = _median_(v,nothing)
	_cmedian_(v::T where T<:AbstractVector, t::S where S<:AbstractVector) = begin 
		d = Dict(ti=>median(v[t.==ti]) for ti in unique(t))
		(x,y) -> d[y]
	end
	_cmajority_(v::T where T<:AbstractVector, ::Void) = _majority_(v, nothing) 
	_cmajority_(v::T where T<:AbstractVector, t::S where S<:AbstractArray) = begin
		#(x,y) -> Dict(k=>_majority_(v[t.==k], nothing) for k in unique(t))[y]
		d = Dict{eltype(t), eltype(v)}()
		for ti in unique(t)
			cm = countmap(v[t.==ti])
			_, idx = findmax(values(cm))
			m = collect(keys(cm))[idx]
			push!(d, ti=>m)
		end
		return (x,y)->d[y]
	end

	# Create dictionary for each processed variable Dict(column=>(isreplaceable, filter function))
	modeldata = Dict{Int, Tuple{Function,Any}}()
	for (idx, v) in dopts
		fmask = .!g.(_variable_(x,idx))					# Mask for samples that are not to be replaced 	
		fvariable = _variable_(x, idx)[fmask]				# Variable values for the mask	
		flabels = (labels isa Void) ? nothing : labels[fmask]		# Labels for the mask
		
		if v=="mean" 
			w = (g, _mean_(fvariable, flabels)) 
		elseif v=="c-mean" 
			w = (g, _cmean_(fvariable, flabels)) 
		elseif v=="median" 
			w = (g, _median_(fvariable, flabels)) 
		elseif v=="c-median" 
			w = (g, _cmedian_(fvariable, flabels)) 
		elseif v=="majority" 
			w = (g, _majority_(fvariable, flabels)) 
		elseif v=="c-majority" 
			w = (g, _cmajority_(fvariable, flabels)) 
		elseif v isa Function
			w = (g, v)
		else # do not filter if option not recognized
			warn("[filterg] Option $(v) not recognized, will not process variable $(idx)") 
			w = (g, (x,y)->x)	
		end
		push!(modeldata, idx => w ) 
	end
	
	# Return model cell
	return FunctionCell(filterg, Model(modeldata), modelprops, filtername)
end


# Execution
filterg(x::T where T<:Union{AbstractArray, CellData}, model::Model, modelprops::Dict) = begin
	
	f = modelprops["targets_function"]
	xc = deepcopy(x)				# Create local copy of input
	labels = _targets_(f, xc)			# Get labels (it is assumed that the labels are readily available)
						
	for (idx, (isreplaceable, ff)) in model.data 	# Iterate over variable index, replaceable and filter functions
		variable = _variable_(xc,idx) 		# Assign temp variable
		rmask = isreplaceable.(variable) 
		
		# Process variable vector
		@inbounds @fastmath variable[rmask] = ff.(variable[rmask], labels[rmask])
	end
	return xc
end



