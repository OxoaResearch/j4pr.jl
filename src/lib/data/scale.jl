###########################
# FunctionCell Interface  #	
###########################
"""
	scaler!([f,] opts)

Returns a cell that scales data piped to it according to the scaling options specified 
in `opts`. If the dataset is labeled, an additional function `f` can be specified to 
obtain the labels calling `LearnBase.targets(f,data)`. The argument `opts` can be a 
string specifying the scaling method or a `Dict(idx=>method)` if different methods 
are to be used for different variables. In this case, 

* `idx` can be an `Int`, `Vector{Int}` or `UnitRange` and specifies the variable indices
* `method` is a string that specifies the method. Available methods: `"mean"`, `"variance"`,
`"domain"`,`"c-mean"`, `"c-variance"` and `"2-sigma"`. Unrecognized options will be ignored. 

# Examples
```
julia> a=[1.0 0 0; 0 1.0 1.0];

julia> w = a |> scaler!("mean")
Data scaler! (mean), 2 -> 2, trained

julia> a |> w
2×3 Array{Float64,2}:
 0.666667  -0.333333  -0.333333
 -0.666667   0.333333   0.333333

julia> a=datacell([1.0 -1 0 0 0; 5 0 1.0 1.0 1.0; 1 2 3 4 5], [0, 0, 1, 1, 1]);

julia> +a
3×5 Array{Float64,2}:
 1.0  -1.0  0.0  0.0  0.0
 5.0   0.0  1.0  1.0  1.0
 1.0   2.0  3.0  4.0  5.0

julia> w=a |> scaler!(Dict(1=>"mean", 2=>"2-sigma"))
Data scaler! (mixed), 3 -> 3, trained

julia> a|>w; +a
3×5 Array{Float64,2}:
 1.0       -1.0       0.0       0.0       0.0     
 0.880132   0.321115  0.432918  0.432918  0.432918
 1.0        2.0       3.0       4.0       5.0     
```
"""
scaler!(f::Function, opts::T where T<:AbstractString) = FunctionCell(scaler!, (f,opts), ModelProperties(), "Data scaler! ("*opts*")")
scaler!(opts::T where T<:AbstractString) = scaler!(identity, opts) 

scaler!(f::Function, opts::T where T<:Dict) = FunctionCell(scaler!, (f, opts), ModelProperties(), "Data scaler! (mixed)")
scaler!(opts::T where T<:Dict) = scaler!(identity, opts)



############################
# DataCell/Array Interface #	
############################
"""
	scaler!(data, [f,] opts)

Scales `data` according to the scaling options specified in `opts`. 
If the dataset is labeled, an additional function `f` can be specified to 
obtain the labels calling `LearnBase.targets(data,f)`. """
# Training
scaler!(x::T where T<:Union{AbstractArray, CellData}, opts) = scaler!(x, identity, opts)

scaler!(x::T where T<:Union{AbstractArray, CellData}, f::Function, opts) = begin
	
	# Get dictionary or construct a proper one from original;
	# Inputs: scaling option and total number of variables 
	_sdict_(x::T where T<:String, m) = Dict(i=>x for i in 1:m) 
	_sdict_(x::T where T<:Dict, m) = begin
		out = Dict{Int, String}()
		for (k,v) in x 
			if(k isa Int)
				@assert (k>=1)&&(k<= m) "[scaler!] Index $(k) for a variable out of bounds."
				push!(out, k=>v) 
			elseif (k isa UnitRange{Int} || k isa Vector{Int})
				@assert (minimum(k)>=1)&&(maximum(k)<= m) "[scaler!] Index $(k) for a variable out of bounds."
				push!(out, (ki=>v for ki in k)...) 
			else
				error("[scaler!] Unsupported variable index syntax.")
			end
		end
		return out
	end
	dopts = _sdict_(opts, nvars(x))
	
	# Determine scaler name (for information only)
	_scalername_(x) = begin 
		if isempty(x) return "Data scaler! ()"
		elseif (length(unique(x)) == 1) return "Data scaler! ("*collect(x)[1]*")"
		else return "Data scaler! (mixed)"
		end
	end
	scalername = _scalername_(values(dopts))
	
	# Get targets (nothing for unlabeled data cells and arrays)
	labels = _targets_(f,x)
	@assert !any(isna.(labels))		
	@assert !any(isnan.(labels))		
	@assert (labels isa Vector)||(labels isa Void) "[scaler!] Labels need to be a Vector or ::Void." 

	# Get priors
	priors = countmapn(labels)	

	# Create model properties
	modelprops = ModelProperties(nvars(x), nvars(x), nothing, 
			Dict("priors" => priors,						# Priors
			"variables_to_process" => collect(keys(dopts))				# Which columns to process (can be any sort of vector)
			)	
	)
	
	# Define processing functions (x - data vector, t - targets array, p - class priors), returns Tuple(mean, variance, clip) 
	# - Scalers not based on class information have also an overloaded version for labeled datasets which reverts to one where 
	#   no labels are present
	# - Class based scalers expect targets to be a vector (not matrix) or default to non-class versions if possible
	_mean_(x::T where T<:AbstractVector, ::Void) = (1.0, -mean(x), false)
	_mean_(x::T where T<:AbstractVector, t::S where S<:AbstractArray) = _mean_(x,nothing)

	_variance_(x::T where T<:AbstractVector, ::Void) = (1/std(x), -mean(x)/std(x), false)
	_variance_(x::T where T<:AbstractVector, t::S where S<:AbstractArray) = _variance_(x,nothing)

	_domain_(x::T where T<:AbstractVector, ::Void) = begin 
		maxv = maximum(x)
		minv = minimum(x)
		return (1/(maxv-minv+eps()), -minv/(maxv-minv+eps()), false)
	end
	_domain_(x::T where T<:AbstractVector, t::S where S<:AbstractArray) = _domain_(x,nothing)

	_cmean_(x::T where T<:AbstractVector, ::Void, p) = _mean_(x,nothing)
	_cmean_(x::T where T<:AbstractVector, t::S where S<:AbstractVector, p) = begin 
		offset = 0.0
		for c in keys(p)
			@inbounds @fastmath d = x[find(isequal.(t,c))]
			@inbounds @fastmath offset += p[c] * mean(d)
		end
		return (1.0, -offset, false)
	end

	_cvariance_(x::T where T<:AbstractVector, ::Void, p) = _variance_(x,nothing)
	_cvariance_(x::T where T<:AbstractVector, t::S where S<:AbstractArray, p) = begin 
		offset = 0.0
		scale = 0.0
		for c in keys(p)
			@inbounds @fastmath d = x[find(isequal.(t,c))]
			@inbounds @fastmath offset += p[c] * mean(d)
			@inbounds @fastmath scale += p[c] * var(d)
		end
		return (1.0/scale, -offset/scale, false)
	end

	_2sigma_(x::T where T<:AbstractVector, ::Void, p) = error("[scaler!] 2-sigma scaling needs labels")
	_2sigma_(x::T where T<:AbstractVector, t::S where S<:AbstractVector, p) = begin 
		offset = 0.0
		scale = 0.0
		for c in keys(p)
			@inbounds @fastmath d = x[find(isequal.(t,c))]
			@inbounds @fastmath offset += p[c] * mean(d)
			@inbounds @fastmath scale += p[c] * var(d)
		end
		scale = 4*sqrt(scale)
		offset = offset - 0.5*scale
		return (1.0/scale, -offset/scale, true)
	end
	
	
	# Create dictionary for each processed variable Dict(column=>(scale, offset, labels))
	modeldata = Dict{Int, Tuple{eltype(1.0),eltype(1.0), Bool}}()
	for (idx, v) in dopts
		variable = _variable_(x,idx)
		if v=="mean" 
			s = _mean_(variable, labels) 
		elseif v=="variance" 
			s = _variance_(variable, labels)
		elseif v=="domain" 
			s = _domain_(variable, labels)
		elseif v=="c-mean" 
			s = _cmean_(variable, labels, priors) 
		elseif v=="c-variance" 
			s = _cvariance_(variable, labels, priors)
		elseif v=="2-sigma" 
			s = _2sigma_(variable, labels, priors)
		else # do not scale if option not recognized
			warn("[scaler!] Option $(v) not recognized, will not process variable $(idx)") 
			s = (1.0, 0.0, false)
		end
		push!(modeldata, idx => s ) 
	end
	
	# Return trained cell
	return FunctionCell(scaler!, Model(modeldata, modelprops), scalername)
end



# Execution
scaler!(x::T where T<:Union{AbstractArray, CellData}, model::Model) = begin
	
	for (idx, (scale, offset, clip)) in model.data
		variable = _variable_(x,idx) 	# Assign temp variable
		
		# Process variable vector
		@inbounds @fastmath variable[:] = variable .* scale + offset
		if (clip) 
			clamp!(variable,0.0,1.0)
		end
	end
	return x
end
