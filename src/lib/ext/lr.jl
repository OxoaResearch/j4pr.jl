##########################
# FunctionCell Interface #	
##########################
"""
	lr(r=0; trans=true, bias=true)

Constructs an untrained cell that when piped data inside (observations and
responses) calculates the regularized (ridge) Least-Squares regression 
coefficients of the data based on the responses and regularization parameter
`r`.

The parameter `r` can be a real scalar, a vector or symetric matrix.

# Keyword arguments (same as in `MultivariateStats`)
  * `trans` whether to use the trasposed form, (default `true` e.g. rows arevariables, columns are observations) 
  * `bias` whether to include a bias term (default `true`)

Read the `MultivariateStats.jl` documentation for more information.  
"""
lr(r=0; trans::Bool=true, bias::Bool=true) = FunctionCell(lr, (r,), Dict(), kwtitle("Linear regressor", [(:trans,trans),(:bias,bias)]); trans=trans, bias=bias) 



############################
# DataCell/Array Interface #	
############################
"""
	lr(x,y, [;kwargs])

Trains a Least-Squares regression function cell. Returns the regression coefficients and bias (if available) 
"""
# Training
lr(x::T where T<:CellDataU, r=0; trans::Bool=true, bias::Bool=true) = lr((getx!(x), zeros(nobs(x))), r; trans=trans, bias=bias)
lr(x::T where T<:CellData, r=0; trans::Bool=true, bias::Bool=true) = lr((getx!(x), gety(x)), r; trans=trans, bias=bias)
lr(x::Tuple{T,S} where T<:AbstractVector where S<:AbstractArray, r=0; trans::Bool=true, bias::Bool=true) = lr((mat(x[1], LearnBase.ObsDim.Constant{2}()), x[2]), r; trans=trans, bias=bias)
lr(x::Tuple{T,S} where T<:AbstractMatrix where S<:AbstractArray, r=0; trans::Bool=true, bias::Bool=true ) = begin
	
	@assert nobs(x[1]) == nobs(x[2])

	# If y vector, leave as is, if matrix, traspose
	if x[2] isa Vector 
		tmp = MultivariateStats.ridge(getobs(x[1]), x[2], r, trans=trans, bias=bias)
	else
		tmp = MultivariateStats.ridge(getobs(x[1]), x[2]', r, trans=trans, bias=bias)
	end
	
	if bias
		a = tmp[1:end-1,:]
		b = tmp[end,:]
	else
		a = tmp
		b = zeros(nvars(x[1]))
	end

	# Build model properties 
	modelprops = Dict("size_in" => size(a,1),		
		   	  "size_out" => size(a,2)
	)
	
	# Returned trained cell
	FunctionCell(lr, Model((a,b)), modelprops, kwtitle("Linear regressor", [(:trans,trans),(:bias,bias)]))	 
end



# Execution
lr(x::T where T<:CellData, model::Model{<:Tuple{<:AbstractArray, <:AbstractArray}}, modelprops::Dict) = datacell(lr(getx!(x), model, modelprops), gety(x)) 	
lr(x::T where T<:AbstractVector, model::Model{<:Tuple{<:AbstractArray, <:AbstractArray}}, modelprops::Dict) = lr(mat(x, LearnBase.ObsDim.Constant{2}()), model, modelprops) 	
lr(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:AbstractArray, <:AbstractArray}}, modelprops::Dict) = begin
	@assert modelprops["size_in"] == nvars(x) "$(modelprops["size_in"]) input variable(s) expected, got $(nvars(x))."	
	return model.data[1]'*x .+ model.data[2] # e.g. aᵀX+b
end
