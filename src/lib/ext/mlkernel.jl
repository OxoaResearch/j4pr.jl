##########################
# FunctionCell Interface #	
##########################
"""
	mlkernel([σ,] κ; center=true)

Constructs an untrained function cell using the memory layout `σ` and kernel `κ`.  

# Arguments
  * `σ::MLKernels.PairwiseFunctions.MemoryLayout` specifies whether to work with row or column major data
  (default `MLKernels.ColumnMajor() i.e. columns are observations)
  * `κ::MLKernels.Kernel` is the kernel object 

# Keyword arguments
  * `center::Bool=true` specifies whether to center the kernel or not

A list of common used kernels in `MLKernels.jl` can be found below (definitions from the documentation of to MLKernels.jl):

# Kernels
  * `(x,y)-> a⋅xᵀy + c		MLKernels.LinearKernel(a>0, c≥0)`
  * `(x,y)-> (a⋅xᵀy+c)ᵈ		MLKernels.PolynomialKernel(a>0, c≥0, d∈Ζ₊)`
  * `(x,y)-> tanh(a⋅xᵀy+c)		MLKernels.SigmoidKernel(a>0, c≥0)`
  * `(x,y)-> exp(a⋅xᵀy)		MLKernels.ExponentiatedKernel(a>0)`
  * `(x,y)-> exp(-α*‖x-y‖)		MLKernels.ExponentialKernel(α>0)`
  * `(x,y)-> exp(-α⋅‖x-y‖²)		MLKernels.SquaredExponentialKernel(α>0)`
  * `(x,y)-> exp(-α⋅‖x-y‖ᵞ)		MLKernels.GammaExponentialKernel(α>0, 0<γ<1)`
  * `(x,y)-> (1 + α⋅‖x-y‖²)⁻ᵝ	MLKernels.RationalQuadraticKernel(α>0, β>0)`
  * `(x,y)-> (1 + α⋅‖x-y‖ᵞ)⁻ᵝ	MLKernels.GammaRationalKernel(α>0, β>0, 0<γ,1)`
  * `(x,y)-> exp{-α⋅Σᵢsin²[p⋅(xᵢ-yᵢ)]}	MLKernels.PeriodicKernel(p>0, α>0)`
  * `(x,y)-> ‖x-y‖²ᵞ		MLKernels.PowerKernel(0<γ≤1) (negative definite)`
  * `(x,y)-> log(1+α⋅‖x-y‖²ᵞ)	MLKernels.LogKernel(α>0, 0<γ≤1) (negative definite)` 

Read the `MLKernels.jl` documentation for more information.  

# Examples
```
julia> K = mlkernel(MLKernels.LinearKernel(1,2), center=false)
ML Kernel σ=MLKernels.PairwiseFunctions.ColumnMajor(), κ=LinearKernel(1.0,2.0), center=false, varying I/O dimensions, untrained

julia> x = [1,2,3]; y=[4,5,6,7];

julia> y |> K(x)
3×4 Array{Float64,2}:
  6.0   7.0   8.0   9.0
 10.0  12.0  14.0  16.0
 14.0  17.0  20.0  23.0
``
"""
mlkernel(κ::MLKernels.Kernel; center::Bool=true) = 
	mlkernel(MLKernels.PairwiseFunctions.ColumnMajor(), κ; center=center)	

mlkernel(σ::MLKernels.PairwiseFunctions.MemoryLayout, κ::MLKernels.Kernel; center::Bool=true) = 
	FunctionCell(mlkernel, (σ,κ), ModelProperties(), "ML Kernel σ=$(σ), κ=$(κ), center=$center"; center=center) 



############################
# DataCell/Array Interface #	
############################
"""
	mlkernel(x, [σ,] κ; center=true)

Trains the function cell using the memory layout `σ`, kernel `κ` and data `x` for future kernel calculations.
"""
# Training
mlkernel(x::T where T<:CellData, κ::MLKernels.Kernel; center::Bool=true) = 
	mlkernel(x, MLKernels.PairwiseFunctions.ColumnMajor(), κ; center=center)

mlkernel(x::T where T<:CellData, σ::MLKernels.PairwiseFunctions.MemoryLayout, κ::MLKernels.Kernel; center::Bool=true) = 
	mlkernel(getx!(x), MLKernels.PairwiseFunctions.ColumnMajor(), κ; center=center)

mlkernel(x::T where T<:AbstractArray, κ::MLKernels.Kernel; center::Bool=true) = 
	mlkernel(x, MLKernels.PairwiseFunctions.ColumnMajor(), κ; center=center)

mlkernel(x::T where T<:AbstractVector, σ::MLKernels.PairwiseFunctions.MemoryLayout, κ::MLKernels.Kernel; center::Bool=true) = 
	mlkernel(mat(x, LearnBase.ObsDim.Constant{2}()), σ, κ; center=center)

mlkernel(x::T where T<:AbstractMatrix, σ::MLKernels.PairwiseFunctions.MemoryLayout, κ::MLKernels.Kernel; center::Bool=true) = 
begin	
	# Build model properties
	modelprops = ModelProperties(nvars(x),nobs(x))
	
	# Returned trained cell
	FunctionCell(mlkernel, Model((getobs(x), σ, κ, center), modelprops), "ML Kernel σ=$(σ), κ=$(κ), center=$center")	 
end



# Execution
mlkernel(x::T where T<:CellData, model::Model{<:Tuple{<:AbstractMatrix,<:MLKernels.PairwiseFunctions.MemoryLayout, <:MLKernels.Kernel, Bool}}) =
	datacell(mlkernel(getx!(x), model), gety(x)) 	
mlkernel(x::T where T<:AbstractVector, model::Model{<:Tuple{<:AbstractMatrix,<:MLKernels.PairwiseFunctions.MemoryLayout, <:MLKernels.Kernel, Bool}}) =
	mlkernel(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
mlkernel(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:AbstractMatrix, <:MLKernels.PairwiseFunctions.MemoryLayout, <:MLKernels.Kernel, Bool}}) =
begin
	out = MLKernels.kernelmatrix(model.data[2], model.data[3], model.data[1], getobs(x))
	if model.data[4]
		MLKernels.centerkernelmatrix!(out)
	end
	return out
end
