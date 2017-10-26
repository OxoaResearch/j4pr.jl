# Simplistic kernel calculation
function kernelize!(out::AbstractArray, x::AbstractArray, y::AbstractArray, kernel::Function; 
		    center::Bool=false, symmetric::Bool=false)
	m = size(x, 2)
	n = size(y, 2)
	
	if !symmetric   # generic case, x and y are different
		@inbounds for j in 1:n
        		yj = view(y,:,j)
        		@simd for i in 1:m
				xi = view(x,:,i)
				out[i,j] = kernel(xi, yj)
        		end
		end
	else
    		@inbounds for j = 1:n
        		yj = view(y,:,j)
        		@simd for i in 1:j
				xi = view(x,:,i)
            			out[i,j] = kernel(xi, yj)
				out[j,i] = out[i,j]
        		end
    		end
	end

	if center
		μi = mean(out, 2)	# kernel mean of individual rows
		μj = mean(out, 1)	# kernel mean of individual columns
		μ = mean(out)		# overall mean
		@inbounds @simd for i in 1:m
			for j in 1:n
				out[i,j] = out[i,j] - μi[i] - μj[j] + μ
			end
		end
	end
	return out
end

kernelize!(out::AbstractArray, x::AbstractArray, kernel::Function; center::Bool=false, symmetric::Bool=true) =
	kernelize!(out, x, x, kernel; center=center, symmetric=symmetric)

function kernelize(x::AbstractArray, y::AbstractArray, kernel::Function; center::Bool=false, symmetric::Bool=false)
	m = size(x, 2)
	n = size(y, 2)
	out = similar(x, m, n)
	kernelize!(out, x, y, kernel; center=center, symmetric=symmetric)
end

kernelize(x::AbstractArray, kernel::Function; center::Bool=false, symmetric::Bool=true) = 
	kernelize(x, x, kernel; center=center, symmetric=symmetric)



##########################
# FunctionCell Interface #	
##########################
"""
	kernel([f];[center=false, symmetric=false])

Constructs an untrained function cell using the kernel function `f` that accepts two vector arguments `x` and `y`, 
and returns a scalar value (default `(x,y)->x'y`). When piped data into, the untrained function cell returns a
trained function cell, retaining the training data.

# Keyword arguments
  * `symmetric::Bool` specifies whether the kernel should be symmetric i.e. calculated using the same observations (default `false`)
  * `center::Bool` specifies whether the center the kernel (default `false`)

A list of common used kernels can be found below (definitions from the documentation of to MLKernels.jl):
# Kernels
  * `(x,y)-> a⋅xᵀy + c		Linear, a>0, c≥0`
  * `(x,y)-> (a⋅xᵀy+c)ᵈ		Polynomial, a>0, c≥0, d∈Ζ₊`
  * `(x,y)-> tanh(a⋅xᵀy+c)		Sigmoid, a>0, c≥0`
  * `(x,y)-> exp(a⋅xᵀy)		Exponentiated, a>0`
  * `(x,y)-> exp(-α*‖x-y‖)		Exponential, α>0`
  * `(x,y)-> exp(-α⋅‖x-y‖²)		Squared exponential, α>0`
  * `(x,y)-> exp(-α⋅‖x-y‖ᵞ)		Gamma exponential, α>0, 0<γ<1`
  * `(x,y)-> (1 + α⋅‖x-y‖²)⁻ᵝ	Rational quadratic, α>0, β>0`
  * `(x,y)-> (1 + α⋅‖x-y‖ᵞ)⁻ᵝ	Gamma-rational, α>0, β>0, 0<γ,1`
  * `(x,y)-> exp{-α⋅Σᵢsin²[p⋅(xᵢ-yᵢ)]}	Periodic kernel, p>0, α>0`
  * `(x,y)-> ‖x-y‖²ᵞ		Power, 0<γ≤1 (negative definite)`
  * `(x,y)-> log(1+α⋅‖x-y‖²ᵞ)	Log, α>0, 0<γ≤1 (negative definite)` 

# Examples
```
julia> k(a,c) = (x,y)->a*x'y+c # linear kernel                                                                                                                                                 
k (generic function with 1 method)                                                                                                                                                             

julia> K = kernel(k(1.0,2.0))                                                                                                                                                                  
Kernel: symmetric=false, varying I/O dimensions, untrained                                                                                                                                     

julia> x=[1,2,3]; y=[4,5,6,7];                                                                                                                                                                    
                                                                                                                                                                                               
julia> y |> K(x)                                                                                                                                                                               
3×4 Array{Int64,2}:        
  6   7   8   9                                                                                                                                                                                
 10  12  14  16                                                                                                                                                                                
 14  17  20  23   
```
"""
kernel(f::Function=(x,y)->x'y; center::Bool=false, symmetric::Bool=false) = 
	FunctionCell(kernel, (f,), ModelProperties(), "Kernel: center=$center, symmetric=$symmetric"; 
	      center=center, symmetric=symmetric) 



############################
# DataCell/Array Interface #	
############################
"""
	kernel(x [,f] [;symetric=false])

Trains the function cell using the kernel function `f` and data `x` which will be used
for future kernel calculations.
"""
# Training
kernel(x::T where T<:CellData, f::Function=(x,y)->x'y; center::Bool=false, symmetric::Bool=false) = 
	kernel(getx!(x), f; center=center, symmetric=symmetric)
kernel(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractArray, f::Function=(x,y)->x'y; 
       center::Bool=false, symmetric::Bool=false) = kernel(x[1], f; center=center, symmetric=symmetric)
kernel(x::T where T<:AbstractVector, f::Function=(x,y)->x'y; center::Bool=false, symmetric::Bool=false) = 
	kernel(mat(x, LearnBase.ObsDim.Constant{2}()), f; center=center, symmetric=symmetric)
kernel(x::T where T<:AbstractMatrix, f::Function=(x,y)->x'y; center::Bool=false, symmetric::Bool=false) = begin
	
	# Build model properties
	modelprops = ModelProperties(nvars(x),nobs(x))
	
	# Returned trained cell
	FunctionCell(kernel, Model((f, getobs(x), center, symmetric), modelprops), 
	      "Kernel: center=$center, symmetric=$symmetric")	 
end



# Execution
kernel(x::T where T<:CellData, model::Model{<:Tuple{<:Function,Matrix,Bool,Bool}}) =
	datacell(kernel(getx!(x), model), gety(x)) 	
kernel(x::T where T<:AbstractVector, model::Model{<:Tuple{<:Function,Matrix,Bool,Bool}}) =
	kernel(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
kernel(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:Function,Matrix,Bool,Bool}}) =
	kernelize(model.data[2], getobs(x), model.data[1] ;center=model.data[3], symmetric=model.data[4])
