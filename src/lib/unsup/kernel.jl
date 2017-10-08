# Simplistic kernel calculation
function kernelize!(out::AbstractArray, x::AbstractArray, y::AbstractArray, kernel::Function; symmetric::Bool=false)
	m = size(x, 2)
	n = size(y, 2)
	
	if !symmetric   # generic case, x and y are different
		@simd for j in 1:n
        		yj = view(y,:,j)
        		for i in 1:m
				@inbounds out[i,j] = kernel(view(x,:,i), yj)
        		end
		end
	else
    		@simd for j = 1:n
        		yj = view(y,:,j)
        		for i in 1:j
            			@inbounds out[i,j] = kernel(view(x,:,i), yj)
				@inbounds out[j,i] = out[i,j]
        		end
    		end
	end
	return out
end

kernelize!(out::AbstractArray, x::AbstractArray, kernel::Function; symmetric::Bool=true) =
	kernelize!(out, x, x, kernel; symmetric=symmetric)

function kernelize(x::AbstractArray, y::AbstractArray, kernel::Function; symmetric::Bool=false)
	m = size(x, 2)
	n = size(y, 2)
	out = similar(x, m, n)
	kernelize!(out, x, y, kernel; symmetric=symmetric)
end

kernelize(x::AbstractArray, kernel::Function; symmetric::Bool=true) = 
	kernelize(x, x, kernel; symmetric=symmetric)



##########################
# FunctionCell Interface #	
##########################
"""
	kernel([f];[symmetric=false])

Constructs an untrained function cell using the kernel function `f` that accepts two vector arguments `x` and `y`, 
and returns a scalar value (default `(x,y)->x'y`). When piped data into, the untrained function cell returns a
trained function cell, retaining the training data.

# Keyword arguments
  * `symmetric::Bool` specifies whether the kernel should be symmetric i.e. calculated using the same observations (default `false`)

# Kernels
List of the commonly used kernels:
    function 				description
    `(x,y)->x'y` 			 `Linear`
    `(x,y)->(x'y+c)^d` 			 `Polynomial`
    `(x,y)->exp(-γ*norm(x-y)^2.0)` 	 `Radial basis function (RBF)`

"""
kernel(f::Function=(x,y)->x'y; symmetric::Bool=false) = 
	FunctionCell(kernel, (f,), ModelProperties(), "Kernel: symmetric=$symmetric"; symmetric=symmetric) 



############################
# DataCell/Array Interface #	
############################
"""
	kernel(x [,f] [;symetric=false])

Trains the function cell using the kernel function `f` and data `x` which will be used
for future kernel calculations.
"""
# Training
kernel(x::T where T<:CellData, f::Function=(x,y)->x'y; symmetric::Bool=false) = 
	kernel(getx!(x), f; symmetric=symmetric)
kernel(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractArray, f::Function=(x,y)->x'y; symmetric::Bool=false) = 
	kernel(x[1], f; symmetric=symmetric)
kernel(x::T where T<:AbstractVector, f::Function=(x,y)->x'y; symmetric::Bool=false) = 
	kernel(mat(x, LearnBase.ObsDim.Constant{2}()), f; symmetric=symmetric)
kernel(x::T where T<:AbstractMatrix, f::Function=(x,y)->x'y; symmetric::Bool=false) = begin
	
	# Build model properties
	modelprops = ModelProperties(nvars(x),nobs(x))
	
	# Returned trained cell
	FunctionCell(kernel, Model((f, getobs(x), symmetric), modelprops), "Kernel: symmetric=$symmetric")	 
end



# Execution
kernel(x::T where T<:CellData, model::Model{<:Tuple{<:Function,Matrix,Bool}}) =
	datacell(kernel(getx!(x), model), gety(x)) 	
kernel(x::T where T<:AbstractVector, model::Model{<:Tuple{<:Function,Matrix,Bool}}) =
	kernel(mat(x, LearnBase.ObsDim.Constant{2}()), model) 	
kernel(x::T where T<:AbstractMatrix, model::Model{<:Tuple{<:Function,Matrix,Bool}}) =
	kernelize(model.data[2], getobs(x), model.data[1] ;symmetric=model.data[3])
