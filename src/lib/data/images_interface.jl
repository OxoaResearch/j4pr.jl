"""
	im2targets(img)

Transforms image `im` into a data representation and returns a tuple `(X,Y)` where 
`X` and `Y` represent the coordinates and pixels values of the image respectively.
The image can be any of the supported `Images.jl` image representations. The rows
(e.g. variables) of `X` will correspond to the `Array` dimensions of `X`.	

Read the `Images.jl` documentation for more on images.

#Examples
```
julia> A = rand(2,2)
2×2 Array{Float64,2}:
 0.825531  0.400416
 0.798304  0.187369

julia> myimg = Images.Gray.(A);

julia> X,Y = im2targets(myimg) # in X, the first line corresponds to the first dimension (e.g. y axis)
([1.0 2.0 1.0 2.0; 1.0 1.0 2.0 2.0], ColorTypes.Gray{Float64}[Gray{Float64}(0.825531), Gray{Float64}(0.798304), Gray{Float64}(0.400416), Gray{Float64}(0.187369)])
```
"""
# Function to transform from/to an image to/from a datacell where the pixel values are the targets
# and the data is represented by the pixel coordinates
function im2targets(img::I where I<:AbstractArray{T,N}) where {T,N}
	# Pre-allocate
	X = zeros(N, length(img))
	Y = Vector{T}(length(img))
	
	for i in eachindex(img)
		idxs = ind2sub(img, i)
		for j in eachindex(idxs)
			X[j,i] = idxs[j]    
		end
		Y[i] = img[i]
	end
	return (X,Y)
end

"""
	targets2im(timg [,dim=Val{2}])

Converts a `DataCell` obtained with `im2targets` back to an image. The data variable `timg` can be either
a tuple of the form `(coordinates, pixels)` or a datacell with the coordinates as data and pixel values as
targets. `dim` is a `Val{T}` and specifies how many dimensions the output image will have. For color images, 
if the channel views are used, `dim` has to be explicitly changed to `Val{3}`.

Read the `Images.jl` documentation for more on images.

#Examples
```
julia> A=rand(2,2) # grayscale image
2×2 Array{Float64,2}:
 0.516513  0.649544
 0.390564  0.455701

julia> img = Images.Gray.(A); D = im2targets(A)
([1.0 2.0 1.0 2.0; 1.0 1.0 2.0 2.0], [0.516513, 0.390564, 0.649544, 0.455701])

julia> rimg = targets2im(D, Val{2}())
2×2 Array{Float64,2}:
 0.516513  0.649544
 0.390564  0.455701
 
julia> D2 = datacell(D)
DataCell, 4 obs, 2 vars, 1 target(s)/obs, 4 distinct values: "0.39056363119663073"(1),"0.4557007500598864"(1),"0.5165134793129167"(1),"0.6495441098194403"(1)

julia> rimg = j4pr.targets2im(D2, Val{2}())
2×2 Array{Float64,2}:
 0.516513  0.649544
 0.390564  0.455701

julia> A=rand(3,2,2)
3×2×2 Array{Float64,3}:
[:, :, 1] =
 0.412733  0.543216
 0.603818  0.521376
 0.738603  0.295367

[:, :, 2] =
 0.701765   0.735347
 0.0223506  0.751951
 0.93287    0.121406

julia> img = j4pr.targets2im(j4pr.im2targets(img)) # FAILS
ERROR: ...

julia> img = j4pr.targets2im(j4pr.im2targets(img),Val{3}())
3×2×2 Array{Float64,3}:
[:, :, 1] =
 0.158023  0.944338
 0.52807   0.40711 
 0.638118  0.853228

[:, :, 2] =
 0.0279565  0.820002 
 0.896604   0.966971 
 0.504744   0.0373515
```
"""
function targets2im(timg::I where I<:Tuple{AbstractArray{T,N}, AbstractVector{V}}, dim::Val{X}=Val{2}()) where {T,N,V,X}
	# Pre-allocate
	M = size(timg[1],1)
	dims = zeros(M)
	for i = 1:M
		dims[i] = maximum(timg[1][i,:]) # maximum(X[i,:])
	end
	img = Array{V,X}((Int.(dims)...))
	img[:] = timg[2][:]		
	return img::Array{V,X}
end

targets2im(timg::T where T<:DataCell, dim::Val{X}=Val{2}()) where {X} = targets2im((getx!(timg), gety!(timg)),dim)




### Note: A function to transform from/to an image to/from a datacell where the pixel values are the data  
# is not needed at this point. Since Images.jl uses Arrays as data container, transforming into a 
# DataCell is pretty trivial. The reverse transform (e.g. visualizing some array) does and should not require
# j4pr at all.
#
# For example, to transform an N-d array into a DataCell with 1 sample:
# julia> A=rand(2,2)
# 2×2 Array{Float64,2}:
#  0.472233  0.936257
#  0.947118  0.737812
#
# julia> img = Images.Gray.(A);  
# 
# julia> typeof(img) # still an array
# Array{ColorTypes.Gray{Float64},2}
#
# julia> j4pr.datacell(j4pr.mat(img[:])) # convert to column matrix
# DataCell, 1 obs, 4 vars, 0 target(s)/obs
#

