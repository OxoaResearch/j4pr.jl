# Function that calculates the number of  relational variables / each adjacency structure
get_width_rv(y::AbstractVector{T}) where T<:Float64 = 1			# regression case
get_width_rv(y::AbstractVector{T}) where T = length(unique(y))::Int	# classification case
get_width_rv(y::AbstractArray) = error("Only vectors supported as targets in relational learning.")



# Function that calculates the priors of the dataset
getpriors(y::AbstractVector{T}) where T<:Float64 = [1.0]	
getpriors(y::AbstractVector{T}) where T = [sum(yi.==y)/length(y) for yi in sort(unique(y))]
getpriors(y::AbstractArray) = error("Only vectors supported as targets in relational learning.")
