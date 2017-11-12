# Function that calculates the number of  relational variables / each adjacency structure
get_size_out(y::AbstractVector{T}) where T<:Float64 = 1			# regression case
get_size_out(y::AbstractVector{T}) where T = length(unique(y))::Int	# classification case
get_size_out(y::AbstractArray) = error("Only vectors supported as targets in relational learning.")



# Function that calculates the priors of the dataset
getpriors(y::AbstractVector{T}) where T<:Float64 = [1.0]	
getpriors(y::AbstractVector{T}) where T = [sum(yi.==y)/length(y) for yi in sort(unique(y))]
getpriors(y::AbstractArray) = error("Only vectors supported as targets in relational learning.")

encode_targets(labels::T where T<:AbstractVector{S}) where S = begin
	ulabels::Vector{S} = sort(unique(labels))
	enc = LabelEnc.NativeLabels(ulabels)
	return (enc, label2ind.(labels,enc))
end

encode_targets(labels::T where T<:AbstractVector{S}) where S<:AbstractFloat = begin
	return (nothing, labels)
end

encode_targets(labels::T where T<:AbstractArray{S}) where S = begin
	error("Targets must be in vector form, other arrays not supported.")
end
