###########################
# Indexing for data cells #
###########################

# Generic setindex! for data cells
setindex!(c::T where T<:CellData, data, inds...) = setindex!(getx!(c), data, inds...)


# Generic indexing operator, no argument
getindex(c::CellData) = j4pr.datacell(eltype(getx!(c))[])
getindex(c::CellData, ::Colon) = c 
getindex(c::CellData, ::Colon, ::Colon) = c

# Single integer indexing (select sample)
getindex(c::Union{CellDataVec, CellDataVecVec}, i::Int64) = DataCell([getx!(c)[i]], [gety!(c)[i]] )				
getindex(c::CellDataVecMat, i::Int64) = DataCell([getx!(c)[i]], gety!(c)[:,i:i] )				
getindex(c::Union{CellDataMat, CellDataMatVec}, i::Int64) = DataCell(getx!(c)[:,i:i], [gety!(c)[i]] )				
getindex(c::CellDataMatMat, i::Int64) = DataCell(getx!(c)[:,i:i], gety!(c)[:,i:i] )				

# Integer + Colon indexing (select variable) - only for matrix data
getindex(c::CellDataMatrices , i::Int64, ::Colon) = DataCell(getx!(c)[i:i,:], gety!(c) )				

# Colon + Integer indexing (select sample) - only for matrix data
getindex(c::Union{CellDataMat, CellDataMatVec}, ::Colon, i::Int64) = DataCell(getx!(c)[:,i:i], [gety!(c)[i]])  					
getindex(c::CellDataMatMat, ::Colon, i::Int64) = DataCell(getx!(c)[:,i:i], gety!(c)[:,i:i])  					

# Range/Vector indexing (select samples) - only for vector data
getindex(c::Union{CellDataVec, CellDataVecVec}, i::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[i], gety!(c)[i])
getindex(c::CellDataVecMat, i::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[i], gety!(c)[:,i])

# Range/Vector + Colon indexing (select variables) - only for matrix data 
getindex(c::Union{CellDataMat, CellDataMatVec}, i::Union{UnitRange{Int64}, Vector{Int64}}, ::Colon) = DataCell(getx!(c)[i,:], gety!(c)[:]) 
getindex(c::CellDataMatMat, i::Union{UnitRange{Int64}, Vector{Int64}}, ::Colon) = DataCell(getx!(c)[i,:], gety!(c)[:,:]) 

# Colon + Range/Vector indexing (select samples) - only for matrix data
getindex(c::Union{CellDataMat, CellDataMatVec}, ::Colon, i::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[:,i], gety!(c)[i])
getindex(c::CellDataMatMat, ::Colon, i::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[:,i], gety!(c)[:,i])

# Range/Vector + Range/Vector indexing (select samples and variables) - only for matrix data
getindex(c::Union{CellDataMat, CellDataMatVec}, i::Union{UnitRange{Int64}, Vector{Int64}}, j::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[i,j], gety!(c)[j])    
getindex(c::CellDataMatMat, i::Union{UnitRange{Int64}, Vector{Int64}}, j::Union{UnitRange{Int64}, Vector{Int64}}) = DataCell(getx!(c)[i,j], gety!(c)[:,j]) 

# Getindex for void arguments (needed to get indexes for CellDataU objects)
getindex(c::T where T<:Void, args...) = nothing



######################################
# Iteration interface for data cells #
######################################
start(c::CellData) = 1
next(c::CellData, state) = (datasubset(c, state), state+1)
done(c::CellData, state) = state > nobs(c)
endof(c::CellData) = nobs(c)
eltype(c::CellData) = typeof(c)
length(c::CellData) = nobs(c)


######################
# Indexing for Pipes #
######################
# Generic setindex! for data cells
setindex!(c::T where T<:PipeGeneric, cells, inds...) = setindex!(getx!(c), cells, inds...)

getindex(ac::T where T<:PipeGeneric, i::Int64) = getx!(ac)[i]
getindex(ac::T where T<:PipeStacked, i::Union{Vector{Int},UnitRange{Int}}) = PipeCell(getx!(ac)[i])
getindex(ac::T where T<:PipeParallel, i::Union{Vector{Int}, UnitRange{Int}}) = PipeCell(getx!(ac)[i], SortedDict(Dict(j=>ac.y[j] for j in i)))
getindex(ac::T where T<:PipeSerial, i::Union{Vector{Int}, UnitRange{Int}}) = PipeCell(getx!(ac)[i], collect(1:length(i)) )



#################################
# Iteration interface for pipes #
#################################
start(p::PipeGeneric) = 1
next(p::PipeGeneric, state) = (getx!(p)[state], state+1)
done(p::PipeGeneric, state) = state > length(getx!(p))
endof(p::PipeGeneric) = length(getx!(p))
eltype(p::PipeGeneric) = eltype(getx!(p))
length(p::PipeGeneric) = length(getx!(p))


# Positional argument selectors
idx(args...) = FunctionCell(getindex, args, "Index selector (args=$args)")
idx(f::Function, args...) = FunctionCell((x,args...)->f(getindex(x,args...)), args, "Index selector+function (args=$args)")
