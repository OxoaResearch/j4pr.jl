##############################################################################################################################
# Type Definition [Cell] e.g. Basic construct used to create classifiers, data transforms and other elementary PR concepts
##############################################################################################################################
"""
	abstract type AbstractCell{T,S,U}

The basic type of J4PR. Most of the types are its subtypes to ensure that operations
between various types of `Cells` are possible, especially when creating `Pipes` that
can have arbitrary levels of nesting.
"""
abstract type AbstractCell{T,S,U} end



"""
	DataCell{T,S,U}(x::T, y::S, f::U, layer::Int, tinfo::String, oinfo::String)

Constructs the basic object for holding data, the `DataCell`. The constructor is parametric so it can be
easily extended to new data types.

# Main `DataCell` constructors
  * `DataCell(x::AbstractArray, tinfo::String="")` creates an `unlabeled` object.
  * `DataCell(x::AbstractArray, y::AbstractVector, tinfo::String="")` creates a `labeled` object. Useful for classification.
  * `DataCell(x::AbstractArray, y::AbstractMatrix, tinfo::String="")` creates a `multi-labeled` object. Useful for classification/regression

For the creation of data cells, the `datacell` function is quite well suited.
"""
#############
# DataCells # 
#############
struct DataCell{T,S,U} <: AbstractCell{T,S,U}
	x::T                                                                                    # Data
    	y::S                                                                                    # Labels/probabilities etc.
    	f::U                                                                                    # Usually a function
	layer::Int										# Layer
    	tinfo::String										# Title or name
    	oinfo::String                                                                        	# Other information
end

DataCell(x::T where T<:AbstractArray, tinfo::String = "") = DataCell(x, nothing, nothing, 0, tinfo, oinfoglobal)
DataCell(x::T where T<:AbstractArray, y::S where S<:Void, tinfo::String = "") = DataCell(x, nothing, nothing, 0, tinfo, oinfoglobal)
DataCell(x::T where T<:AbstractArray, y::S where S<:AbstractArray{Void}, tinfo::String = "") = DataCell(x, nothing, nothing, 0, tinfo, oinfoglobal)
DataCell(x::T where T<:AbstractArray, y::S where S<:AbstractArray, tinfo::String = "") = DataCell(x, checktargets(x,y), nothing, 0, tinfo, oinfoglobal)



"""
	FunctionCell{T,S,U,V,W}(x::T, y::S, f::U, fargs::V<:Tuple, fkwargs::W<:Tuple, layer::Int, tinfo::String, oinfo::String)

Constructs the basic object for processing data, the `FunctionCell`. The constructor is parametric so new varieties of 
`FunctionCells` can be aliased to cover new functionality.

# Main `FunctionCell` constructors
  * `FunctionCell(f::Function, fargs::Tuple, tinfo=string(f); fkwargs...)` creates a `fixed` FunctionCell or transform.
  * `FunctionCell(f::Function, fargs::Tuple, mprops::Dict, tinfo=string(f); fkwargs...)` creates an `untrained` FunctionCell.
  * `FunctionCell(f::Function, m::Model, mprops::Dict, tinfo=string(f); fkwargs...)` creates an `trained` FunctionCell 
  
where, `m` is the model, an object of type Model{T} (which is in itself a simple wrapper).

  In the above constructors, `f` is an overloaded function that, depending on circumstances, is applied on the input data piped
or called by the cell, `fargs` are `f`'s arguments, `tinfo` is the title and `fkwargs` are the keyword arguments of `f`. For 
fixed function cells, `f` is the the transform applied on the input; for untrained function cells, `f` is the training algorithm which
generates a model and some arbitrary properties of the model, such as priors, expected input/output dimensions etc. 
generated in training. The body of `f` for untrained cells should return a trained cell where `f` is the model execution function, 
`m` is the model and `mprops` the model properties.

The easyest way to construct cells, especially is using annonymous functions is to call the `functioncell` function which is a 
wrapper around the constructor. As a general rule, data piped to a `function cell` is placed as the first argument
of the function that has been wrapped around. 
"""
#################
# FunctionCells #
#################
struct FunctionCell{T,S,U, V<:Tuple, W<:Tuple} <: AbstractCell{T,S,U}
	x::T                                                                                    # Model
    	y::S                                                                                    # Model properties
    	f::U                                                                                    # Function for transforming/training/predicting
	fargs::V										# Function arguments
	fkwargs::W										# Function keyword arguments
	layer::Int										# Layer
    	tinfo::String										# Title or name
    	oinfo::String                                                                        	# Other information
end

"""
	Model{T}(data::T)		

Basic wrapper around data to designate a model.
"""
struct Model{T} data::T end

# Constructor for fixed FunctionCells e.g. simple transforms
FunctionCell(f::U where U<:Function, fargs::V where V<: Tuple, tinfo::String = string(f); fkwargs...) = 
	FunctionCell(nothing, nothing, f, fargs, ntuple(i->fkwargs[i], Val{length(fkwargs)}), 0, tinfo, oinfoglobal)

# Constructor for untrained FunctionCells e.g. untrained classifiers or trainable transforms
FunctionCell(f::U where U<:Function, fargs::V where V<:Tuple, mprops::Dict, tinfo::String = string(f); fkwargs...) = 
	FunctionCell(nothing, mprops, f, fargs, ntuple(i->fkwargs[i], Val{length(fkwargs)}), 0, tinfo, oinfoglobal)

# Constructor for trained FunctionCells
FunctionCell(f::U where U<:Function, m::T where T<:Model, mprops::Dict, tinfo::String = string(f); fkwargs...) = 
	FunctionCell(m, mprops, f, (), ntuple(i->fkwargs[i], Val{length(fkwargs)}), 0, tinfo, oinfoglobal)



"""
	PipeCell{T,S,U}(x::T, y::S, f::U, layer::Int, tinfo::String, oinfo::String)

Constructs the basic object for complex processing data, the `PipeCell`. The constructor is parametric so it can be
easily extended to new types of processing.

# Main `PipeCell` constructors
  * `PipeCell(x::T where T<:PTuple{Abstractcell}, tinfo::String = "")` creates a `stacked pipe`
  * `PipeCell(x::T where T<:PTuple{AbstractCell}, dispatch::SortedDict, tinfo::String = "")` creates a `parallel pipe`
  * `PipeCell(x::T where T<:PTuple{AbstractCell}, order::Vector{Int}, tinfo::String="")` creates a `serial pipe`

where `const PTuple{T} = Tuple{Vararg{<:T}}`.

The concepts are as follows: when piping data to a stacked pipe, the data is individually piped to each AbstractCell 
of the pipe. For parallel pipes, dispatch allows to select which element of the collection (e.g. `Vector`, `Tuple`, data cell etc.)
gets piped where. Serial pipes process data in a serial manner. Pipes can be created also by using the functions 
`pipestack`, `pipeparallel` and `pipeserial`. 
"""
#############
# PipeCells #
#############
struct PipeCell{T,S,U} <: AbstractCell{T,S,U}
	x::T                                                                                    # List of Data/Function/PipeCells
    	y::S                                                                                    # Processing information
    	f::U                                                                                    # <unused so far>
	layer::Int										# Layer
    	tinfo::String										# Title or name
    	oinfo::String                                                                        	# Other information
end

const PTuple{T} = Tuple{Vararg{<:T}}

# General constructors for Stacked pipes (e.g. input routed to all cells)
PipeCell(x::T where T<:PTuple{AbstractCell}, tinfo::String = "") = PipeCell(x, nothing, nothing, countlayers(x)+1, tinfo, oinfoglobal )

# General constructor for Parallel pipes (e.g. input routed according to some dispatch information to cells)
PipeCell(x::T where T<:PTuple{AbstractCell}, dispatch::SortedDict, tinfo::String = "") = begin
	@assert length(x) == length(keys(dispatch)) "Parallel pipe element number does not match dispatch information."	
	PipeCell(x, dispatch, nothing, countlayers(x)+1, tinfo, oinfoglobal )
end

# General constructor for Serial pipes (e.g. input passes from element to element of the pipe according to some order)
PipeCell(x::T where T<:PTuple{AbstractCell}, order::Vector{Int}, tinfo::String="") = begin
	@assert length(x) >= length(order) "Serial pipe must contain at least as many elements as order vector."	
	PipeCell(x, order, nothing, countlayers(x)+1, tinfo, oinfoglobal )
end
