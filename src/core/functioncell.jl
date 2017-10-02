"""
	FunctionCell{T,S,U,V,W}(x::T, y::S, f::U, fargs::V<:Tuple, fkwargs::W<:Tuple, layer::Int, tinfo::String, oinfo::String)

Constructs the basic object for processing data, the `FunctionCell`. The constructor is parametric so new varieties of 
`FunctionCells` can be aliased to cover new functionality.

# Main `FunctionCell` constructors
  * `FunctionCell(f::Function, fargs::Tuple, tinfo=string(f); fkwargs...)` creates a `fixed` FunctionCell or transform.
  * `FunctionCell(f::Function, fargs::Tuple, mprops::ModelProperties, tinfo=string(f); fkwargs...)` creates an `untrained` FunctionCell.
  * `FunctionCell(f::Function, m::Model, tinfo=string(f); fkwargs...)` creates an `trained` FunctionCell 
  
where, `m` is the model, an object of type Model{T} (which is in itself a simple wrapper).

  In the above constructors, `f` is an overloaded function that, depending on circumstances, is applied on the input data piped
or called by the cell, `fargs` are `f`'s arguments, `tinfo` is the title and `fkwargs` are the keyword arguments of `f`. For 
fixed function cells, `f` is the the transform applied on the input; for untrained function cells, `f` is the training algorithm which
generates a model and some arbitrary properties of the model, such as priors, expected input/output dimensions etc. 
generated in training. The body of `f` for untrained cells should return a trained cell where `f` is the model execution function, 
`m` is the model and `mprops` the model properties.
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
	ModelProperties([idim ,odim, labels, other])

Basic type used to describe the properties of a model.

# Arguments
  * `idim::Int` is the required input data dimension (0 stands for any dimension)
  * `odim::Int` is the expected dimension of the output of the model (0 stands for any dimension)
  * `labels::LabelEncoding` the encoding of the labels
  * `other::Dict` are other properties of the data i.e. priors, custom execution arguments
"""
struct ModelProperties{T<:LabelEncoding, S<:Dict}
	idim::Int
	odim::Int	
	labels::T
	other::S
end

ModelProperties(idim::Int=0, odim::Int=0, enclabels::LabelEncoding=LabelEnc.NativeLabels([],Val{0})) =
	ModelProperties(idim, odim, enclabels, Dict())	

labelencn(labels::T where T<:AbstractVector{S}) where S = begin
	ulabels::Vector{S} = sort(unique(labels))
 	return LabelEnc.NativeLabels(ulabels)
end

ModelProperties(idim::Int, odim::Int, labels::Void, other::Dict=Dict()) =
	ModelProperties(idim, odim, LabelEnc.NativeLabels([],Val{0}), other)	

ModelProperties(idim::Int, odim::Int, labels::T where T<:AbstractVector, other::Dict=Dict()) =	# Known dimensions and labels 
	ModelProperties(idim, odim, labelencn(labels), other)

ModelProperties(idim::Int, odim::Int, labels::T where T<:AbstractMatrix, other::Dict=Dict()) =	# Known dimensions and labels 
	ModelProperties(idim, odim, labelenc(labels), other)

Base.show(io::IO, mprop::ModelProperties{T,S}) where {T,S} = begin 
	println("Model properties")
	println("`- I/O dimensions: $(mprop.idim)/$(mprop.odim)")
	println("`- label encoding: $(typeof(mprop.labels))")
	print("`- other: $(mprop.other)")
end



"""
	Model(data [,properties])		

Basic wrapper around model data and propertiesm, containing all the data needed for the execution of a model,
given an execution function of the form:
	`f(x, m)` 
where `x` is the input data and `m::Model` the model.

# Arguments
  * `data` is the input data on which the model is applied
  * `properties::ModelProperties` are the properties of the model, containing parameters necessary for its correct operation
"""
struct Model{T, S<:ModelProperties} 
	data::T 
	properties::S
end

Model() = Model(nothing, ModelProperties()) 							# Empty model

Model(data) = Model(data, ModelProperties())							# Model without properties

Model(properties::S where S<:ModelProperties) = Model(nothing, properties)			# Model without data

Model(model::T where T<:Model, properties::S where S<:ModelProperties) =			# Grabdata from a model and replace properties 
	Model(model.data, properties)

Base.show(io::IO, m::Model) = begin
	println("Model")
	println("`- data: $(typeof(m.data))")
	print("`- properties: Int/Int/$(typeof(m.properties.labels))/$(typeof(m.properties.other))")
end


# Constructor for fixed FunctionCells e.g. simple transforms
FunctionCell(f::U where U<:Function, fargs::V where V<: Tuple, tinfo::String = string(f); fkwargs...) = 
	FunctionCell(nothing, nothing, f, fargs, ntuple(i->fkwargs[i], Val{length(fkwargs)}), 0, tinfo, oinfoglobal)

# Constructor for untrained FunctionCells e.g. untrained classifiers or trainable transforms
FunctionCell(f::U where U<:Function, fargs::V where V<:Tuple, mprops::ModelProperties, tinfo::String = string(f); fkwargs...) = 
	FunctionCell(nothing, mprops, f, fargs, ntuple(i->fkwargs[i], Val{length(fkwargs)}), 0, tinfo, oinfoglobal)

# Constructor for trained FunctionCells (no arguments, keyword arguments, the model should fully describe the operation)
FunctionCell(f::U where U<:Function, m::T where T<:Model, tinfo::String = string(f)) = 
	FunctionCell(m, nothing, f, (), (),  0, tinfo, oinfoglobal)



########################
# FunctionCell aliases #
########################
const CellFunF{U<:Function} = 			 FunctionCell{Void, Void, U}			# Fixed function cell: simple transforms
const CellFunU{S<:ModelProperties, U<:Function}= FunctionCell{Void, S, U}			# Untrained function cell: untrained classifier or transform
const CellFunT{T<:Model, S<:Void, U<:Function} = FunctionCell{T, S, U}				# Trained function cell: trained classifier, transform
const CellFun{T,S, U<:Function} = 		 FunctionCell{T, S, U}



##############################################################################################################################
# Operators [Function cells]																								 #
##############################################################################################################################

# Pipe operators for function cells (a different execution method for each FunctionCell variety: fixed, untrained and trained)
|>(x::T where T, c::S where S<:CellFunF) = getf!(c)(x, c.fargs...; c.fkwargs...) 		
|>(x::T where T, c::S where S<:CellFunU) = getf!(c)(x, c.fargs...; c.fkwargs...)	
|>(x::T where T, c::S where S<:CellFunT) = begin
	# Check input dimension (if input dimension is 0, skip the check, input dimension can vary)
	getx!(c).properties.idim != 0 && getx!(c).properties.idim !=nvars(x) && 
		error("Expected $(getx!(c).properties.idim) input variables(s), got $(nvars(x)).")
	return getf!(c)(x, getx!(c))
end



"""
	functioncell([x,y,z])

Calls the `FunctionCell` constructor for one, two or three arguments. If no arguments are provided, 
a `FunctionCell` that expects as input a collection with contents compatible with the arguments of
the `FunctionCell` constructor is returned.
"""
functioncell() = FunctionCell((args)->FunctionCell(args...), (), "FunctionCell creator") 	
functioncell(x::T) where T = FunctionCell(x)					# calls to convert methods
functioncell(x::T, y::S) where {T,S}= FunctionCell(x, y)			# Calls any 2 argument parametric constructor 
functioncell(x::T, y::S, z::U) where {T,S,U}= FunctionCell(x, y, z)		# Calls any 3 argument parametric constructor
