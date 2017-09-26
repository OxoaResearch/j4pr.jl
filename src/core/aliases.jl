###################
# Generic aliases #
###################
const DataElement{T<:Union{Number, AbstractString, Symbol}} = Union{T,Nullable{T}}
const FixedElement{T<:Union{Number, AbstractString, Symbol}} = T
const NullableElement{T<:Union{Number, AbstractString, Symbol}} = Nullable{T}



####################
# DataCell aliases #
####################
const CellDataVec{T<:AbstractVector} = DataCell{T, Void, Void}
const CellDataVecVec{T<:AbstractVector, S<:AbstractVector} = DataCell{T, S, Void}
const CellDataVecMat{T<:AbstractVector, S<:AbstractMatrix} = DataCell{T, S, Void}
const CellDataMat{T<:AbstractMatrix} = DataCell{T, Void, Void}
const CellDataMatVec{T<:AbstractMatrix, S<:AbstractVector} =  DataCell{T, S, Void}
const CellDataMatMat{T<:AbstractMatrix, S<:AbstractMatrix} =  DataCell{T, S, Void}
const CellDataVectors{T<:AbstractVector, S<:Union{AbstractArray, Void}} = DataCell{T, S, Void}	
const CellDataMatrices{T<:AbstractMatrix, S<: Union{AbstractArray, Void}} = DataCell{T, S, Void}
const CellDataU{T<:AbstractArray} = DataCell{T, Void, Void} 					# Data cell, unlabeled 
const CellDataL{T<:AbstractArray, S<:AbstractVector} = DataCell{T, S, Void}			# Data cell labeled
const CellDataLL{T<:AbstractArray, S<:AbstractMatrix} = DataCell{T, S, Void}			# Data cell multiple labels
const CellData{T<:AbstractArray, S} = DataCell{T, S, Void}
#const TrainableData{U <:Real, T<:AbstractMatrix{U}, S<:AbstractArray} = DataCell{T, S, Void} 	



########################
# FunctionCell aliases #
########################
const CellFunF{V<:Function} = 			FunctionCell{Void, Void, V}			# Fixed function cell: simple transforms
const CellFunU{S<:Dict, V<:Function} =		FunctionCell{Void, S, V}			# Untrained function cell: untrained classifier or transform
const CellFunT{T<:Model, S<:Dict, V<:Function} =FunctionCell{T, S, V}				# Trained function cell: trained classifier, transform
const CellFun{T,S, V<:Function} = 		FunctionCell{T, S, V}



#####################
# PipeCell aliases  #
#####################

# Stacked pipe aliases
const PipeStacked = PipeCell{<:PTuple{AbstractCell}, Void, Void}				# Stacked pipe (generic)
const PipeStackedF = PipeCell{<:PTuple{CellFunF}, Void, Void}  					# Stacked pipe (fixed Cells)
const PipeStackedU = PipeCell{<:PTuple{CellFunU}, Void, Void}  					# Stacked pipe (untrained Cells)
const PipeStackedT =  PipeCell{<:PTuple{CellFunT}, Void, Void} 					# Stacked pipe (trained Cells)

# Parallel pipe aliases
const PipeParallel = PipeCell{<:PTuple{AbstractCell}, <:SortedDict, Void}			#  Parallel pipe (generic)
const PipeParallelF = PipeCell{<:PTuple{CellFunF}, <:SortedDict, Void} 				#  Parallel pipe (fixed Cells)
const PipeParallelU = PipeCell{<:PTuple{CellFunU}, <:SortedDict, Void}				#  Parallel pipe (untrained Cells)
const PipeParallelT = PipeCell{<:PTuple{CellFunT}, <:SortedDict, Void} 				#  Parallel pipe (trained Cells)

# Serial pipe aliases 
const PipeSerial = PipeCell{<:PTuple{AbstractCell}, <:Vector{Int}, Void}			# Serial pipe (generic)
const PipeSerialF = PipeCell{<:PTuple{CellFunF}, <:Vector{Int}, Void}				# Serial pipe (fixed Cells)
const PipeSerialU = PipeCell{<:PTuple{CellFunU}, <:Vector{Int}, Void} 				# Serial pipe (untrained Cells)
const PipeSerialT = PipeCell{<:PTuple{CellFunT}, <:Vector{Int}, Void} 				# Serial pipe (trained Cells)

# Pipe alias
const PipeGeneric = PipeCell{<:PTuple{AbstractCell}, <:Any, <:Any}				# Any type of pipe
