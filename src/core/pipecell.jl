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



##############################
# Piping operators for Pipes #
##############################

# Pipe operator (stacked pipes)
|>(x::T where T, p::PipeStacked) = pipestack(ntuple(i->p[i](x), Val{length(p)})) # Pipe operators for generic stacked pipes
    


# Pipe operator (parallel pipes)
|>(x::T where T<:CellData, p::PipeParallel) = begin	                        		# Pipe operator for generic parallel pipes
	# Read dispatch information
	dispatch = gety!(p)
	K = collect(keys(dispatch))

	# Pass parts from the input data 
	pipeparallel(ntuple( (i,K=K,dispatch=dispatch)->p[K[i]](varsubset(x,dispatch[K[i]])), Val{length(p)}), dispatch)
end

# Pipe operator (parallel pipes)
|>(x::T where T, p::PipeParallel) = begin	                        			# Pipe operator for generic parallel pipes
	# Read dispatch information
	dispatch = gety!(p)
	K = collect(keys(dispatch))

	# Pass parts from the input data 
	pipeparallel(ntuple( (i,K=K,dispatch=dispatch)->p[K[i]](x[dispatch[K[i]]]), Val{length(p)}), dispatch)
end



# Pipe operator (serial pipes)
|>(x::T where T, p::Union{PipeSerialF, PipeSerialT}) = begin					# Pipe operator for fixed or trained serial pipes (e.g. pass data through the pipe's cells)
    	# Read order information
	order = gety!(p)

	# Start passing data through pipe
	xout = x;                                                                               # Initialize output with the input
    	for i in order		                                                        	# Pass data through each stage of the pipe
        	xout = xout |> p[i]
    	end
    	return xout #data
end

|>(x::T where T, p::PipeSerialU) = begin                                                        # Pipe operator for untrained serial pipes (e.g. train function cells sequentially)
    	# Read order information
	order = gety!(p)

	tp = pipeserial((x |> p[order[1]],))		                                                
	
	# Train first element of the untrained serial pipe and create serial pipe
	for i in order[2:end]		                                                        # Iterate through the rest of the untrained pipe elements and:
        	inp = x |> tp                                                                   # - run data through the trained pipe
        	out = inp |> p[i]                                                               # - train current pipe element
		tp = pipeserial((getx!(tp)...,out))                                             # - add newly trained element in the pipe
    	end
    return tp #trained serial pipe
end

|>(x::T where T, p::PipeSerial) = begin                                                        	# Pipe operator for untrained serial pipes
    	# Read order information
	order = gety!(p)

    	inp = x                                                                                 
	outp = x
	ndo = 0; # (non-data-output) its value indicates if pipe was fully processed or not 			
	processed_Cells = ()                                                           
	
	# Iterate through the pipe and multiply elements (i order, j just an index in the order vector)
	for (j,i) in enumerate(order)	
		
		# Pass data through each element of the pipe	
		outp = inp |> p[i]                                                           
		#info("\t\t Pipe element $i done...")

		# Check if output at this stage is data or not
		if ~(outp isa CellFun) && ~(outp isa PipeGeneric)
			processed_Cells = (processed_Cells...,p[i])				# Output is data, pipe element was fixed/trained, insert as is 
			inp = outp				  				# former output becomes next cells input	
		else 										# Output is not data, store output, the rest of the elements are copied in the output
			processed_Cells = (processed_Cells...,outp)
			ndo = j;
		end
		
		# Check whether data will not fully pass through the pipe
		if ndo > 0 && ndo < length(p)	
			info("[operators] Pipe was partially processed ($(j)/$(length(p)) elements)." )
			# Copy the rest of the pipe unchanged
			for k = j+1:length(order)
				processed_Cells = (processed_Cells..., p[order[k]])
			end
			break;
		end
    	end
	
	# If at all stages data was returned, return data, else return processed serial pipe
	if ndo == 0 return outp
	else return pipeserial(processed_Cells)
	end
end



##############################################################################################################################
# Functions needed to construct pipes 
##############################################################################################################################

# Function that creates stacked pipes
"""
	pipestack(x)	

Creates a `stacked pipe`.
"""
pipestack(x::T where T<:DataElement) = x							# leave simple stuff alone
pipestack(x::T where T<:AbstractArray) = x                       				# leave arrays alone			
pipestack(x::T where T<:PTuple{DataElement}) = mat(collect(x))					# tuples of elements become 1 column matrices (e.g. 1 obs) 
pipestack(x::T where T<:PTuple{AbstractArray}) = dcat(x...) 					# concatenate no matter what type of array
pipestack(x::PTuple{T} where T<:CellData) = vcat(x...) 						# concatenate only datacells of the same type	
pipestack(x::T where T<:PTuple{CellData}) = flatten(x, PipeStacked)				# call flatten if the data cell types differ	
pipestack(x::T where T<:PTuple{AbstractCell}) = PipeCell(x)              			# tuples of cells call the constructor (e.g. for pipes)
pipestack(x) = pipestack(map(AbstractCell, x)) 							# Make Cells out of arguments, then call pipestack again


# Function that creates parallel pipes
"""
	pipeparallel(x)	

Creates a `parallel pipe`.
"""
pipeparallel(x::T where T<:DataElement, dispatch::SortedDict) = x
pipeparallel(x::T where T<:AbstractArray, dispatch::SortedDict) = x                       				
pipeparallel(x::T where T<:PTuple{DataElement}, dispatch::SortedDict) = mat(collect(x)) 
pipeparallel(x::T where T<:PTuple{AbstractArray}, dispatch::SortedDict) = dcat(x...) 
pipeparallel(x::PTuple{T} where T<:CellData, dispatch::SortedDict) = vcat(x...)
pipeparallel(x::T where T<:PTuple{CellData}) = flatten(x, PipeParallel)				# if no dispatch is specified, (e.g. hcat, concatenate data)	
pipeparallel(x::T where T<:PTuple{CellData}, dispatch::SortedDict) = flatten(x, PipeStacked)	# if dispatch is specified (e.g. output of pipe, concatenate variables)	
pipeparallel(x::T where T<:PTuple{AbstractCell}, dispatch::SortedDict=SortedDict(Dict(i=>i for i in 1:length(x)))) = PipeCell(x, dispatch)	
pipeparallel(x, dispatch::SortedDict=SortedDict(Dict(i=>i for i in 1:length(x)))) = pipeparallel(map(AbstractCell, x)) 


# Function that creates serial pipes
"""
	pipeserial(x)	

Creates a `serial pipe`.
"""
#pipeserial{T<:PTuple{CellData}}(x::T, order::Vector{Int}=collect(1:length(x))) = Cell(x, order)# call flatten if the data cell types differ	
pipeserial(x::T where T<:PTuple{AbstractCell}, order=collect(1:length(x))) = PipeCell(x, order)				
pipeserial(x, order=collect(1:length(x))) = pipeserial(map(AbstractCell, x), order) 
pipeserial(x::T where T<:PTuple, order=collect(1:length(x))) = flatten(x, PipeSerial)


#####################################
# Concatenation operators for Pipes #
#####################################
vcat(ac::T where T<:AbstractCell, args...) = pipestack((ac, args...))	                        # Vertical concatenation creates stacked pipes
vcat(ac::T...)  where T<:AbstractCell= pipestack(ac)                   	  			# Vertical concatenation creates stacked pipes
hcat(ac::T where T<:AbstractCell, args...) = pipeparallel((ac, args...))                       	# Horizontal concatenation creates parallel pipes
hcat(ac::T...) where T<:AbstractCell = pipeparallel(ac)                				# Horizontal concatenation creates parallel pipes
+(ac::T where T<:AbstractCell, args...) = pipeserial((ac, args...))                            	# Summation operator creates serial pipes
+(ac::T...) where T<:AbstractCell = pipeserial(ac)             					# Summation operator creates serial pipes



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



"""
	flatten(x, t, [tn=t])

Low-level function used for generic pipe creation. `x` should be a `PTuple{<:AbstractCell}` and
`t` one of the aliases `PipeStacked`, `PipeParalel` or `PipeSerial`. Any element of `x` that
is not a `Cell` will be automatically transformed into one. `tn` is used in recursing the pipe
and is used to match the types of parent and nested pipes (e.g. if they match one can flatten them).
"""
flatten(v, op, newop=op) = begin
	#info("!")
	out = ()								# Result of the flattened v 
	pipeout = ()								# Result of the flattened Pipe
	for c in v								# Loop through all elements of the Pipe/Tuple
		# Recursion
		if c isa CellFun || c isa CellData 
			out=(out..., c)						# If current element is a function or data cell, add to list of cells
		elseif c isa PipeGeneric					# If it is a Pipe, flatten it (if possible)
			# Once entering the nested pipe, change the op
			# so that it matches the pipe type
			arg = ()
			if c isa PipeStacked 
				newop = PipeStacked
			elseif c isa PipeParallel 
				newop = PipeParallel
				arg = (SortedDict(Dict(i=>i for i in 1:length(c))),)
			elseif c isa PipeSerial 
				newop = PipeSerial
				arg = ([i for i in 1:length(c)],)
			else 
				error("[flatten]$(typeof(v)) cannot be part of a pipe")
			end
			pipeout= flatten(c,newop)				# Call recursively function, specifying the type 

			if op<:newop 						# If Pipe element is the same as its parent Pipe
				out=(out..., pipeout...)			# Concatenate their elements	
			else 
				out = (out...,newop((pipeout,arg...))) 		# Otherwise, concatenate elements of parent with 
			end							# flattened Pipe element
		else								# If element is not a Cell, trasform it into one
			nc = AbstractCell(c)
			out=(out...,nc)
		end
	end

	# Check if we are at the top layer (e.g. v is a Tuple) and if so, construct Pipe;
	# If not at the top layer (e.g. in a pipe, return flattened pipe)
	if v isa Tuple 								
		try	
			if all(c isa CellData for c in v)
				# Check if all label information is identical accross cells
				# - if so, add it to the flattened dataset,
				# - otherwise, concatenate only data content and issue warning
				if !all( isequal(gety!(v[i]),gety!(v[i+1])) for i in 1:length(v)-1 ) 
				  	labels = nothing	
					#warn("[flatten] Label information will be discarded (inconsistent across elements).")
				else
					labels = gety!(v[1])
				end

				# Return concatenated datacell
				if (length(v) > 1)
					if op<:PipeStacked
						return datacell(dcat(getx!.(v)), labels)
					elseif op<:PipeParallel
						return datacell(ocat(getx!.(v)), labels)
					else
						error("[flatten] Could not concatenate data cells, unsupported op=$(op)")
					end
				else
					return datacell(getx!(v[1]), labels)
				end
			end
		catch
			error("[flatten] Flattening failed. Possible reasons:\n
	 			 - inconsistent dimensions or data being concatenated
				 - label size did not match data size. ")
		end
		
		if op<:PipeStacked arg = ()
		elseif op<:PipeParallel arg = (SortedDict(Dict(i=>i for i in 1:length(out))),)
		elseif op<:PipeSerial arg = ([i for i in 1:length(out)],)
		else error("[flatten] Wrong operation.")
		end
		warn("[flatten] Any custom dispatch/order information for parallel/serial pipes is lost.")
		return op((out,arg...))
	else
		return out 
	end
end
