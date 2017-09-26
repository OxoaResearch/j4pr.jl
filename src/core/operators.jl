##############################################################################################################################
# Operators [Data cells]												     #
##############################################################################################################################

# Vertical datacell concatenation (e.g. data concatenation) and auxiliary methods
dcat(x::T where T<:PTuple{<:Void}) = nothing
dcat(x::T where T<:Array{<:Void}) = nothing
dcat(::Void) = nothing 
dcat(::Void,::Void) = nothing
dcat(x,::Void) = dcat(x)
dcat(::Void,x) = dcat(x)
dcat(x::AbstractVector) = getobs(x)
dcat(x::AbstractMatrix) = getobs(x)
dcat(x::Matrix) = x 
dcat(x::Vector) = x
dcat(x::AbstractVector, y::AbstractVector) = vcat(mat(x, ObsDim.Constant{2}()), mat(y,ObsDim.Constant{2}()))
dcat(x::AbstractMatrix, y::AbstractVector) = vcat(dcat(x), mat(y, ObsDim.Constant{2}()))
dcat(x::AbstractVector, y::AbstractMatrix) = vcat(mat(x, ObsDim.Constant{2}()), dcat(y))
dcat(x::AbstractMatrix, y::AbstractMatrix) = vcat(dcat(x), dcat(y))
dcat(x,y,z...) = dcat(dcat(x,y),z...)
dcat(x) = dcat(x...)
vcat(c::T...) where T<:CellData = begin
	@assert all(labx == laby for labx in gety!.(c), laby in gety!.(c)) "[vcat] 'y' fields have to be identical for all DataCells."
	datacell(dcat(getx!.(c)), gety!(c[1]))
end


# Horizontal datacell concatenation (e.g. observation concatenation) and auxiliary methods
ocat(x::T where T<:PTuple{<:Void}) = nothing
ocat(x::T where T<:Array{<:Void}) = nothing
ocat(::Void) = nothing 
ocat(::Void,::Void) = nothing
ocat(x,::Void) = ocat(x)
ocat(::Void,x) = ocat(x)
ocat(x::AbstractVector) = getobs(x)
ocat(x::AbstractMatrix) = getobs(x)
ocat(x::Vector) = x
ocat(x::Matrix) = x
ocat(x::AbstractVector, y::AbstractVector) = vcat(ocat(x), ocat(y))
ocat(x::AbstractMatrix, y::AbstractVector) = hcat(ocat(x), mat(y, ObsDim.Constant{2}()))
ocat(x::AbstractVector, y::AbstractMatrix) = hcat(mat(x, ObsDim.Constant{2}()), ocat(y))
ocat(x::AbstractMatrix, y::AbstractMatrix) = hcat(ocat(x), ocat(y))
ocat(x,y,z...) = ocat(ocat(x,y),z...)
ocat(x) = ocat(x...)
hcat(c::T...) where T<:CellData = datacell(ocat(getx!.(c)), ocat(gety!.(c)))



# Piping operators
|>(x::T where T<:AbstractArray, c::S where S<:CellData) = begin
    	info("[operators] Creating new data cell with Array contents..." )
    	datacell(x)
end

|>(x::T where T<:Tuple, c::S where S<:CellData) = begin 					# Generally used in the case: (data, labels ) |> datacell([])
	if (nobs(c) > 0)
		info("[operators] Creating new data cell appending Tuple contents..." )
		datacell(dcat(x[1],getx!(c)), dcat(x[2], gety!(c)))
	else
		info("[operators] Creating new data cell with Tuple contents..." )
		datacell(x...)
	end
end


|>(x1::T where T<:CellDataU, x2::S where S<:CellDataU) = begin                 			# Generally used to glue together two DataCells
	@assert(nobs(x1) == nobs(x2))
	datacell(dcat(getx!(x1),getx!(x2)))
end

|>(x1::T where T<:CellDataU, x2::S where S<:CellData) = begin                 			# Generally used to glue together two DataCells
	@assert(nobs(x1) == nobs(x2))
	datacell(dcat(getx!(x1),getx!(x2)), gety!(x2))
end

|>(x1::T where T<:CellData, x2::S where S<:CellDataU) = begin                 			# Generally used to glue together two DataCells
	@assert(nobs(x1) == nobs(x2))
	datacell(dcat(getx!(x1),getx!(x2)), gety!(x1))
end

|>(x1::T where T<:CellData, x2::S where S<:CellData) = begin                 			# Generally used to glue together two DataCells
	@assert(nobs(x1) == nobs(x2))
	if isequal(gety!(x1), gety!(x2))	
		datacell(dcat(getx!(x1),getx!(x2)), gety!(x1))
	else
		datacell(dcat(getx!(x1),getx!(x2)), dcat(gety!(x1),gety!(x2)))
	end
end



##############################################################################################################################
# Operators [Function cells]																								 #
##############################################################################################################################

# Pipe operators for function cells (a different execution method for each FunctionCell variety: fixed, untrained and trained)
|>(x::T where T, c::S where S<:CellFunF) = getf!(c)(x, c.fargs...; c.fkwargs...)
|>(x::T where T, c::S where S<:CellFunU) = getf!(c)(x, c.fargs...; c.fkwargs...)
|>(x::T where T, c::S where S<:CellFunT) = getf!(c)(x, getx!(c), gety!(c) ; c.fkwargs...)


##############################################################################################################################
# Operators for Pipes
##############################################################################################################################

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
# Operators [AbstractCells] 
##############################################################################################################################

# Simple math operators (quite outdated but may still be useful)
+(ac::T where T<:AbstractCell) = ac.x
-(ac::T where T<:AbstractCell) = ac.y
~(ac::T where T<:AbstractCell) = dump(ac)



# Concatenation (Pipe creation operators)
vcat(ac::T where T<:AbstractCell, args...) = pipestack((ac, args...))	                        # Vertical concatenation creates stacked pipes
vcat(ac::T...)  where T<:AbstractCell= pipestack(ac)                   	  			# Vertical concatenation creates stacked pipes
hcat(ac::T where T<:AbstractCell, args...) = pipeparallel((ac, args...))                       	# Horizontal concatenation creates parallel pipes
hcat(ac::T...) where T<:AbstractCell = pipeparallel(ac)                				# Horizontal concatenation creates parallel pipes
+(ac::T where T<:AbstractCell, args...) = pipeserial((ac, args...))                            	# Summation operator creates serial pipes
+(ac::T...) where T<:AbstractCell = pipeserial(ac)             					# Summation operator creates serial pipes



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
