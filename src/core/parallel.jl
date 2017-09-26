# Function that performes the calculation: takes data from the input channel and puts it on the output channel
function rcell(x::T where T<:j4pr.AbstractCell, ci::S, co::S) where S<:Union{Channel, RemoteChannel}
	
	while true			
		# Take data and perform processing 
		try 
			data = take!(ci)
			put!(co, data |> x) 
		catch exception
			if (exception isa InterruptException )					# Task received an interrupt 
				println("InterruptException() caught, exiting.")
				break 
			end
			
			if (exception isa InvalidStateException ) 				# Task could not write/read channels
				println("InvalidStateException() caught, exiting.") 		# probably because another task/process 
				break  								# closed them.
			end
			
			if (exception isa RemoteException ) 					# Task could not write/read channels
				println("RemoteException() caught, exiting.") 			# probably because another task/process 
				break  								# closed them.
			end
			
			@show exception
			println("Could not execute operation")
		end #try
		
		# Loop exit condition	
	end #while

end



# Small interrupt function that acts on the result of the @async_cell and @remote_cell macros
function interrupt(t::Tuple{T,S1,S2} where T<:Task where S1<:Channel where S2<:Channel)
	
	# Map tuple to local vars
	task,ic,oc = t
	
	# Check that the channels are open
	@assert isopen(ic) && isopen(oc)
	try
		@schedule Base.throwto(task, InterruptException())	
		return true
	catch
		println("Could not interrupt $(task).")	
		return false
	end

	
end


function interrupt(t::Tuple{Int,S1,S2} where S1<:RemoteChannel where S2<:RemoteChannel)
	
	# Map tuple to local vars
	id,ic,oc = t
	
	try
		interrupt(id)
		try
			close(ic);
			close(oc);
		catch
			println("Could not close channels.")	
			return false
		end
		return true
	catch
		println("Could not interrupt process on worker $(id).")	
		return false
	end
end



# Macro for asynchronous cell execution in the current process
"""
	@async_cell x

The macro generates two unbuffered channels, for input and output, and an starts a task that
waits for data from the input channel, passes it through `x` and puts the result 
on the ouput channel.
"""
macro async_cell(x)
	ex=quote begin
		local ci = Channel(0) 								# Input channel
		local co = Channel(0)								# Output channel
		
		local t = @async j4pr.rcell($x, ci, co)						# Call the cell server function asynchronously (e.g. waits for data)
		
		# Bind channels (when interrupting task, they will automatically close)
		bind(ci,t)
		bind(co,t)
		(t,ci,co)									# Return the channels and termination condition
	end # local block
	
	end #quote

	return esc(ex)
end



# Macro for asynchronous cell execution in the current process
# that supports specifying the input/output channels as well
"""
	@async_cell x ci co

The macro uses the channels `ci` and `co` as input and output and starts a task that
waits for data from the input channel, passes it through `x` and puts the result
on the ouput channel.
"""
macro async_cell(x, ci, co)
	ex=quote begin
		# Basic checks
		@assert $ci isa Channel && isopen($ci)
		@assert $co isa Channel && isopen($co)
		
		local t = @async j4pr.rcell($x, $ci, $co)					# Call the cell server function asynchronously (e.g. waits for data)
		
		# Bind channels (when interrupting task, they will automatically close)
		bind(ci,t)
		bind(co,t)
		(t,$ci,$co)									# Return the channels and termination condition
	end # local block
	end #quote

	return esc(ex)
end



# Macro for asynchronous cell execution on a remote process
"""
	@remote_cell id x

The macro generates two unbuffered remote channels, for input and output, and starts a task 
on worker `id` that waits for data from the input channel, passes it through `x` and puts 
the result on the ouput channel.
"""
macro remote_cell(id::Int, x)
	ex=quote begin
		local ci = RemoteChannel(()->Channel(0) )					# Input channel
		local co = RemoteChannel(()->Channel(0) )					# Output channel
		local worker 									# Variable that holds the id of the worker
		if !($(id) in workers()) 							# Check if the specified worker exists
			error("[@remote_cell] Worker $($(id)) not present.")
		else
			worker = $(id)	
		end
		local t = @async remote_do(j4pr.rcell, worker, $x, ci, co)			# Start on a remote worker the cell server
		(worker,ci,co)									# Return the channels and termination condition
	end # local block
	end #quote

	return esc(ex)
end



# Macro for asynchronous cell execution on a remote process
# that supports specifying the input/output channels as well
"""
	@remote_cell id x ci co

The macro uses the two remote channels `ci` and `co` as input and output 
and starts a task on worker `id` that waits for data from the input channel, 
passes it through `x` and puts the result on the ouput channel.
"""
macro remote_cell(id::Int, x, ci, co)
	ex=quote begin
		# Basic checks
		@assert $ci isa RemoteChannel
		@assert $co isa RemoteChannel
			
		local worker 									# Variable that holds the id of the worker
		if !($(id) in workers()) 							# Check if the specified worker exists
			error("[@remote_cell] Worker $($(id)) not present.")
		else
			worker = $(id)	
		end
			
		local t = @async remote_do(j4pr.rcel, worker, $x, $ci, $co)			# Start on a remote worker the cell server
		(worker,$ci,$co)								# Return the channels and termination condition
	end # local block
	end #quote

	return esc(ex)
end



# Experimental stuff
function rcell(x::T where T<:j4pr.AbstractCell, ci::Vector{S}, co::Vector{S}) where S<:Union{Channel, RemoteChannel}
	
	while true			
		# Take data and perform processing 
		try
			if length(ci) == 1
				data = take!(ci[1])
			else
				data = Any[take!(c) for c in ci] 
			end
			r = data |> x
			for c in co
				@async put!(c, r)
			end
		catch exception
			if (exception isa InterruptException )					# Task received an interrupt 
				println("InterruptException() caught, exiting.")
				break 
			end
			
			if (exception isa InvalidStateException ) 				# Task could not write/read channels
				println("InvalidStateException() caught, exiting.") 		# probably because another task/process 
				break  								# closed them.
			end
			
			if (exception isa RemoteException ) 					# Task could not write/read channels
				println("RemoteException() caught, exiting.") 			# probably because another task/process 
				break  								# closed them.
			end
			
			@show exception
			println("Could not execute operation")
		end #try
		
		# Loop exit condition	
	end #while
end


macro async_cell_exp(x, ci, co)
	ex=quote begin
		# Basic checks
		local t = @async j4pr.rcell($x, $ci, $co)					# Call the cell server function asynchronously (e.g. waits for data)
	end # local block
	end #quote

	return esc(ex)
end
