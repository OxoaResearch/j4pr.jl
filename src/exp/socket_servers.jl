# Reload J4PR just in case ;)
module socket_servers

export start_sync_sock_server, start_async_sock_server



# Function that simply listens to a socket, takes the message and returns it
function start_sync_sock_server(socketpath, readyc=RemoteChannel(()->Channel(1)))
	@sync begin
		println("[PRE-listen]")
		issocket(socketpath) && rm(socketpath)
		server = listen(socketpath)
		println("[POST-listen]")

		println("Listening...")
		msg = ""
		while msg != "END"
			sock = accept(server)
			msg = readline(sock);
			println("Nope, $(msg) is not it")
		end
		println("yes, thats it")
		put!(readyc,1)
	end
	return 0
end



# Function that spawns an asynchronous task that performs stuff
# and returns the processed data to the calling routine. 
# Like the main server which can be stopped by "END", the async
# task can be stopped by "TERMINATE"
function start_async_sock_server(socketpath, readyc=RemoteChannel(()->Channel(1)))
	
	# A small processing function
	f(x) = begin
		waittime=10*rand()
		println("\t[f] Processing... (wait time $(waittime)[s])")
		sleep(waittime)
		maximum(x)
	end

	#Function that reads a matrix from an input channel and returns the output of function f on an output channel
	my_processing(chi::Channel, cho::Channel, f::Function, ready::Condition) = begin
		while (true)
			msg = take!(chi)
			println("[MY_PROCESSING] Received $(msg) from channel, processing...")
			
			if replace(msg,"\n","") == "TERMINATE"
				println("[MY_PROCESSING] Terminating async process ...")
				notify(ready)
				break
			end
			
			try
				iv=[eval(parse(msg))...]
				ov = Float64(f(iv))
				print("[MY_PROCESSING] Putting output value on channel ...")
				put!(cho, ov)
				notify(ready)
				println("OK.")

			catch
				println("[MY_PROCESSING] Error, vector conversion failed. Waiting for new data...")
				put!(cho,nothing)
				notify(ready)
			end
		end
	end
	
	# Define channels
	chi = Channel{String}(1024)
	cho = Channel{Any}(10)
	asyncready = Condition()

	# Start asynchronous task
	@async my_processing(chi, cho, f, asyncready) 
	asyncrunning = true


	# Start the server
	@sync begin
		# Check for and remove any unix socket if present	
		issocket(socketpath) && rm(socketpath)
		

		# Start listening
		server = listen(socketpath)
		while (true)
			println("[MAIN] Listening...")
			sock = accept(server)
			msg = readline(sock);

			if (replace(msg,"\n","") != "END" )														# Check for "END" message, if so terminate server
			
				# Part run only when asynchronous task is running
				################################################
				if (asyncrunning)																	# If asynchronous task is running, 
					put!(chi, msg) 																	# Send message to asynchronous task
					wait(asyncready)																# wait for the ready condition then
					if isready(cho)																	# check if the channel has data
						println("[MAIN] Output is: $(take!(cho))")									# and if it has print it
					else
						# Condition was triggered however channel is not ready, process exited
						asyncrunning = false														# mark the task as not running anymore
					end
				end
				################################################
			else
				println("[MAIN] Terminating main process ...")
				put!(readyc,1)
				exit(0)
			end #if 
		end #while
	end # @sync block
end

end # end module




		
