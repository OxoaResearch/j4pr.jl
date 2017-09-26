# Execution time; to send data, do in the TERMINAL  $echo "message" | socat - UNIX-CONNECT:/tmp/tmpfifo

if nworkers() < 2
	newproc = addprocs(1)
	@show workers()
end

# Make module available (it is assumed we are in j4pr root directory)
@everywhere include("src/exp/socket_servers.jl")

# Assign worker, unix socket
pid = workers()[1]			# which worker
sockname = "/tmp/tmpsocket"		# which unix socket
cc=RemoteChannel(()->Channel(1)) 	# notification channel
#srv = socket_servers.start_sync_sock_server
srv = socket_servers.start_async_sock_server


# start server and wait for notification that the worker is no longer running
@async remote_do(srv, pid, sockname, cc)
wait(cc)


