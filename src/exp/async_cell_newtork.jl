# Define a dataset
a = "This is a string"

# Define lambda functions that perform actions

fsw = (x)->split(x," ")
fsl = (x)->split(x,"")


# Define the cells that will be nodes
w1 = j4pr.functioncell(x->x) 	# 1. copy-er
w2 = j4pr.functioncell(fsw)  	# 2. word splitter
w3 = j4pr.functioncell(fsl)  	# 3. letter counter
w41 = j4pr.functioncell(length)	# 4. length
w42 = j4pr.functioncell(length)  	# 5. length
w5 = j4pr.functioncell((x)->begin 
		str = "the answer is: "*string(max(x...)/min(x...))
		@show str
		str
	end
) # 6. divide

# Define some form of connectivity
#	      / w2 -> w41 \
# Cin -> w1 ->		   -> w5 -> Cout  
# 	      \ w3 -> w42 /

W=[w1,w2,w3,w41,w42,w5]
V= [ (0,1), (1,2), (1,3),(2,4), (3,5), (4,6), (5,6), (6,-1)] #vertices with indices from W

# Parse vertices and create async cells
V = unique(V)
C = [Channel(0) for i in 1:length(V)]
Cin = Channel(0);
Cout = Channel(0);

for (iw, w) in enumerate(W)
	ic = []; #input channels
	oc = []; #output channels
	for (iv, v) in enumerate(V)
		if iw == v[1] && v[2] != -1 oc=[oc...,C[iv]] end
		if iw == v[2] && v[1] != 0  ic=[ic...,C[iv]] end
		if iw == v[2] && v[1] == 0  ic=[ic...,Cin]  end
		if iw == v[1] && v[2] == -1 oc=[oc...,Cout] end
	end
	j4pr.@async_cell_exp w ic oc
end


# Put input, take output
print("Putting...")
tic()
put!(Cin, a)
println("OK")

println("The end with result $(take!(Cout))")
toc()

ps = w1+[w2+w41;w3+w42]+w5 
@time r=a|> ps 
@show r
[close(c) for c in C];



