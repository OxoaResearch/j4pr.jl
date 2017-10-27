function t_operators()

N = 10	# variables
M = 25  # samples

vectordata = rand(M)
matrixdata = rand(N,M)
labels = round.(2*rand(M))
multilabels = round.(2*rand(2,M))

A01 = j4pr.datacell(vectordata)
A02 = j4pr.datacell(vectordata, labels)
A03 = j4pr.datacell(vectordata, multilabels)
A04 = j4pr.datacell(matrixdata)
A05 = j4pr.datacell(matrixdata, labels)
A06 = j4pr.datacell(matrixdata, multilabels)
AV = [A01, A02, A03, A04, A05, A06]

# Data cells 
#println("Checking that data cell operators work... ")
#print("\t|> operator...")
for (i,a) in enumerate(AV)
	for (j,b) in enumerate(AV)
		try
			a|>b
			Test.@test true	
		catch
			Test.@test false
		end
	end
end
#println("PASSED")

#print("\tvcat operator...")
for (i,a) in enumerate(AV)
	for (j,b) in enumerate(AV)
		try
			[a;b]
			Test.@test true	
		catch
			Test.@test false
		end
	end
end
#println("PASSED")

#print("\thcat operator...")
for (i,a) in enumerate(AV)
	for (j,b) in enumerate(AV)
		try
			[a;b]
			Test.@test true	
		catch
			Test.@test false
		end
	end
end
#println("PASSED")



# Function cells
W = j4pr.functioncell(x->x)
PT = [W;W]
PPV = j4pr.pipeparallel((W,W), DataStructures.SortedDict(Dict(1=>1, 2=>1)))
PPM = [W W]
PS = W+W

#println("Checking that function cell operators work... ")
#print("\t|> for stacked pipes...")
for (i,a) in enumerate(AV)
	try
		a |> PT
		Test.@test true
	catch
		Test.@test false
	end
end
#println("PASSED")

#print("\t|> for parallel pipes...")
for (i,a) in enumerate(AV)
	try
		if i < 4
			a |> PPV # Send data cells with vector content to their corresponding parallel pipe
		else 
			a |> PPM # Send data cells with matrix content to their corresponding parallel pipe
		end
		Test.@test true
	catch
		Test.@test false
	end
end
#println("PASSED")

#print("\t|> for serial pipes...")
for (i,a) in enumerate(AV)
	try
		a |> PS
		Test.@test true
	catch
		Test.@test false
	end
end
#println("PASSED")

end
