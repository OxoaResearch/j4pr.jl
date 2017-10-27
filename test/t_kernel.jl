# Tests for kernels 
function t_kernel()


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

# Test kernels 
F = [(x,y)->x'y, 		# a linear kernel
     (x,y)->(x'y+1.)^2.		# a polynomial kernel
    ]

for tr in [A01, A02, A03, vectordata]
	for ts in [A01, A02, A03, vectordata]
		for f in F
			Test.@test try 
				ts |> (tr |> j4pr.kernel(f)); 
				ts |> (tr |> j4pr.kernel(f,center=true)); 
				true
			catch 
				false
			end
		end
	end
end

for tr in [A04, A05, A06, matrixdata]
	for ts in [A04, A05, A06, matrixdata]
		for f in F
			Test.@test try 
				ts |> (tr |> j4pr.kernel(f));
				ts |> (tr |> j4pr.kernel(f,center=true));
				true
			catch
				false
			end
		end
	end
end

# Make small test for kernelize with symmetric=false (call with one argument)
Test.@test try 
	j4pr.kernelize(rand(10,2), F[1])
	j4pr.kernelize(rand(10,2), F[1], center=true)
	true
catch
	false
end



end

