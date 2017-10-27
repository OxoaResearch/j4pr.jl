# Tests for distances
function t_mds()


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


for tr in [A01, A02, A03, vectordata]
	Test.@test try 
		tr |> j4pr.mds(1); 
		true
	catch 
		false
	end
end

for tr in [A04, A05, A06, matrixdata]
	Test.@test try 
		tr |> j4pr.ica(2); 
		true
	catch 
		false
	end

end

end
