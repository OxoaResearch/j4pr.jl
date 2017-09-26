# Tests for distances
function t_lda()


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


Base.Test.@test try 
	A05 |> (A05 |> j4pr.lda(true) )
	+A05 |> (A05 |> j4pr.lda(true) )
	
	A05 |> (A05 |> j4pr.lda(false) )
	+A05 |> (A05 |> j4pr.lda(false) )
	
	A05 |> (A05 |> j4pr.ldasub(true) )
	+A05 |> (A05 |> j4pr.ldasub(true) )
	
	A05 |> (A05 |> j4pr.ldasub(false) )
	+A05 |> (A05 |> j4pr.ldasub(false) )
	true
catch 
	false
end


end

