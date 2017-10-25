# Tests for distances
function t_lr()


N = 10	# variables
M = 25  # samples

vectordata = rand(M)
matrixdata = rand(N,M)

labels = 3*vectordata+0.1*randn(M)
multilabels = [collect(1:10) 3*collect(1:10)]'*matrixdata+0.1*randn(2,M)

A01 = j4pr.datacell(vectordata)
A02 = j4pr.datacell(vectordata, labels)
A03 = j4pr.datacell(vectordata, multilabels)
A04 = j4pr.datacell(matrixdata)
A05 = j4pr.datacell(matrixdata, labels)
A06 = j4pr.datacell(matrixdata, multilabels)


Base.Test.@test try 
	for A in [A01, A02, A03, A04, A05, A06]
		w = A |> j4pr.lr(1)
		w2 = A |> j4pr.lr(1,bias=false)
		A |> w
		A |> w2
	end
	true
catch 
	false
end


end

