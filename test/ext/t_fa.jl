# Tests for Factor Analysis
function t_fa()


N = 2	# variables
M = 250 # samples

matrixdata = randn(N,M)
labels = round.(2*rand(M))
multilabels = round.(2*rand(2,M))

A01 = j4pr.datacell(matrixdata)
A02 = j4pr.datacell(matrixdata, labels)
A03 = j4pr.datacell(matrixdata, multilabels)


for tr in [A01, A02, A03, matrixdata]
	for ts in [A01, A02, A03, matrixdata]
		Base.Test.@test try 
			Wfa = tr |> j4pr.fa()
			Wfar = j4pr.far(Wfa)
			ts |> Wfa 
			(ts |> Wfa) |> Wfar
			true
		catch 
			false
		end
	end
end

end

