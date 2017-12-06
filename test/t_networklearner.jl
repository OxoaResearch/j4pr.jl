# Tests for network learning 
function t_networklearner()

	pixel_adjacency(X) = begin 
		n = size(X,2)
		A = spzeros(Int,n,n)
		for i in 1:n
			for j in i:n
				if i!=j && abs(X[1,i] - X[1,j]) < 2.0 && abs(X[2,i] - X[2,j]) < 2.0
					A[i,j] = 1
					A[j,i] = 1
				end
			end
		end
		return A
	end

	Ac = j4pr.DataGenerator.fish(20)   # classification dataset

	fl_train = j4pr.knn
	fl_exec = (m,x) ->j4pr.knn(x,m.x)
	fr_train = j4pr.knn
	fr_exec = (m,x) ->j4pr.knn(x,m.x)
	Adj = [NetworkLearning.adjacency(pixel_adjacency(+Ac))]

	w = j4pr.networklearner(Adj, fl_train, fl_exec, fr_train, fr_exec, maxiter=3)

	Test.@test try 
		wt = Ac |> w
		NetworkLearning.add_adjacency!(wt.x.data, Adj)
		result = Ac |> wt
		true
	catch 
		false
	end
end
