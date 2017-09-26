# Tests for knn classification/regression 
function t_knn()

Ac = j4pr.DataGenerator.iris()   # classification dataset
Ar = j4pr.DataGenerator.boston() # regression dataset


tol  = 1e-6; # tolerance when comparing results
Wclass = [j4pr.knn(k, smooth=s, leafsize=l, metric=m) for k in [1,5,10], 
	  						  s in [:ml, :laplace,:none, :dist], 
							  l in [10,20], 
							  m in [Distances.Euclidean(), Distances.Cityblock()]
	]

	

Wreg = [j4pr.knnr(k, smooth=s, leafsize=l, metric=m) for k in [1,5,10], 
	  						  s in [:ml, :dist], 
							  l in [10,20], 
							  m in [Distances.Euclidean(), Distances.Cityblock()]
	]

# Test classification
for w in Wclass
	Base.Test.@test try 
		wt1 = Ac |> w
		wt2 = j4pr.strip(Ac) |> w

		result = Ac |> wt1
		result2 = +Ac |> wt2
	
		sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end
end

# Test regression 
for w in Wreg
	Base.Test.@test try 
		wt1 = Ar |> w
		wt2 = j4pr.strip(Ar) |> w

		result = Ar |> wt1
		result2 = +Ar |> wt2
	
		sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end
end

end
