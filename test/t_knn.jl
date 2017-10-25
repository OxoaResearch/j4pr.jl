# Tests for knn classification/regression 
function t_knn()

Ac = j4pr.DataGenerator.iris()   # classification dataset
Ar = j4pr.DataGenerator.boston() # regression dataset


Wclass_nn = [j4pr.knn(k, smooth=s, leafsize=l, metric=m) for k in [1,5,10], 
	  						  s in [:ml, :laplace, :mest, :none, :dist], 
							  l in [10,20], 
							  m in [Distances.Euclidean(), Distances.Cityblock()]
	]

Wclass_r = [j4pr.knn(k, smooth=s, leafsize=l, metric=m) for k in [1.0,5.0,10.0], 
	  						  s in [:ml, :laplace, :mest, :none, :dist], 
							  l in [10,20], 
							  m in [Distances.Euclidean(), Distances.Cityblock()]
	]
	

Wreg_nn = [j4pr.knnr(k, smooth=s, leafsize=l, metric=m) for k in [1,5,10], 
	  						  s in [:ml, :dist], 
							  l in [10,20], 
							  m in [Distances.Euclidean(), Distances.Cityblock()]
	]

Wreg_r = [j4pr.knnr(k, smooth=s, leafsize=l, metric=m) for k in [1.0,5.0,10.0], 
	  						  s in [:ml, :dist], 
							  l in [10,20], 
							  m in [Distances.Euclidean(), Distances.Cityblock()]
	]

# Test classification
for w in Wclass_nn
	Base.Test.@test try 
		wt = Ac |> w
		result = Ac |> wt
		true
	catch 
		false
	end
end

for w in Wclass_r
	Base.Test.@test try 
		wt = Ac |> w
		result = Ac |> wt
		true
	catch 
		false
	end
end

# Test regression 
for w in Wreg_nn
	Base.Test.@test try 
		wt = Ar |> w
		result = Ar |> wt
		true
	catch 
		false
	end
end

for w in Wreg_r
	Base.Test.@test try 
		wt = Ar |> w
		result = Ar |> wt
		true
	catch 
		false
	end
end


end
