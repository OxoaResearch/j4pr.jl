# Tests for the parzen density estimator and classifier 
function t_parzen()

Ac = j4pr.DataGenerator.iris()   	# classification dataset
Ade = j4pr.datacell(collect(-1:0.1:1)) 	# density estimation dataset

Wclass = [j4pr.parzen(h, window=Φ, metric=m) for h in [0.1,1,3], 
	  						  Φ in [:hat, :linear, :gaussian, :exponential, :cosine], 
							  m in [Distances.Euclidean(), Distances.Cityblock()]
	]

	

Wde = [j4pr.parzen(h, window=Φ, metric=m) for h in [0.1,1,3], 
	  						  Φ in [:hat, :linear, :gaussian, :exponential, :cosine], 
							  m in [Distances.Euclidean(), Distances.Cityblock()]
	]

# Test classification
for w in Wclass
	Base.Test.@test try 
		wt1 = Ac |> w
		wt2 = j4pr.strip(Ac) |> w

		result = Ac |> wt1
		result2 = +Ac |> wt2
		true	
	catch 
		false
	end
end

# Test density estimation 
for w in Wde
	Base.Test.@test try 
		wt1 = Ade |> w
		wt2 = +Ade |> w

		result = Ade |> wt1
		result2 = +Ade |> wt2
		true	
	catch 
		false
	end
end

# Test classification on a 1-D dataset
A1d = j4pr.datacell([rand(100);2*rand(100)],[zeros(100);ones(100)])
Base.Test.@test try 
	wt1 = A1d |> Wclass[1]
	result = A1d |> wt1
	true
	catch 
		false
	end

# Test Printer
Base.Test.@test try
	buf = IOBuffer()
	
	# test show for classifier
	wt1 = A1d |> Wclass[1]
	Base.show(buf,wt1.x.data)
	
	# test show for estimator
	wt1 = +A1d |> Wclass[1]
	Base.show(buf,wt1.x.data)
	
	true #works
	catch
		false
	end
end
