# Tests for the linear discriminant classifier 
function t_lindisc()

Ac = j4pr.DataGenerator.fish(20)   # classification dataset


tol  = 1e-6; # tolerance when comparing results
Wclass = j4pr.lindisc(rand(),rand())
	

# Test classification
Test.@test try 
	wt1 = Ac |> Wclass
	wt2 = j4pr.strip(Ac) |> Wclass
	result = Ac |> wt1
	result2 = +Ac |> wt2
	sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end


# Test classification on a 1-D dataset
A1d = j4pr.datacell([rand(100);2*rand(100)],[zeros(100);ones(100)])
wt1 = A1d |> Wclass
Test.@test try 
	wt2 = j4pr.strip(A1d) |> Wclass
	result = A1d |> wt1
	result2 = +A1d |> wt2
	sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end

# Test Printer
Test.@test try
	buf = IOBuffer()
	Base.show(buf,wt1.x.data)
	true #works
	catch
		false
	end

end
