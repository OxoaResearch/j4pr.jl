# Tests for the quadratic discriminant classifier 
function t_quaddisc()

Ac = j4pr.DataGenerator.iris()   # classification dataset


tol  = 1e-6; # tolerance when comparing results
Wclass = j4pr.quaddisc(rand(),rand())
	

# Test classification
Base.Test.@test try 
	wt1 = Ac |> Wclass
	wt2 = j4pr.strip(Ac) |> Wclass
	result = Ac |> wt1
	result2 = +Ac |> wt2
	sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end

end

