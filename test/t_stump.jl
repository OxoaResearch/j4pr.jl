# Tests for the Decision stump classifier and regressor 
function t_stump()

Ac = j4pr.DataGenerator.fish(20)  # classification dataset
Ar = j4pr.DataGenerator.fishr(20) # regression dataset


tol  = 1e-6; # tolerance when comparing results
Wclass = [j4pr.stump(vartypes=v, nthresh=n, split=s, count=c, prop=p, crit=cr ) for 
	  						v in [:nominal, :real, Dict(1=>:nominal)], 
	  						n in [2,5,100], 
							s in [:linear,:density], 
							c in [0,10],
							p in [0.0],
							cr in [:gini,:entropy,:misclassification]
]

# Test classification
for w in Wclass
	Test.@test try 
		wt = Ac |> w
		result = Ac |> wt
		result2 = +Ac |> wt
		sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end
end

# Test classification on a 1-D dataset
A1d = j4pr.datacell([rand(100);2*rand(100)],[zeros(100);ones(100)])
Test.@test try 
	wt1 = A1d |> Wclass[1]
	result = A1d |> wt1
	buf = IOBuffer()
	Base.show(buf,wt1.x.data)
	true
	catch 
		false
	end

# Test Prin# model=:linear fit is not tested because it is unstable in certain situations  
Wreg = [j4pr.stumpr(model=m, errcrit=ec, vartypes=v, nthresh=n, split=s, count=c, prop=p) for 
							m in [:mean, :median],
						        ec in [(x,y)->mean((x-y).^2), (x,y)->mean(abs.(x-y))],
							v in [:nominal, :real, Dict(1=>:nominal)], 
	  						n in [2,5], 
							s in [:linear,:density], 
							c in [0,10],
							p in [0.0]
]

# Test regression 
for w in Wreg
	Test.@test try 
		wt = Ar |> w

		result = Ar |> wt
		result2 = +Ar |> wt
	
		sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end
end

# Test regression on a 1-D dataset
A1d = j4pr.datacell([rand(100);2*rand(100)],rand(200))
Test.@test try 
	wt1 = A1d |> Wreg[1]
	result = A1d |> wt1
	buf = IOBuffer()
	Base.show(buf,wt1.x.data)
	true
	catch 
		false
	end


end

