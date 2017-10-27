# Tests for distances
function t_affinityprop()



X = [randn(10,200)*2+rand(10,200) rand(10,200)+10 -5+randn(10,400)]

Test.@test try 
	W = [	j4pr.affinityprop(),
		j4pr.affinityprop(maxiter=200, tol=1e-5,damp=0.7)]
	Xd = j4pr.datacell(X)	
	
	for w in W
		for dtr in [X,Xd], dts in [X,Xd]
			wt = dtr |> w
			dts |> wt
		end
	end
	true
catch 
	false
end


end

