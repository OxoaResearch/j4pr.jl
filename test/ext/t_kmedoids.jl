# Tests for distances
function t_kmedoids()



X = [randn(10,200)*2+rand(10,200) rand(10,200)+10 -5+randn(10,400)]
C = X |> j4pr.dist(X)

Test.@test try 
	W = [	j4pr.kmedoids(2),
      		j4pr.kmedoids(3,Distances.Jaccard()), 
		j4pr.kmedoids!([1,2,3]),
		j4pr.kmedoids!([1,2,3], Distances.Jaccard()) ]

	Xd = j4pr.datacell(X)	
	
	for w in W
		for dtr in [X,Xd], dts in [X,Xd]
			C = dtr|>j4pr.dist(dtr)
			wt = C |> w
			dts |> j4pr.dist(dtr) |> wt
		end
	end
	true
catch 
	false
end


end

