# Tests for distances
function t_kmeans()



X = [randn(10,200)*2+rand(10,200) rand(10,200)+10 -5+randn(10,400)]

Test.@test try 
	W = [	j4pr.kmeans(2),
      		j4pr.kmeans(3,Distances.Jaccard()), 
		j4pr.kmeans!(rand(10,3)),
	        j4pr.kmeans!(rand(10,3), Distances.Jaccard()) ]

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

