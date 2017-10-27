# Tests for distances
function t_dbscan()



X = [randn(10,200)*2+rand(10,200) rand(10,200)+10 -5+randn(10,400)]

Test.@test try
	
	f1(x,y) = Distances.pairwise(Distances.Euclidean(),x,y)
	f2(dist,agg)= (p,S) -> agg([dist(p, j4pr.getobs(S,i)) for i in j4pr.nobs(S)])

	W = [	j4pr.dist(X)+j4pr.dbscan(0.5, 10, f1), # this version of dbscan expects distance matrix input
      		j4pr.dbscan(0.7, f2(Distances.euclidean, mean), min_neighbors=2, min_cluster_size=5) 
	]
	
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
