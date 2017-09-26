# Tests for Decision trees/ Random forest
function t_decisiontree()

Ac = j4pr.DataGenerator.iris()   # classification dataset
Ar = j4pr.DataGenerator.boston() # regression dataset


Wclass = [j4pr.tree(), j4pr.randomforest(), j4pr.adaboostump()] # only default arguments

Wreg = [j4pr.treer(), j4pr.randomforestr()]			# only default arguments	
	

# Test classification
for w in Wclass
	Base.Test.@test try 
		wt1 = Ac |> w
		wt2 = j4pr.strip(Ac) |> w

		result = Ac |> wt1
		result2 = Ac |> wt2
	
		# Do not check decisions, they may vary based on sampling etc.
		true
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
		result2 = Ar |> wt2
	
		# Do not check decisions, they may vary based on sampling etc.
		true
	catch 
		false
	end
end

end
