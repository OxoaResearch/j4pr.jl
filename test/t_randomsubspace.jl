# Tests for the random subspace ensemble
function t_randomsubspace()

	## Test the random subspace ensemble on toy data

	X = [1 2 3; -1 -2 3; 0 0 3] # data
	y = [1,1,0]		    # labels

	f_exec(model,X) = X.+model # the execution function adds to the input the value contained in the model
	
	# Create an ensemble that for each variable (i.e. line) of X, takes the maximum, which is '3' in all cases. 'y' is not used as
	# the maximum function only takes the first argument of the input `x` (`x` is represented in the training function by Tuple(X,y)
	ensemble = j4pr.RandomSubspace.randomsubspace_train(X, y, 5, 1, x->maximum(x[1],2), f_exec, j4pr.ClassifierCombiner.NoCombiner(), true)

	# Test that the `trained` ensemble members are matrices of a single element of value 3
	Test.@test all((ensemble.members[i] ==Matrix([3]) for i in eachindex(ensemble.members)))

	
	# Run the ensemble execution (we expect that for the lines specified by ensemble.idx the corresponding values form ensemble.members are added)
	Xt = [1 1 1;2 2 2;3 3 3]
	ensemble_results = j4pr.RandomSubspace.randomsubspace_exec(ensemble,Xt)

	Test.@test all((ensemble_results[i] == f_exec(ensemble.members[i], Xt[ensemble.idx[i],:]) for i in eachindex(ensemble.members)))

	
	
	## Test the random subspace ensemble on the fish dataset and a classifier (the test is just a functionality assertion, does not verify values)
	A = j4pr.DataGenerator.fish(20) 			# load data
	tr,ts = j4pr.splitobs(j4pr.shuffleobs(A), 0.5)		# shuffle and split data
	w=j4pr.parzen(0.1)				
	
	C = 2; # number of classes (for the 'fish dataset')
	L = 5; # ensemble size
	M = 2; # variables per ensemble memeber

	# Loop through combiners (must be non-trainable), train subspace on `tr` and run it on `ts`
	for comb in [j4pr.ClassifierCombiner.GeneralizedMeanCombiner(L,C,1.0), j4pr.ClassifierCombiner.ProductCombiner(L,C)]
		Test.@test try 
			wes=j4pr.randomsubspace(w, L, M, comb, parallel_execution=false) # ensemble of 5 members, 2 variables, serial execution
			wep=j4pr.randomsubspace(w, L, M, comb, parallel_execution=true)  # ensemble of 5 members, 2 variables, parallel execution
	
			# Train
			west = wes(tr)
			wept = wep(tr)
			
			# Execute 
			west(ts);  	
			west(ts); 
		
			# If we're here, everything worked
			true
		catch
			false
		end

	end

	# Test variable sampling without replacing
	Test.@test try
		wes2=j4pr.randomsubspace(w, 2, 1, j4pr.ClassifierCombiner.NoCombiner(),false, parallel_execution=false)
		west2 = wes2(tr)
		west2(+ts);
		true
	catch
		false
	end

end
