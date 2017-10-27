# Tests for the AdaBoost ensemble
function t_adaboost()

	# M1
 	Test.@test try
		L = 10 # ensemble size

		# Prepare data
		D=j4pr.DataGenerator.fish(20);
		(tr,ts)=j4pr.splitobs(j4pr.shuffleobs(D),0.3);
		yorig=-D; 
		yu=sort(unique(yorig)); 
		ytr=Int.(j4pr.ohenc_integer(-tr, yu)); # knn works only with integer labels 
		
		# Construct execution function (has to return a vector of labels)
		f_exec=(m,x)->j4pr.targets(indmax, j4pr.DecisionStump.stump_exec(m,j4pr.getobs(x))) 
		
		# Train 
		ae=j4pr.AdaBoost.adaboost_train(+tr, ytr, L, x->j4pr.DecisionStump.stump_train(x[1],x[2]), f_exec, j4pr.AdaBoost.AdaBoostM1())
		
		# Execute
		j4pr.AdaBoost.adaboost_exec(ae,+D) 
		true
	catch 
		false
	end
	
	
	
	# M2	
 	Test.@test try
		L = 10 # ensemble size
		
		# Prepare data
		D=j4pr.DataGenerator.fish(20);
		(tr,ts)=j4pr.splitobs(j4pr.shuffleobs(D),0.3);
		yorig=-D; 
		yu=sort(unique(yorig)); 
		ytr=Int.(j4pr.ohenc_integer(-tr, yu)); 
		
		# Construct execution function (has to return a matrix of observation probabilities)
		f_exec=(m,x)->j4pr.DecisionStump.stump_exec(m,j4pr.getobs(x))
		
		# Train
		ae=j4pr.AdaBoost.adaboost_train(+tr, ytr, L, x->j4pr.DecisionStump.stump_train(x[1],x[2]), f_exec, j4pr.AdaBoost.AdaBoostM2())
        	
		# Execute
		j4pr.AdaBoost.adaboost_exec(ae,+D)
		true
	catch 
		false
	end


	# AdaBoost FunctionCell interface
	A = j4pr.DataGenerator.fish(20) 			# load data
	tr,ts = j4pr.splitobs(j4pr.shuffleobs(A), 0.5)		# shuffle and split data
	w=j4pr.parzen(0.1) 					# define classifier
	L=10	
	
	Test.@test try 
		m1 = j4pr.adaboost(w, L; boost_type=j4pr.AdaBoost.AdaBoostM1()) 
		m2 = j4pr.adaboost(w, L; boost_type=j4pr.AdaBoost.AdaBoostM2()) 

		# Train
		m1t = m1(tr)
		m2t = m2(tr)
		
		# Execute 
		m1t(ts);  	
		m2t(ts); 
		
		# If we're here, everything worked
		true
		
	catch
		false
	end
	
end
