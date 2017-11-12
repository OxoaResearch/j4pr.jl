# Tests for the NetworkLearner module
function t_networklearner()

# Test transform methods for relational learners
LEARNER = [j4pr.NetworkLearner.SimpleRN,
	   j4pr.NetworkLearner.WeightedRN,
	   j4pr.NetworkLearner.BayesRN,
	   j4pr.NetworkLearner.ClassDistributionRN]
N = 5							# number of observations
C = 2; 							# number of classes
A = [							# adjacency matrix		
 0.0  0.0  1.0  1.0  0.0;
 0.0  0.0  1.0  0.0  0.0;
 1.0  1.0  0.0  1.0  1.0;
 1.0  0.0  1.0  0.0  0.0;
 0.0  0.0  1.0  0.0  0.0;
]
Ad = j4pr.NetworkLearner.adjacency(A); 

X = [							# local model estimates (2 classes)
 1.0  1.0  1.0  0.0  0.0;
 0.5  1.0  0.0  1.5  0.0
]

y = [1, 1, 1, 2, 2]					# labels

result = [
 [0.5  1.0  0.5  1.0  1.0; 				# validation data for SimpleRN
  0.5  0.0  0.5  0.0  0.0]
, 
 [0.5  1.0  0.5   1.0   1.0;				# validation data for WeightedRN 
  0.75 0.0  0.75  0.25  0.0]
, 		
 [1.60199  1.51083  1.60199  1.51083  1.51083;		# validation data for BayesRN
  1.14384  1.28768  1.14384  1.28768  1.28768]
,
 [0.300463  0.600925  0.300463  0.416667  0.600925; 	# validation data for ClassDistributionRN
  0.800391  0.125     0.800391  0.125     0.125]   
]

Xo = zeros(size(X));					# ouput (same size as X)
Xon = zeros(size(X));					# normalized ouput (same size as X)

tol = 1e-5
for li in 1:length(LEARNER)
	rl = j4pr.NetworkLearner.fit(LEARNER[li], Ad, X, y; priors=ones(length(unique(y))),normalize=false)
	rln = j4pr.NetworkLearner.fit(LEARNER[li], Ad, X, y; priors=ones(length(unique(y))),normalize=true)
        j4pr.NetworkLearner.transform!(Xo, rl, Ad, X, y);
        j4pr.NetworkLearner.transform!(Xon, rln, Ad, X, y);
	Test.@test all(abs.(Xo - result[li]) .<= tol);	# external validation
	Test.@test Xon ≈ (Xo./sum(Xo,1))		# normalization validation
end



# NetworkLearner tests
Ntrain = 100						# Number of training observations
Ntest = 10						# Number of testing observations					
inferences = [:ic, :rl]					# Collective inferences
rlearners = [:rn, :wrn, :bayesrn, :cdrn]		# Relational learners
nAdj = 2						# Number of adjacencies to generate	
X = rand(1,Ntrain); 					# Training data

for tL in [:regression, :classification]		# Learning scenarios 
	# Initialize data            
	if tL == :regression
		ft=x->vec(LearnBase.targets(x))
		y = vec(sin.(X)); 
		Xo = zeros(1,Ntest)
		
		# Train and test methods for local model 
		fl_train = (x)->mean(x[1]); 
		fl_exec=(m,x)->x.-m;

		# Train and test methods for relational model
		fr_train=(x)->sum(x[1],2);
		fr_exec=(m,x)->sum(x.-m,1)
	else 
		ft=x->LearnBase.targets(indmax,x)
		y = rand([1,2,3],Ntrain) # generate 3 classes 
		C = length(unique(y))
		Xo = zeros(3,Ntest)
		
		# Train and test methods for local model
		fl_train = (x)->zeros(3,1);
		fl_exec=(m,x)->abs.(x.-m);
		
		# Train and test methods for relational model
		fr_train=(x)->sum(x[1],2);
		fr_exec=(m,x)->rand(3,size(x,2))
	end

	amv = sparse.(full.(Symmetric.([sprand(Float64, Ntrain,Ntrain, 0.5) for i in 1:nAdj])));
	adv = j4pr.NetworkLearner.adjacency.(amv); 

	for infopt in inferences
		for rlopt in rlearners  
			Test.@test try
				# Train NetworkLearner
				nlmodel=j4pr.NetworkLearner.fit(j4pr.NetworkLearner.NetworkLearnerOutOfGraph, X, y, 
				       adv, fl_train, fl_exec,fr_train,fr_exec;
				       learner=rlopt, 
				       inference=infopt,
				       use_local_data=true,
				       f_targets=ft,
				       normalize=false, maxiter = 5
				)

				# Test NetworkLearner
				Xtest = rand(1,Ntest)

				# Add adjacency
				amv_t = sparse.(full.(Symmetric.([sprand(Float64, Ntest,Ntest, 0.7) for i in 1:nAdj])));
				adv_t = j4pr.NetworkLearner.adjacency.(amv_t); 
				j4pr.NetworkLearner.add_adjacency!(nlmodel, adv_t)
				
				#Run NetworkLearner
				j4pr.NetworkLearner.transform!(Xo, nlmodel, Xtest);
				true
			catch
				false
			end
		end
	end
end

end
