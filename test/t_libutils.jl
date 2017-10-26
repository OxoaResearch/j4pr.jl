# Test library utility functions
function t_libutils()
	
	# Test confusion matrix

	references = ["a","a","c","c","b","b"];
	predictions = ["a","b","c","c","a","a"];

	C = confusionmatrix(predictions,references)
	Cref = [1.0  2.0  0.0; 1.0  0.0  0.0; 0.0  0.0  2.0]
	Base.Test.@test C == Cref
	
	Cn = confusionmatrix(predictions,references;normalize=true)
	Cref = [0.5  1.0  0.0; 0.5  0.0  0.0; 0.0  0.0  1.0]
	Base.Test.@test Cn == Cref
	
	Cp = confusionmatrix(predictions,references;positive="a")
	Cref = [1.0 2.0; 1.0  2.0]
	Base.Test.@test Cp == Cref

	Cpn = confusionmatrix(predictions,references;normalize=true,positive="a")
	Cref = [0.5  0.5; 0.5  0.5]
	Base.Test.@test Cpn == Cref

	# the 'showmatrix' option is not tested.
	#confusionmatrix(predictions,references;normalize=true,positive="a",showmatrix=true);
end
