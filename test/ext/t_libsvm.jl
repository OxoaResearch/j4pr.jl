# Tests for libsvm
function t_libsvm()

iris_class = j4pr.DataGenerator.iris()
regression_values = j4pr.ohenc_integer(j4pr.gety(iris_class), unique(j4pr.gety(iris_class)))
iris_reg = j4pr.datacell(j4pr.getx(iris_class), regression_values)

d = Distances.Euclidean()
tol  = 0.2; # tolerance when comparing results

Wclass = [j4pr.libsvm(), j4pr.libsvm(d), 			# LIBSVM.SVC
     j4pr.libsvm(svmtype=LIBSVM.NuSVC), j4pr.libsvm(d, svmtype=LIBSVM.NuSVC),
     j4pr.libsvm(svmtype=LIBSVM.NuSVC), j4pr.libsvm(d, svmtype=LIBSVM.NuSVC),
     j4pr.libsvm(svmtype=LIBSVM.OneClassSVM), j4pr.libsvm(d, svmtype=LIBSVM.OneClassSVM)]

Wreg = [j4pr.libsvm(svmtype=LIBSVM.NuSVR, probability=false), 
     j4pr.libsvm(d, svmtype=LIBSVM.NuSVR, probability=false),
     j4pr.libsvm(svmtype=LIBSVM.EpsilonSVR, probability=false), 
     j4pr.libsvm(d, svmtype=LIBSVM.EpsilonSVR, probability=false)]
	

# Test classification
for w in Wclass
	Test.@test try 
		wt1 = iris_class |> w
		wt2 = j4pr.strip(iris_class) |> w

		result = iris_class |> wt1
		result2 = iris_class.x |> wt2
		
		#println("Not matching: ",sum(abs.(result.x - result2) .>= tol) )
		sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end
end

# Test regression
for w in Wreg
	Test.@test try 
		wt1 = iris_reg |> w
		wt2 = j4pr.strip(iris_reg) |> w

		result = iris_reg |> wt1
		result2 = iris_reg.x |> wt2
		#println("Not matching: ",sum(abs.(result.x - result2) .>= tol) )
		sum(abs.(result.x - result2) .>= tol) > 0 ? false : true
	catch 
		false
	end
end

end
