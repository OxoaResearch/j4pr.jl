function t_roc()

	# Generate log-likelihood scores; classes fully overlap - the only one to guarantee that 
	# the calculated measure actually work
	tar = collect(-1:0.01:1);
	non = -tar; 


	x=(1+[tar;non])./2;
	X=[x';(1-x)'];
	y=[fill("a",length(tar));fill("b",length(non))];

	# Define functions to verify that ROC optimization works
	calculate_measure(y::AbstractVector{T}, yest::AbstractVector{T}, class::T, measure::j4pr.ROC.TPr) where T = 
		sum((y.==class) .& (yest.==class)) ./ (sum(y.==class)+eps())

	calculate_measure(y::AbstractVector{T}, yest::AbstractVector{T}, class::T, measure::j4pr.ROC.TNr) where T = 
		sum((y.!=class) .& (yest.!=class)) ./ (sum(y.!=class)+eps())

	calculate_measure(y::AbstractVector{T}, yest::AbstractVector{T}, class::T, measure::j4pr.ROC.FPr) where T = 
		sum((y.!=class) .& (yest.==class)) ./ (sum(y.!=class)+eps())

	calculate_measure(y::AbstractVector{T}, yest::AbstractVector{T}, class::T, measure::j4pr.ROC.FNr) where T = 
		sum((y.==class) .& (yest.!=class)) ./ (sum(y.==class)+eps())
		

	perfmetric = [j4pr.ROC.TPr(), j4pr.ROC.FPr(), j4pr.ROC.TNr(), j4pr.ROC.FNr()]; 
	desiredval = collect(0:0.1:1);
	tol = 0.01
	class = ["a","b"];
	methods=[:j4pr,:ra]
	yu = sort(unique(y))
	for c in class
		for pm in perfmetric
			for dv in desiredval
				for m in methods
					op = j4pr.ROC.findop(X, y, c, pm, dv, m, 10_000); # large number of ops for low tolerances
					Xop = op.weights*X;
					yest = yu[j4pr.targets(indmax,Xop)];
					ev = calculate_measure(y, yest, c, pm)
					Base.Test.@test abs(dv-ev) <=tol
				end
			end
		end
	end

	op = j4pr.ROC.findop(X, y, "a", j4pr.ROC.FPr(), 0.1);
	
	# Test changing an op
	pos = 2
	j4pr.ROC.changeop!(op,pos)
	Base.Test.@test op.pos == pos

	j4pr.ROC.changeop!(op,j4pr.ROC.FPr(),0.0)
	Base.Test.@test op.pos == size(op.rocdata,1)
	
	# Test simple op
	sop = j4pr.ROC.simpleop(op); Base.Test.@test sop.weights == op.weights
	sop = j4pr.ROC.simpleop([1,2,3]); Base.Test.@test sop.weights == diagm([1.0,2.0,3.0])
	
	
	
	# Test for the FunctionCell interface
	D=j4pr.DataGenerator.iris();
	(tr,ts)=j4pr.splitobs(j4pr.shuffleobs(D),0.7);
	w=j4pr.libsvm();
	wt=w(tr);
	out=tr|>wt;
	
	# Test op search
	Base.Test.@test try
		uop=j4pr.findop("virginica",j4pr.ROC.FPr(),0.3);
		cop = uop(out); 
		pt=wt + cop;
		+D |> pt;
		true
	catch
		false
	end
	
	# Test changing an op
	pos = 2
	j4pr.changeop!(cop,pos)
	Base.Test.@test cop.x.data.pos == pos

	j4pr.changeop!(cop,j4pr.ROC.FPr(),0.0)
	Base.Test.@test cop.x.data.pos == size(cop.x.data.rocdata,1)
	
	
	# Test simple op
	uop=j4pr.findop("virginica",j4pr.ROC.FPr(),0.3);
	cop = uop(out)
	sop = j4pr.simpleop(cop)
	Base.Test.@test sop.x.data.weights == cop.x.data.weights
	
	sop = j4pr.simpleop([1,2,3]); 
	Base.Test.@test sop.x.data.weights == diagm([1.0,2.0,3.0])
	
end
