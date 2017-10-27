# Basic tests that check that no plot crashes
function t_plotting()

	X = rand(2,10)			# random 2-D data
	y = round.(rand(10)) 		# random labels

	Du = j4pr.datacell(X)		# unlabeled datacell
	Dl = j4pr.datacell(X,y)		# labeled datacell

	pl = j4pr.lineplot(1) 		# plots first variable when piped data
	pl2 = j4pr.lineplot(sin,1) 	# plots the sine of first variable when piped data
	psc = j4pr.scatterplot() 	# plots scatterplot
	pd1 = j4pr.densityplot1d() 	# plots densityplot 1d
	pd2 = j4pr.densityplot2d() 	# plots densityplot 2d

	for p in [pl, pl2, psc, pd1, pd2]
		Test.@test try 
			Du |> p;
			Dl |> p;
			true
		catch
			false
		end
	end

	if (VERSION <= v"0.6")
	Test.@test try
		j4pr.rocplot((X./sum(X,1),y)|>j4pr.findop(1,j4pr.ROC.TPr(),1.));
		true
	catch
		false
	end
	end

end
