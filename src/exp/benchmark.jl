module benchmark

__precompile__(true)
reload("j4pr")	
using DataArrays
import j4pr

function benchmark_training(a::T where T<:j4pr.CellData, w::S where S<:j4pr.CellFun)
	
	a[:,1:2] |> w
	
	@printf("* %s:", w.tinfo)
	@time out = a|>w
end
	

	
function benchmark_execution(a::T where T<:j4pr.CellData, w::S where S<:j4pr.AbstractCell)
			
	out1 = [];
	out2 = [];
	out3 = [];

	adc = deepcopy(a)
	ad = DataArray(deepcopy(a))
	aa = deepcopy(j4pr.getx(a))

	@printf("* %s ...\n", w.tinfo)
	
	# Test datacell interface
	@printf("\tDataCell |> :")
	#try
		deepcopy(adc[:,1:10]) |> w;
		@time out1 = adc |> w
	#catch
	#	@printf("FAIL\n")
	#end

	# Test DataArray interface
	#@printf("\tDataArray |> :")
	#try
	#	deepcopy(ad[:,1:10]) |>w
	#	@time out2 = ad |> w
	#catch
	#	@printf("FAIL\n")
	#end
	
	# Test Array interface
	@printf("\tArray |> :")
	try
		deepcopy(aa[:,1:10]) |>w
		@time out3 = aa |> w
	catch
		@printf("FAIL\n")
	end
	
	# Test returned values
	try
		all(out1.x.==out3) ? @printf("\tValues check OK.") : begin
			@printf("\tValues check failed.")
			# Specify which version(s) disagree ...
		end
	catch
		@printf("\tCould not execute values check.")
	end
	
	@printf("\n\n")
	return out1, out2, out3
end



function benchmark_execution_pipe(a::T where T<:j4pr.CellData, w::S where S<:j4pr.PipeGeneric)
	@printf("\n\n=============== Testing %s =========================\n",w.tinfo)
	@printf("Compiling pipe elements ...");
	a[:,1] |> w;
	
	@printf("Passing %s data through pipe ...", typeof(a));
	@time a |> w

end



function run_benchmarks_basic()
	
	@printf("\n\nRunning some basic tests ...");
	
	# Get some dataset
	#A= dgenr("datasets", "iris")

	#A = datacell( rand(10000,100), round(rand(10000)*20) )
	#A=datacell( [[1,2,1,1,1] [1,1,0,0,0.0] ["a","v","x","v","a"]] )
	
	M=100;
	N=100;
	C = 3;
	A = j4pr.datacell( rand(M, N), round.((C-1)*rand(N)) );
	
	# Execution only cells
	W = [		j4pr.cslice([0,1]), 
      			j4pr.filterdomain!(Dict(i=>[0 10] for i in size(A,1))),
			j4pr.sample(0.5),
	     ]
	# Trainable cells
	Wt = [		j4pr.scaler!("mean"),
			j4pr.filterg("mean"),
			j4pr.ohenc("binary"),
			j4pr.ohenc("integer"),
	     ]


		
	# Benchmark training for trainable cells 
	@printf("\n\n=============== Training [Basic] =========================\n")
	out = Any[]
	for (i,w) in enumerate(Wt)
		push!(out, benchmark_training(A,w))
	end
		
	# Benchmark execution for executable only cells	
	@printf("\n\n=============== Testing [Basic] =========================\n")
	for w in W
		#@show w.m
		benchmark_execution(A,w)
	end
	
	# Benchmark execution for trainable cells	
	for w in out 
		benchmark_execution(A,w)
	end

end



function run_benchmarks_pipe()
	
	@printf("\n\nRunning some more advanced tests ...");
	
	# Get some dataset
	#A= dgenr("datasets", "iris")

	#A = datacell( rand(100,10000), round.(rand(10000)*20) )
	#A=datacell( [[1,2,1,1,1] [1,1,0,0,0.0] ["a","v","x","v","a"]] )
	M=100;
	N=10000;
	C = 3;
	A = 	j4pr.datacell( round.(50*rand(M, N)), round.((C-1)*rand(N)) ); # training dataset
	At = 	j4pr.datacell( round.(60*rand(M, N)), round.((C-1)*rand(N)) ); # testing dataset

	
	#Build pipe
	
	Wclass =	j4pr.cslice([0,1]) 
	Wdom = 		j4pr.filterdomain!(Dict(i=>[0 1] for i in 1:50))
	Wprocmv = 	j4pr.filterg("majority")
	Wremove =	j4pr.FunctionCell(getindex, (setdiff(1:100,collect(1:5)),:),"Data remover")
	Wsample =	j4pr.sample(0.5)
	Wuscale = 	j4pr.scaler!(Dict(i=>"mean" for i in 1:50))
	Wbu =		j4pr.ohenc("binary")
	Wiu	=	j4pr.ohenc("integer")
	
	# Train pipe
	@printf("\nPreliminary training ...\n")
	Wb = A |> Wbu
	PIPE = deepcopy(A) |> (deepcopy(A) |> (Wb+Wremove+Wdom+Wprocmv+Wuscale+Wsample))

	benchmark_execution(At,PIPE)
	return A, At, PIPE
end

run_benchmarks_basic();

a,at,pipe = run_benchmarks_pipe()


end
