function t_pipe_creation()

N = 10	# variables
M = 25  # samples
A = rand(N,M)
A0 = j4pr.datacell(A)
A1 = j4pr.datacell(A, round.(rand(M)))
A2 = j4pr.datacell(A, round.(rand(2,M)))

W1 = j4pr.ohenc("binary")
W2 = j4pr.scaler!(Dict(1=>"mean"))
W3 = j4pr.functioncell((x)->x)

PT = [W1;W2]
PP=[W1 W2]
PS = W1+W2



# Simple cell checks
#print("Type checks for function cells before passing data... ")
Base.Test.@test W1 isa j4pr.CellFunU && !(W1 isa j4pr.CellFunF) && !(W1 isa j4pr.CellFunT)
Base.Test.@test W2 isa j4pr.CellFunU && !(W2 isa j4pr.CellFunF) && !(W2 isa j4pr.CellFunT)
Base.Test.@test W3 isa j4pr.CellFunF && !(W3 isa j4pr.CellFunU) && !(W3 isa j4pr.CellFunT)

#println("PASSED")

#print("Type checks for function cells after passing data... ")
tmp = A0 |> W1
Base.Test.@test tmp isa j4pr.CellFunT && !(tmp isa j4pr.CellFunF) && !(tmp isa j4pr.CellFunU)
tmp = A0 |> W2
Base.Test.@test tmp isa j4pr.CellFunT && !(tmp isa j4pr.CellFunF) && !(tmp isa j4pr.CellFunU)
Base.Test.@test A0 |> W3 isa j4pr.CellDataU
Base.Test.@test A1 |> W3 isa j4pr.CellDataL
Base.Test.@test A2 |> W3 isa j4pr.CellDataLL
#println("PASSED")



# Simple pipe checks
#print("Type checks for [simple] pipes before passing data... ")
Base.Test.@test PT isa j4pr.PipeStackedU && !(PT isa j4pr.PipeStackedF) && !(PT isa j4pr.PipeStackedT)
Base.Test.@test PP isa j4pr.PipeParallelU && !(PP isa j4pr.PipeParallelF) && !(PP isa j4pr.PipeParallelT)
Base.Test.@test PS isa j4pr.PipeSerialU && !(PS isa j4pr.PipeSerialF) && !(PS isa j4pr.PipeSerialT)
#println("PASSED")

#print("Type checks for [simple] pipes after passing data... ")
tmp = A0 |> PT
Base.Test.@test tmp isa j4pr.PipeStackedT && !(tmp isa j4pr.PipeStackedF) && !(tmp isa j4pr.PipeStackedU)
tmp = [A0|>W1; A1|>W2] 
Base.Test.@test tmp isa j4pr.PipeStackedT && !(tmp isa j4pr.PipeStackedF) && !(tmp isa j4pr.PipeStackedU)
tmp = A0 |> PP
Base.Test.@test tmp isa j4pr.PipeParallelT && !(tmp isa j4pr.PipeParallelF) && !(tmp isa j4pr.PipeParallelU)
tmp = [A0|>W1 A1|>W2] 
Base.Test.@test tmp isa j4pr.PipeParallelT && !(tmp isa j4pr.PipeParallelF) && !(tmp isa j4pr.PipeParallelU)
tmp = A0 |> PS
Base.Test.@test tmp isa j4pr.PipeSerialT && !(tmp isa j4pr.PipeSerialF) && !(tmp isa j4pr.PipeSerialU)
tmp = (A0|>W1)+(A0|>W2)
Base.Test.@test tmp isa j4pr.PipeSerialT && !(tmp isa j4pr.PipeSerialF) && !(tmp isa j4pr.PipeSerialU)
#println("PASSED")


# Generic pipe checks
#print("Type checks for [generic] pipes before passing data... ")
PTG = [W2+W1;W2;W3] 
PPG = [W2+W1 W2 W1] 
PSG = (W2+W1)+[W2;W1] 
Base.Test.@test PTG isa j4pr.PipeStacked && !(PTG isa j4pr.PipeParallel) && !(PTG isa j4pr.PipeSerial)
Base.Test.@test PPG isa j4pr.PipeParallel && !(PPG isa j4pr.PipeStacked) && !(PPG isa j4pr.PipeSerial)
Base.Test.@test PSG isa j4pr.PipeSerial && !(PSG isa j4pr.PipeStacked) && !(PSG isa j4pr.PipeParallel)
tmp = [[W1;W2;W3]+W3 W1 [W1 W2]] 
Base.Test.@test tmp isa j4pr.PipeParallel && !(tmp isa j4pr.PipeStacked) && !(tmp isa j4pr.PipeSerial)
#println("PASSED")

#print("Type checks for [generic] pipes before passing data... ")
Base.Test.@test A0 |> (A0 |> PTG) isa j4pr.CellDataU
Base.Test.@test A0 |> PPG isa j4pr.PipeParallel 
Base.Test.@test A0 |> (A0|> PPG)  isa j4pr.CellDataU
Base.Test.@test A0 |> (A0 |> PSG) isa j4pr.PipeSerial 
Base.Test.@test A0|> (A0 |> (A0 |> PSG)) isa j4pr.CellDataU 
#println("PASSED")

end
