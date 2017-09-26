function t_data_cells()

N = 10	# variables
M = 25  # samples

vectordata = rand(M)
matrixdata = rand(N,M)
labels = round.(2*rand(M))
multilabels = round.(2*rand(2,M))

A01 = j4pr.datacell(vectordata)
A02 = j4pr.datacell(vectordata, labels)
A03 = j4pr.datacell(vectordata, multilabels)
A04 = j4pr.datacell(matrixdata)
A05 = j4pr.datacell(matrixdata, labels)
A06 = j4pr.datacell(matrixdata, multilabels)

# Simple cell checks
#print("Checking data cell generated through their aliases ... ")
Base.Test.@test A01 isa j4pr.CellDataVec
Base.Test.@test A02 isa j4pr.CellDataVecVec
Base.Test.@test A03 isa j4pr.CellDataVecMat

Base.Test.@test A04 isa j4pr.CellDataMat
Base.Test.@test A05 isa j4pr.CellDataMatVec
Base.Test.@test A06 isa j4pr.CellDataMatMat

Base.Test.@test (A01 isa j4pr.CellDataU) && !(A01 isa j4pr.CellDataL) && !(A01 isa j4pr.CellDataLL) 
Base.Test.@test (A04 isa j4pr.CellDataU) && !(A04 isa j4pr.CellDataL) && !(A04 isa j4pr.CellDataLL) 

Base.Test.@test !(A02 isa j4pr.CellDataU) && (A02 isa j4pr.CellDataL) && !(A02 isa j4pr.CellDataLL) 
Base.Test.@test !(A05 isa j4pr.CellDataU) && (A05 isa j4pr.CellDataL) && !(A05 isa j4pr.CellDataLL) 

Base.Test.@test !(A03 isa j4pr.CellDataU) && !(A03 isa j4pr.CellDataL) && (A03 isa j4pr.CellDataLL) 
Base.Test.@test !(A06 isa j4pr.CellDataU) && !(A06 isa j4pr.CellDataL) && (A06 isa j4pr.CellDataLL) 
#println("PASSED")

end
