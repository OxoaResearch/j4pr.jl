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
Test.@test A01 isa j4pr.CellDataVec
Test.@test A02 isa j4pr.CellDataVecVec
Test.@test A03 isa j4pr.CellDataVecMat

Test.@test A04 isa j4pr.CellDataMat
Test.@test A05 isa j4pr.CellDataMatVec
Test.@test A06 isa j4pr.CellDataMatMat

Test.@test (A01 isa j4pr.CellDataU) && !(A01 isa j4pr.CellDataL) && !(A01 isa j4pr.CellDataLL) 
Test.@test (A04 isa j4pr.CellDataU) && !(A04 isa j4pr.CellDataL) && !(A04 isa j4pr.CellDataLL) 

Test.@test !(A02 isa j4pr.CellDataU) && (A02 isa j4pr.CellDataL) && !(A02 isa j4pr.CellDataLL) 
Test.@test !(A05 isa j4pr.CellDataU) && (A05 isa j4pr.CellDataL) && !(A05 isa j4pr.CellDataLL) 

Test.@test !(A03 isa j4pr.CellDataU) && !(A03 isa j4pr.CellDataL) && (A03 isa j4pr.CellDataLL) 
Test.@test !(A06 isa j4pr.CellDataU) && !(A06 isa j4pr.CellDataL) && (A06 isa j4pr.CellDataLL) 


# Test data/observation concatenation
v=[1,2,3]; m=rand(3,3)
Test.@test j4pr.dcat((nothing,nothing)) == nothing
Test.@test j4pr.dcat(nothing) == nothing
Test.@test j4pr.dcat(nothing,nothing) == nothing
Test.@test j4pr.dcat(nothing,v) == v
Test.@test j4pr.dcat(v,nothing) == v
Test.@test j4pr.dcat(v) == v
Test.@test j4pr.dcat(view(v,:)) == v
Test.@test j4pr.dcat(m) == m
Test.@test j4pr.dcat(view(m,:,:)) == m
Test.@test j4pr.dcat(v,v) == [v';v']
Test.@test j4pr.dcat(v,m) == [v';m]
Test.@test j4pr.dcat(m,v) == [m;v']
Test.@test j4pr.dcat(m,m) == [m;m]
Test.@test j4pr.dcat(m,m,v) == [m;m;v']


Test.@test j4pr.ocat((nothing,nothing)) == nothing
Test.@test j4pr.ocat(nothing) == nothing
Test.@test j4pr.ocat(nothing,nothing) == nothing
Test.@test j4pr.ocat(nothing,v) == v
Test.@test j4pr.ocat(v,nothing) == v
Test.@test j4pr.ocat(v) == v
Test.@test j4pr.ocat(view(v,:)) == v
Test.@test j4pr.ocat(m) == m
Test.@test j4pr.ocat(view(m,:,:)) == m
Test.@test j4pr.ocat(v,v) == [v;v]
Test.@test j4pr.ocat(v,m) == [v m]
Test.@test j4pr.ocat(m,v) == [m v]
Test.@test j4pr.ocat(m,m) == [m m]
Test.@test j4pr.ocat(m,m,v) == [m m v]


# Test iteration
Test.@test try
	for i in j4pr.datacell(m,v) i end
	true
catch
	false
end

end
