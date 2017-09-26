function t_interfaces()

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
AV = [A01, A02, A03, A04, A05, A06]

# Simple getindex type stability checks
#print("Checking getindex type stability for data cells... ")

# Index for ::Void and empty index
Base.Test.@test nothing[1] isa Void
Base.Test.@test nothing[:,1] isa Void
Base.Test.@test nothing[:,[1,2]] isa Void
Base.Test.@test nothing[:,1:2] isa Void

# Index with single integer (works for all data types e.g. vector and matrix)
for A in AV
	Base.Test.@test A[5] isa typeof(A)
end

# Integer + Colon (works only for matrix data) 
for A in AV[4:end]
	Base.Test.@test A[5,:] isa typeof(A)
end

# Colon + Integer (works only for matrix data)
for A in AV[4:end]
	Base.Test.@test A[:,5] isa typeof(A)
end

# Range/Vector (works only for vector data)
for A in AV[1:3]
	Base.Test.@test A[[1,2,3]] isa typeof(A)
	Base.Test.@test A[1:3] isa typeof(A)
end

# Range/Vector + Colon (works only for matrix data)
for A in AV[4:end]
	Base.Test.@test A[[1,2,3],:] isa typeof(A)
	Base.Test.@test A[1:3,:] isa typeof(A)
end

# Colon + Range/Vector (works only for matrix data)
for A in AV[4:end]
	Base.Test.@test A[:,[1,2,3]] isa typeof(A)
	Base.Test.@test A[:,1:3] isa typeof(A)
end
# Range/Vector + Range Vector (works only for matrix data)
for A in AV[4:end]
	Base.Test.@test A[[1,2,3],[1,2,3]] isa typeof(A)
	Base.Test.@test A[1:3,[1,2,3]] isa typeof(A)
	Base.Test.@test A[1:3,[1,2,3]] isa typeof(A)
	Base.Test.@test A[1:3,1:3] isa typeof(A)
end
#println("PASSED")



# Simple getindex type stability checks
#print("Checking setindex! for vector and matrix data... ")

Xvec = rand(3)
Xmat = rand(2,2)

for A in AV[1:3]
	A[1:length(Xvec)] = Xvec
	Base.Test.@test A.x[1:length(Xvec)] == Xvec
end

for A in AV[4:end]
	A[1:size(Xmat,1),1:size(Xmat,2)] = Xmat
	Base.Test.@test A.x[1:size(Xmat,1),1:size(Xmat,2)]  == Xmat
end
#println("PASSED")



# Simple pipe iteration checks
W1 = j4pr.ohenc("binary")
W2 = j4pr.scaler!(Dict(1=>"mean"))
W3 = j4pr.functioncell((x)->x)

PT = [W1;W2]
PP=[W1 W2]
PS = W1+W2

PC = [PT [PS;PP]]

#print("Checking pipe iteration... ")
for P in [PT, PP, PS]
	i = 1
	for W in P
		Base.Test.@test W == P.x[i]
		i+=1
	end
end

for (i,P) in enumerate(PC) 
	Base.Test.@test P == PC.x[i] 
end
#println("PASSED")

end
