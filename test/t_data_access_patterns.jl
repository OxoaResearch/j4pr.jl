#= Testing data access patterns

The following methods will be evaluated:
 	_variable_
	varsubset
 	datasubset
	getx/getx!
	getindex 
	setindex
=#

function t_data_access_patterns()

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
AVv = [A01, A02, A03]
AVm = [A04, A05, A06]



# Data selection
# --------------

# Select observations
#println("Checking data selection...")
#print("\t|> Observations: datasubset - getindex - getx/getx!...")
for idx in [1,1:1,1:2,[1,2]]
	# Test for vector data
	for A in AVv
		Base.Test.@test all(+j4pr.datasubset(A,idx) .== +A[idx] .== j4pr.getx(A)[idx] .== j4pr.getx!(A)[idx])
	end
	
	# Test for matrix data
	for A in AVm
		Base.Test.@test all(+j4pr.datasubset(A,idx) .== +A[:,idx] .== j4pr.getx(A)[:,idx] .== j4pr.getx!(A)[:,idx])
	end
end
#println("PASSED")

# Select variables 
#print("\t|> Variables: varsubset - _variable_ - getindex - getx/getx!...")
for idx in [1,1:1,1:2,[1,2]]
	# Test for vector data
	for A in AVv
		if idx isa Int 	# idx has to be 1 otherwise most of the variable access methods fail, as they should; 
				# Note that there is no getindex method for A
			Base.Test.@test all(+j4pr.varsubset(A,idx) .== j4pr._variable_(A,idx) .== +A .== j4pr.getx(A)[:,idx] .== j4pr.getx!(A)[:,idx])
		end
	end
	
	# Test for matrix data
	for A in AVm
		if idx isa Int
			# data cell indexing returns a matrix data if single row selected, result must be transposed as all the other methods use vectors
			Base.Test.@test all(+j4pr.varsubset(A,idx) .== j4pr._variable_(A,idx) .== (+A[idx,:])' .== j4pr.getx(A)[idx,:] .== j4pr.getx!(A)[idx,:])
		else
			Base.Test.@test all(+j4pr.varsubset(A,idx) .== j4pr._variable_(A,idx) .== +A[idx,:] .== j4pr.getx(A)[idx,:] .== j4pr.getx!(A)[idx,:])
		end
	end
end
#println("PASSED")

# Selecting observations + variables is done by applying successively one of the methods for variable selection and observation selection 



# Data modification
# -----------------

# Modify observations
#println("Checking data modification...")
#print("\t|> Observations: datasubset - settindex - getx! and NOT getx...")
for idx in [1,1:1,1:2,[1,2]]
	# Test for vector data
	for A in AVv
		# datasubset approach
		Ac = deepcopy(A)
		newvals = idx isa Int ? rand() : rand(length(idx))
		j4pr.datasubset(Ac,idx)[:] = newvals;
		Base.Test.@test all(+j4pr.datasubset(Ac,idx) .== newvals)

		# setindex! approach
		Ac = deepcopy(A)
		newvals = idx isa Int ? rand() : rand(length(idx))
		Ac[idx] = newvals;
		Base.Test.@test all(+Ac[idx] .== newvals)

		# getx! approach
		Ac = deepcopy(A)
		newvals = idx isa Int ? rand() : rand(length(idx))
		j4pr.getx!(Ac)[idx] = newvals;
		Base.Test.@test all(j4pr.getx!(Ac)[idx] .== newvals)
		
		# getx approach (MUST NOT MODIDY)
		Ac = deepcopy(A)
		newvals = idx isa Int ? rand() : rand(length(idx))
		j4pr.getx(Ac)[idx] = newvals;
		Base.Test.@test all(j4pr.getx!(Ac)[idx] .!= newvals)
		Base.Test.@test all(j4pr.getx!(Ac)[idx] .== j4pr.getx!(A)[idx])
	end
	
	# Test for matrix data
	for A in AVm
		# datasubset approach
		Ac = deepcopy(A)
		newvals = rand(j4pr.nvars(Ac), length(idx))
		j4pr.datasubset(Ac,idx)[:] = newvals;
		Base.Test.@test all(+j4pr.datasubset(Ac,idx) .== newvals)

		# setindex! approach
		Ac = deepcopy(A)
		newvals = rand(j4pr.nvars(Ac), length(idx))
		Ac[:,idx] = newvals;
		Base.Test.@test all(+Ac[:,idx] .== newvals)

		# getx! approach
		Ac = deepcopy(A)
		newvals = rand(j4pr.nvars(Ac), length(idx))
		j4pr.getx!(Ac)[:,idx] = newvals;
		Base.Test.@test all(j4pr.getx!(Ac)[:,idx] .== newvals)
		
		# getx approach (MUST NOT MODIDY)
		Ac = deepcopy(A)
		newvals = rand(j4pr.nvars(Ac), length(idx))
		j4pr.getx(Ac)[:,idx] = newvals;
		Base.Test.@test all(j4pr.getx!(Ac)[:,idx] .!= newvals)
		Base.Test.@test all(j4pr.getx!(Ac)[:,idx] .== j4pr.getx!(A)[:,idx])
	end
end
#println("PASSED")

# Modify variables
#print("\t|> Variables: varsubset - _variable_ - settindex - getx! and NOT getx...")
for idx in [1,1:1,1:2,[1,2]]
	# Test for vector data
	for A in AVv
		if idx isa Int
		# varsubset approach
		Ac = deepcopy(A)
		newvals = rand(j4pr.nobs(A))
		j4pr.varsubset(Ac,idx)[:] = newvals;
		Base.Test.@test all(+j4pr.varsubset(Ac,idx) .== newvals)

		# _variable_ approach
		Ac = deepcopy(A)
		newvals = rand(j4pr.nobs(A))
		j4pr._variable_(Ac,idx)[:] = newvals;
		Base.Test.@test all(+j4pr._variable_(Ac,idx) .== newvals)
		
		# setindex! approach
		Ac = deepcopy(A)
		newvals = rand(j4pr.nobs(A))
		Ac[:] = newvals;
		Base.Test.@test all(+Ac .== newvals)

		# getx! approach
		Ac = deepcopy(A)
		newvals = rand(j4pr.nobs(A))
		j4pr.getx!(Ac)[:,idx] = newvals;
		Base.Test.@test all(j4pr.getx!(Ac)[:,idx] .== newvals)
		
		# getx approach (MUST NOT MODIDY)
		Ac = deepcopy(A)
		newvals = rand(j4pr.nobs(A))
		j4pr.getx(Ac)[:,idx] = newvals;
		Base.Test.@test all(j4pr.getx!(Ac)[:,idx] .!= newvals)
		Base.Test.@test all(j4pr.getx!(Ac)[:,idx] .== j4pr.getx!(A)[:,idx])
		end
	end
	# Test for matrix data
	for A in AVm
		# varsubset approach
		Ac = deepcopy(A)
		newvals = rand(size(j4pr.varsubset(Ac,idx)[:]))
		j4pr.varsubset(Ac,idx)[:] = newvals;
		Base.Test.@test all(+j4pr.varsubset(Ac,idx) .== newvals)

		# _variable_ approach
		Ac = deepcopy(A)
		newvals = rand(size(j4pr._variable_(Ac,idx)))
		j4pr._variable_(Ac,idx)[:] = newvals;
		Base.Test.@test all(+j4pr._variable_(Ac,idx) .== newvals)
		
		# setindex! approach
		Ac = deepcopy(A)
		newvals = rand(size(Ac[idx,:]))
		Ac[idx,:] = newvals;
		Base.Test.@test all(+Ac[idx,:] .== newvals)

		# getx! approach
		Ac = deepcopy(A)
		newvals = rand(size(j4pr.getx!(Ac)[idx,:]))
		j4pr.getx!(Ac)[idx,:] = newvals;
		Base.Test.@test all(j4pr.getx!(Ac)[idx,:] .== newvals)
		
		# getx approach (MUST NOT MODIDY)
		Ac = deepcopy(A)
		newvals = rand(size(j4pr.getx!(Ac)[idx,:]))
		j4pr.getx(Ac)[idx,:] = newvals;
		Base.Test.@test all(j4pr.getx!(Ac)[idx,:] .!= newvals)
		Base.Test.@test all(j4pr.getx!(Ac)[idx,:] .== j4pr.getx!(A)[idx,:])
	end
end
#println("PASSED")

end
