# Network learning
module NetworkLearner
	
	using LearnBase, MLDataPattern, LightGraphs, SimpleWeightedGraphs
	#export ... 
	
	include("adjacency.jl") 								# Adjacency graph (`obtain_ad_graph`) 
	include("rlearners.jl")
	include("cinference.jl")

	abstract type AbstractNetworkLearner end
	
	# NetworkLearner for out of graph computations 
	mutable struct NetworkLearnerOModel{T,U,S,V,
				     	    R<:AbstractRelationalLearner,
					    C<:AbstractCollectiveInferer,
					    A<:Vector{<:AbstractAdjacency}} <: AbstractNetworkLearner 			
		Ml::T										# local model
		f_exec_local::U									# local model execution function
		Mr::S										# relational model
		f_exec_rel::V									# relational model execution function
		Rl::Type{R}										# relational learner
		Ci::Type{C}										# collective inferer	
		Adj::A										# adjacency information
	end
	
	# NetworkLearner for ingraph computations
	mutable struct NetworkLearnerIModel{T,U,S,V,R,C,A} <: AbstractNetworkLearner 			
		Ml::T										# local model
		f_exec_local::U									# local model execution function
		Mr::S										# relational model
		f_exec_rel::V									# relational model execution function
		Rl::R										# relational learner
		Ci::C										# collective inferer	
		Adj::A										# adjacency information
	end
	
	# Aliases

	# Printers
	Base.show(io::IO, m::NetworkLearnerOModel) = println("Network learner, out-of-graph computation")
	
	Base.dump(io::IO, m::NetworkLearnerOModel) = begin 
		println("Network learner, out-of-graph computation")
		print(io,"`- local model: $(m.lmodel)")
		print(io,"`- relational model: $(m.rmodel)")
		print(io,"`- relational learner: $(m.rlearner)")
		print(io,"`- collective inferer: $(m.ci)")
		print(io,"`- adjacency: $(m.A)")
	
	end


	
	#########################
	# Out-of-graph learning #
	#########################
	function fit(::Type{NetworkLearnerOModel}, X::T, y::S, Adj::A, Rl::R, Ci::C, f_train_local::U, f_exec_local::U2, f_train_rel::U3, f_exec_rel::U4; 
	      		normalize::Bool=true, use_local_data::Bool=true) where {T<:AbstractMatrix, 
	 			S<:AbstractArray, 
				A<:Vector{<:AbstractAdjacency}, 
				R<:Type{<:AbstractRelationalLearner}, 
				C<:Type{<:AbstractCollectiveInferer}, 
				U, U2, U3, U4 
			}
		# Step 0: pre-process input arguments and retrieve sizes
		
		# Function that calculates the number of  relational variables / each adjacency structure
		_width_rv_(y::AbstractVector{T}) where T<:Float64 = 1			# regression case
		_width_rv_(y::AbstractVector{T}) where T = length(unique(y))::Int	# classification case
		_width_rv_(y::AbstractArray) = error("Only vectors supported as targets in relational learning")

		n = nobs(X)						# number of observations
		c = _width_rv_(y)					# number of relational variables / adjacency
		m = c * length(Adj)					# number of relational variables

		# Pre-allocate relational variables array	
		if use_local_data					# Local observation variable data is used
			Xr = zeros(m+size(X,1), n)
			Xr[1:size(X,1),:] = X
			offset = size(X,1)					
		else							# Only relational variables are used
			Xr = zeros(m,n)				
			offset = 0
		end
		
		# Step 1: train and execute local model
		Dl = (X,y)
		Ml = f_train_local(Dl); 
		Xl = f_exec_local(Ml,X);
		
		# Step 2: Get relational variables 
		for (i,Ai) in enumerate(Adj)				# Loop through adjacency vector and
			ri = fit(Rl, Ai, Xl)				#  - train relational learner using adjacency information and local model output
			Xs = datasubset(Xr, 
		   			offset+(i-1)*c+1 : offset+i*c,	#  - subset of data to be filled
		   			ObsDim.Constant{1}()
			)	
			transform!(Xs, ri, Ai, Xl; normalize=normalize) #  - apply relational learner
		end
		
		# Step 3 : train relational model 
		Dr = (Xr,y)
		Mr = f_train_rel(Dr)

		# Step 4: remove adjacency data 
		sAdj = strip_adjacency.(Adj)	
		
		# Step 5: return network learner 
		return NetworkLearnerOModel(Ml, f_exec_local, Mr, f_exec_rel, Rl, Ci, sAdj)
	end
	
	# It may be necessary to add adjacency information to the model, regarding the test data

	# Execution methods i.e. transform(::AbstractNetworkLearner, X)
	# function transform(m::NetworkLearnerOModel)
		# Step 0: Make initializations and pre-allocations 	
		# Step 1: Apply local model, get initial estimates, decisions
		# Step 2: Apply collective inference
		# transform!(out, CI, R, A, X)
	# end

	#####################
	# In-graph learning #
	#####################
	# TODO: Generate hypotheses regarding on the variability of the network information, i.e.
	# - wether new vertices are added to the existing strucure or not, mutating the initial model
	# - wether new edges can be established if a transitory edge appeared at some point 
	# ...

end


