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
		fl_exec::U									# local model execution function
		Mr::S										# relational model
		fr_exec::V									# relational model execution function
		Rl::Type{R}									# relational learner
		Ci::Type{C}									# collective inferer	
		Adj::A										# adjacency information
		use_local_data::Bool								# whether to use local data
		# TODO: Remove fields below, move to relation rl, ci objects	
		# TODO: Change fields from Type{T} to T i.e. NetworkLearner uses instances not types, 
		#  this implies changing the the fit methods as well.
		m::Int										# number of relational variables
		c::Int										# number of relational variables / adjacency
		priors::Vector{Float64}								# class priors
		normalize::Bool									# whether to normalize local estimates for the relational learners 
		f_targets::Function								# function employed to obtain decisions
	end
	
	
	
	# Aliases
	#const NetworkLearnerOModelEmptyAdj{T,U,S,V,R,C,A<:Vector{<:EmptyAdjacency}} = NetworkLearnerOModel{T,U,S,V,R,C,A}
	#const NetworkLearnerOModelPartialAdj{T,U,S,V,R,C,A<:Vector{<:PartialAdjacency}} = NetworkLearnerOModel{T,U,S,V,R,C,A}
	


	# Printers
	#Base.show(io::IO, m::NetworkLearnerOModel) = println("Network learner, out-of-graph computation")
	
	Base.show(io::IO, m::NetworkLearnerOModel) = begin 
		println("Network learner, out-of-graph, $(m.m) relational variables, $(m.c) adjacencies")
		print(io,"`- local model: "); println(io, m.Ml)
		print(io,"`- relational model: "); println(io, m.Mr)
		print(io,"`- relational learner: "); println(io, m.Rl)
		print(io,"`- collective inferer: "); println(io, m.Ci)
		print(io,"`- adjacency: "); println(io, m.Adj)	
		print(io,"`- priors: "); println(io, m.priors)	
		println(io,"`- use local data: $(m.use_local_data), normalize: $(m.normalize)");
	end


	# Function that calculates the number of  relational variables / each adjacency structure
	get_width_rv(y::AbstractVector{T}) where T<:Float64 = 1			# regression case
	get_width_rv(y::AbstractVector{T}) where T = length(unique(y))::Int	# classification case
	get_width_rv(y::AbstractArray) = error("Only vectors supported as targets in relational learning.")

	# Function that calculates the priors of the dataset
	getpriors(y::AbstractVector{T}) where T<:Float64 = [1.0]	
	getpriors(y::AbstractVector{T}) where T = [sum(yi.==y)/length(y) for yi in sort(unique(y))]
	getpriors(y::AbstractArray) = error("Only vectors supported as targets in relational learning.")



	#########################
	# Out-of-graph learning #
	#########################
	function fit(::Type{NetworkLearnerOModel}, X::AbstractMatrix, y::AbstractArray, Adj::A where A<:Vector{<:AbstractAdjacency}, 
	      		fl_train, fl_exec, fr_train, fr_exec; 
	      		priors::Vector{Float64}=getpriors(y), learner::Symbol=:wvrn, inference::Symbol=:rl, 
			normalize::Bool=true, use_local_data::Bool=true, f_targets::Function=x->targets(indmax,x), 
			tol::Float64=1e-6, κ::Float64=1.0, α::Float64=0.99, maxiter::Int=1000) 

		# TODO: Write argument parsing, add aditional keyword arguments ...

	end
	
	
	function fit(::Type{NetworkLearnerOModel}, X::T, y::S, Adj::A, Rl::R, Ci::C, fl_train::U, fl_exec::U2, fr_train::U3, fr_exec::U4, 
	      		priors::Vector{Float64}=getpriors(y); normalize::Bool=true, use_local_data::Bool=true, f_targets::Function=x->targets(indmax,x)) where {
				T<:AbstractMatrix, 
	 			S<:AbstractArray, 
				A<:Vector{<:AbstractAdjacency}, 
				R<:Type{<:AbstractRelationalLearner}, 
				C<:Type{<:AbstractCollectiveInferer}, 
				U, U2, U3, U4 
			}
		
		# Step 0: pre-process input arguments and retrieve sizes
		n = nobs(X)									# number of observations
		c = get_width_rv(y)								# number of relational variables / adjacency
		m = c * length(Adj)								# number of relational variables

		@assert c == length(priors) "Found $c classes, the priors indicate $(length(priors))."
		
		# Pre-allocate relational variables array	
		if use_local_data								# Local observation variable data is used
			Xr = zeros(m+size(X,1), n)
			Xr[1:size(X,1),:] = X
			offset = size(X,1)					
		else										# Only relational variables are used
			Xr = zeros(m,n)				
			offset = 0
		end
		
		# Step 1: train and execute local model
		Dl = (X,y)
		Ml = fl_train(Dl); 
		Xl = fl_exec(Ml,X);
		
		# Step 2: Get relational variables by training and executing the relational learner 
		for (i,Ai) in enumerate(Adj)		
			
			# Train relational learner using adjacency information and local model output
			ri = fit(Rl, Ai, Xl)				

			# Get subset from the output where the relational data will go
			Xs = datasubset(Xr, offset+(i-1)*c+1 : offset+i*c, ObsDim.Constant{1}())	
			
			# Apply relational learner
			transform!(Xs, ri, Ai, Xl, priors; normalize=normalize) 
		end
		
		# Step 3 : train relational model 
		Dr = (Xr,y)
		Mr = fr_train(Dr)

		# Step 4: remove adjacency data 
		sAdj = AbstractAdjacency[];
		for i in 1:length(Adj)
			push!(sAdj, strip_adjacency(Adj[i]))	
		end

		# Step 5: return network learner 
		return NetworkLearnerOModel(Ml, fl_exec, Mr, fr_exec, Rl, Ci, sAdj, m, c, 
			      			priors, use_local_data, normalize, f_targets)
	end
	

	# Execution methods 
	function transform(m::NetworkLearnerOModel, X::T) where T<:AbstractMatrix
		# Step 0: Make initializations and pre-allocations 	
		C = nobs(m.priors)								# number of output variables i.e. estimates
		m = size(X,1)
		n = nobs(X)									# number of observations
		out = zeros(C,n)
		
		# Pre-allocate relational dataset
		if m.use_local_data
			Xr = zeros(m.m+m, n)							# relational variables number + local variable number
			Xr[1:m,:] = X								# allocate current data to relational dataset	
			offset = m
		else										# Only relational variables are used
			Xr = zeros(m.m,n)				
			offset = 0
		end

		# Step 1: Apply local model, get initial estimates, decisions
		Xl = m.fl_exec(m.Ml, X)
		
		# Step 2: Apply collective inference
		transform!(out, m.Ci, m.Rl, m.Adj, X)	
	     	# transform!(out, CI, R, A, X)
		
	     	# Step 3: Return output estimates
		return out
	end



	# It may be necessary to add adjacency information to the model, regarding the test data
	function add_adjacency!(m::T, A::Vector{S}) where {T<:NetworkLearnerOModel, S<:AbstractAdjacency}
		@assert length(A) == length(m.Adj) "New adjacency vector must have a length of $(length(m.Adj))."
		m.Adj = A				
	end
		
	function add_adjacency!(m::T, A::Vector{S}) where {T<:NetworkLearnerOModel, S}
		@assert length(A) == length(m.Adj) "Adjacency data vector must have a length of $(length(m.Adj))."
		m.Adj = adjacency.(A)
	end






	#####################
	# In-graph learning #
	#####################
	# TODO: Generate hypotheses regarding on the variability of the network information, i.e.
	# - wether new vertices are added to the existing strucure or not, mutating the initial model
	# - wether new edges can be established if a transitory edge appeared at some point 
	# ...

	# NetworkLearner for ingraph computations
	mutable struct NetworkLearnerIModel{T,U,S,V,
				     	    R<:AbstractRelationalLearner,
					    C<:AbstractCollectiveInferer,
					    A<:Vector{<:AbstractAdjacency}} <: AbstractNetworkLearner 			
		Ml::T										# local model
		fl_exec::U									# local model execution function
		Mr::S										# relational model
		fr_exec::V									# relational model execution function
		Rl::Type{R}									# relational learner
		Ci::Type{C}									# collective inferer	
		Adj::A										# adjacency information
	end
	
end


