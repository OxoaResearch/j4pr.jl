# Network learning
module NetworkLearner
	
	using LightGraphs, SimpleWeightGraphs
	#export ... 
	
	include("adjacency.jl") 								# Adjacency graph (`obtain_ad_graph`) 
	include("rlearners.jl")
	include("cinference.jl")

	abstract type AbstractNetworkLearner end
	
	# NetworkLearner for out of graph computations 
	mutable struct NetworkLearnerOModel{T,S,R,C,U} <: AbstractNetworkLearner 			
		lmodel::T									# local model
		rmodel::S									# relational model
		rlearner::R									# relational learner
		ci::C										# collective inferer	
		A::U										# adjacency information
	end
	
	# NetworkLearner for ingraph computations
	mutable struct NetworkLearnerIModel{T,S,R,C,U} <: AbstractNetworkLearner 			
		lmodel::T									# local model
		rmodel::S									# relational model
		rlearner::R									# relational learner
		ci::C										# collective inferer	
		A::U										# adjacency information
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
		
	end
	#########################
	# Out-of-graph learning #
	#########################

	# X - input data, A - list of adjacency strucures, fa - local model, fb - relational model, R - relational learner, CI - collective inferer)
	# function fit(::Type{NetworkLearnerOModel}, X, A, fa, fb, R, CI;kwargs...)
		# Step 0 : pre-process training arguments and pre-allocate relational model output Xr 
		# Step 1 : train local model
			# Ml = fa(X); 
			# Xl = fit(Ml, X)	   
		# Step 2 : for each element in A, Ai
			# obtain adjacency graph
			# transform!(Xr, R, Ai, X; normalize=true)
		# Step 3 : train relational model 
			# if use_local_data
			# 	Mr = fb([X;Xl])
			# else
			#	Mr = fb(Xl)
			# end

		# Step pre-4: analyze A and see what to save from it: either nothing or a list of functions, etc.
		# Step 4: return NetworkLearnerOModel(Ml, Mr, R, CI, A) 
	# end
	
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


