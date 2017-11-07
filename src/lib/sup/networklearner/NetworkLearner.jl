# Network learning
module NetworkLearner
	
	using LearnBase, MLDataPattern, LightGraphs, SimpleWeightedGraphs
	
	export AbstractNetworkLearner,
		NetworkLearnerOutOfGraph,
		NetworkLearnerInGraph,
		fit, 
		transform, transform!, 
		add_adjacency!
	
	abstract type AbstractNetworkLearner end
	
	include("adjacency.jl") 								# Adjacency-related structures 
	include("rlearners.jl")									# Relational learners
	include("cinference.jl")								# Collective inference algorithms		
	include("utils.jl")
	include("outlearning.jl")								# Out-of-graph learning
	include("inlearning.jl")								# In-graph learning

end


