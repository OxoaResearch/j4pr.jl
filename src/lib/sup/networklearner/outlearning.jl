# NetworkLearner for out of graph computations 
mutable struct NetworkLearnerOutOfGraph{T,U,S,V,
				    R<:Vector{<:AbstractRelationalLearner},
				    C<:AbstractCollectiveInferer,
				    A<:Vector{<:AbstractAdjacency},
				    L<:Union{Void, MLLabelUtils.LabelEncoding}} <: AbstractNetworkLearner 			 
	Ml::T											# local model
	fl_exec::U										# local model execution function
	Mr::S											# relational model
	fr_exec::V										# relational model execution function
	RL::R											# relational learner
	Ci::C											# collective inferer	
	Adj::A											# adjacency information
	use_local_data::Bool									# whether to use local data
	target_enc::L										# target encoding
	size_in::Int										# expected input dimensionality
	size_out::Int										# expected output dimensionality
end



# Aliases
#const NetworkLearnerOutOfGraphEmptyAdj{T,U,S,V,R,C,A<:Vector{<:EmptyAdjacency}} = NetworkLearnerOutOfGraph{T,U,S,V,R,C,A}
#const NetworkLearnerOutOfGraphPartialAdj{T,U,S,V,R,C,A<:Vector{<:PartialAdjacency}} = NetworkLearnerOutOfGraph{T,U,S,V,R,C,A}



# Printers
#Base.show(io::IO, m::NetworkLearnerOutOfGraph) = println("Network learner, out-of-graph computation")

Base.show(io::IO, m::NetworkLearnerOutOfGraph) = begin 
	println("NetworkLearner, $(m.size_in)×$(m.size_out), out-of-graph, $(length(m.Adj)) adjacencies")
	print(io,"`- local model: "); println(io, m.Ml)
	print(io,"`- relational model: "); println(io, m.Mr)
	print(io,"`- relational learners: "); println(io, m.RL)
	print(io,"`- collective inferer: "); println(io, m.Ci)
	print(io,"`- adjacency: "); println(io, m.Adj)	
	println(io,"`- use local data: $(m.use_local_data)");
	println(io,"`- targets: $(m.target_enc isa Void ? "not encoded" : "encoded")");
end



####################
# Training methods #
####################
function fit(::Type{NetworkLearnerOutOfGraph}, X::AbstractMatrix, y::AbstractArray, Adj::A where A<:Vector{<:AbstractAdjacency}, 
		fl_train, fl_exec, fr_train, fr_exec; 
		priors::Vector{Float64}=getpriors(y), learner::Symbol=:wvrn, inference::Symbol=:rl, 
		normalize::Bool=true, use_local_data::Bool=true, f_targets::Function=x->targets(indmax,x), 
		tol::Float64=1e-6, κ::Float64=1.0, α::Float64=0.99, maxiter::Int=100, bratio::Float64=0.1) 

	# Parse, transform input arguments
	κ = clamp(κ, 1e-6, 1.0)
	α = clamp(α, 1e-6, 1.0-1e-6)
	tol = clamp(tol, 0.0, Inf)
	maxiter = ifelse(maxiter<=0, 1, maxiter)
	bratio = clamp(bratio, 1e-6, 1.0-1e-6)
	@assert all((priors.>=0.0) .& (priors .<=1.0)) "All priors have to be between 0.0 and 1.0."
	
	# Parse relational learner argument and generate relational learner type
	if learner == :rn
		Rl = SimpleRN
	elseif learner == :wrn
		Rl = WeightedRN
	elseif learner == :cdrn 
		Rl = ClassDistributionRN
	elseif learner == :bayesrn
		Rl = BayesRN
	else
		warn("Unknown relational learner. Defaulting to :wrn.")
		Rl = WeightedRN
	end

	# Parse collective inference argument and generate collective inference objects
	if inference == :rl
		Ci = RelaxationLabelingInferer(maxiter, tol, f_targets, κ, α)
	elseif inference == :ic
		Ci = IterativeClassificationInferer(maxiter, tol, f_targets)
	elseif inference == :gs
		Ci = GibbsSamplingInferer(maxiter, tol, f_targets, ceil(Int, maxiter*bratio))
	else
		warn("Unknown collective inferer. Defaulting to :rl.")
		Ci = RelaxationLabelingInferer(maxiter, tol, f_targets, κ, α)
	end
	
	fit(NetworkLearnerOutOfGraph, X, y, Adj, Rl, Ci, fl_train, fl_exec, fr_train, fr_exec; 
		priors=priors, normalize=normalize, use_local_data=use_local_data)
end



function fit(::Type{NetworkLearnerOutOfGraph}, X::T, y::S, Adj::A, Rl::R, Ci::C, fl_train::U, fl_exec::U2, fr_train::U3, fr_exec::U4; 
		priors::Vector{Float64}=getpriors(y), normalize::Bool=true, use_local_data::Bool=true) where {
			T<:AbstractMatrix, 
			S<:AbstractArray, 
			A<:Vector{<:AbstractAdjacency}, 
			R<:Type{<:AbstractRelationalLearner}, 
			C<:AbstractCollectiveInferer, 
			U, U2, U3, U4 
		}
	
	# Step 0: pre-process input arguments and retrieve sizes
	size_in = size(X,1)									# number of local variables
	size_out = get_size_out(y)								# number of relational variables / adjacency
	n = nobs(X)										# number of observations
	m = length(Adj) * size_out								# number of relational variables

	@assert size_out == length(priors) "Found $c classes, the priors indicate $(length(priors))."
	
	# Pre-allocate relational variables array	
	if use_local_data									# Local observation variable data is used
		Xr = zeros(size_in+m, n)
		Xr[1:size_in,:] = X
		offset = size_in					
	else											# Only relational variables are used
		Xr = zeros(m,n)				
		offset = 0
	end
	
	# Encode targets
	(t_enc, yₑ) = encode_targets(y)


	# Step 1: train and execute local model
	Dl = (X,yₑ)
	Ml = fl_train(Dl); 
	Xl = fl_exec(Ml,X);
	

	# Step 2: Get relational variables by training and executing the relational learner 
	RL = [fit(Rl, Aᵢ, Xl, yₑ; priors=priors, normalize=normalize) for Aᵢ in Adj]		# Train relational learners				

	Xrᵢ = zeros(size_out,n)									# Initialize temporary storage	
	for (i,(RLᵢ,Aᵢ)) in enumerate(zip(RL,Adj))		
		
		# Apply relational learner
		transform!(Xrᵢ, RLᵢ, Aᵢ, Xl, yₑ)

		# Update relational data output		
		Xr[offset+(i-1)*size_out+1 : offset+i*size_out, :] = Xrᵢ									
	end
	

	# Step 3 : train relational model 
	Mr = fr_train((Xr,yₑ))

	# Step 4: remove adjacency data 
	Adj_s = AbstractAdjacency[];
	for i in 1:length(Adj)
		push!(Adj_s, strip_adjacency(Adj[i]))	
	end


	# Step 5: return network learner 
	return NetworkLearnerOutOfGraph(Ml, fl_exec, Mr, fr_exec, RL, Ci, Adj_s, use_local_data, t_enc, size_in, size_out)
end


#####################
# Execution methods #
#####################
function transform(model::M, X::T) where {M<:NetworkLearnerOutOfGraph, T<:AbstractMatrix}
	Xo = zeros(model.size_out, nobs(X))
	transform!(Xo, model, X)
	return Xo
end

function transform!(Xo::S, model::M, X::T) where {M<:NetworkLearnerOutOfGraph, T<:AbstractMatrix, S<:AbstractMatrix}
	
	# Step 0: Make initializations and pre-allocations 	
	m = size(X,1)										# number of input variables
	n = nobs(X)										# number of observations
	l = length(model.Adj)									# number of adjacencies in the model

	@assert model.size_in == m "Expected input dimensionality $(model.size_in), got $m."
	@assert size(Xo) == (model.size_out, n) "Output dataset size must be $(model.size_out)×$n."
	
	if model.use_local_data									# Pre-allocate relational dataset based on local data use
		Xr = zeros(model.size_out*l+m, n)						#  - dimensions = relational variables number + local variable number
		Xr[1:m,:] = X									#  - initialize (local) dimensions	
		offset = m
	else											# Skip local dataset dimensions
		Xr = zeros(model.size_out*l, n)				
		offset = 0
	end


	# Step 1: Apply local model, initialize output
	Xl = model.fl_exec(model.Ml, X)
	@assert size(Xo) == size(Xl) "Local model output size is $(size(Xl)) and NetworkLearner expected output size $(size(Xo))." 	
	Xo[:] = Xl 


	# Step 2: Apply collective inference
	transform!(Xo, model.Ci, model.Mr, model.fr_exec, model.RL, model.Adj, offset, Xr)	
	

	# Step 3: Return output estimates
	return Xo
end



# It may be necessary to add adjacency information to the model, regarding the test data
function add_adjacency!(model::M, Av::Vector{T}) where {M<:NetworkLearnerOutOfGraph, T<:AbstractAdjacency}
	@assert length(Av) == length(model.Adj) "New adjacency vector must have a length of $(length(model.Adj))."
	model.Adj = Av				
end
	
function add_adjacency!(model::M, Av::Vector{T}) where {M<:NetworkLearnerOutOfGraph, T}
	@assert length(Av) == length(model.Adj) "Adjacency data vector must have a length of $(length(model.Adj))."
	model.Adj = adjacency.(Av)
end
