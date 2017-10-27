# Module that exports a few functions that when called return DataCell datasets
module DataGenerator
	import j4pr;
	export iris, quakes, boston, normal2d, fish, fishr, fnoise;

	"""
		iris()
	
	DataCell version of the `iris` dataset. The `RDatasets` package is required. 
	"""
	# Iris dataset
	function iris()
		irisdf = j4pr.rdataset("iris");
		return j4pr.datacell(Array(irisdf[:,1:4])', Array(irisdf[:,5]), name="Iris Dataset")
	end



	"""
		quakes(thresh::Float64=5.0)
	
	DataCell version of the `quakes` dataset. The `thresh` parameter is used to threshold 
	a continuous variable, in this case the quake magnitude to obtain the labels 
	(e.g. 0 for < `thresh`, 1 >= `thresh`). The `RDatasets` package is required. 
	"""
	# Quakes dataset
	function quakes(thresh=5.0) 
		quakedf = j4pr.rdataset("quakes")
		labels = Array(quakedf[:,4].>=thresh)
		return j4pr.datacell(Array(quakedf[:,[1,2,3,5]])', labels, name="Quakes Dataset (threshold >=$thresh)")
	end



	"""
		boston()
	
	DataCell version of the `boston` dataset. The `RDatasets` package is required. 
	"""
	# Boston dataset
	function boston()
		bostondf = j4pr.rdataset("Boston")
		return j4pr.datacell(Array{typeof(1.0)}(bostondf[:,1:13])', Array(bostondf[:,14]), name="Boston Dataset")
	end



	"""
		normal2d(N::Int=100)
	
	Dataset containing two classes and two variables  with `N` samples (default `100`), normally distributed 
	so that the class distributions look perpendicular and overlapping; the mean vectors for each class 
	and variable vary a little each time a new dataset is generated.
	"""
	# Two gaussians
	function normal2d(N::Int=100)
		N0 = round(Int, N/2)
		N1 = N - N0
		X = [	randn(N0)+round.(rand([1,2,3])) randn(N0)+round.(rand([1,2,3])); 		# class 0
       			5*randn(N1)+round.(rand([4,5,8])) 0.1*randn(N1)+round.(rand([1,2,3])); 		# class 1  
		]'
		y = [zeros(N0);ones(N1)]
		order = shuffle(1:N)
		return j4pr.datacell(X[:,order], y[order], name="Two 2-D Gaussians")
	end
	


	"""
		fish(N::Int=100, noise::Float64=0.1)
	
	The `Fish dataset` is a 2-D dataset portraying a fish which represents one class, with the background as other class; noise
	can be added. `N` is the size of the grid (default `100`, e.g. grid of 100×100 points) and `noise` (default `0.1`) is the 
	fraction of samples with their labels flipped (e.g. noisy).

	Dataset based on the "Fish data" from `Kuncheva L. "Combining Pattern Classifiers", 2-nd Ed. ISBN 978-1-118-31523-1`
	"""
	# Fish 
	function fish(N::Int=100, noise::Float64=0.01)
		
		data = zeros(2,N^2)

		# generate the 2-D grid
		lab1 = BitVector(N*N);
		lab2 = BitVector(N*N);
		for (i,y) in enumerate(1:N), (j,x) in enumerate(1:N)
			xn = x/N; yn = y/N;
			lab1[(i-1)*N+j] = xn^3 -2xn*yn + 1.6yn^2 < 0.4
			lab2[(i-1)*N+j] = -xn^3 + 2*sin(xn)*yn + yn < 0.7
			data[1,(i-1)*N+j] = x	# x dimension 1'st variable
			data[2,(i-1)*N+j] = y	# y dimension 2'nd variable
		end
		l = xor.(lab1, lab2)
		nnoisy = round(Int, N^2*noise)
		if nnoisy > 0.0
			nidx = randperm(N^2)[1:nnoisy] # get indexes of noisy samples
			l[nidx]=.!l[nidx]
		end

		labels = map(x->x ? "fish" : "background", l)
		return j4pr.datacell(data, labels[:], name="Fish Dataset, classification")
	end



	"""
		fishr(N::Int=100, noise::Float64=0.1)
	
	Version of the `fish` dataset for regression problems. 

	Dataset based on the "Fish data" from `Kuncheva L. "Combining Pattern Classifiers", 2-nd Ed. ISBN 978-1-118-31523-1`
	"""
	function fishr(N::Int=100, noise::Float64=0.01)
		
		data = zeros(2,N^2)

		# generate the 2-D grid
		lab1 = falses(N*N);
		lab2 = falses(N*N);
		for (i,y) in enumerate(1:N), (j,x) in enumerate(1:N)
			xn = x/N; yn = y/N;
			lab1[(i-1)*N+j] = xn^3 -2xn*yn + 1.6yn^2 < 0.4
			lab2[(i-1)*N+j] = -xn^3 + 2*sin(xn)*yn + yn < 0.7
			data[1,(i-1)*N+j] = x	# x dimension 1'st variable
			data[2,(i-1)*N+j] = y	# y dimension 2'nd variable
		end
		l = xor.(lab1, lab2)
		nnoisy = round(Int, N^2*noise)
		labels = map(x->x ? 1.0 : 0.0, l)
		if nnoisy > 0.0
			nidx = randperm(N^2)[1:nnoisy] # get indexes of noisy samples
			labels[nidx]+=rand(nnoisy)
		end

		return j4pr.datacell(data, labels[:], name="Fish Dataset, regression")
	end



	"""
		fnoise(points, f, fnoise)
	
	Generates a dataset containing the `points` as data and the function `f` 
	applied to these points as values. Noise is added through `fnoise`. The
	full formula for data generation is `y = f.(points) + fnoise.(f.(points))`

	# Arguments
	  * `points::AbstractVector` is the vector of points
	  * `f` is a function applied to each point (default `sin`)
	  * `fnoise` is a function applied to each value of the function (default `x->x+rand()`)

	# Examples
	```
	```
	"""
	function fnoise(points::AbstractVector, f::Function=sin, fnoise::Function=x->x+rand())
		X = points
		y = f.(points)
		y = y + fnoise.(y)
		return j4pr.datacell(X,y, name="Function $f, with noise")
	end

end

