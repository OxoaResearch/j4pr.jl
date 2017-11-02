# j4pr Feature Roadmap 

## Data Processing
	* Unsupervised
		- Filter (data-based)						[OK][v.0.1.0]
		- Domain filtering 						[OK][v.0.1.0]
		- Scaling (data-based)						[OK][v.0.1.0]
		- Sampling (data-based)						[OK][v.0.1.0]
		- One-hot encoding (binary, integer)				[OK][v.0.1.0]
	
	* Supervised
		- Filter (class-based)						[OK][v.0.1.0]
		- Scaling (class-based)						[OK][v.0.1.0]
		- Sampling (class-based)					[OK][v.0.1.0]
		- Class slicing							[OK][v.0.1.0]
		- Label processing i.e. adding, removing 			[OK][v.0.1.0]
	
	* Data generation							
		- Various datasets, artificial and not so artificial		[OK][v.0.1.0]

## Unsupervised Learning
	* Transforms:	 
		- Whitening --> MultivariateStats.jl				[OK][v.0.1.0]
		- PCA --> MultivariateStats.jl					[OK][v.0.1.0] 
		- PPCA (probabilitstic PCA) --> MultivariateStats.jl		[OK][v.0.1.0]
		- KPCA (Kernel PCA) --> MultivariateStats.jl			[OK][v.0.1.1]
		- LDA/LDA subspace --> MultivariateStats.jl			[OK][v.0.1.0] 
		- ICA --> MultivariateStats.jl					[OK][v.0.1.0] 
		- MDS --> MultivariateStats.jl					[OK][v.0.1.0]
		- Distances/Proximity mapping --> Distances.jl			[OK][v.0.1.0] 
		- Factor Aralysis -->Multivariatestats.jl			[OK][v.0.1.1]
		- Kernels i.e. kernel trick --> j4pr.jl, MLKernels.jl		[OK][v.0.1.1]
		- Sammon transform --> ?					[Unknown version]
		- tSNE --> TSne.jl 						[Unknown version] 
	 	- Chernoff --> ?						[Unknown version]

	* Clustering:
		- hierarchical clustering --> Clustering.jl 			[Unknown version] 
		- k-means --> Clustering.jl					[OK][v.0.1.0] 
		- k-medoids --> Clustering.jl					[OK][v.0.1.0] 
		- fuzzy c-means --> Clustering.jl				[Unknown version]	
		- Affinity propagation --> Clustering.jl			[OK][v.0.1.0] 
		- DBSCAN --> Clustering.jl					[OK][v.0.1.0] 
		- k-centers --> ?						[Unknown version]
		- EM --> ?							[Unknown version]



## Supervised Learning
	* Basic classifiers:
		- kNN --> NearestNeighbours.jl + j4pr.jl			[OK][v.0.1.0]
		- SVM --> LIBSVM.jl						[OK][v.0.1.0] 
		- XGBoost --> XGBoost.jl					[Unknown version]
		- Naive Bayes --> j4pr.jl					[Unknown version]
		- Decision Stump -->j4pr.jl					[OK][v.0.1.1]
		- Decision Tree --> DecisionTree.jl				[OK][v.0.1.0] 
		- Random Forest --> DecisionTree.jl				[OK][v.0.1.0] 
		- Adaboosted stumps --> DecisionTree.jl				[OK][v.0.1.0] 
		- Perceptron --> j4pr.jl					[Unknown version]
		- Linear Bayes --> j4pr.jl					[OK][v.0.1.0]
		- Quadratic Bayes --> j4pr.jl					[OK][v.0.1.0]
		- Logistic Regression --> j4pr.jl				[Unknown version]
		- Neural Networks --> KNet.jl, Mocha.jl etc.			[Unknown version]
		- Parzen density estimation/classifier --> j4pr.jl		[OK][v.0.1.0]
		- RBF classification/regression/density estimation		[Unknown version]

	* Meta classifiers:
		- Combiners
			- voting -->j4pr.jl					[OK][v.0.1.0]
			- averaging -->j4pr.jl					[OK][v.0.1.0]
			- NaiveByes --> j4pr.jl					[OK][v.0.1.0]
			- DT, BKS -->j4pr.jl					[OK][v.0.1.1]
		- Boosting 
			- Adaboost M1 --> j4pr.jl				[OK][v.0.1.0]
			- Adaboost M2 --> j4pr.jl				[OK][v.0.1.0]
			- Arc-x4 --> j4pr.jl					[Unknown version]	
		- Gradient Boosting --> ?					[Unknown version]
		- Random subspace --> j4pr.jl					[OK][v.0.1.0]
		- Network classifiers --> j4pr.jl				[TODO][v.0.1.2]
		- Rotation Forest --> j4pr.jl					[Unknown version]	

	* Evaluation & Optimization 
		- cross-validation --> MLBase.jl?				[TODO][v.0.1.2?] 
		- loss-functions --> MLLabelUtils.jl, LossFunctions.jl		[OK][v.0.1.0]
		- ROC Analysis --> ROCAnalysis.jl+j4pr.jl			[OK][v.0.1.1]
		- Hyperparameter optimization --> MLBase.jl?			[TODO][v.0.1.2?]

	* Feature Selection
		- forward, backward, nested ,etc. --> j4pr.jl			[TO DO][v.0.1.2?]

## Regression
	* Basic Regression:
		- Linear --> MultivariateStats.jl				[OK][v.0.1.0]
		- Ridge --> MultivariateStats.jl				[OK][v.0.1.0]
		- LASSO --> Lasso.jl (builds on GLM.jl)				[Unknown version]
	 	- SVM --> LIBSVM.jl						[OK][v.0.1.0]
	 	- kNN --> NearestNeighbors.jl + j4pr.jl				[OK][v.0.1.0]	
		- Decision Stump -->j4pr.jl					[OK][v.0.1.1]
		- Decision Tree --> DecisionTree.jl				[OK][v.0.1.0]
		- Random Forest --> DecisionTree.jl				[OK][v.0.1.0]



## Data manipulation and I/O (video/audio/text/...)
	* Data I/O:
		- JuliaDB							[Unknown version]
		- For dataframes --> DataFramesIO.jl				[Unknown version]
		- Data loading --> [view the list](https://github.com/svaksha/Julia.jl/blob/master/IO.md) [Unknown version]
	
	* Analysis algorithms:
		- Video data + analysis (libav, ffmpeg)-->VideoIO.jl		[NOT NEEDED] 
	 	- Image analysis --> Images.jl and other			[DONE] 
	 	- Text Analysis --> TextAnalysis.jl				[NOT NEEDED] 
		- Signal analysis --> TimeSeries.jl, DSP.jl			[NOT NEEDED] 



## Feature creation
	* Word vectors : Word2Vec.jl, AdaGram.jl				[Unknown version (need to modify source code, wrapping not possible now)] 
	* Deep learning: Mocha.jl, Knet.jl					[Unknown version] 



## Tools
	* Image viewer --> ImageView.jl						[Unknown version] 
	* Image viewer in the console --> ImageInTerminal.jl			[OK][v.0.1.0] 
	* JSON parser --> JSON.jl						[Unknown version] 
	* Plotting --> UnicodePlots.jl						[OK][v.0.1.0] 
	* Video (creation+display) --> j4pr.jl 					[Unknown version]



# Longer term
	* Optimization								[Unknown version]
	* Reinforcement Learning						[Unknown version]
	* Online Learning							[Unknown version]
	* Streaming data processor						[Unknown version]
	* Asynchronous multiple pipeline handler				[Unknown version] 


# Resources
[General](http://ucidatascienceinitiative.github.io/IntroToJulia/)

[A.I](https://github.com/svaksha/Julia.jl/blob/master/AI.md)

[OnlineAI](https://github.com/tbreloff/OnlineAI.jl)

[I/O](https://github.com/svaksha/Julia.jl/blob/master/IO.md)
