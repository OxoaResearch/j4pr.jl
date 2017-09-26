# j4pr Features Roadmap 

## Unsupervised Learning
	* Transforms:	 
		- Whitening --> MultivariateStats.jl				[OK]
		- PCA --> MultivariateStats.jl					[OK] 
		- PPCA (probabilitstic PCA) --> MultivariateStats.jl		[OK]
		- LDA/LDA subspace --> MultivariateStats.jl			[OK] 
		- ICA --> MultivariateStats.jl					[OK] 
		- MDS --> MultivariateStats.jl					[OK]
		- Distances/Proximity mapping --> Distances.jl			[OK] 
		- Sammon transform --> ?					[Unknown version]
		- tSNE --> TSne.jl 						[Unknown version] 
	 	- Chernoff --> ?						[Unknown version]
		- Other (e.g. sigmoid,etc.) --> NumericFuns.jl, NumericExtensions.jl [Unknown version] 

	* Clustering:
		- hierarchical clustering --> Clustering.jl 			[Unknown version] 
		- k-means --> Clustering.jl					[OK] 
		- k-medoids --> Clustering.jl					[OK] 
		- fuzzy c-means --> Clustering.jl				[Unknown version]	
		- Affinity propagation --> Clustering.jl			[OK] 
		- DBSCAN --> Clustering.jl					[OK] 
		- k-centers --> ?						[Unknown version]
		- EM --> ?							[Unknown version]



## Supervised Learning
	* Basic classifiers:
		- kNN --> NearestNeighbours.jl + j4pr.jl			[OK]
		- SVM --> LIBSVM.jl						[OK] 
		- XGBoost --> XGBoost.jl					[Unknown version]
		- Naive Bayes --> j4pr.jl					[Unknown version]
		- Decision Stump -->j4pr.jl					[Unknown version]
		- Decision Tree --> DecisionTree.jl				[OK] 
		- Random Forest --> DecisionTree.jl				[OK] 
		- Adaboosted stumps --> DecisionTree.jl				[OK] 
		- Perceptron --> j4pr.jl					[Unknown version]
		- Linear Bayes --> j4pr.jl					[OK]
		- Quadratic Bayes --> j4pr.jl					[OK]
		- Logistic Regression --> j4pr.jl				[Unknown version]
		- Neural Networks --> KNet.jl, Mocha.jl etc.			[Unknown version]
		- Parzen density estimation/classifier --> j4pr.jl		[OK]
		- RBF classification/regression/density estimation		[Unknown version]

	* Meta classifiers:
		- Combiners --> j4pr.jl						[OK]
		- Ada-Boosting --> j4pr.jl					[OK]
		- Gradient Boosting --> ?					[Unknown version]
		- Random subspace --> j4pr.jl					[OK]
		- Network classifiers --> j4pr.jl				[Unknown version]
		- Rotation Forest --> j4pr.jl					[Unknown version]	

	* Evaluation 
		- cross-validation --> j4pr.jl					[Unknwon version]
		- loss-functions --> MLLabelUtils.jl, LossFunctions.jl		[OK]

	* Feature Selection
		- forward, backward, nested ,etc. --> j4pr.jl			[Unknown version]

## Regression
	* Basic Regression:
		- Linear --> MultivariateStats.jl				[OK]
		- Ridge --> MultivariateStats.jl				[OK]
		- Partial LS --> MultivariateStats.jl (future proposal)		[Unknown version]
		- LASSO --> Lasso.jl (builds on GLM.jl)				[Unknown version]
	 	- SVM --> LIBSVM.jl						[OK]
	 	- kNN --> NearestNeighbors.jl + j4pr.jl				[OK]	
		- Decision Tree --> DecisionTree.jl				[OK]
		- Random Forest --> DecisionTree.jl				[OK]



## Graph algorithms
	* Investigate LightGraphs.jl, LightGraphsExtras.jl			[Unknown version]


	
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
	* Image viewer in the console --> ImageInTerminal.jl			[OK] 
	* JSON parser --> JSON.jl						[Unknown version] 
	* Plotting --> UnicodePlots.jl						[OK] 
	* Video (creation+display) --> j4pr.jl 					[Unknown version]



# Longer term
	* Continue work on combiners: BKS, decision templates,			[Unknown version]
	  weighted voting (regression training) for continuous outputs
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
