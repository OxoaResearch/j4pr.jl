0.1.2 / TBD 
==================
  * 

0.1.1 / 2017-10-27 
==================
  * Moved code relating to different types in separate files
  * Changed model properties from Dict to ModelProperties
  * Wrapped MLKernels.jl for efficient kernel matrix calculation
  * Added support for generating custom kernels matrices i.e. kernel trick, less efficient 
  * Added decision stump classifier, regressor
  * Added DT and BKS label combiners
  * Added support for operating point search and creation
  * Added an ROC plot, confusion matrix  
  * Wrapped kernel PCA, disabled for now (requires latest MultivariateStats.jl)
  * Wrapped Factor Analysis, disabled for now (requires latest MultivariateStats.jl)
  * Various bugfixes, improved unit test coverage

0.1.0 / 2017-09-24
==================
  * Initial commit
