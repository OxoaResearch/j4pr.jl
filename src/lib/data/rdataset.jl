"""
	rdataset()
	rdataset(dataset)
	rdataset(package,dataset)

Wrapper for RDatasets dataset generation. At some point it should convert to cells...
For more information on the available packages and datasets in each package do:
 	RDatasets.packages() 		# lists available packages
	RDatasets.datasets("package")   # lists available datasets within the package "package"
"""
rdataset() = println("Use RDatasets.packages() and Rdatasets.datasets(\"package\") to view availabe R datasets.")

rdataset(dataset_name::AbstractString) = begin
	dsets = RDatasets.datasets(); 
	j = findin(dsets[:,:Dataset], [dataset_name]); 
	RDatasets.dataset(dsets[j,:Package]..., dsets[j,:Dataset]...)
end

rdataset(package_name::AbstractString, dataset_name::AbstractString) = RDatasets.dataset(package_name, dataset_name)

