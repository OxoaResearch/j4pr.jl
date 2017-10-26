# Load j4pr
reload("j4pr")

# [core]
include("t_data_cells.jl");
include("t_interfaces.jl");
include("t_operators.jl");
include("t_pipe_creation.jl");
include("t_data_access_patterns.jl");
include("t_coreutils.jl");
Base.Test.@testset "[j4pr: /src/core]" begin
	Base.Test.@testset "Data cells: types, aliases" begin t_data_cells(); end
	Base.Test.@testset "Interfaces: interation, types" begin t_interfaces(); end
	Base.Test.@testset "Operators: piping, concatenation" begin t_operators(); end
	Base.Test.@testset "Pipe creation: types and results" begin t_pipe_creation(); end
	Base.Test.@testset "Data access patterns: geting/modifying observations/variables" begin t_data_access_patterns(); end
	Base.Test.@testset "coreutils.jl (utility functions)" begin t_coreutils(); end
end



# [/lib/ext]
### include("ext/t_kpca.jl")
### include("ext/t_fa.jl")
include("ext/t_dist.jl")
include("ext/t_pca.jl")
include("ext/t_whiten.jl")
include("ext/t_ppca.jl")
include("ext/t_ica.jl")
include("ext/t_mds.jl")
include("ext/t_lda.jl")
include("ext/t_lr.jl")
include("ext/t_kmeans.jl")
include("ext/t_kmedoids.jl")
include("ext/t_affinityprop.jl")
include("ext/t_dbscan.jl")
include("ext/t_decisiontree.jl")
include("ext/t_mlkernel.jl")
Base.Test.@testset "[j4pr: /src/lib/ext]" begin
	### Base.Test.@testset "kpca.jl (KPCA)" begin t_kpca(); end
	### Base.Test.@testset "fa.jl (Factor analysis)" begin t_fa(); end
	Base.Test.@testset "dist.jl (Distances)" begin t_dist(); end
	Base.Test.@testset "whiten.jl (Whitening)" begin t_whiten(); end
	Base.Test.@testset "pca.jl (PCA)" begin t_pca(); end
	Base.Test.@testset "ppca.jl (PPCA)" begin t_ppca(); end
	Base.Test.@testset "ica.jl (ICA)" begin t_ica(); end
	Base.Test.@testset "mds.jl (MDS)" begin t_mds(); end
	Base.Test.@testset "lda.jl (LDA)" begin t_lda(); end
	Base.Test.@testset "lr.jl (Linear regression)" begin t_lr(); end
	Base.Test.@testset "kmeans.jl (K-means clustering)" begin t_kmeans(); end
	Base.Test.@testset "kmedoids.jl (K-medoids clustering)" begin t_kmedoids(); end
	Base.Test.@testset "affinityprop.jl (Affinity propagation clustering)" begin t_affinityprop(); end
	Base.Test.@testset "dbscan.jl (DBSCAN clustering)" begin t_dbscan(); end
	Base.Test.@testset "decisiontree.jl (DT/RF/stump regression/classification)" begin t_decisiontree(); end
	Base.Test.@testset "mlkernel.jl (ML kernels)" begin t_mlkernel(); end
end



# [/lib/sup]
include("t_knn.jl")
include("t_lindisc.jl")
include("t_quaddisc.jl")
include("t_parzen.jl")
include("t_loss.jl")
include("t_combiners.jl")
include("t_randomsubspace.jl")
include("t_adaboost.jl")
include("t_stump.jl")
include("t_roc.jl")
Base.Test.@testset "[j4pr: /src/lib/sup]" begin
	Base.Test.@testset "knn.jl (knn regression/classification)" begin t_knn(); end
	Base.Test.@testset "lindisc.jl (Linear discriminant classifier)" begin t_lindisc(); end
	Base.Test.@testset "quaddisc.jl (Quadratic discriminant classifier)" begin t_quaddisc(); end
	Base.Test.@testset "parzen.jl (Parzen density estimator/classifier)" begin t_parzen(); end
	Base.Test.@testset "loss.jl (Loss functions)" begin t_loss(); end
	Base.Test.@testset "combiners.jl (Label/output combiners)" begin t_combiners(); end
	Base.Test.@testset "randomsubspace.jl (Random subspace ensemble)" begin t_randomsubspace(); end
	Base.Test.@testset "adaboost.jl (AdaBoost ensemble)" begin t_adaboost(); end
	Base.Test.@testset "stump.jl (Decision stump classifier and regressor)" begin t_stump(); end
	Base.Test.@testset "roc.jl (ROC curves and operating points)" begin t_roc(); end
end

if (VERSION <= v"0.6") # LIBSVM does not work with Julia 0.7
	include("ext/t_libsvm.jl")
	Base.Test.@testset "[j4pr: /src/lib/ext] (Julia 0.6 ONLY)" begin
		Base.Test.@testset "libsvm.jl (LIBSVM classifier/regressor)" begin t_libsvm(); end
	end
end

# [/lib/unsup]
include("t_kernel.jl")
Base.Test.@testset "[j4pr: /src/lib/unsup]" begin
	Base.Test.@testset "kernel.jl (kernels)" begin t_kernel(); end
end

# [/exp]
include("exp/t_plotting.jl")
Base.Test.@testset "[j4pr: /src/exp]" begin
	Base.Test.@testset "plotting.jl" begin t_plotting(); end
end
