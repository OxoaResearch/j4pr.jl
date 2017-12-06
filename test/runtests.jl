# Load j4pr
using j4pr
using DataStructures, DataArrays, LearnBase, MLKernels, MLLabelUtils, Distances, Images

if (VERSION > v"0.7-")
	using Test
end

# [core]
include("t_data_cells.jl");
include("t_interfaces.jl");
include("t_operators.jl");
include("t_pipe_creation.jl");
include("t_data_access_patterns.jl");
include("t_coreutils.jl");
Test.@testset "[j4pr: /src/core]" begin
	Test.@testset "Data cells: types, aliases" begin t_data_cells(); end
	Test.@testset "Interfaces: interation, types" begin t_interfaces(); end
	Test.@testset "Operators: piping, concatenation" begin t_operators(); end
	Test.@testset "Pipe creation: types and results" begin t_pipe_creation(); end
	Test.@testset "Data access patterns: geting/modifying observations/variables" begin t_data_access_patterns(); end
	Test.@testset "coreutils.jl (utility functions)" begin t_coreutils(); end
end



# [/lib]
include("t_libutils.jl")
Test.@testset "[j4pr: /src/lib]" begin
	Test.@testset "libutils.jl (utility functions for libraries)" begin t_libutils(); end
end


# [/lib/data]
include("t_libdata.jl")
Test.@testset "[j4pr: /src/lib/data]" begin
	Test.@testset "Data container processing functions (class slicing, filtering, encoding)" begin t_libdata(); end
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
include("ext/t_decisiontree.jl")
include("ext/t_mlkernel.jl")
Test.@testset "[j4pr: /src/lib/ext]" begin
	### Test.@testset "kpca.jl (KPCA)" begin t_kpca(); end
	### Test.@testset "fa.jl (Factor analysis)" begin t_fa(); end
	Test.@testset "dist.jl (Distances)" begin t_dist(); end
	Test.@testset "whiten.jl (Whitening)" begin t_whiten(); end
	Test.@testset "pca.jl (PCA)" begin t_pca(); end
	Test.@testset "ppca.jl (PPCA)" begin t_ppca(); end
	Test.@testset "ica.jl (ICA)" begin t_ica(); end
	Test.@testset "mds.jl (MDS)" begin t_mds(); end
	Test.@testset "lda.jl (LDA)" begin t_lda(); end
	Test.@testset "lr.jl (Linear regression)" begin t_lr(); end
	Test.@testset "kmeans.jl (K-means clustering)" begin t_kmeans(); end
	Test.@testset "kmedoids.jl (K-medoids clustering)" begin t_kmedoids(); end
	Test.@testset "affinityprop.jl (Affinity propagation clustering)" begin t_affinityprop(); end
	Test.@testset "decisiontree.jl (DT/RF/stump regression/classification)" begin t_decisiontree(); end
	Test.@testset "mlkernel.jl (ML kernels)" begin t_mlkernel(); end
end



# [/lib/sup]
include("t_lindisc.jl")
include("t_quaddisc.jl")
include("t_parzen.jl")
include("t_loss.jl")
include("t_combiners.jl")
include("t_randomsubspace.jl")
include("t_adaboost.jl")
include("t_stump.jl")
include("t_networklearner.jl")
Test.@testset "[j4pr: /src/lib/sup]" begin
	Test.@testset "lindisc.jl (Linear discriminant classifier)" begin t_lindisc(); end
	Test.@testset "quaddisc.jl (Quadratic discriminant classifier)" begin t_quaddisc(); end
	Test.@testset "parzen.jl (Parzen density estimator/classifier)" begin t_parzen(); end
	Test.@testset "loss.jl (Loss functions)" begin t_loss(); end
	Test.@testset "combiners.jl (Label/output combiners)" begin t_combiners(); end
	Test.@testset "randomsubspace.jl (Random subspace ensemble)" begin t_randomsubspace(); end
	Test.@testset "adaboost.jl (AdaBoost ensemble)" begin t_adaboost(); end
	Test.@testset "stump.jl (Decision stump classifier and regressor)" begin t_stump(); end
	Test.@testset "networklearner.jl (Network Learning)" begin t_networklearner(); end
end

if v"0.6" <= VERSION < v"0.7-"
	include("ext/t_dbscan.jl")
	include("ext/t_libsvm.jl")
	Test.@testset "[j4pr: /src/lib/ext] (Julia 0.6 ONLY)" begin
		Test.@testset "dbscan.jl (DBSCAN clustering)" begin t_dbscan(); end
		Test.@testset "libsvm.jl (LIBSVM classifier/regressor)" begin t_libsvm(); end
	end
	
	include("t_knn.jl")
	include("t_roc.jl")
	Test.@testset "[j4pr: /src/lib/sup] (Julia 0.6 ONLY)" begin
		Test.@testset "knn.jl (knn regression/classification)" begin t_knn(); end
		Test.@testset "roc.jl (ROC curves and operating points)" begin t_roc(); end
	end
end

# [/lib/unsup]
include("t_kernel.jl")
Test.@testset "[j4pr: /src/lib/unsup]" begin
	Test.@testset "kernel.jl (kernels)" begin t_kernel(); end
end

# [/exp]
include("exp/t_plotting.jl")
Test.@testset "[j4pr: /src/exp]" begin
	Test.@testset "plotting.jl" begin t_plotting(); end
end
