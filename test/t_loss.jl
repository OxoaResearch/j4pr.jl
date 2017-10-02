# Tests for loss function functionality 
function t_loss()

Ac = j4pr.DataGenerator.iris()   # classification dataset
Ar = j4pr.DataGenerator.boston() # regression dataset


# Test losses for classification
tr,ts = j4pr.splitobs(j4pr.shuffleobs(Ac),0.5)
wt = tr |> j4pr.knn(5,smooth=:ml)
ts = j4pr.getobs(ts)
wlc1 = j4pr.loss(()->MLLabelUtils.convertlabel(MLLabelUtils.LabelEnc.OneOfK{Float64}, -ts, wt.x.properties.labels.label)::Matrix{Float64 }) # for array input
wlc2 = j4pr.loss((x)->MLLabelUtils.convertlabel(MLLabelUtils.LabelEnc.OneOfK{Float64}, x, wt.x.properties.labels.label)::Matrix{Float64 }) # for Tuple/datacell input
	r1 = +ts |> wt+wlc1
	r2 = ts |> wt+wlc2
	r3 = j4pr.strip(ts|>wt) |> wlc2
Base.Test.@test  r1==r2==r3	

# Test losses for classification
tr,ts = j4pr.splitobs(j4pr.shuffleobs(Ar),0.5)
wt = tr |> j4pr.knnr(5,smooth=:ml)
ts = j4pr.getobs(ts)
wlc1 = j4pr.loss(()->-ts, vec) # for array input
wlc2 = j4pr.loss(identity,vec) # for Tuple/datacell input
	r1 = +ts |> wt+wlc1
	r2 = ts |> wt+wlc2
	r3 = j4pr.strip(ts|>wt) |> wlc2
Base.Test.@test  r1==r2==r3	


end
