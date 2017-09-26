#!/bin/julia
reload("j4pr")

(tr,ts) = j4pr.splitobs(j4pr.shuffleobs(j4pr.DataGenerator.iris()));

Ws=tr |> j4pr.parzen(0.1,window=:hat); 
Wg=tr |> j4pr.parzen(0.1,window=:gaussian); 
We=tr |> j4pr.parzen(0.1,window=:exponential); 

clf = (tr |> j4pr.parzen(0.1, window=w) for w in [:square, :gaussian, :exponential])
data = (j4pr.sample(j4pr.DataGenerator.iris(),N)[1] for N in [100,1000,10_000])

println 
t=zeros(3,3)
for (i,clf) in enumerate(clf)
	println("$(clf)")
	for (j,D) in enumerate(data)
		D[:,1:2] |> clf
		@time  D |> clf
	end
end
