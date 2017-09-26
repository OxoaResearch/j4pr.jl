##########################
# FunctionCell Interface #	
##########################
"""
	loss([x, f_targets=identity, f_output=identity, losstype=LossFunctions.L2DistLoss(), avgmode=LossFunctions.AvgMode.Mean()])

Computes the loss of `x`. Here `x` is assumend to contain information on both the output of a classifier or regressor as well as
the target information e.g. labels or regression values.

# Arguments
  * `x` is the data. Can be a labeled `DataCell`, `Tuple` of the form `(<output data>, <target data>)` or an `Array` of outputs; in this latter case
`f_targets` needs to be specified
  * `f_targets` is a function that is applied to the labels of `x`, before applying the loss function
  * `f_output` is a function that is applied to the data of `x`, before applying the loss function
  * `losstype`::LeanBase.Loss is the loss object describing the type of loss applied
  * `avgmode`::LossFunctions.AverageMode is the loss aggregation object which describes how individual observation losses are combined

It is assumed that the inputs of `f_targets` and `f_output` are known, as well as their outputs. Generally, only one of the functions is
actually needed, transforming either targets or output to a common format. 

Check the `LossFunctions.jl` and `MLLabelUtils.jl` documentation for further details.

# Examples
```
# Train a small classifier and check the mean squared error
julia> using j4pr; tr, ts = splitobs(shuffleobs(DataGenerator.iris())); LC = lindisc(); LCT = tr |> LC;         

julia> LC                                                                                                                                                                                             
Linear discriminant classifier: r1=0.0, r2=0.0, no I/O size information, untrained                                                                                                                    

julia> LCT                                                                                                                                                                                            
Linear discriminant classifier: r1=0.0, r2=0.0, 4 -> 3, trained        

julia> ff1=x->MLLabelUtils.convertlabel(MLLabelUtils.LabelEnc.OneOfK,x,LCT.y["labels"]) # ff1 processes existing labels                                                                                                                
(::#43) (generic function with 1 method)  

julia> ff2=x->MLLabelUtils.convertlabel(MLLabelUtils.LabelEnc.OneOfK, targets(indmax,x)) #ff2 processes the outputs                                                                                                         
(::#47) (generic function with 1 method)     

julia> ff1(-ts)                                                                                                                                                                                       
3×45 Array{Int64,2}:                                                                                                                                                                                  
 1  1  0  1  1  0  0  0  1  0  1  0  0  1  0  1  0  0  1  0  1  1  0  1  1  0  1  0  0  0  0  0  0  0  0  1  0  0  0  1  1  0  1  1  1                                                                
 0  0  1  0  0  1  1  1  0  1  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  1  0  0  1  0  0  0  0  0  0  0                                                                
 0  0  0  0  0  0  0  0  0  0  0  1  1  0  1  0  0  1  0  1  0  0  1  0  0  1  0  1  1  0  1  1  1  1  0  0  1  0  1  0  0  1  0  0  0                                                                
                                                                                                                                                                                                         
julia> ff2(+ts|>LCT)                                                                                                                                                                                  
3×45 Array{Int64,2}:                                                                                                                                                                                  
 1  1  0  1  1  0  0  0  1  0  1  0  0  1  0  1  0  0  1  0  1  1  0  1  1  0  1  0  0  0  0  0  0  0  0  1  0  0  0  1  1  0  1  1  1                                                                
 0  0  1  0  0  1  1  1  0  1  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  1  0  1  0  0  0  0  1  0  0  1  0  0  0  0  0  0  0                                                                
 0  0  0  0  0  0  0  0  0  0  0  1  1  0  1  0  0  1  0  1  0  0  1  0  0  1  0  0  1  0  1  1  1  1  0  0  1  0  1  0  0  1  0  0  0       

# Calculating the error:
julia> j4pr.loss(ts|> LCT, ff1, ff2)                                                                                                                                                                  
0.014814814814814815         
```
"""
# Create fixed function cell (no `x`)
loss(f_targets::F=identity, f_output::G=identity, 
	losstype::L=LossFunctions.L2DistLoss(), avgmode::A=LossFunctions.AvgMode.Mean()) where {F<:Function, G<:Function, L<:LearnBase.Loss, A<:LossFunctions.AverageMode} = 
begin
	
	FunctionCell(loss, (f_targets, f_output, losstype, avgmode), "Loss: type=$(losstype), avgmode=$(avgmode)") 
end

# Work with labeled data cells
loss(x::T where T<:CellDataL, f_targets::F=identity, f_output::G=identity, 
     losstype::L=LossFunctions.L2DistLoss(), avgmode::A=LossFunctions.AvgMode.Mean()) where {F<:Function, G<:Function, L<:LearnBase.Loss, A<:LossFunctions.AverageMode} = 
begin
	return loss(f_targets(gety!(x)), f_output((getx!(x))), losstype, avgmode)
	#       	y         	     ŷ
end

# Work with Tuples
loss(x::T where T<:Tuple{U,V} where {U,V}, f_targets::F=identity, f_output::G=identity,
     losstype::L=LossFunctions.L2DistLoss(), avgmode::A=LossFunctions.AvgMode.Mean()) where {F<:Function, G<:Function, L<:LearnBase.Loss, A<:LossFunctions.AverageMode} = 
begin
	return loss(f_targets(x[2]), f_output(x[1]), losstype, avgmode)
	#                y               ŷ
end

# Work with Array input (output); In this case labels need to be specified through f_targets (e.g. f_targets=()->Labels::Array{...})
loss(x::T where T<:AbstractArray, f_targets::F, f_output::G=identity,
     losstype::L=LossFunctions.L2DistLoss(), avgmode::A=LossFunctions.AvgMode.Mean()) where {F<:Function, G<:Function, L<:LearnBase.Loss, A<:LossFunctions.AverageMode} = 
begin
	return loss(f_targets(), f_output(x), losstype, avgmode)
	#                y               ŷ
end

#Define low level loss function
loss(y::T where T<:AbstractArray, ŷ::S where S<:AbstractArray, losstype::U where U<:LearnBase.Loss, avgmode::V where V<:LossFunctions.AverageMode ) = LossFunctions.value(losstype, y, ŷ, avgmode)



