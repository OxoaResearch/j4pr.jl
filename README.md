# j4pr.jl 

*j4pr is a small library and Julia package wrapper designed 
to simplify some of the common tasks encountered in generic 
pattern recognition / machine learning / a.i. workflows 
and to act as a learning resource, easily expandable, 
for parctitioners in these domains. It does not aim to 
extensively cover a specific topic, like most Julia packages, 
but rather to provide some practical consistency and ease of
use for existing algorithms as well as providing new ones.*

## First glance and philosophy

j4pr provides either natively or by means of wrapping around, 
algorithms for clustering, classification, regression, 
data manipulation, error assessment and terminal-based plotting.
While type stability is not enforced, it still is pretty speedy, 
due to the nature of Julia itself. So far, it has some basic 
capabilities for parallelization and provides a few generic 
frameworks for combining various learning algorithms. It is 
currently under heavy development and some bugs are expected 
to be around.

A simple example:
```julia
julia> using j4pr; j4pr.version() # use j4pr and print the version string
#
#  _  _
# (_\/_)                  |  This is a small library and package wrapper written at 0x0α Research.
# (_/\_)                  |  Type "?j4pr" for general documentation. 
#    _ _   _  _____ _ _   |  Look inside src/j4pr.jl for a list of available algorithms.
#   | | | | |/____ / ` |  |  
#   | | |_| | |  | | /-/  |  Version 0.1.0 "The Monolith" revision: 165 (2017-09-24)
#  _/ |\__  | |  | | |    |  
# |__/    |_|_|  |_|_|    |  License: MIT, view ./LICENSE.md for details.



julia> data = DataGenerator.iris() # get the iris dataset
# Iris Dataset, 150 obs, 4 vars, 1 target(s)/obs, 3 distinct values: "virginica"(50),"setosa"(50),"versicolor"(50)

julia> (tr,ts)=splitobs(shuffleobs(data),0.3); # split dataset

julia> tr
# [*]DataCell, 45 obs, 4 vars, 1 target(s)/obs, 3 distinct values: "virginica"(15),"versicolor"(12),"setosa"(18)

julia> clf = knn(5, smooth=:ml) # 5-nn classifier, max-likelihood posterior smoothing
# 5-NN classifier: smooth=ml, no I/O size information, untrained

julia> clf_trained = clf(tr)
# 5-NN classifier: smooth=ml, 4->3, trained

julia> +ts |> clf_trained # get output probs for test data
# 3×105 Array{Float64,2}:
#  0.0  1.0  0.0  1.0  0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  1.0  …  1.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  1.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0
#  0.8  0.0  0.6  0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.0  0.0  1.0  0.4  0.0  0.0  0.0     0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.2  1.0  0.0  0.0  0.8  1.0  0.0  1.0  0.0  0.0  0.0
#  0.2  0.0  0.4  0.0  1.0  0.0  1.0  0.0  0.0  0.0  1.0  1.0  0.0  0.0  0.6  1.0  1.0  0.0     0.0  0.0  0.0  0.0  1.0  1.0  0.0  0.8  0.0  0.0  0.0  0.2  0.0  0.0  0.0  1.0  1.0  0.0

julia> +data |> clf_trained + lineplot(2,width=100) # print output probs for whole dataset, second class
     ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐ 
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣷⣿⠉⠉⢹⣾⢸⣿⣿⡇⡏⡇⡏⠉⢹⠀⠀⡏⢹⡏⠉⠉⠉⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀⢸⣿⢸⣿⣿⡇⡇⡇⡇⠀⢸⠀⠀⡇⢸⡇⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠀⠀⢸⣿⢸⣿⣿⡇⡇⡇⡇⠀⢸⠀⠀⡇⢸⡇⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⠉⠈⣿⠀⠀⢸⡏⠉⣿⣿⣷⠁⠉⠁⠀⢸⡏⠉⠁⠈⠁⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⠀⢸⡇⠀⣿⣿⣿⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⠀⢸⡇⠀⣿⣿⣿⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠁⠀⠀⠈⠁⠀⣿⣿⣿⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠉⣿⠉⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⢰⡇⣾⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⢸⡇⣿⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⢸⡇⣿⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⢸⡇⠀⠀⢸⡇⢸⡇⣿⢰⡇⢸⢸⠀⣾⠀⠀⠀⣾⠀⠀⠀ │ 
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣿⠀⢸⡇⠀⠀⢸⡇⢸⡇⣿⢸⡇⢸⢸⠀⣿⠀⠀⠀⣿⠀⠀⠀ | 
   0 │⢀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣇⣀⣀⣿⣀⣸⣇⣀⣀⣸⣇⣸⣇⣿⣸⣇⣸⢸⣀⣿⣀⣀⣀⣿⣀⡀⠀ | 
     └────────────────────────────────────────────────────────────────────────────────────────────────────┘ 
     0                                                                                                  200
```

A slightly more complex example:
```julia
julia> data = DataGenerator.iris()
       (tr,ts)=splitobs(shuffleobs(data),0.3); # split dataset
       clf = knn(1);  # base classifier
       L = 5          # ensemble size
       C = 3          # 3 classes
       ensemble = pipestack(        # stacked classifier ensemble
                            Tuple(  # 'pipestack' works with Tuples
                                  clf(f[1]) for f in kfolds(tr,5) # ~36 observations for training
                                 )
                           ) + meancombiner(L,C;α=1.0) # generalized mean combiner, averages individual classifier predictions

# Serial Pipe, 2 element(s), 2 layer(s), generic

julia> +ensemble
# 2-element PTuple{j4pr.AbstractCell}:
# `- Stacked Pipe, 5 element(s), 1 layer(s), trained
# `- Generalized mean combiner: α=1.0, 15->3, trained

julia> +data |> ensemble
# 3×150 Array{Float64,2}:
#  1.0          1.0          1.0          1.0          1.0          1.0          1.0          …  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16
#  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16     2.22045e-16  2.22045e-16  2.22045e-16  0.2          2.22045e-16  2.22045e-16  2.22045e-16
#  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16  2.22045e-16     1.0          1.0          1.0          0.8          1.0          1.0          1.0        
```



## Overview

The package provides three main constructs or types designed to describe data 
``DataCell``, functions ``FunctionCell`` i.e. data transforms, training 
and execution methods for classification, regression etc. and processing pipelines 
``PipeCell`` i.e. successions of arbitrary operations. Aside from the 
objects themselves, several operators, conversion methods and iteration interfaces
are provided that allow combining and working with the objects, with the aim 
of obtaining arbitrarily complex structures.

The data container is integrated
with the [MLDataPattern.jl API](https://github.com/JuliaML/MLDataPattern.jl)
allowing for ellegant and efficient operations. An unrelated example of the concept
behind the function container i.e. ``FunctionCell`` can be found in 
[this](https://julialang.org/blog/2017/08/dsl) post. ``PipeCell`` is a container
for ``FunctionCell`` objects as well as some information specifying  how data is processed
by these i.e. passed sequentially through each, sent to each in parallel etc.

- **Data container** 
    
    The main data container for j4pr is the `DataCell`. This is a simply a wrapper
    around either one or two `AbstractArray` objects. The convention throughout j4pr
    is that the first dimension of arrays is the variable dimension while the second is 
    the observation dimension. If the data is a `Vector`, its dimension is the observation
    dimension. One can:

    - Create a 'unlabeled','labeled' or 'multi-labeled' ``DataCell``:
      ```julia
      julia> X1=[0;1]; X2=[1 2; 3 4;5 6]; y1=[1,2]; y2=rand(2,2);
      
      julia> using j4pr;
      
      julia> datacell(X1,y1)
      # DataCell, 2 obs, 1 vars, 1 target(s)/obs, 2 distinct values: "2"(1),"1"(1)
      
      julia> datacell(X1,y2)
      # DataCell, 2 obs, 1 vars, 2 target(s)/obs
      
      julia> datacell(X2,y1)
      # DataCell, 2 obs, 3 vars, 1 target(s)/obs, 2 distinct values: "2"(1),"1"(1)
      
      julia> datacell(X2,y2)
      # DataCell, 2 obs, 3 vars, 2 target(s)/obs
      
      julia> datacell(X1)
      # DataCell, 2 obs, 1 vars, 0 target(s)/obs
      
      julia> datacell(X2)
      # DataCell, 2 obs, 3 vars, 0 target(s)/obs
      ```
    
    - Index similar to an ``Array``:
      ```julia
      julia> A=datacell(X2,y2);

      julia> A[1] # first observation
      # DataCell, 1 obs, 3 vars, 2 target(s)/obs

      julia> A[[1,3],:] # fist and third observations
      # DataCell, 2 obs, 2 vars, 2 target(s)/obs

      julia> A[[1],[2]] # first observation, second variable
      # DataCell, 1 obs, 1 vars, 2 target(s)/obs
      
      julia> datasubset(A,1) # first observation (SubArrays)
      # [*]DataCell, 1 obs, 3 vars, 2 target(s)/obs

      julia> varsubset(A,2) # second variable (SubArrays)
      # [*]DataCell, 2 obs, 1 vars, 2 target(s)/obs
      
      julia> A[[1,3],2] = [0,0]; # change values of second variable, 1'st and 3'rd observation
      
      julia> A.x # field 'x' is data, 'y' is labels
      # 3×2 Array{Int64,2}:
      #  1  0
      #  3  4
      #  5  0
       
      julia> X2
      # 3×2 Array{Int64,2}:
      #  1  0
      #  3  4
      #  5  0
      ```

    - Shortcut access to the 'data' and 'label' contents:
      ```julia
      julia> A=datacell(X1,y1;name="My data")
      # My data, 2 obs, 1 vars, 1 target(s)/obs, 2 distinct values: "2"(1),"1"(1)

      julia> +A # access data
      # 2-element Array{Int64,1}:
      #  0
      #  1
      
      julia> -A # access labels
      # 2-element Array{Int64,1}:
      #  1
      #  2
      ```

    - Concatenate several 'unlabeled' `DataCell` objects:
      ```julia
      julia> A=datacell(rand(3)); B=datacell(100*rand(2,3));

      julia> C=[A;B] # variable concatenation
      # DataCell, 3 obs, 3 vars, 0 target(s)/obs

      julia> +C
      # 3×3 Array{Float64,2}:
      #  0.32635   0.211486   0.0696501
      #  54.1882    6.56315   82.4691   
      #  80.4626   78.2586    57.8038   
      
      julia> D=[A A] # observation concatenation
      # DataCell, 6 obs, 1 vars, 0 target(s)/obs
      
      julia> +D
      # 6-element Array{Float64,1}:
      #  0.32635  
      #  0.211486
      #  0.0696501
      #  0.32635
      #  0.211486
      #  0.0696501
      ```
    
    - The concatenation of 'labeled' and 'unlabeled' ``DataCell`` objects is more restrictive:
      ```julia
      julia> A=datacell(rand(3),[0,0,1]); B=datacell(100*rand(2,3)); C=datacell([1.0,2.0,3.0],[1,2,1]);

      julia> -[A C] # for observation concatenation, labels are kept
      # 6-element Array{Int64,1}:
      #  0
      #  0
      #  1
      #  1
      #  2
      #  1
       
      julia> -[A;C] # fails: for variable concatenation, labels have to be equal 
      # ERROR: AssertionError: [vcat] 'y' fields have to be identical for all DataCells.
      # Stacktrace:
      #  ...
      
      julia> -[A;A] # works
      # 3-element Array{Int64,1}:
      #  0
      #  0
      #  1
       
      julia> C=[A;B] # variable concatenation with 'unlabeled' DataCells silently drops the labels
      # DataCell, 3 obs, 3 vars, 0 target(s)/obs
      ```
     
     It is important to note that most code throughout j4pr supports implicitly 
     besides `DataCell` also arrays (considered unlabeled data) or
     tuples of two arrays (considered labeled data), a convention used also in 
     MLDataPattern.jl. The term 'label' is used here more
     as a convention however, in a general sense, it should be interpreted as any
     value dependent on the data through a relation of the form 
     `label = some_property_or_function(data)`.

- **Wrapping functions** 
    
    The `FunctionCell` type is ment to be a wapper around functions that perform
    fixed operations on data or, train and apply models. Its operation is defined
    by overloading the `|>` operator as well as call methods, making the
    instantiated objects act in a function-like manner. As a toy example, let us 
    consider three functions: one that returns the input `foo`, one that constructs
    a simple model - the mean of the input - `bar`, and one that executes the model - 
    subtracts from the input the mean - `baz`.
    
    For the 'fixed' `FunctionCell`, the wrapping is straightforward: 
    ```julia
    julia> foo(x) = x;

    julia> Wfoo = FunctionCell(foo,(),"My foo")
    # My foo, no I/O size information, fixed

    julia> Wfoo(1) # same as 1 |> Wfoo
    # 1
    ```
   
    To obtain 'trained' FunctionCells, one has to define a small training wrapper:
    ```julia
    bar(x) = mean(x); baz(x,m) = x-m;
    train(x) = begin # 'x' stands everywhere for input data
        # Generate model data
        model_data = Model(bar(x)) # it is wrapped by a 'Model' type   
      				   # 'Model' has one field only, 'data' 
        # Construct execution function based on 'baz' (has to have 3 input arguments: data, model and model properties)
        exec_func = (x,m,mp)->baz(x,m.data) # construct execution function
					           
        # Construct 'trained' function cell that uses the execution function defined above
        out = FunctionCell(exec_func, model_data, Dict(),"Trained using bar")
    end;
    ```

    To obtain 'untrained' FunctionCells, one has to wrap the `train` function previously defined:
    ```julia
    to_train() = FunctionCell(train, (), Dict(), "Expects data to train") # create untrained FunctionCell
    ```
  
    Now, the basic functionality is covered:
    ```julia
    julia> Au = to_train() # no training arguments required
    # Expects data to train, no I/O size information, untrained

    julia> train_data = 5*rand(10); # the training data 

    julia> test_data = rand(10); # test data

    julia> At = Au(train_data) # train; or: train_data |> Au
    # Trained using bar, no I/O size information, trained

    julia> At(test_data) # execute; or: test_data |> At
    # 10-element Array{Float64,1}:
    #  -1.57413
    #  -1.95684
    #  -1.38696
    #  -1.23955
    #  -1.80348
    #  -1.57977
    #  -1.33085
    #  -1.08819
    #  -1.64787
    #  -1.91666
    ```
  
    One can already go beyond the basic functionality using simple Julia constructs:
    ```julia
    julia> At = map(Au, [[1,2,3],[3,4,5],[0,0,0]]) # get three models
    # 3-element Array{j4pr.FunctionCell{j4pr.Model{Float64},Dict{Any,Any},##11#12,Tuple{},Tuple{}},1}:
    #  Trained using bar, no I/O size information, trained
    #  Trained using bar, no I/O size information, trained
    #  Trained using bar, no I/O size information, trained

    julia> results = [At[i](test_data) for i in 1:length(At)] # apply each model to 'test_data'
    # 3-element Array{Array{Float64,1},1}:
    #  [-1.51074, -1.89345, -1.32357, -1.17616, -1.74009, -1.51638, -1.26746, -1.0248, -1.58448, -1.85327]
    #  [-3.51074, -3.89345, -3.32357, -3.17616, -3.74009, -3.51638, -3.26746, -3.0248, -3.58448, -3.85327]
    #  [0.489263, 0.106547, 0.676427, 0.823845, 0.259914, 0.483618, 0.73254, 0.975199, 0.41552, 0.146731] 
  
    julia> D=rand(2,5); # dataset with 2 variables, 5 observations

    julia> Au.([D[i,:] for i in 1:size(D,1)]) # train a model/variable :)
    # 2-element Array{j4pr.FunctionCell{j4pr.Model{Float64},Dict{Any,Any},##11#12,Tuple{},Tuple{}},1}:
    #  Trained using bar, no I/O size information, trained
    #  Trained using bar, no I/O size information, trained
    ```

    Basically, the main ideea behing function cells, one that is being used throughout j4pr is
    to be able to do:
    ```julia
    U = algorithm(train_args...)             # create untrained model
    T = algorithm(train_data, train_args...) # create trained model, or
    T = U(data) 			 
    T(test_data) # execute model on test data
    ```
    while keeping the same function signature i.e. `train_args` as in the original methods
    that were wrapped.
    
- **Processing pipelines**

    Pipelines represent ways of processing data. There are several alternatives 
    to the ones here, one good example being [Lazy.jl](https://github.com/MikeInnes/Lazy.jl).
    j4pr pipelines, namely the `PipeCell` type, can only be created from other `Cell`-like
    objects, meaning `DataCell`, `FunctionCell` or `PipeCell` itself. Three types
    of pipelines can be created: serial pipelines - data is passed from one 
    pipe element to another in a sequential manner, stacked pipes - the 
    same data is passed to all (or some) of the elements of the pipe and parallel
    pipelines - some elements of the input data are passed to some elements of the pipe
    (obviously, such assumption must hold in practice in order to be applicable).
    Although more complicated examples can be contrived, let us look at some simple ones:
    ```julia
    julia> wa=FunctionCell(x->x*"A"); wb = FunctionCell(x->x*"B");

    julia> se = wa+wb;    # serial pipe
           st = [wa;wb];  # stacked pipe
	   pp = [wa wb];  # parallel pipe
    
    julia> se
    # Serial Pipe, 2 element(s), 1 layer(s), fixed
    
    julia> st
    # Stacked Pipe, 2 element(s), 1 layer(s), fixed
    
    julia> pp
    # Parallel Pipe, 2 element(s), 1 layer(s), fixed

    julia> +se
    # 2-element PTuple{j4pr.FunctionCell{Void,Void,U,Tuple{},Tuple{}} where U}:
    # `- #35, no I/O size information, fixed
    # `- #37, no I/O size information, fixed

    julia> "" |> se
    # "AB"
    
    julia> "" |> st
    # 2×1 Array{String,2}:
    #  "A"
    #  "B"
    
    julia> ["1","2"] |> pp
    # 2×1 Array{String,2}:
    #  "1A"
    #  "2B"
    
    julia> pg = se + st
    # Serial Pipe, 2 element(s), 2 layer(s), generic

    julia> "" |> pg
    # 2×1 Array{String,2}:
    #  "ABA"
    #  "ABB"

    julia> ["","."] |> [pg pg]
    # 4×1 Array{String,2}:
    #  "ABA" 
    #  "ABB" 
    #  ".ABA"
    #  ".ABB"
    
    julia> longpipe_1 = [pg pg]+[se se]
    # Serial Pipe, 2 element(s), 4 layer(s), generic
    
    julia> ["","."] |> longpipe_1
    # 2×1 Array{String,2}:
    # "ABAAB"
    # "ABBAB"
	
    julia> longpipe_2 = [pg pg]+[se se se se]
    # Serial Pipe, 2 element(s), 4 layer(s), generic

    julia> ["","."] |> longpipe_2
    # 4×1 Array{String,2}:
    #  "ABAAB" 
    #  "ABBAB" 
    #  ".ABAAB"
    #  ".ABBAB"
    ```
    
    This documentation portion is somewhat incomplete as there are many aspects on pipes
    that can be covered. I recommend experimenting with the concepts presented above in 
    order to get a better grasp of how pipes can be efficiently used for your own 
    workflow.

- **Algorithms**

    So far, j4pr wraps most of [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl), 
    [Clustering.jl](https://github.com/JuliaStats/Clustering.jl) as
    well as [LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl), 
    [Distances.jl](https://github.com/JuliaStats/Distances.jl), 
    [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) 
    and uses 
    functionality from [UnicodePlots.jl](https://github.com/Evizero/UnicodePlots.jl), 
    [JuliaML](https://github.com/JuliaML) and some other nice packages. 
    It provides also implementations (as submodules) for kNN classification
    and regression (based on [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl)), 
    Parzen window density estimation and classification, linear and quadratic discriminants
    as well as generic frameworks for classifier combiners, random subspace 
    ensembles and boosting (AdaBoost M1 and M2).

    Although it is difficult to provide a detailed roadmap, 
    future releases will include, among other, some integration with JuliaDB, 
    implementations of radial basis function classification and regression, 
    network classification, online learning mechanisms, extensions of the parallel
    framework i.e. parallel pipeline execution and hopefully, 
    some online data collection and processing methods.
    Be sure to check doc/roadmap.md for details.
    Suggestions are always welcomed ;)


## Documentation

Most of the documentation is provided in Julia's native docsystem. 
Unfortunately, due to time constraints, a more detailed documentation
is not feasible at this point. Yet, the code is commented and should 
be pretty easy to get around. Most functions and algorithms are 
documented. For example information on the `Parzen` classifier/density 
estimator can be accessed by writing in the REPL:

```
?j4pr.parzen
```



## Installation

This package is not registered in `METADATA.jl` and can cannot 
be installed from the REPL. It can be downloaded either from 
GitHub or from [here](https://oxoaresearch.com/wp-content/uploads/2017/09/j4pr.tar.gz). It is recommended to add the
path to `j4pr.jl` to the `LOAD_PATH` i.e. add
`push!(LOAD_PATH, "/path/to/j4pr/dot/jl/")` to `~/.juliarc`.



## License

This code has an MIT license and therefore it is free.



## Credits

This work would not have been possible without the excellent
work done by the Julia language and package developers. 



## FAQ

- Are there any plans to make `j4pr.jl` a Julia package ?
  
  At this point, no. It does not follow the main concepts
  of a package nor does it aim to. If the feedback received
  is positively positive, maybe. Otherwise, it should be 
  considered an unofficial resource of hacks, tricks and 
  algorithms that exist outside the Julia ecosystem.
	
- Can I contribute ?

  Yes, contributions are encouraged however, not at this point. By 
  the end of this year it should be possible. You can
  report bugs by e-mailing at j4pr@oxoaresearch.com .

- Can I make Julia packages out of `j4pr` submodules ?
  
  Yes, however support will most likely not be available.
