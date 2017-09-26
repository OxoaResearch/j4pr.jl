# Experiment that should test J4PR functionality
module experiment3

reload("j4pr")
using DataFrames
import j4pr
# Generate data and write it to file
# TODO

# Read data and preproces 
# -----------------------

# Read from csv
data = permutedims(readcsv("/home/zgornel/projects/j4pr/trunk/data/sample.csv"),(2,1));

label_col = 7

# Apply data transforms
# ---------------------

# 1. Process independently missing values on columns
data_filt = data |> (data |> j4pr.filterg(Dict(1=>"majority", 
					2=>(x,y)->"missing",
					3=>(x,y)->zero(typeof(x)),
					4:5=>"majority",
					6=>(x,y)->"missing")
				   )
		 )

# 2. define a domain for a variable
dom_x1 = Dict(2=>x->x in ["0","1","a"] ? x : 0);
data_filt |> j4pr.filterdomain!(dom_x1);



# 3. Run one-hot encoder 
data_ohenc = data_filt[1:6,:] |> j4pr.ohenc(j4pr._variable_(data_filt,1:6), Dict(1:6=>"binary"))
labels = Vector{String}(data_filt[7,:])
dc = j4pr.datacell(data_ohenc, labels)


end # module end
