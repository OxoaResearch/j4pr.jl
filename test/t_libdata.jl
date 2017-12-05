# Test data container processing functionality
function t_libdata()
	
#############################################################
# Test data generator (test that datasets can be generated) #
#############################################################
Test.@test try j4pr.DataGenerator.iris();true catch; false end
Test.@test try j4pr.DataGenerator.quakes();true catch; false end
Test.@test try j4pr.DataGenerator.boston();true catch; false end
Test.@test try j4pr.DataGenerator.normal2d();true catch; false end
Test.@test try j4pr.DataGenerator.fish(10);true catch; false end
Test.@test try j4pr.DataGenerator.fishr();true catch; false end
Test.@test try j4pr.DataGenerator.fnoise(rand(10),x->2x);true catch; false end



###############################################################
# Test images interface (test that datasets can be generated) #
###############################################################
A = [1.0 1.1; 1.2 1.3]
img = Images.Gray.(A)
Test.@test j4pr.im2targets(A) == ([1.0 2.0 1.0 2.0; 1.0 1.0 2.0 2.0], 
			    	  Images.ColorTypes.Gray{Float64}[Images.Gray{Float64}(1.0), Images.Gray{Float64}(1.2), 
							   Images.Gray{Float64}(1.1), Images.Gray{Float64}(1.3)])
Test.@test j4pr.targets2im(j4pr.datacell(j4pr.im2targets(Images.Gray.(A)))) == A



########################
# Test label utilities #
########################
C = 2
N = 10
A = rand(10)
l = rand(Bool,N)
l2 = rand(Bool,C,N)

# addlabels
for Di in [j4pr.datacell(A), j4pr.datacell(A,l), j4pr.datacell(A,l2)]
	for li in [l,l2]
		Di2 = j4pr.addlabels(Di,li)
		Test.@test j4pr.getx!(Di2) == j4pr.getx!(Di) 
		Test.@test j4pr.gety!(Di2) == j4pr.dcat(j4pr.gety!(Di),li)
		Test.@test j4pr.gety!(j4pr.unlabel(Di2)) == nothing
	end
end

# labelize
D = j4pr.datacell(A,l2)

wl = j4pr.labelize(l, true)
wl2 = j4pr.labelize(l, false)
Test.@test j4pr.gety!(D |> wl) == l
Test.@test j4pr.gety!(D |> wl2) == j4pr.dcat(l2,l)


wl = j4pr.labelize(x->x[1], true)
wl2 = j4pr.labelize(x->x[1], false)
Test.@test j4pr.gety!(D |> wl) == l2[1,:]
Test.@test j4pr.gety!(D |> wl2) == j4pr.dcat(l2,l2[1,:])
Test.@test try
	j4pr.unlabel(D) |> wl
	false
catch
	true # has to fail
end

A=[1 2 3; 4 5 6; 7 8 9]
wl = j4pr.labelize(1, true)
wl2 = j4pr.labelize(1, false)
wl3 = j4pr.labelize(indmax, 1:2, true)
wl4 = j4pr.labelize(indmax, 1:2, false)
D = j4pr.datacell(A)
Test.@test j4pr.getx!(D |> wl) == A[2:3,:]
Test.@test j4pr.gety!(D |> wl) == A[1,:]
Test.@test j4pr.getx!(D |> wl2) == A
Test.@test j4pr.gety!(D |> wl2) == A[1,:]
Test.@test j4pr.getx!(D |> wl3) == A[3:3,:]
Test.@test j4pr.gety!(D |> wl3) == 2*ones(3)
Test.@test j4pr.getx!(D |> wl4) == A
Test.@test j4pr.gety!(D |> wl4) == 2*ones(3)



#####################################
# Test class slicing, domain filter #
#####################################

A = [1 2 3;
     4 5 6]
l = [1 1 2]
l2 = [0 1 0;
      1 0 1]

Du = j4pr.datacell(A)
Dl = j4pr.datacell(A,l)
Dl2 = j4pr.datacell(A,l2)

# class slicing
Test.@test j4pr.cslice(Du,[1,]) isa j4pr.DataCell{<:SubArray}
Test.@test j4pr.getobs(j4pr.getx!(j4pr.cslice(Dl,[2]))) == Matrix([3;6])
Test.@test j4pr.getobs(j4pr.getx!(j4pr.cslice(Dl2,[1],1))) == Matrix([2;5])

# domain filter
A = [1.0 2 3;
     4 5 6;
     7 8 9;
     10 11 12;
     13 14 15]

l = [1 1 2]
l2 = [0 1 0;
      1 0 1]

Du = j4pr.datacell(A)
Dl = j4pr.datacell(A,l)
Dl2 = j4pr.datacell(A,l2)
(+Dl2,-Dl2) |> j4pr.filterdomain!(Dict(1=>x->x>2 ? NaN : x)) # the '3' from the first row becomes NaN
Dl2 |> j4pr.filterdomain!(Dict(2:3=>[4 9])) # the '9' from the third row becomes NaN
Dl2 |> j4pr.filterdomain!(Dict([4]=>[11 13; 14 16])) # the '10' from the fourth row becomes NaN
Dl2 |> j4pr.filterdomain!(Dict([5]=>15)) # the '13' and '14' from the fifth row become NaN

Dl2f = j4pr.getx!(Dl2)
v = [ 
       1.0    2.0    NaN;
       4.0    5.0    6.0;
       7.0    8.0    NaN;  
       NaN     11.0   12.0;
       NaN    NaN     15.0
] 
Test.@test isequal(Dl2f,v)



########################
# Test one hot encoder #
########################

A = [1.0 2 3;
     4 5 6;
    ]

# binary encoder 
w = A|> j4pr.ohenc("binary")
T = [1. 6; 2 6]
Test.@test isequal(T|>w, [1.0  NaN;  
			0.0  NaN;  
   			0.0  NaN;
 			NaN  0.0;
 			NaN  0.0;
 			NaN  1.0])

# integer encoder
w = A|> j4pr.ohenc("integer")
T = [1. 6; 2 6]
Test.@test isequal(T|>w, [1.0  NaN; NaN 3.0]) 

# mixed case
A = [1.0 2 3;
     4 5 6;
     7 8 9]
w = A|> j4pr.ohenc(Dict(1=>"binary",2=>"integer"))
T = [1. 6; 2 6; -1 -2]
Test.@test isequal(T|>w, [1.0  NaN; # binary 
			0.0  NaN;   # binary
   			0.0  NaN;   # binary
 			NaN  3.0;   # integer
 			-1   -2])   # as input



#################
# Test sampling #
#################
A = j4pr.datacell([1 2 3; 4 5 6],["a","a","b"])
(As, idxs) =  A |> j4pr.sample(7)
Test.@test j4pr.nobs(As) == 7
Test.@test j4pr.getx!(A)[:,idxs] == j4pr.getobs(j4pr.getx!(As)) 

(As, idxs) =  A |> j4pr.sample(0.5)
Test.@test j4pr.nobs(As) == 2
Test.@test j4pr.getx!(A)[:,idxs] == j4pr.getobs(j4pr.getx!(As)) 

(As, idxs) =  A |> j4pr.sample(Dict("a"=>10, "b"=>2))
Test.@test j4pr.nobs(As) == 12
Test.@test j4pr.getx!(A)[:,idxs] == j4pr.getobs(j4pr.getx!(As)) 
y = j4pr.getobs(j4pr.gety!(As)) 
Test.@test sum(y.=="a") == 10 && sum(y.=="b") == 2



#######################################################################
# Test generic filter (verify that works, functionality is not tested #
#######################################################################

for fo in ["mean", "c-mean", "median", "c-median", "majority", "c-majority", (x,y)->1.0, Dict(1=>"mean",2=>(x,y)->x)]
	w = j4pr.filterg(fo)
	Au = j4pr.datacell(rand(2,10))
	Al = j4pr.datacell(rand(2,10), rand([1,2],10))
	
	Test.@test try
       		Au |> (Au |> w)
     		Al |> (Al |> w)
		true
	catch
		false
	end
end



###############################################################
# Test scaler (verify that works, functionality is not tested #
###############################################################

for so in ["mean", "c-mean", "variance", "c-variance", "domain", Dict(1=>"mean", 2=>"variance")]
	w = j4pr.scaler!(so)
	Au = j4pr.datacell(rand(2,10))
	Al = j4pr.datacell(rand(2,10), rand([1,2],10))
	
	Test.@test try
       		Au |> (Au |> w)
     		Al |> (Al |> w)
		true
	catch
		false
	end
end

# Separate case for 2-sigma scaling, has to fail for unlabeled containers
w = j4pr.scaler!("2-sigma")
Au = j4pr.datacell(rand(2,10))
Al = j4pr.datacell(rand(2,10), rand([1,2],10))

Test.@test try 
	Au |> (Au |>w)
	false
catch
	true
end

Test.@test try 
	Al |> (Al |>w)
	true
catch
	false
end


end

