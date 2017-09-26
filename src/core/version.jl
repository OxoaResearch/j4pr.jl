"""
	version(vstr::String=j4pr_version)

Prints the current J4PR version and ASCII art. The version is
contained in the global constant `j4pr_version`.
"""
function version(vstr::String=j4pr_version)
	rev = ""
	date = ""
	try 
		readmehtod = x->nothing
		if (VERSION > v"0.6")
			readmethod = x->read(x,String) 			# Julia 0.7+
		else
			readmethod = x->readstring(x) 			# Julia 0.6
		end
		rev = open(pipeline(`svn info /home/zgornel/projects/j4pr`,`grep Revision`)) do x
				replace(split(readmethod(x)," ")[2],"\n","")
		end

		date = open(pipeline(`svn info /home/zgornel/projects/j4pr`,`grep Date`)) do x
				split(split(readmethod(x),"Date:")[2]," ")[2]
		end
	catch
		rev = "162+"
		date = "2017-09-22"
	end

	vers ="
 _  _
(_\\/_)                  |  This is a small library and package wrapper written at 0x0α Research.
(_/\\_)                  |  Type \"?j4pr\" for general documentation. 
   _ _   _  _____ _ _   |  Look inside src/j4pr.jl for a list of available algorithms.
  | | | | |/____ / ` |  |  
  | | |_| | |  | | /-/  |  Version $(vstr) \"The Monolith\" revision: $(rev) ($(date))
 _/ |\\__  | |  | | |    |  
|__/    |_|_|  |_|_|    |  License: MIT, view ./LICENSE.md for details.
\n\n"

	print(vers)
end
