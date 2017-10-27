"""
	version(vstr::String=j4pr_version)

Prints the current J4PR version and ASCII art. The version is
contained in the global constant `j4pr_version`.
"""
function version(vstr::String=j4pr_version)
	commit = ""
	date = ""
	try 
		readmehtod = x->nothing
		if v"0.6" <= VERSION < v"0.7-"
			readmethod = x->readstring(x) 			# Julia 0.6
		else
			readmethod = x->read(x,String) 			# Julia 0.7+
		end
		commit = open(`git show --oneline -s`) do x
				readmethod(x)[1:7]
		end

		date = open(`git show -s --format="%ci"`) do x
			readmethod(x)[1:10]
		end
	catch
		commit = "67a5c10+"
		date = "2017-10-27"
	end

	vers ="
 _  _
(_\\/_)                  |  This is a small library and package wrapper written at 0x0α Research.
(_/\\_)                  |  Type \"?j4pr\" for general documentation. 
   _ _   _  _____ _ _   |  Look inside src/j4pr.jl for a list of available algorithms.
  | | | | |/____ / ` |  |  
  | | |_| | |  | | /-/  |  Version $(vstr) \"The Monolith\" commit: $(commit) ($(date))
 _/ |\\__  | |  | | |    |  
|__/    |_|_|  |_|_|    |  License: MIT, view ./LICENSE.md for details.
\n\n"

	print(vers)
end
