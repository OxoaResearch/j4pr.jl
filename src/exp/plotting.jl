# First basic plot function
generatecolors() = [:red, :blue, :green, :yellow, :white, :cyan, :magenta]



# Lineplots TODO: Add LearnBase.ObsDim support so that one can either plot samples or variables (so far only variable plotting supported)
lineplot(xidx::Int=1; color=:white, kwargs...) = FunctionCell(lineplot, (xidx,), "Line Plot (xidx=$xidx)"; color=color, kwargs...)
lineplot(x::T where T<:CellData, xidx::Int=1; color=:white, kwargs...) = lineplot(getobs(_variable_(x,xidx)); color=color, kwargs...)
lineplot(x::T where T<:AbstractArray, xidx::Int=1; color=:white, kwargs...) = UnicodePlots.lineplot(getobs(_variable_(x,xidx)); color=color, kwargs...)
lineplot(f::Function, xidx::Int=1; color=:white, kwargs...) = FunctionCell(x->UnicodePlots.lineplot(f, getobs(_variable_(x,xidx)); color=color, kwargs...)
									   , (), "Line Plot (f=$f, xidx=$xidx)")
lineplot(x::T where T<:CellData, f::Function, xidx::Int=1; color=:white, kwargs...) = UnicodePlots.lineplot(f, getobs(_variable_(x,xidx)); color=color, kwargs...)

#=
# Cumbersome for vector of functions; TODO: UnicodePlots.lineplot! should be used by superimposing the plots of vf[i]( getobs(_variable_(x,xidx)) )
lineplot(vf::Vector{Function}, idx::Int=1; kwargs...) = FunctionCell(x->UnicodePlots.lineplot([y->f(y) for f in vf], minimum(x), maximum(x); kwargs...)
								, (), "Line Plot (vf=$(vf), idx=$idx")
lineplot{T <: CellData}(x::T, vf::Vector{Function}, idx::Int=1; kwargs...) = begin
	data = getobs(_variable_(x,idx))
	UnicodePlots.lineplot([, data, minimum(data), maximum(data); kwargs...)
end
=#



# Scatterplots
scatterplot(xidx::Int=1, yidx::Int=2; kwargs...) = j4pr.FunctionCell(scatterplot, (xidx, yidx), "Scatter Plot (xidx=$xidx, yidx=$yidx)"; kwargs...)
scatterplot(dc::T where T<:CellDataU, xidx::Int=1, yidx::Int=2; color=:white, kwargs...) = scatterplot(getx!(dc), xidx, yidx; color=color, kwargs...)
scatterplot(dc::T where T<:CellDataL, xidx::Int=1, yidx::Int=2; kwargs...) = scatterplot(strip(dc), xidx, yidx; kwargs...)

scatterplot(x::T where T<:AbstractMatrix, xidx::Int=1, yidx::Int=2; color=:white, kwargs...) = begin

	@assert nvars(x) > 1 "[scatterplot] Dimensionality of the data has to be larger than 1."
	
	x1 = getobs(_variable_(x,xidx))
	x2 = getobs(_variable_(x,yidx))
	maxx1 = maximum(x1)
	minx1 = minimum(x1)
	maxx2 = maximum(x2)
	minx2 = minimum(x2)
	
	# Plot
	p = UnicodePlots.scatterplot(x1, x2, xlim=[minx1,maxx1], ylim=[minx2, maxx2]; color=color, kwargs...)		
end

scatterplot(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractVector, xidx::Int=1, yidx::Int=2; kwargs...) = begin
	
	@assert nvars(x[1]) > 1 "[scatterplot] Dimensionality of the data has to be larger than 1."
	
	# Read needed data
	data = getobs(x[1])	
	labels = getobs(x[2]) 
	classes = sort(unique(labels))
	
	# Plot 1-st class
	x1 = getobs(_variable_(data,xidx))
	x2 = getobs(_variable_(data,yidx))
	
	colors = generatecolors()
	cmask = labels .== classes[1]
	x1c = x1[cmask]
	x2c = x2[cmask]
	
	maxx1 = maximum(x1)
	minx1 = minimum(x1)
	maxx2 = maximum(x2)
	minx2 = minimum(x2)
	
	p = UnicodePlots.scatterplot(x1c, x2c, color = colors[1], xlim=[minx1,maxx1], ylim=[minx2, maxx2]; kwargs...)		
	
	# Plot the rest of the classes
	for i in 2:length(classes)
		cmask = labels .== classes[i]
		x1c = getobs(_variable_(data,xidx))[cmask]
		x2c = getobs(_variable_(data,yidx))[cmask]
		UnicodePlots.scatterplot!(p, x1c, x2c, color = colors[(i-1)%length(colors)+1])
 	end
	p	
end



# 2-D density plots
densityplot2d(xidx::Int=1, yidx::Int=2; kwargs...) = j4pr.FunctionCell(densityplot2d, (xidx, yidx), "Density Plot 2D (xidx=$xidx, yidx=$yidx)"; kwargs...)
densityplot2d(dc::T where T<:CellDataU, xidx::Int=1, yidx::Int=2; color=:white, kwargs...) = densityplot2d(getx!(dc), xidx, yidx; color=color, kwargs...)
densityplot2d(dc::T where T<:CellDataL, xidx::Int=1, yidx::Int=2; kwargs...) = densityplot2d(strip(dc), xidx, yidx; kwargs...)

densityplot2d(x::T where T<:AbstractMatrix, xidx::Int=1, yidx::Int=2; color=:white, kwargs...) = begin

	@assert nvars(x) > 1 "[scatterplot] Dimensionality of the data has to be larger than 1."
	
	x1 = getobs(_variable_(x,xidx))
	x2 = getobs(_variable_(x,yidx))
	maxx1 = maximum(x1)
	minx1 = minimum(x1)
	maxx2 = maximum(x2)
	minx2 = minimum(x2)
	
	# Plot
	p = UnicodePlots.densityplot(x1, x2, xlim=[minx1,maxx1], ylim=[minx2, maxx2]; color=color, kwargs...)		
end

densityplot2d(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractVector, xidx::Int=1, yidx::Int=2; kwargs...) = begin
	
	@assert nvars(x[1]) > 1 "[scatterplot] Dimensionality of the data has to be larger than 1."
	
	# Read needed data
	data = getobs(x[1])	
	labels = getobs(x[2]) 
	classes = sort(unique(labels))
	
	# Plot 1-st class
	x1 = getobs(_variable_(data,xidx))
	x2 = getobs(_variable_(data,yidx))
	
	colors = generatecolors()
	cmask = labels .== classes[1]
	x1c = x1[cmask]
	x2c = x2[cmask]
	
	maxx1 = maximum(x1)
	minx1 = minimum(x1)
	maxx2 = maximum(x2)
	minx2 = minimum(x2)
	
	p = UnicodePlots.densityplot(x1c, x2c, color = colors[1], xlim=[minx1,maxx1], ylim=[minx2, maxx2]; kwargs...)		
	
	# Plot the rest of the classes
	for i in 2:length(classes)
		cmask = labels .== classes[i]
		x1c = getobs(_variable_(data,xidx))[cmask]
		x2c = getobs(_variable_(data,yidx))[cmask]
		UnicodePlots.densityplot!(p, x1c, x2c, color = colors[(i-1)%length(colors)+1])
 	end
	p	

end



# Histogram ... of sorts, using lineplot
densityplot1d(xidx::Int=1; nbins=100, kwargs...) = j4pr.FunctionCell(densityplot1d, (xidx,), "Density Plot (xidx=$xidx, nbins=$nbins)"; nbins=nbins, kwargs...)
densityplot1d(dc::T where T<:CellDataU, xidx::Int=1; nbins=100, color=:white, kwargs...) = densityplot1d(getx!(dc), xidx; nbins=nbins, color=color, kwargs...)
densityplot1d(dc::T where T<:CellDataL, xidx::Int=1; nbins=100, kwargs...) = densityplot1d(strip(dc), xidx; nbins=nbins, kwargs...)
densityplot1d(x::T where T<:AbstractMatrix, xidx::Int=1; nbins=100, color=:white, kwargs...) = densityplot1d(getobs(_variable_(x,xidx)), xidx; nbins=nbins, color=color, kwargs...)

densityplot1d(x::T where T<:AbstractVector, xidx::Int=1; nbins=100, color=:white, kwargs...) = begin

	h = fit(Histogram, x, closed=:left, nbins=nbins)
	r = h.edges[1]
	x1 = r[1:length(r)-1]+0.5*step(r)
	x2 = h.weights
	
	p = UnicodePlots.lineplot(x1,x2; color=color, kwargs...) 
end

densityplot1d(x::Tuple{T,S} where T<:AbstractArray where S<:AbstractVector, xidx::Int=1; nbins=nbins, kwargs...) = begin
	
	# Read needed data
	data = getobs(_variable_(x[1],xidx))	
	labels = getobs(x[2]) 
	classes = sort(unique(labels))

	colors = generatecolors()
	cmask = labels .== classes[1]
	maxx = maximum(data)
	minx = minimum(data)
	miny = 0
	maxy = 0
	
	# search first to find maximum height for plot
	for c in classes
		h = fit(Histogram, data[labels.==c], nbins=nbins)
		maxy = max(maxy,maximum(h.weights))
	end
	
	# Plot 1-st class
	h = fit(Histogram, data[cmask], nbins=nbins)
	r = h.edges[1]
	x = r[1:length(r)-1]+0.5*step(r)
	y = h.weights
	maxy = max(maxy,maximum(y))
	p = UnicodePlots.lineplot(x, y, color=colors[1], xlim=[minx,maxx], ylim=[miny, maxy]; kwargs...) 
	
	# Plot the rest of the classes
	for i in 2:length(classes)
		cmask = labels .== classes[i]
		h = fit(Histogram, data[cmask], nbins=nbins)
		r = h.edges[1]
		x = r[1:length(r)-1]+0.5*step(r)
		y = h.weights
		maxy = max(maxy,maximum(y))
		UnicodePlots.lineplot!(p, x, y, color = colors[(i-1)%length(colors)+1], xlim=[minx,maxx],ylim=[miny,maxy] ; kwargs...)
 	end
	p	
end



# ROC curve plots 
using j4pr.ROC: ComplexOP, AbstractPerfMetric, TPr, FPr, TNr, FNr 

rocplot(x::CellFunT{<:Model{<:ComplexOP}}, xmetric::AbstractPerfMetric=FPr(), 
		ymetric::AbstractPerfMetric=TPr(); color=:white, kwargs...) = 
	rocplot(getx!(x), xmetric, ymetric; color=color, kwargs...)

rocplot(model::Model{<:ComplexOP}, xmetric::AbstractPerfMetric=FPr(), 
		ymetric::AbstractPerfMetric=TPr(); color=:white, kwargs...) = 
	rocplot(model.data, xmetric, ymetric; color=color, kwargs...)

rocplot(op::ComplexOP, xmetric::AbstractPerfMetric=FPr(), ymetric::AbstractPerfMetric=TPr(); color=:white, kwargs...)=
begin
	
	_get_data_(::TPr, op::ComplexOP) = 1-op.r.pmiss # TPr
	_get_data_(::FNr, op::ComplexOP) = op.r.pmiss   # FNr
	_get_data_(::FPr, op::ComplexOP) = op.r.pfa   	# FPr
	_get_data_(::TNr, op::ComplexOP) = 1-op.r.pfa  	# TNr
	
	# Get data from op
	x = _get_data_(xmetric, op)
	y = _get_data_(ymetric, op)

	p = UnicodePlots.lineplot(x, y; color=color, kwargs...) 
end
