"""
	abstract type AbstractCell{T,S,U}

The basic type of J4PR. Most of the types are its subtypes to ensure that operations
between various types of `Cells` are possible, especially when creating `Pipes` that
can have arbitrary levels of nesting.
"""
abstract type AbstractCell{T,S,U} end



##############################################################################################################################
# Operators [AbstractCells] 
##############################################################################################################################

# Simple math operators (quite outdated but may still be useful)
+(ac::T where T<:AbstractCell) = ac.x
-(ac::T where T<:AbstractCell) = ac.y
~(ac::T where T<:AbstractCell) = dump(ac)



# Define any new aliases used throughout subsequent code
const PTuple{T} = Tuple{Vararg{<:T}}
