import Crayons;
import OhMyREPL

function generate_j4pr_colorscheme()
	color_scheme = OhMyREPL.Passes.SyntaxHighlighter.ColorScheme()

	# Keyword 
	OhMyREPL.Passes.SyntaxHighlighter.keyword!(color_scheme, Crayons.Crayon(foreground = :white, bold=true))

	# Number 
	OhMyREPL.Passes.SyntaxHighlighter.number!(color_scheme, Crayons.Crayon(foreground = :white, bold=true))

	# Text 
	OhMyREPL.Passes.SyntaxHighlighter.text!(color_scheme, Crayons.Crayon(foreground = :light_gray, bold=false))

	# Symbol 
	OhMyREPL.Passes.SyntaxHighlighter.symbol!(color_scheme, Crayons.Crayon(foreground = :magenta, bold=true))

	# String
	OhMyREPL.Passes.SyntaxHighlighter.string!(color_scheme, Crayons.Crayon(foreground = :light_magenta, bold=true))

	# Operators 
	OhMyREPL.Passes.SyntaxHighlighter.op!(color_scheme, Crayons.Crayon(foreground = :light_cyan, bold=true))

	# Macros 
	OhMyREPL.Passes.SyntaxHighlighter.macro!(color_scheme, Crayons.Crayon(foreground = :light_green, bold=true))

	# Comment
	OhMyREPL.Passes.SyntaxHighlighter.comment!(color_scheme, Crayons.Crayon(foreground = :dark_gray, bold=true))

	# Argument definitions 
	OhMyREPL.Passes.SyntaxHighlighter.argdef!(color_scheme, Crayons.Crayon(foreground = :light_blue, bold=true))

	# Function definitions 
	OhMyREPL.Passes.SyntaxHighlighter.function_def!(color_scheme, Crayons.Crayon(foreground = :default, bold=true))

	# Function call 
	OhMyREPL.Passes.SyntaxHighlighter.call!(color_scheme, Crayons.Crayon(foreground = :default, bold=true))

	# Error 
	OhMyREPL.Passes.SyntaxHighlighter.error!(color_scheme, Crayons.Crayon(foreground = :light_red, bold=false))

	return color_scheme
end

function apply_j4pr_colorscheme()
	
	# Apply colorscheme
	j4prscheme = generate_j4pr_colorscheme()
	OhMyREPL.Passes.SyntaxHighlighter.add!("j4pr", j4prscheme)
	OhMyREPL.colorscheme!("j4pr")

	# Set bracket highlighting
	c = Crayons.Crayon(background = :white, foreground=:black)
	OhMyREPL.Passes.BracketHighlighter.setcrayon!(c)

	# Bracket autocompletion off
	OhMyREPL.enable_autocomplete_brackets(false)

	# Prompt
	OhMyREPL.input_prompt!("julia-[j4πr]>", :light_green)
	#OhMyREPL.output_prompt!("", :white)
end

apply_j4pr_colorscheme()
