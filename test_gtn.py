from juliacall import Main as jl
jl.seval("import Pkg; Pkg.activate(\"tn_env\")")
jl.seval("using GenericTensorNetworks")
print("GTN OK")
