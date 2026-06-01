import Pkg
Pkg.activate("tn_env")
using GenericTensorNetworks
using Graphs
import JSON

while !eof(stdin)
    line = readline(stdin)
    isempty(strip(line)) && continue
    data = JSON.parse(line)
    n = data["n"]; edges = data["edges"]
    Jvals = Float64.(data["J"]); h = Float64.(data["h"])

    g = SimpleGraph(n)
    for e in edges; add_edge!(g, e[1], e[2]); end

    Jmap = Dict{Tuple{Int,Int},Float64}()
    for (k, e) in enumerate(edges)
        a, b = min(e[1],e[2]), max(e[1],e[2])
        Jmap[(a,b)] = Jvals[k]
    end
    Jordered = [Jmap[(min(src(e),dst(e)), max(src(e),dst(e)))]
                for e in Graphs.edges(g)]

    problem = SpinGlass(g; J=Jordered, h=h)
    res = solve(problem, SingleConfigMax())[]
    cfg = [Int(c) for c in res.c.data]
    println(JSON.json(Dict("energy"=>res.n, "config"=>cfg)))
    flush(stdout)
end
