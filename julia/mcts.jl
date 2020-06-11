using Random
using Parameters
#using Debugger

Random.seed!(1)



abstract type TreeSearch end

@with_kw mutable struct Mcts <: TreeSearch
	depth::Int64 = 12
	iters::Int64 = 100
	tMax::Int64 = 200

	c::Float64 = 1.0

	mdp = nothing
	nnet = nothing
	reuseTree::Bool = false
	pi0 = nothing

	#Tree = Set()
	#Nsa = Dict() 
	#Ns = Dict() 
	#Q = Dict() 

	Tree = Set{Array{Float64,1}}()
	Nsa = Dict{Tuple{Array{Float64,1}, Float64}, Float64}() 
	Ns = Dict{Array{Float64,1}, Float64}() 
	Q = Dict{Tuple{Array{Float64,1}, Float64}, Float64}() 
end

function init!(m::Mcts; 
               mdp = nothing, 
			   piRollout = nothing,
			   depth::Int64 = 12,
			   iters::Int64 = 100,
			   explorConst::Float64 = 1.0,
			   tMaxRollouts::Int64 = 200,
			   reuseTree::Bool = true,
			   nnet = nothing)
	m.depth = depth
	m.iters = iters
	m.c = explorConst
	m.tMax = tMaxRollouts
	m.reuseTree = reuseTree
	m.mdp = mdp
	m.nnet = nnet
	m.pi0 = piRollout

	resetTree!(m)
end

function resetTree!(m::Mcts)
	m.Tree = Set{Array{Float64,1}}()
	m.Nsa = Dict{Tuple{Array{Float64,1}, Float64}, Float64}() 
	m.Ns = Dict{Array{Float64,1}, Float64}() 
	m.Q = Dict{Tuple{Array{Float64,1}, Float64}, Float64}() 

	sizehint!(m.Tree, m.iters)
	sizehint!(m.Nsa, m.iters)
	sizehint!(m.Ns, m.iters)
	sizehint!(m.Q, m.iters)
end

function act(m::Mcts, state::Array{Float64,1}, obstacles::Array{Tuple{Float64, Float64}, 1})::Float64
	if m.reuseTree
		resetTree!(m)
	end
	action = selectAction(m, state, m.depth)
	return action
end

function selectAction(m::Mcts, s::Array{Float64,1}, d::Int64)::Float64
	for i in 1:m.iters
		simulate(m, s, d, m.pi0) 
	end
	q, action = maximum([(m.Q[(s,a)], a) for a in actions(m.mdp, s)])
	#println("q=$q")
	return action
end

function simulate(m::Mcts, s::Array{Float64,1}, d::Int64, pi0)::Float64
	if isEnd(m.mdp, s)>0 #[1]
		return 0
	end
	if d == 0 # we stop exploring the tree, just estimate Qval here
		return rollout(m, s, m.tMax, pi0)
	end
	if !in(s, m.Tree)
		for a in actions(m.mdp, s)
			m.Nsa[(s,a)], m.Ns[s], m.Q[(s,a)] =  0, 1, 0.
		end
		push!(m.Tree, s)
		return rollout(m, s, m.tMax, pi0)
	end
	a::Float64 = maximum([(m.Q[(s,a)]+m.c*sqrt(log(m.Ns[s])/(1e-5 + m.Nsa[(s,a)])), a) for a in actions(m.mdp, s)])[2]
	sp::Vector{Float64}, r::Float64, _ = sampleSuccReward(m.mdp, s, a)
	q::Float64 = r + discount(m.mdp) * simulate(m, sp, d-1, pi0)
	m.Nsa[(s,a)] += 1
	#println(typeof(s))
	m.Ns[s] += 1
	m.Q[(s,a)] += (q-m.Q[(s,a)])/m.Nsa[(s,a)]
	return q
end

function rollout(m::Mcts, s::Array{Float64,1}, d::Int64, pi0)::Float64
	if d==0 || isEnd(m.mdp, s)>0#[1]
		return 0
	else
		a::Float64 = pi0(m.mdp, s)
		sp::Vector{Float64}, r::Float64, _ = sampleSuccReward(m.mdp, s, a)
		return r + discount(m.mdp) * rollout(m, sp, d-1, pi0)
	end
end
