using Random
using Parameters
#using Debugger
using DataStructures

Random.seed!(1)

function actUCS(mdp::ActMdp, s::Vector{Float64})::Tuple{Float64, Vector{Float64}, Vector{Float64}}
	# frontier(state, pastCost)
	frontier = PriorityQueue{Vector{Float64} , Float64}()
	# explored[s] = (pastCost, prev_s, prev_a, prev_cost)
	explored = Dict{Vector{Float64}, Tuple{Float64, Vector{Float64}, Float64, Float64}}()
	# previous[sp] = (s, a, cost)
	previous = Dict{Vector{Float64}, Tuple{Vector{Float64}, Float64, Float64}}()

	#sizehint!(frontier, 1500000)
	sizehint!(explored, 1_500_000)
	sizehint!(previous, 1_500_000)

	frontier[s] = 0.0
	previous[s] = ([0., 0., -1.0], 0.0, 0.0)

	while true
		s, pastCost = dequeue_pair!(frontier)
		prev_s, prev_a, prev_cost = previous[s]
		explored[s] = (pastCost, prev_s, prev_a, prev_cost)
		if isEnd(mdp, s) > 0
			global minCost = pastCost
			break
		end
		for res in succAndCost(mdp, s)
			a, sp, cost = res
			if haskey(explored, sp)
				continue
			end
			newCost = pastCost + cost
			if !haskey(frontier, sp) || newCost < frontier[sp]
				frontier[sp] = newCost
				previous[sp] = (s, a, cost)
			end
		end
	end

	#sf = deepcopy(s)
	sf = copy(s)

	# recover history
	println("n_states explored : ", length(explored))
	history = Tuple{Float64, Float64, Vector{Float64}}[]
	sequence = Float64[]
	while s[3] > 0 # !isnothing(s)
		pastCost, prev_s, prev_a, prev_cost = explored[s]
		if prev_s[3] > 0 # !isnothing(prev_s)
			push!(history, (prev_a, pastCost, prev_s))
			push!(sequence, prev_a)
		end
		s = prev_s
	end

	return minCost, reverse(sequence), sf
end

function actDP(mdp::ActMdp, s::Vector{Float64})
	futureCost = Dict{Vector{Float64}, Tuple{Float64, Float64, Vector{Float64}, Float64}}()

	sizehint!(futureCost, 2500000)

	function recurse(mdp::ActMdp, s::Vector{Float64})::Float64
		if haskey(futureCost, s)
			return futureCost[s][1]
		end
		if isEnd(mdp, s) > 0
			return 0
		end
		succ = []
		for res in succAndCost(mdp, s)
			a, sp, cost = res
			push!(succ, (cost + recurse(mdp, sp), a, sp, cost))
		end
		futureCost[s] = minimum(succ)
		#futureCost[s] = minimum([(cost + recurse(sp), a, sp, cost) for a, sp, cost in succAndCost(mdp, s)])
		return futureCost[s][1]
	end

	minCost = recurse(mdp, s)

	# recover history
	println("n_states explored : ", length(futureCost))
	history = []
	sequence = []
	while isEnd(mdp, s) == 0
		totalCost, a, sp, cost = futureCost[s]
		push!(history, (a, totalCost, s))
		push!(sequence, a)
		s = sp
	end

	#println("cost: ", minCost)
	#println("history: ", history)
	return minCost, sequence, history[end][3]
end
