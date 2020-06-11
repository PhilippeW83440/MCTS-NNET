using Random
using Parameters
using Debugger

Random.seed!(1)

GOAL = "goal"
COLLISION = "collision"
ONGOING = "ongoing"


abstract type Mdp end

@with_kw mutable struct ActMdp <: Mdp
	s_max::Float64 = 200.0
	sd_max::Float64 = 30.0
	gamma::Float64 = 1.0

	dt::Float64 = 0.250 # ms
	tMax::Float64 = 25 # secs

	goal::Vector{Float64} = [200.0, 30.0] # column vector
	obstacles::Array{Tuple{Float64, Float64}} = [] # list of (s,t) tuples
	dcol::Float64 = 10.0 # meters

	start_state::Vector{Float64} = [0.0, 20.0, 0.0] # s, sd (column vector)
	state::Vector{Float64} = [0.0, 20.0] # s, sd (column vector)
	actions::Vector{Float64} = [-4, -2, -1, 0, 1, 2]
	# Transtion model
	Ts::Array{Float64,2} = [1.0 dt 0.0; 0  1 0; 0 0 1]
	Ta::Array{Float64,1} = [dt^2/2; dt; 0]
	Tt::Array{Float64,1} = [0.0; 0; dt]

end

"""
mdp init
"""
function init!(m::ActMdp; 
				s_max::Float64 = 200.0, 
				sd_max::Float64 = 30.0, 
				gamma::Float64 = 1.0, 
				dt::Float64 = 0.25, 
				tMax::Float64 = 30.0, 
				goal::Vector{Float64} = [200.0, 30.0],
				obstacles=[],
				dcol::Float64 = 10.0,
				start_state::Vector{Float64} = [0.0, 20.0, 0.0],
				actions::Vector{Float64} = [-4.0, -2, -1, 0, 1, 2])

	m.s_max = s_max
	m.sd_max = sd_max
	m.gamma = gamma

	m.dt = dt
	m.tMax = tMax

	m.goal[1], m.goal[2] = goal[1], goal[2]
	m.obstacles = deepcopy(obstacles)
	m.dcol = dcol

	m.start_state = start_state
	m.state = start_state
	m.actions = actions
end

"""
mdp startState
"""
function startState(m::ActMdp)
	#return Tuple(m.start_state)
	return m.start_state
end

"""
mdp isEnd
"""
function isEnd(m::ActMdp, s::Array{Float64,1})::Int64
	ret = 0
	if s[1] >= m.goal[1]
		ret = 2 # true, "goal"
	elseif s[2] < 0 || s[3] > m.tMax
		ret = 1 # true, "collision" # handle driving backward as collision
	elseif s[3] > m.obstacles[end][1] # small optim
		ret = 0 # false, "ongoing"
	else
		for obstacle in m.obstacles
			#s_cross, t_cross = obstacle
			t_cross, s_cross = obstacle
			if s[3] == t_cross
				#println("==========> DBG: t_cross=$t_cross s_cross=$s_cross s_ego=$(s[1])")
				if abs(s_cross - s[1]) <= m.dcol
					ret = 1 # true, "collision"
					return ret
				end
			end
		end
	end
	return ret #false, "ongoing"
end

"""
mdp actions
"""
function actions(m::ActMdp, s::Array{Float64,1})::Vector{Float64}
	return m.actions
end

"""
mdp discount
"""
function discount(m::ActMdp)
	return m.gamma
end

"""
mdp pi0
"""
#function pi0(m::ActMdp, s)
#	return rand(actions(m, s)::Array)
#end

# use a generic function
pi0 = (m::ActMdp,s)-> rand(actions(m, s)::Array)::Float64

"""
mdp succProbReward
	returns sp, proba=T(s,a,sp), R(s,a,sp)
"""
function succProbReward(m::ActMdp, s::Array{Float64,1}, a::Float64)
	# UNUSED
	results = []
	return results
end

"""
mdp sampleSuccReward
"""
#function sampleSuccReward(m::ActMdp, s::Array{Float64,1}, a::Float64)::Tuple{Vector{Float64},Float64,Tuple{Bool, String}}
function sampleSuccReward(m::ActMdp, s::Array{Float64,1}, a::Float64)::Tuple{Vector{Float64},Float64,Int64}
	sp = m.Ts * s + m.Ta*a + m.Tt
	done = isEnd(m, sp)
	if sp[2] < 0 || 1==done #"collision" == done[2]
		r = -1.0
	elseif a <= -4
		r = -0.002
	else
		r = -0.001
	end
	#return Tuple(sp), r
	return sp, r, done
end

function succAndCost(m::ActMdp, s::Array{Float64, 1})::Array{Tuple{Float64, Vector{Float64}, Float64}}
	res = Tuple{Float64, Vector{Float64}, Float64}[] # (action, nextState, cost)
	for a in actions(m, s)
		sp, r, _ = sampleSuccReward(m, s, a)
		push!(res, (a, sp, -r))
	end
	return res
end
