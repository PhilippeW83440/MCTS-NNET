using Random
using Parameters
using Debugger

using Statistics
using Printf
using Logging
using LoggingExtras

#using Traceur
#using Profile

Random.seed!(1)

io = open("log.txt", "a")
logio = SimpleLogger(io)
logerr = ConsoleLogger(stderr)
#logger = FileLogger(Dict(Logging.Info => "info.log", Logging.Error => "error.log"))
#logger = IOLogger()
#global_logger(logger)

demux_logger = TeeLogger(
    #MinLevelLogger(FileLogger("info.log"), Logging.Info),
    MinLevelLogger(logio, Logging.Info),
    MinLevelLogger(logerr, Logging.Info),
);


export ActMdp
include("mdp.jl")
include("mcts.jl")
include("search.jl")
include("mpc.jl")


abstract type Scn end

@with_kw mutable struct ScnMulti{T} <: Scn
	nobjs::Int32 = 10
	start = Vector{T}[]
	train_set = Vector{T}[]
	dev_set = Vector{T}[]
	test_set = Vector{T}[]
end

"""
scn init
"""
function init!(scn::ScnMulti; 
				nobjs=10)

	scn.nobjs = nobjs
	scn.start = randomStartState(scn)
	scn.train_set = [randomStartState(scn) for i in 1:10000]
	scn.dev_set = [randomStartState(scn) for i in 1:100]
	scn.test_set = [randomStartState(scn) for i in 1:100]
end


"""
scn randomStartState
"""
function randomStartState(scn::ScnMulti)
	state = []
	for n in 1:floor(Int, scn.nobjs/2)
		x = convert(Float64, rand(0:50))
		y = rand(25:190)
		vx = rand(10:25)
		vy = rand(0:5)
		obj = [x, y, vx, vy]
		push!(state, obj)
	end

	for n in 1:floor(Int, scn.nobjs/2)
		x = convert(Float64, rand(150:200))
		y = rand(25:190)
		vx = -rand(10:25)
		vy = -rand(0:5)
		obj = [x, y, vx, vy]
		push!(state, obj)
	end
	return state
end


"""
scn projectState: onto x=100
"""
function projectState(scn::ScnMulti, state)::Vector{Tuple{Float64, Float64}}
	obstacles = Tuple{Float64, Float64}[] # list of (s, t) tuples
	for s in state
		x,y,vx,vy = s
		t_cross = floor( ((100-x)/vx)/0.25 ) * 0.25 # TODO CLEAN THAT
		y_cross = y + t_cross * vy
		#push!(obstacles, (y_cross, t_cross))
		push!(obstacles, (t_cross, y_cross))
	end
	sort!(obstacles)
	# return [obstacles[1]] # just for tests/checks XXX
	return obstacles
end


# julia 1.3.1 used
# run: julia scn.jl
# or : julia scn.jl dp
# or : julia scn.jl mcts


# 1) Create a scenario or scene
scn = ScnMulti{Float64}()
init!(scn)
println(scn.nobjs)

# 2) Get projected state
pstate = projectState(scn, scn.start)
println(pstate)
#exit()

# 3) Setup ActMdp with list of spatio-temporal obstacles
#mdp = ActMdp{Float64}()
mdp = ActMdp()
init!(mdp, obstacles=pstate)
println(mdp.obstacles)
println(mdp.goal)

println(isEnd(mdp, startState(mdp)))
println(actions(mdp, startState(mdp)))
for i in 1:10
	println(pi0(mdp, startState(mdp)))
end

s=startState(mdp)

for i in 1:100
	a = pi0(mdp, s)
	sp, r, done = sampleSuccReward(mdp, s, a)
	println("(s, a, r, sp)=($s, $a, $r, $sp)")
	global s = sp # cf https://discourse.julialang.org/t/error-variable-inside-a-loop/21504
	if done>0#[1]
		println(done)
		break
	end
end


# Default to MCTS
length(ARGS) > 0 ? algo=ARGS[1] : algo="mcts"

#mcts = Mcts{Int64, Float64}()

if algo == "mcts"
	agent = Mcts()
	init!(agent, mdp=mdp, piRollout=pi0)
	print(agent)
	print(agent.mdp)
elseif algo == "mpc"
	mdlParams = MdlParams()
	mpcParams = MpcParams()
	agent = MpcModel(mpcParams, mdlParams)
	println("mpc agent: ", agent)
elseif algo == "dp"
	actSearch = actDP
elseif algo == "ucs"
	actSearch = actUCS
end

# test_algo.py

metric_scores = []
metric_hardbrakes = []
metric_steps_to_goal = []
metric_steps_to_collision = []
metric_speed_at_collision = []
metric_runtime = []
success = 0


for (num_s, scn_start) in enumerate(scn.dev_set)
	#println("num_s = $num_s")
	#if num_s  != 95
	#	continue
	#end
	println("Test $num_s: success=$success/$(num_s-1) scn_start=$scn_start")
	obstacles = projectState(scn, scn_start)
	init!(mdp, obstacles=obstacles)
	#println("obstacles=$(mdp.obstacles)")
	s = mdp.state
	score, hardbrakes, speed_at_collision = 0, 0, nothing
	sequence = []
	runtimes = []
	if algo == "mcts" || algo == "mpc"
		algo == "mcts" && resetTree!(agent)
		for steps in 1:1000
			#runtime = @elapsed a = pi0(mdp, s)
			#@time a = act(mcts, s)
			#@trace a = act(mcts,s)
			#@trace(act(mcts, s), modules=[Main])
			#@profile a = act(mcts,s)
			runtime = @elapsed a = act(agent, s, obstacles)
			push!(runtimes, runtime)
			#println("(s, a)=($s, $a)")
			push!(sequence, a)
			if a <= -4.0
				hardbrakes += 1
			end
			sp, r, done = sampleSuccReward(mdp, s, a)
			println("profiles: $(sp[1]), $(sp[2]), $a")
			score += r
			if done>0#[1]
				if done==2 #[2] == "goal"
					global success += 1
					push!(metric_steps_to_goal, steps)
				elseif done==1 #[2] == "collision"
					push!(metric_steps_to_collision, steps)
					speed_at_collision = sp[2]
					push!(metric_speed_at_collision, speed_at_collision)
				end
				break
			end
			s = sp
		end
		runtime = mean(runtimes)
	elseif algo == "dp" || algo == "ucs"
		mdp.tMax = 12.0
		#@time cost, history = actDP(mdp, s)
		runtime = @elapsed score, sequence, s = actSearch(mdp, s)
		steps = s[3]/mdp.dt + 1
		hardbrakes = length(filter(x -> x<=-4.0, sequence))
		if score < 1
			global success += 1
			push!(metric_steps_to_goal, steps)
		else
			speed_at_collision = s[2]
			push!(metric_steps_to_collision, steps)
			push!(metric_speed_at_collision, speed_at_collision)
		end
	end
	push!(metric_runtime, runtime)
	push!(metric_scores, score)
	push!(metric_hardbrakes, hardbrakes)
	with_logger(demux_logger) do
		@info("Test $num_s: score $score hardbrakes $hardbrakes TTG $(s[3]) runtime $runtime collision_speed $speed_at_collision")
		@info("  actions $sequence")
	end

end

score = mean(metric_scores)
success_rate = success/length(scn.dev_set)
runtime = mean(metric_runtime)
hardbrakes = mean(metric_hardbrakes)
length(metric_steps_to_goal) > 0 ? steps_to_goal = mean(metric_steps_to_goal) : steps_to_goal = nothing
length(metric_steps_to_collision) > 0 ? steps_to_collision = mean(metric_steps_to_collision) : steps_to_collision = nothing
length(metric_speed_at_collision) > 0 ? speed_at_collision = mean(metric_speed_at_collision) : speed_at_collision = nothing

with_logger(demux_logger) do
	@info("METRICS mean values => score $score, success_rate $success_rate, runtime $runtime, hardbrakes $hardbrakes, steps_to_goal $steps_to_goal, steps_to_collision $steps_to_collision, speed_at_collision $speed_at_collision")
end
	

close(io)
