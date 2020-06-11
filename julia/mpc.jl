using JuMP, Ipopt
using Statistics
using LinearAlgebra


mutable struct MdlParams
    dsafety::Float64
    dt::Float64
    MdlParams(dsafety=10.0, dt=0.250) = new(dsafety, dt)
end

mutable struct MpcParams
	N::Int64
	Q::Array{Float64,2}
	R::Array{Float64,2}
	MpcParams(N=16, Q=Diagonal([1, 50]), R=Diagonal([0.001])) = new(N, Q, R)
end


mutable struct MpcModel
	s0::Array{JuMP.NonlinearParameter, 1}
	sRef::Array{JuMP.NonlinearParameter, 2}
	uRef::Array{JuMP.NonlinearParameter, 2}

	obstacles::Array{JuMP.NonlinearParameter, 2}

	s::Array{JuMP.VariableRef, 2}
	u::Array{JuMP.VariableRef, 2}
	λ::Array{JuMP.VariableRef, 1}

	costS::JuMP.NonlinearExpression
	costU::JuMP.NonlinearExpression
	mdl::JuMP.Model

    dt::Float64
	N::Int64

	function MpcModel(mpcParams::MpcParams, mdlParams::MdlParams)
		println("Building mpc controller ...")

		m = new()

		# get model parameters
		dt = mdlParams.dt
		dsafety = mdlParams.dsafety

		# get mpc parameters
		N = mpcParams.N
		Q = mpcParams.Q
		R = mpcParams.R

		# Create Model
		mdl = Model(Ipopt.Optimizer)
		set_optimizer_attributes(mdl, "max_iter" => 5000)

		@variable(mdl, 0 <= λ[i=1:10] <=1 , start = 1) 

		# Create variables (will be optimized)
		smin::Vector{Float64} = [0.0; 0.0] # s sd
		smax::Vector{Float64} = [1000.0; 25.0] # s sd
		@variable(mdl, smin[i] <= s[i=1:2, t=1:(N+1)] <= smax[i], start = [0.0, 20.0][i]) 

		umin::Vector{Float64} = [-4.0] # decel max
		umax::Vector{Float64} = [2.0] # accel max
		@variable(mdl, umin[i] <= u[i=1:1, t=1:N] <= umax[i], start = 0) 

		@NLparameter(mdl, s0[i=1:2] == [0.0, 20.0][i])
		sref::Vector{Float64} = [200.0; 20.0] # s sd
		@NLparameter(mdl, sRef[j=1:2,i=1:N+1] == sref[j])
		@NLparameter(mdl, uRef[j=1:1,i=1:N] == 0)

		# System dynamics
		@NLconstraint(mdl, [i=1:2], s[i,1] == s0[i]) # initial condition

		for i=1:N # CA model
			@constraint(mdl, s[1,i+1] == s[1,i] + dt*s[2,i] + 0.5*dt^2*u[1,i]) # s
			@constraint(mdl, s[2,i+1] == s[2,i] + dt*u[1,i]) # sd
		end

		# Cost definitions
		# Input cost
		# ---------------------------------
		@NLexpression(mdl, costU, 0.5*sum(R[j,j] * sum((u[j,i] - uRef[j,i])^2 for i=1:N) for j = 1:1))
		
		# State cost
		# ---------------------------------
		@NLexpression(mdl, costS, 0.5*sum(Q[j,j] * sum((s[j,i] - sRef[j,i])^2 for i=2:N+1) for j = 1:2))

		@NLexpression(mdl, binaryλ, 10*sum((λ[i]-0.5)^2 for i =1:10))
		
		# Objective function
		#@NLobjective(mdl, Min, costS + costU - binaryλ)
		@NLobjective(mdl, Min, costS - binaryλ)
		#@NLobjective(mdl, Min, costS + costU - 1000*(λ-0.5)^2)
		#@NLobjective(mdl, Min, costS - (λ-0.5)^2)

		#for i=1:10
		#	@constraint(mdl, λ[i]*(1-λ[i]) == 0) # Binary ...
		#end

		# First solve
		println("Attempting first solve ...")
		optimize!(mdl)
		sol_stat = termination_status(mdl)
		println("Finished solve 1: $sol_stat")

		#delete(mdl, con)

		m.mdl = mdl
		m.λ = λ
		m.dt = dt
		m.N = N
		m.s0 = s0
		m.s = s
		m.u = u
		m.costS = costS
		m.costU = costU
		m.sRef = sRef
		m.uRef = uRef
		return m
	end
end

#function SolveMpcProblem(mdl::MpcModel, mpcSol::MpcSol, sCurr::Array{Float64}, sRef::Array{Float64,2}, uRef::Array{Float64,2})
function act(mpc::MpcModel, state::Array{Float64}, obstacles::Array{Tuple{Float64, Float64}, 1})::Float64

	# update current initial condition
	for i in 1:length(state)-1
		set_value(mpc.s0[i], state[i])
	end

	#println("state=$state")
	constr = []
	count, i = 1, 1
	while i<=min(10,length(obstacles)) && count <= 10 
		tcross, scross = obstacles[i]
		tcrossd = floor(Int, (tcross-state[3])/mpc.dt)
		#println("tcross=$tcross tcrossd=$tcrossd scross=$scross")
		constrained::Bool = (tcrossd > 0) && (tcrossd < mpc.N)
		if constrained
			#con = @constraint(mpc.mdl, (mpc.s[1,tcrossd+1] - scross)^2 >= 12.0^2  )
			#push!(constr, con)

			# It is a hand-engineered OR (Ideally λ should be a binary variable)
			con = @constraint(mpc.mdl, mpc.λ[count] * (mpc.s[1,tcrossd+1]  - (scross + 12)) >= 0)
			push!(constr, con)
			con1 = @constraint(mpc.mdl, (1.0 - mpc.λ[count]) * (mpc.s[1,tcrossd+1] - (scross - 12))  <= 0)
			push!(constr, con1)
			count += 1
		end
		i += 1
	end

	println("Attempting solve ...")
	optimize!(mpc.mdl)
	sol_stat = termination_status(mpc.mdl)
	println("Finished solve: $sol_stat")

	sol_u = value.(mpc.u)
	#println("sol_u $sol_u")
	sol_s = value.(mpc.s)
	#println("sol_s $sol_s")

	sol_λ = value.(mpc.λ)
	println("sol_λ $sol_λ")

	for i in 1:length(constr)
		delete(mpc.mdl, constr[i])
	end
	#exit()

	return sol_u[1]
end


#################### Not Used #######################

# abstract type OCP end
# 
# @with_kw mutable struct Mpc <: OCP
# 	T::Float64 = 20.0
# 	dt::Float64 = 0.250
# 	n::Int64 = T/dt
# 	dsafety::Float64 = 10.0
# 
# 	smin::Vector{Float64} = [0.0; 0.0]
# 	smax::Vector{Float64} = [1000.0; 25.0]
# 	umin::Float64 = -4.0
# 	umax::Float64 = 2.0
# 
# 	# Transtion model
# 	Ts::Array{Float64,2} = [1.0 dt; 0  1]
# 	Ta::Array{Float64,1} = [dt^2/2; dt]
# 
# 	s::Array{JuMP.VariableRef, 2} = nothing
# 
# 	mdp = nothing
# 	model = nothing
# end


