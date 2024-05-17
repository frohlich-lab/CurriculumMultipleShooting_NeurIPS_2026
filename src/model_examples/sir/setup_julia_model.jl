using ComponentArrays, OrdinaryDiffEq, ForwardDiff, Lux, Random, ModelingToolkit
using CSV, DataFrames, LinearAlgebra, Zygote, SciMLSensitivity, Optimization, Printf
using OptimizationOptimisers

my_glorot_uniform(rng, dims...) = Float64.(Lux.glorot_uniform(rng, dims...; gain=0.1))

function build_sir_net(hidden_layers, width_size, actvation_function, rng)
    first_layer = Dense(3, width_size, actvation_function; init_bias=Lux.zeros64, init_weight=my_glorot_uniform)
    intermediate_layers = [Dense(width_size, width_size, actvation_function; init_bias=Lux.zeros64, init_weight=my_glorot_uniform) for _ in 1:hidden_layers-1]
    last_layer = Dense(width_size, 1; init_bias=Lux.zeros64, init_weight=my_glorot_uniform)
    nn = Lux.Chain(first_layer, intermediate_layers..., last_layer) |> f64
    p_nn, st = Lux.setup(rng, nn)
    return p_nn, nn, st
end

function sir!(du, u, p, t)
    S, I, R, T = u
    @unpack β, γ, δ, p_nn = p

    nnₜ = nn([S, I, R], p_nn, st)[1][1]

    du[1] = - β*S*(I)/u0[1]
    du[2] = β*S*(I)/u0[1] - γ*I - nnₜ*I/u0[1]
    du[3] = γ*I + δ*T
    du[4] =  nnₜ*I/u0[1] - δ*T

    return nothing
end

function _loss(p, uprob, data, solver, tsave, sensealg)
    _uprob = remake(uprob, p = p)
    sol = solve(_uprob, solver, abstol=1e-8, reltol=1e-8, sensealg=sensealg, saveat=tsave)
    if sol.retcode === ReturnCode.Success
        # Compute the full loss specified in the paper
        infected = @view data[:, 1]
        dead = @view data[:, 2]
        recovered = @view data[:, 3]
        loss = sum(abs2, log.(abs.(infected)) .- log.(abs.(sol[2, :] .+ sol[4, :])))
        loss += sum(abs2, log.(abs.(recovered + dead)) .- log.(abs.(sol[3, :])))
        return loss
    else
        return Inf64
    end
end
