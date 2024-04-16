using ComponentArrays, OrdinaryDiffEq, ForwardDiff, Lux, Random, ModelingToolkit
using CSV, DataFrames, LinearAlgebra, Zygote, SciMLSensitivity

my_glorot_uniform(rng, dims...) = Float64.(Lux.glorot_uniform(rng, dims...; gain=0.1))

function build_lv_net(hidden_layers, width_size, actvation_function, rng)
    first_layer = Dense(2, width_size, actvation_function; init_bias=Lux.zeros64, init_weight=my_glorot_uniform)
    intermediate_layers = [Dense(width_size, width_size, actvation_function; init_bias=Lux.zeros64, init_weight=my_glorot_uniform) for _ in 1:hidden_layers-1]
    last_layer = Dense(width_size, 2; init_bias=Lux.zeros64, init_weight=my_glorot_uniform)
    nn = Lux.Chain(first_layer, intermediate_layers..., last_layer) |> f64
    p_nn, st = Lux.setup(rng, nn)
    return p_nn, nn, st
end

function _loss(p, oprob, data, solver, tsave, sensealg)
    _oprob = remake(oprob, p = p)
    sol = solve(_oprob, solver, abstol=1e-8, reltol=1e-8, sensealg=sensealg, saveat=tsave)
    if sol.retcode === ReturnCode.Success
        return sum((Array(sol) - data).^2)
    else
        return Inf64
    end
end

function lv!(du, u, p, t)
    prey, predator = u
    @unpack α, δ, p_nn = p

    du_nn = nn([prey, predator], p_nn, st)[1]

    du[1] = α*prey + du_nn[1] # prey
    du[2] = du_nn[2] - δ*predator # predator
    return nothing
end
