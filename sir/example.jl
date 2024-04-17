using ComponentArrays, OrdinaryDiffEq, ForwardDiff, Lux, Random, ModelingToolkit
using CSV, DataFrames, LinearAlgebra, Zygote, SciMLSensitivity, Optimization, Printf
using OptimizationOptimisers
include(joinpath(@__DIR__, "setup_julia_model.jl"))

# Let us consider the Korean data. Note initial conditions differ for the UK data
data_korea = CSV.read(joinpath(@__DIR__, "..", "Intermediate", "SIR", "Korea_data.csv"), DataFrame)
u0 = Float64[51000000.0, 578, 24, 10]
data_train = data_korea[!, [:infected, :dead, :recovered]] |> Matrix
tspan = (0.0, maximum(data_korea[!, :time]))
tsave = data_korea[!, :time]

rng = Random.default_rng()
Random.seed!(rng, 123)
p_nn, nn, st = build_sir_net(1, 10, Lux.relu, rng)
p_mechanistic = (β = 0.15, γ = 0.013, δ = 0.01)
p_combined = ComponentArray(merge(p_mechanistic, (p_nn=p_nn,)))

uprob = ODEProblem(sir!, u0, tspan, p_combined)

# For optimization
_callback = function (p, l)
    if p.iter % 100 == 0
        @printf("iter = %d and l = %.2f\n", p.iter, l)
    end
    return false
end
compute_loss = (p, u) -> _loss(p, uprob, data_train, Tsit5(), tsave, ForwardDiffSensitivity())
f = OptimizationFunction(compute_loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(f, p_combined, Float64[])
# Run Adam for some iterations
optsol = solve(optprob, OptimizationOptimisers.Adam(0.01), maxiters=15000, callback=_callback)

using Plots
sol = solve(uprob, Tsit5(), abstol=1e-8, reltol=1e-8, saveat=tsave, p = optsol.u)
plot(data_train[:, 1], seriestype = :scatter, label = "Data infected")
plot!(sol[3, :] + sol[4, :], label = "Prediction infected")

plot(data_train[:, 2] + data_train[:, 3], seriestype = :scatter, label = "Data: Recovered + dead")
plot!(abs.(sol[3, :]), label = "Prediction: Recovered + dead")

# We can also benchmark gradient and Hessian runtime
using BenchmarkTools
_compute_loss = (p) -> _loss(p, uprob, data_train, Tsit5(), tsave, ForwardDiffSensitivity())
bgrad =  @benchmark ForwardDiff.gradient(_compute_loss, p_combined) # ≈ 44ms
bhess =  @benchmark ForwardDiff.hessian(_compute_loss, p_combined) # ≈ 1.3s
