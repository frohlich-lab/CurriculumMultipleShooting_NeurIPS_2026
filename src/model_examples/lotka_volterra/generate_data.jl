using OrdinaryDiffEq, CSV, DataFrames, Random, ModelingToolkit

function lv_true!(du, u, p, t)
    prey, predator = u
    @unpack α, β, γ, δ, = p

    du[1] = α*prey - β*prey*predator # prey
    du[2] = γ*prey*predator - δ*predator # predator
    return nothing
end

Random.seed!(123)
u0 = [0.44249296, 4.6280594]
ptrue = (α = 1.3, β = 0.9, γ = 0.8, δ = 1.8)
lv_prob = ODEProblem(lv_true!, u0, (0.0, 10.0), ptrue)
dirsave = joinpath(@__DIR__, "..", "Intermediate", "Lotka_volterra")

# Training data
sol_true = solve(lv_prob, Vern9(), abstol=1e-12, reltol=1e-12, saveat=0:0.1:0.1*61)
data_train = Array(sol_true) + randn(size(sol_true))*0.1
train_df = DataFrame(t = sol_true.t,
                     u1 = data_train[1, :],
                     u2 = data_train[2, :])
CSV.write(joinpath(dirsave, "Train_data.csv"), train_df)

# Validation data
lv_prob = remake(lv_prob, tspan = (0.0, 20.0))
sol_validation = solve(lv_prob, Vern9(), abstol=1e-12, reltol=1e-12, saveat=collect(0:0.1:0.1*61) .+ maximum(sol_true.t))
data_validation = Array(sol_validation) + randn(size(sol_true))*0.1
validation_df = DataFrame(t = sol_validation.t,
                          u1 = data_validation[1, :],
                          u2 = data_validation[2, :])
CSV.write(joinpath(dirsave, "Validation_data.csv"), validation_df)
