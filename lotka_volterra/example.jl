include(joinpath(@__DIR__, "setup_julia_model.jl"))

# Read training data
train_data_df = CSV.read(joinpath(@__DIR__, @__DIR__, "..", "Intermediate", "Lotka_volterra", "Train_data.csv"), DataFrame)
tsave_train = train_data_df[!, "t"]
train_data = transpose(train_data_df[!, ["u1", "u2"]] |> Array) |> Array

# Build network
rng = Random.default_rng()
Random.seed!(rng, 123)
p_nn, nn, st = build_lv_net(2, 5, Lux.tanh, rng)

# Setup ODEProblem
p_mechanistic = (α = 1.3, δ = 1.8)
u0 = [0.44249296, 4.6280594]
p_combined = ComponentArray(merge(p_mechanistic, (p_nn=p_nn,)))
uprob = ODEProblem(lv!, u0, (0.0, 10.0), p_combined)

# Compute loss and gradient with different methods
compute_loss = (x) -> _loss(x, uprob, train_data, Tsit5(), tsave_train, InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
loss = compute_loss(p_combined)
grad_f = ForwardDiff.gradient(compute_loss, p_combined)
hess_f = ForwardDiff.hessian(compute_loss, p_combined)
grad_z = Zygote.gradient(compute_loss, p_combined)
