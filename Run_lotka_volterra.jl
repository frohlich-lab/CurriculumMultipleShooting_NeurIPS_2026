using ComponentArrays, OrdinaryDiffEq, ForwardDiff, Lux, Random, ModelingToolkit
using Zygote, ReverseDiff, SciMLSensitivity
using LinearAlgebra, FiniteDifferences, Plots
using OptimizationOptimisers, OptimizationOptimJL
using DataFrames, CSV, BenchmarkTools
include("Common.jl")

my_glorot_uniform(rng, dims...) = Float64.(Lux.glorot_uniform(rng, dims...; gain=0.1))

function build_lv_net(hidden_layers, width_size, actvation_function, rng)
    first_layer = Dense(2, width_size, actvation_function; init_bias=Lux.zeros64, init_weight=my_glorot_uniform)
    intermediate_layers = [Dense(width_size, width_size, actvation_function; init_bias=Lux.zeros64, init_weight=my_glorot_uniform) for _ in 1:hidden_layers-1]
    last_layer = Dense(width_size, 2; init_bias=Lux.zeros64, init_weight=my_glorot_uniform)
    nn = Lux.Chain(first_layer, intermediate_layers..., last_layer) |> f64
    p_nn, st = Lux.setup(rng, nn)
    return p_nn, nn, st
end

# Dynamics
function lv!(du, u, p, t)
    prey, predator = u
    @unpack α, δ, p_nn = p

    du_nn = nn([prey, predator], p_nn, st)[1]

    du[1] = α*prey + du_nn[1] # prey
    du[2] = du_nn[2] - δ*predator # predator
    return nothing
end

function get_mtk_prob(p, oprob)
    _p = p .* 0 .+ collect(1:length(p))
    sys = structural_simplify(modelingtoolkitize(remake(oprob, p=_p)))
    _oprob = ODEProblem(sys, [], (0.0, 10.0), [], jac=true)
    imap = floor.(Int64, _oprob.p.tunable[1])

    imap_rev = collect(1:length(imap))
    for i in eachindex(imap)
        imap_rev[imap[i]] = i
    end

    return _oprob, imap, imap_rev
end

function my_loss(p, oprob, data, solver, sensealg, abstol, reltol)
    _oprob = remake(oprob, p = p)
    tsave = 0:0.1:0.1*61
    sol = solve(_oprob, solver, abstol=abstol, reltol=reltol, sensealg=sensealg, saveat=tsave)
    if sol.retcode === ReturnCode.Success
        return sum((Array(sol) - data).^2)
    else
        return Inf64
    end
end

function grad_loss(_p, oprob, data, solver, sensealg, abstol, reltol, imap)
    p = deepcopy(_p)
    if !isnothing(imap)
        p .= p[imap]
    end
    _oprob = remake(oprob, p=p)
    _loss = (p) -> my_loss(p, _oprob, data, solver, sensealg, abstol, reltol)
    loss = _loss(p)
    runtime = 0
    if sensealg === :ForwardDiff
        grad = ForwardDiff.gradient(_loss, p)
        runtime = @benchmark _ = $ForwardDiff.gradient($_loss, $p)
        runtime = median(runtime.times) / 1e9
    elseif sensealg === :FiniteDiff
        grad = FiniteDifferences.grad(central_fdm(5, 1), _loss, p)[1]
        runtime = 0.0
    else
        grad = Zygote.gradient(_loss, p)[1]
        runtime = @benchmark _ = $Zygote.gradient($_loss, $p)
        runtime = median(runtime.times) / 1e9
    end
    # We want time in seconds
    return loss, grad, runtime
end

function saveinfo(grad, dirsave, iteration, runtime, method, mtk::Bool)
    path_save_grad = joinpath(dirsave, "Grad_" * method * "_" * iteration * ".csv")
    save_p(path_save_grad, grad)
    # Save Meta
    data_meta = DataFrame(iter = iteration, runtime = runtime, method=method, mtk=mtk)
    CSV.write(joinpath(dirsave, "Runtime.csv"), data_meta, append=isfile(joinpath(dirsave, "Runtime.csv")))
end

function benchmark_grad(path_dir, p_nn)

    # Setup parameter vector and ODEProblem
    p_mechanistic = (α = 1.3, δ = 1.8)
    u0_map = [0.44249296, 4.6280594]
    p_combined = ComponentArray(merge(p_mechanistic, (p_nn=p_nn,)))
    oprob_ude = ODEProblem(lv!, u0_map, (0.0, 10.0), p_combined)
    # Measurment data
    _data_obs = Array(CSV.read("Data_test1.csv", DataFrame))
    data_obs = Matrix(_data_obs[:, 2:end]')

    # Get the MTK oprob
    oprob_mtk, imap, imap_rev = get_mtk_prob(p_combined, oprob_ude)

    # Setup benchmark testing gradients
    dirsave = joinpath(path_dir, "Julia_it_res")
    !isdir(dirsave) && mkdir(dirsave)
    dir_iter_files = joinpath(path_dir, "Param_it")
    param_it_files = filter(x -> occursin("Param_it", x), readdir(dir_iter_files))

    for (i, param_it_file) in pairs(param_it_files)
        @info "i = $i"

        iteration = match(r"(\d+)", param_it_file).captures[1]
        _parameters = CSV.read(joinpath(dir_iter_files, param_it_file), DataFrame)[!, :val]
        p_combined .= _parameters

        # ForwardDiff
        loss, grad, runtime = grad_loss(p_combined, oprob_ude, data_obs, Tsit5(), :ForwardDiff, 1e-8, 1e-8, nothing)
        saveinfo(grad, dirsave, iteration, runtime, "ForwardDiff", false)
        loss, grad, runtime = grad_loss(p_combined, oprob_mtk, data_obs, Tsit5(), :ForwardDiff, 1e-8, 1e-8, imap)
        saveinfo(grad[imap_rev], dirsave, iteration, runtime, "ForwardDiff_mtk", true)

        # Interpolating adjoint
        loss, grad, runtime = grad_loss(p_combined, oprob_ude, data_obs, Tsit5(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), 1e-8, 1e-8, nothing)
        saveinfo(grad, dirsave, iteration, runtime, "InterRDVJP", false)
        loss, grad, runtime = grad_loss(p_combined, oprob_mtk, data_obs, Tsit5(), InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), 1e-8, 1e-8, imap)
        saveinfo(grad[imap_rev], dirsave, iteration, runtime, "InterRDVJP_mtk", true)
        loss, grad, runtime = grad_loss(p_combined, oprob_mtk, data_obs, Tsit5(), InterpolatingAdjoint(autojacvec=EnzymeVJP()), 1e-8, 1e-8, imap)
        saveinfo(grad[imap_rev], dirsave, iteration, runtime, "InterEnzyme_mtk", true)

        # Gauss adjoint
        loss, grad, runtime = grad_loss(p_combined, oprob_ude, data_obs, Tsit5(), GaussAdjoint(autojacvec=ReverseDiffVJP(true)), 1e-8, 1e-8, nothing)
        saveinfo(grad, dirsave, iteration, runtime, "GaussRDVJP", false)
        loss, grad, runtime = grad_loss(p_combined, oprob_mtk, data_obs, Tsit5(), GaussAdjoint(autojacvec=ReverseDiffVJP(true)), 1e-8, 1e-8, imap)
        saveinfo(grad[imap_rev], dirsave, iteration, runtime, "GaussRDVJP_mtk", true)
        loss, grad, runtime = grad_loss(p_combined, oprob_mtk, data_obs, Tsit5(), GaussAdjoint(autojacvec=EnzymeVJP()), 1e-8, 1e-8, imap)
        saveinfo(grad[imap_rev], dirsave, iteration, runtime, "GaussEnzyme_mtk", true)

        # Reference gradient + solution
        loss, grad, runtime = grad_loss(p_combined, oprob_ude, data_obs, Vern8(), :FiniteDiff, 1e-12, 1e-12, nothing)
        saveinfo(grad, dirsave, iteration, runtime, "FiniteDiff", false)
    end

    return nothing
end

@assert length(ARGS) == 4 "Error ARGS must be of length 4"

path_dir = joinpath(@__DIR__, ARGS[2])
hidden_layers = parse(Int64, ARGS[3])
width_size = parse(Int64, ARGS[4])
activation_function = Lux.tanh

!isdir(path_dir) && mkdir(path_dir)

rng = Random.default_rng()
Random.seed!(rng, 123)
p_nn, nn, st = build_lv_net(hidden_layers, width_size, activation_function, rng)

if ARGS[1] == "Benchmark_grad"
    benchmark_grad(path_dir, p_nn)
end

if ARGS[1] == "Generate_random_parameters"
    @info "Generating startguesses"
    diriter = joinpath(path_dir, "Param_it")
    !isdir(diriter) && mkdir(diriter)
    save_p_iter(diriter, Tsit5(), 50, p_nn)
    save_meta(joinpath(path_dir, "Meta_nn.csv"), hidden_layers, width_size, activation_function)
end
