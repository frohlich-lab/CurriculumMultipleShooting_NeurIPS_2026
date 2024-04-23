using OrdinaryDiffEq, ParameterizedFunctions, LinearAlgebra, BenchmarkTools
using CSV, DataFrames, Sundials, SciMLBase

include(joinpath(@__DIR__, "models.jl"))

function benchmark_solver(prob, solver, atols, rtols, alg, solref)
    @info "Testing $alg"
    uend = zeros(length(prob.u0), length(rtols))
    runtimes = zeros(length(rtols))
    for i in eachindex(rtols)
        sol = solve(prob, solver, abstol=atols[i], reltol=rtols[i], saveat=Float64[])
        runtime = @benchmark $solve($prob, $solver, abstol=$atols[$i], reltol=$rtols[$i]) evals=1000 samples=1 seconds=100
        runtimes[i] = median(runtime.times) / 1e9
        uend[:, i] .= sol[end]
    end
    return runtimes, get_acc(uend, solref)
end

function get_acc(uend, solref)
    return [norm(solref[end] - uend[:, i]) for i in 1:size(uend)[2]]
end

# Save for plotting
function saveres(model_test, res, abstols, reltols)
    dirsave = joinpath(@__DIR__, "..", "Intermediate", "Benchmark_solvers")
    dfsave = DataFrame()
    for (solver, _res) in res
        _dfsave = DataFrame(solver = solver,
                            abstol = abstols,
                            reltol = reltols,
                            runtime = _res[1],
                            error = _res[2])
        dfsave = vcat(dfsave, _dfsave)
    end
    CSV.write(joinpath(dirsave, "Runtime_" * model_test * ".csv"), dfsave)
    return nothing
end

function test_model(model_test, setups)
    prob, abstols, reltols = get_model(model_test)
    if model_test in ["Rober", "Hires", "Pollu"]
        solref = solve(prob, Rodas4P(), abstol=1e-14, reltol=1e-14)
    else
        solref = solve(prob, Vern9(), abstol=1e-14, reltol=1e-14)
    end
    res = Dict()
    for setup in setups
        times, acc = benchmark_solver(prob, setup[1], abstols, reltols, setup[2], solref)
        res[setup[2]] = [times, acc]
    end
    # Get accuarcy for the diffrax data
    dirsave = joinpath(@__DIR__, "..", "Intermediate", "Benchmark_solvers")
    diffrax_df = CSV.read(joinpath(dirsave, "Diffrax" * model_test * ".csv"), DataFrame)
    uend = Array(Array(diffrax_df[!, r"^u"])')
    acc = get_acc(uend, solref)
    if model_test in ["RigidBody", "LotkaVolterra"]
        res["Tsit5_diffrax"] = [diffrax_df[!, :runtime], acc]
    else
        res["Kvaerno5_diffrax"] = [diffrax_df[!, :runtime], acc]
    end

    saveres(model_test, res, abstols, reltols)
    return nothing
end

setups = [[DP5(), "DP5"], [Tsit5(), "Tsit5"], [Vern7(), "Vern7"]]
test_model("LotkaVolterra", setups)

setups = [[DP5(), "DP5"], [Tsit5(), "Tsit5"], [Vern7(), "Vern7"]]
test_model("LotkaVolterra", setups)

setups = [[CVODE_BDF(), "CVODE_BDF"], [Rodas5P(), "Rodas5P"], [Kvaerno5(), "Kvaerno5"], [QNDF(), "QNDF"]]
test_model("Rober", setups)

setups = [[CVODE_BDF(), "CVODE_BDF"], [Rodas5P(), "Rodas5P"], [Kvaerno5(), "Kvaerno5"], [QNDF(), "QNDF"]]
test_model("Hires", setups)

setups = [[CVODE_BDF(), "CVODE_BDF"], [Rodas5P(), "Rodas5P"], [Kvaerno5(), "Kvaerno5"], [QNDF(), "QNDF"]]
test_model("Pollu", setups)
