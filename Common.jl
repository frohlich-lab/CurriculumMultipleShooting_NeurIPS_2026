using Printf

function save_lv_setup(dir_save, p_combined, hidden_layers, width_size, actvation_function)
    !isdir(dir_save) && mkdir(dir_save)
    save_p(joinpath(dir_save, "Parameter.csv"), p_combined)
    save_meta(joinpath(dir_save, "Meta.csv"), hidden_layers, width_size, actvation_function)
end

# Function for writing parameters to file
function save_p(io, p_combined)

    params_val = collect(p_combined)
    params_names = Vector{String}(undef, length(params_val))
    k = 1
    for name in keys(p_combined)
        if typeof(p_combined[name]) <: Real
            params_names[k] = string(name)
            k += 1
            continue
        end
        # Dive deeper (handle layers + bias)
        for layer_name in keys(p_combined[name])
            nweights = length(p_combined[name][layer_name][:weight])
            weight_names = Vector{String}(undef, nweights)
            _l = 1
            for _i in 1:size(p_combined[name][layer_name][:weight])[2]
                for _j in 1:size(p_combined[name][layer_name][:weight])[1]
                    weight_names[_l] = string(layer_name) .* "_w_" * string(_j) * "_" * string(_i)
                    _l += 1
                end
            end
            nbias = length(p_combined[name][layer_name][:bias])
            bias_names = string(layer_name) .* "_bias_" .* string.(1:nbias)
            params_names[k:(k+nweights-1)] .= weight_names
            params_names[(k+nweights):(k+nweights+nbias-1)] .= bias_names
            k += nweights + nbias
        end
    end

    data_save = DataFrame(val = params_val,
                          name = params_names)
    CSV.write(io, data_save)
end

function save_meta(io, hidden_layers, width_size, actvation_function)
    datasave = DataFrame(hidden_layers=hidden_layers,
                         width_size=width_size,
                         actvation_function=string(actvation_function))
    CSV.write(io, datasave)
end

_callback = function (p, l, dir_save, iter_save)
    if p.iter % 100 == 0
        @printf("iter = %d and l = %.2f\n", p.iter, l)
    end
    if p.iter == 1
        @printf("iter = %d and l = %.2f\n", p.iter, l)
        path_save = joinpath(dir_save, "Param_it" * string(p.iter) * ".csv")
        save_p(path_save, p.u)
    end
    if p.iter % iter_save == 0
        path_save = joinpath(dir_save, "Param_it" * string(p.iter) * ".csv")
        save_p(path_save, p.u)
    end
    return false
end

function save_p_iter(path_dir, solver, itersave, p_nn)

    _data_obs = Array(CSV.read("Data_test1.csv", DataFrame))
    data_obs = Matrix(_data_obs[:, 2:end]')

    p_mechanistic = (α = 1.3, δ = 1.8)
    u0_map = [0.44249296, 4.6280594]
    p_combined = ComponentArray(merge(p_mechanistic, (p_nn=p_nn,)))
    oprob_ude = ODEProblem(lv!, u0_map, (0.0, 10.0), p_combined)
    _myloss = (x, p) -> my_loss(x, oprob_ude, data_obs, solver, ForwardDiffSensitivity())
    Random.seed!(123)
    f = OptimizationFunction(_myloss, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(f, p_combined, Float64[])

    callback = (p, l) -> _callback(p, l, path_dir, itersave)
    _ = solve(optprob, OptimizationOptimisers.Adam(0.1), maxiters=9000, callback=callback)
    return nothing
end

function my_loss(p, oprob, data, solver, sensealg)
    _oprob = remake(oprob, p = p)
    tsave = 0:0.1:0.1*61
    sol = solve(_oprob, solver, abstol=1e-8, reltol=1e-8, sensealg=sensealg, saveat=tsave)
    if sol.retcode === ReturnCode.Success
        return sum((Array(sol) - data).^2)
    else
        return Inf64
    end
end
