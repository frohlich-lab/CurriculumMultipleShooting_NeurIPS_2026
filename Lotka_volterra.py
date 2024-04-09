from diffrax import Tsit5
import jax.flatten_util
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import equinox as eqx 
import jax
import jax.random as jr
import numpy
import pandas as pd
import os
import re
import timeit
import sys

# Local files 
import Common

jax.config.update("jax_enable_x64", True)

# Setup the true solution
def vector_field(t, u, args):
    prey, predator = u
    α, β, γ, δ = args
    d_prey = α * prey - β * prey * predator
    d_predator = γ * prey * predator - δ * predator
    return jnp.array([d_prey, d_predator])


class Func(eqx.Module):

    dyn_param: jax.Array
    mlp: eqx.nn.MLP
    
    def __init__(self, key, dir_meta, path_parameters, *args, **kwargs):
        super().__init__(**kwargs)
        # Fix that dynparam are also read
        self.dyn_param, self.mlp = Common.mlp_from_dir(dir_meta, path_parameters, key)
        
    def __call__(self, t, u, args):
        α, δ = self.dyn_param
        prey, predator = u
        nn_out = self.mlp(u)
        d_prey = α*prey + nn_out[0]
        d_predator = nn_out[1] - δ*predator
        return jnp.array([d_prey, d_predator])
    
class UDEProblem(eqx.Module):
    func: Func

    def __init__(self, key, dir_meta, path_parameters=None, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(key=key, dir_meta=dir_meta, path_parameters=path_parameters)

    def __call__(self, u0, tsave):
        t0 = 0
        t1 = 10.0
        sol = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
                                  Tsit5(),
                                  t0, 
                                  t1, 
                                  None, 
                                  u0, 
                                  saveat=tsave, 
                                  throw=False,
                                  stepsize_controller=diffrax.PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0, rtol=1e-8, atol=1e-8))
        return sol

@eqx.filter_value_and_grad
def grad_loss(model, u_data, u0, tsave):
    u_pred = model(u0, tsave)
    return jnp.sum((u_pred.ys - u_data)**2)

@eqx.filter_jit
def make_step(u0, tsave, u_data, model, opt_state):
    loss, grads = grad_loss(model, u_data, u0, tsave)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def run_timing(model, u_data, u0, tsave):
    def benchmark_func():
        _loss, _grad = grad_loss(model, u_data, u0, tsave)
        return None
    duration = timeit.Timer(benchmark_func).repeat(20, number=1)
    return duration

def run_benchmark(path_dir):

    # Read data to match against
    obs_df = pd.read_csv("Data_test1.csv")
    _ts = jnp.array(obs_df["t"].values)
    tsave = diffrax.SaveAt(ts=_ts)
    _u_data = numpy.zeros((len(_ts), 2))
    _u_data[:, 0] = obs_df["u1"]
    _u_data[:, 1] = obs_df["u2"]
    u_data = jnp.array(_u_data)
    u0 = jnp.array([0.44249296, 4.6280594])

    # jnp.iterable(sol_true.result) to check retcode
    # Formulate UDEProblem model
    dir_param_it = path_dir + "/Param_it/"
    parameters_it = os.listdir(dir_param_it)
    for i in range(len(parameters_it)):
        print("i = {}".format(i))
        path_parameters = dir_param_it + "/" + parameters_it[i]
        iteration = re.findall(r'\d+', parameters_it[i])[0]
        key = jr.PRNGKey(123)
        model = UDEProblem(key=key, dir_meta=path_dir, path_parameters=path_parameters)
        loss, grad = grad_loss(model, u_data, u0, tsave)
        duration = numpy.median(run_timing(model, u_data, u0, tsave))

        # Save gradient
        grad_vec = jax.flatten_util.ravel_pytree(grad)[0]
        grad_names = pd.read_csv(path_parameters)["name"].values
        df_save = pd.DataFrame({'val' : grad_vec, 
                                'name' : grad_names})
        dir_save = path_dir + "/Diffrax_it_res/"
        if not os.path.isdir(dir_save):
            os.mkdir(dir_save)
        df_save.to_csv(dir_save + "Grad_diffrax_" + iteration + ".csv")
        # Save runtime
        df_runtime = pd.DataFrame({'runtime' : [duration], 
                                'iter' : [iteration], 
                                'method' : ["Diffrax"]})
        if os.path.isfile(dir_save + "Runtime.csv"):
            mode = "a"
        else:
            mode = "w"
        df_runtime.to_csv(dir_save + "Runtime.csv", mode=mode)

    return None

path_dir = sys.argv[1]

run_benchmark(path_dir)
