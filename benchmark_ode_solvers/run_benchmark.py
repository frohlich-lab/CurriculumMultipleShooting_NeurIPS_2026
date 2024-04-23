import timeit
import diffrax
import pandas as pd
import numpy as np
import equinox as eqx
import pandas as pd
import os

import benchmark_ode_solvers.models as models

def save_result(model_test, u_end, runtime):
    dir_save = os.path.join("Intermediate", "Benchmark_solvers")
    if not os.path.isdir(dir_save):
        os.mkdir(dir_save)
    path_save = os.path.join(dir_save, "Diffrax" + model_test + ".csv")
    df_save = pd.DataFrame(u_end, columns=["u" + str(i+1) for i in range(u_end.shape[1])])
    df_save["runtime"] = runtime
    df_save.to_csv(path_save)

@eqx.filter_jit
def run(ode_problem, solver, t_span, controller, max_steps):
    sol = ode_problem.solve(solver, t_span, controller, max_steps)
    return sol

def time_run(ode_func, solver, t_span, controller, max_steps):
    # Compile and get solution
    sol = run(ode_func, solver, t_span, controller, max_steps)  
    time = np.median(timeit.repeat(lambda: run(ode_func, solver, t_span, controller, max_steps), repeat=1000, number=1))
    return time, sol.ys

def test_model(model_test, solver):
    # Some models demand more time-steps than others
    if model_test == "RigidBody":
        max_steps = 10000000
    else:
        max_steps = 4094 # 2**12

    ode_problem, t_span, rel_tols, abs_tols = models.get_model(model_test)
    runtime = np.zeros(len(rel_tols))
    u_end = np.zeros((len(rel_tols), len(ode_problem.u0)))
    for k in range(len(rel_tols)):
        # Different controllers for stiff-models
        if model_test in ["Rober", "Hires", "Pollu"]:
            controller = diffrax.PIDController(rtol=rel_tols[k], atol=abs_tols[k], 
                                               pcoeff=0.4, icoeff=0.3, dcoeff=0)
        else:
            controller = diffrax.PIDController(rtol=rel_tols[k], atol=abs_tols[k], 
                                               pcoeff=0.0, icoeff=1.0, dcoeff=0)

        runtime[k], u_end[k, :] = time_run(ode_problem, solver, t_span, 
                                           controller, max_steps)
    
    save_result(model_test, u_end, runtime)

test_model("LotkaVolterra", diffrax.Tsit5())

test_model("RigidBody", diffrax.Tsit5())

test_model("Rober", diffrax.Kvaerno5())

test_model("Hires", diffrax.Kvaerno5())

test_model("Pollu", diffrax.Kvaerno5())
