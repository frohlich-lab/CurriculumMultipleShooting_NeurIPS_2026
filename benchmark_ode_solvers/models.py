import jax
import jax.numpy as jnp
import equinox as eqx
import typing
import diffrax

jax.config.update("jax_enable_x64", True)

class ODEProblem(eqx.Module):
    u0 : jax.Array
    f : typing.Callable
    def __init__(self, u0, f) -> None:
        self.u0 = u0
        self.f = f
    
    def solve(self, solver, t_span, controller, max_steps):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.f),
            solver,
            t_span[0],
            t_span[1],
            None,
            self.u0,
            max_steps=max_steps,
            stepsize_controller=controller)
        return sol

def f_lv(t, y, args):
    a, b, c, d = 1.5, -1, -3, 1
    x = y[0]
    y = y[1]
    return jnp.array([a * x + b * x * y, c * y + d * x * y])

def f_rb(t, y, args):
    y1, y2, y3 = y
    dy1  = -2*y2*y3
    dy2  = 1.25*y1*y3
    dy3  = -0.5*y1*y2 + 0.25*jnp.sin(t)**2
    return jnp.array([dy1, dy2, dy3])

def f_ro(t, y, args):
    y1, y2, y3 = y
    k1, k2, k3 = 0.04,3e7,1e4
    dy1 = -k1*y1+k3*y2*y3
    dy2 =  k1*y1-k2*y2**2-k3*y2*y3
    dy3 =  k2*y2**2
    return jnp.array([dy1, dy2, dy3])

def f_hi(t, y, args):
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    dy1 = -1.71*y1 + 0.43*y2 + 8.32*y3 + 0.0007
    dy2 = 1.71*y1 - 8.75*y2
    dy3 = -10.03*y3 + 0.43*y4 + 0.035*y5
    dy4 = 8.32*y2 + 1.71*y3 - 1.12*y4
    dy5 = -1.745*y5 + 0.43*y6 + 0.43*y7
    dy6 = -280.0*y6*y8 + 0.69*y4 + 1.71*y5 - 0.43*y6 + 0.69*y7
    dy7 = 280.0*y6*y8 - 1.81*y7
    dy8 = -280.0*y6*y8 + 1.81*y7
    return jnp.array([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8])

def f_po(t, y, args):
    k1= 35e0
    k2= 266e2
    k3= 123e5
    k4= 86e-3
    k5= 82e-3
    k6= 15e5
    k7= 13e-3
    k8= 24e5
    k9= 165e5
    k10= 9e4
    k11= 22e-1
    k12= 12e5
    k13= 188e1
    k14= 163e5
    k15= 48e7
    k16= 35e-3
    k17= 175e-1
    k18= 1e9
    k19= 444e12
    k20= 124e4
    k21= 21e1
    k22= 578e1
    k23= 474e-1
    k24= 178e4
    k25= 312e1
    r1  = k1 *y[0]
    r2  = k2 *y[1]*y[3]
    r3  = k3 *y[4]*y[1]
    r4  = k4 *y[6]
    r5  = k5 *y[6]
    r6  = k6 *y[6]*y[5]
    r7  = k7 *y[8]
    r8  = k8 *y[8]*y[5]
    r9  = k9 *y[10]*y[1]
    r10 = k10*y[10]*y[0]
    r11 = k11*y[12]
    r12 = k12*y[9]*y[1]
    r13 = k13*y[13]
    r14 = k14*y[0]*y[5]
    r15 = k15*y[2]
    r16 = k16*y[3]
    r17 = k17*y[3]
    r18 = k18*y[15]
    r19 = k19*y[15]
    r20 = k20*y[16]*y[5]
    r21 = k21*y[18]
    r22 = k22*y[18]
    r23 = k23*y[0]*y[3]
    r24 = k24*y[18]*y[0]
    r25 = k25*y[19]

    dy1  = -r1-r10-r14-r23-r24+r2+r3+r9+r11+r12+r22+r25
    dy2  = -r2-r3-r9-r12+r1+r21
    dy3  = -r15+r1+r17+r19+r22
    dy4  = -r2-r16-r17-r23+r15
    dy5  = -r3+r4+r4+r6+r7+r13+r20
    dy6  = -r6-r8-r14-r20+r3+r18+r18
    dy7  = -r4-r5-r6+r13
    dy8  = r4+r5+r6+r7
    dy9  = -r7-r8
    dy10 = -r12+r7+r9
    dy11 = -r9-r10+r8+r11
    dy12 = r9
    dy13 = -r11+r10
    dy14 = -r13+r12
    dy15 = r14
    dy16 = -r18-r19+r16
    dy17 = -r20
    dy18 = r20
    dy19 = -r21-r22-r24+r23+r25
    dy20 = -r25+r24
    return jnp.array([dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9, dy10, dy11, 
                      dy12, dy13, dy14, dy15, dy16, dy17, dy18, dy19, dy20])

def get_model(model):
    if model == "LotkaVolterra":
        t_span = [0.0, 10.0]
        u0 = jnp.array([1.0, 1.0])
        rel_tols = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
        abs_tols = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13)
        ode_problem = ODEProblem(u0, f_lv)

    if model == "RigidBody":
        t_span = [0.0, 100.0]
        u0 = jnp.array([1.0, 0.0, 0.9])
        abs_tols = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13)
        rel_tols = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
        ode_problem = ODEProblem(u0, f_rb)

    if model == "Rober":
        t_span = [0.0, 1e5]
        u0 = jnp.array([1.0, 0.0, 0.0])
        abs_tols = (1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12)
        rel_tols = (0.001, 0.0001, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)
        ode_problem = ODEProblem(u0, f_ro)

    if model == "Hires":
        t_span = [0.0, 321.8122]
        u0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057])
        abs_tols = (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10)
        rel_tols = (0.1, 0.01, 0.001, 0.0001, 1.0e-5, 1.0e-6)
        ode_problem = ODEProblem(u0, f_hi)        

    if model == "Pollu":
        t_span = [0.0, 60.0]
        u0 = jnp.zeros(20)
        u0 = u0.at[1].set(0.2)
        u0 = u0.at[3].set(0.04)
        u0 = u0.at[6].set(0.1)
        u0 = u0.at[7].set(0.3)
        u0 = u0.at[8].set(0.01)
        u0 = u0.at[16].set(0.007)
        abs_tols = (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8)
        rel_tols = (0.1, 0.01, 0.001, 0.0001)
        ode_problem = ODEProblem(u0, f_po)        

    return ode_problem, t_span, rel_tols, abs_tols
