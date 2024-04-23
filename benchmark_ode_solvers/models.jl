f_lv = @ode_def LotkaVolterra begin
    dx = a * x - b * x * y
    dy = -c * y + d * x * y
end a b c d

f_rb = @ode_def RigidBodyBench begin
    dy1  = -2*y2*y3
    dy2  = 1.25*y1*y3
    dy3  = -0.5*y1*y2 + 0.25*sin(t)^2
end

f_ro = @ode_def begin
    dy₁ = -k₁*y₁+k₃*y₂*y₃
    dy₂ =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
    dy₃ =  k₂*y₂^2
end k₁ k₂ k₃

f_hi = @ode_def Hires begin
    dy1 = -1.71*y1 + 0.43*y2 + 8.32*y3 + 0.0007
    dy2 = 1.71*y1 - 8.75*y2
    dy3 = -10.03*y3 + 0.43*y4 + 0.035*y5
    dy4 = 8.32*y2 + 1.71*y3 - 1.12*y4
    dy5 = -1.745*y5 + 0.43*y6 + 0.43*y7
    dy6 = -280.0*y6*y8 + 0.69*y4 + 1.71*y5 -
             0.43*y6 + 0.69*y7
    dy7 = 280.0*y6*y8 - 1.81*y7
    dy8 = -280.0*y6*y8 + 1.81*y7
end

function f_po(dy,y,p,t)
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
    r1  = k1 *y[1]
    r2  = k2 *y[2]*y[4]
    r3  = k3 *y[5]*y[2]
    r4  = k4 *y[7]
    r5  = k5 *y[7]
    r6  = k6 *y[7]*y[6]
    r7  = k7 *y[9]
    r8  = k8 *y[9]*y[6]
    r9  = k9 *y[11]*y[2]
    r10 = k10*y[11]*y[1]
    r11 = k11*y[13]
    r12 = k12*y[10]*y[2]
    r13 = k13*y[14]
    r14 = k14*y[1]*y[6]
    r15 = k15*y[3]
    r16 = k16*y[4]
    r17 = k17*y[4]
    r18 = k18*y[16]
    r19 = k19*y[16]
    r20 = k20*y[17]*y[6]
    r21 = k21*y[19]
    r22 = k22*y[19]
    r23 = k23*y[1]*y[4]
    r24 = k24*y[19]*y[1]
    r25 = k25*y[20]

    dy[1]  = -r1-r10-r14-r23-r24+r2+r3+r9+r11+r12+r22+r25
    dy[2]  = -r2-r3-r9-r12+r1+r21
    dy[3]  = -r15+r1+r17+r19+r22
    dy[4]  = -r2-r16-r17-r23+r15
    dy[5]  = -r3+r4+r4+r6+r7+r13+r20
    dy[6]  = -r6-r8-r14-r20+r3+r18+r18
    dy[7]  = -r4-r5-r6+r13
    dy[8]  = r4+r5+r6+r7
    dy[9]  = -r7-r8
    dy[10] = -r12+r7+r9
    dy[11] = -r9-r10+r8+r11
    dy[12] = r9
    dy[13] = -r11+r10
    dy[14] = -r13+r12
    dy[15] = r14
    dy[16] = -r18-r19+r16
    dy[17] = -r20
    dy[18] = r20
    dy[19] = -r21-r22-r24+r23+r25
    dy[20] = -r25+r24
end

function get_model(model::String)
    # Lotka Volterra
    if model == "LotkaVolterra"
        p = [1.5, 1, 3, 1]
        tspan = (0.0, 10.0)
        u0 = [1.0, 1.0]
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f_lv, u0, tspan, p)
        reltols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        abstols = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    end

    # Rigid body
    if model == "RigidBody"
        u0 = [1.0, 0.0, 0.9]
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f_rb, u0, (0.0,100.0))
        abstols = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
        reltols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    end

    # Rober model
    if model == "Rober"
        u0 = [1.0,0.0,0.0]
        p = [0.04,3e7,1e4]
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f_ro, u0, (0.0,1e5), p)
        abstols = [1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12]
        reltols = [0.001, 0.0001, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]
    end

    # Hires stiff model
    if model == "Hires"
        u0 = zeros(8)
        u0[1] = 1
        u0[8] = 0.0057
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f_hi, u0, (0.0, 321.8122))
        abstols = [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
        reltols = [0.1, 0.01, 0.001, 0.0001, 1.0e-5, 1.0e-6]
    end

    if model == "Pollu"
        u0 = zeros(20)
        u0[2]  = 0.2
        u0[4]  = 0.04
        u0[7]  = 0.1
        u0[8]  = 0.3
        u0[9]  = 0.01
        u0[17] = 0.007
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(f_po, u0, (0.0,60.0))
        abstols = [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]
        reltols = [0.1, 0.01, 0.001, 0.0001]
    end

    return prob, abstols, reltols
end
