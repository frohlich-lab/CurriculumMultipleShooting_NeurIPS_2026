runJulia="julia --project=. --threads=1"

eval "$(conda shell.bash hook)"
conda activate Test_UDE

# width = 5, hidden layers = 2
if [ $1 == "depth1_width1" ];then
    ${runJulia} Run_lotka_volterra.jl Generate_random_parameters width1 2 5
    ${runJulia} Run_lotka_volterra.jl Benchmark_grad width1 2 5
    python Lotka_volterra.py depth1_width1
fi

# width = 10, hidden layers = 2
if [ $1 == "depth1_width2" ];then
    ${runJulia} Run_lotka_volterra.jl Generate_random_parameters depth1_width2 2 10
    ${runJulia} Run_lotka_volterra.jl Benchmark_grad depth1_width2 2 10
    python Lotka_volterra.py depth1_width2
fi

# width = 15, hidden layers = 2
if [ $1 == "depth1_width3" ];then
    ${runJulia} Run_lotka_volterra.jl Generate_random_parameters depth1_width3 2 15
    ${runJulia} Run_lotka_volterra.jl Benchmark_grad depth1_width3 2 15
    python Lotka_volterra.py depth1_width3
fi

# width = 5, hidden layers = 3
if [ $1 == "depth2_width1" ];then
    ${runJulia} Run_lotka_volterra.jl Generate_random_parameters depth2_width1 3 5
    ${runJulia} Run_lotka_volterra.jl Benchmark_grad depth2_width1 3 5
    python Lotka_volterra.py depth2_width1
fi

# width = 10, hidden layers = 3
if [ $1 == "depth2_width2" ];then
    ${runJulia} Run_lotka_volterra.jl Generate_random_parameters depth2_width2 3 10
    ${runJulia} Run_lotka_volterra.jl Benchmark_grad depth2_width2 3 10
    python Lotka_volterra.py depth2_width2
fi

# width = 15, hidden layers = 3
if [ $1 == "depth2_width3" ];then
    ${runJulia} Run_lotka_volterra.jl Generate_random_parameters depth2_width3 3 15
    ${runJulia} Run_lotka_volterra.jl Benchmark_grad depth2_width3 3 15
    python Lotka_volterra.py depth2_width3
fi
