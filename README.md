# About

> **Curriculum Multiple Shooting for Robust Training of Neural and Universal Differential Equations** has been accepted at NeurIPS 2026. This repository contains the Python (`diffrax`/`equinox`) NODE and UDE benchmarks used in the paper: Lotka–Volterra (UDE and NODE), SIR UDE, double pendulum UDE, and the high-dimensional FitzHugh–Nagumo reaction–diffusion system (UDE and NODE).

This repository provides example models for training Universal Differential Equations (UDEs) where the ODE vector field integrates both a mechanistic component and a universal approximation element, such as a neural network. Overall, this project has three aims:

1. Assess both [diffrax](https://github.com/patrick-kidger/diffrax) and [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) in terms of runtime and gradient accuracy. This is relevant as both software have fundamentally different approach to handling gradient computations.
2. Evaluate various training schemes for UDE models. Lessons learned from this will likely aid in training other models that incorporate both mechanistic and data-driven components.
3. Develop a PEtab extension to support UDEs, compatible with Diffrax, PEtab.jl, and AMICI.

## SciML PEtab extension

The draft for the PEtab SciML extension can be found [here](https://docs.google.com/document/d/1TKB3rbUShur5KIOuj9r2XdQ6k5acrkkdcWQYK3jDA5U/edit).

## Lotka volterra

This example replicates the scenario described in the UDE [paper](https://arxiv.org/abs/2001.04385). An example script for evaluating training loss and gradient, as well as validation loss, is available in the `lotka_volterra` folder.

Briefly, the model is given by:

$$\frac{\mathrm{d} prey}{\mathrm{d} t} = \alpha * prey + N[1] $$

$$\frac{\mathrm{d} predator}{\mathrm{d} t} = N[2] - \delta * predator$$

Where $N$ here is the output of a small feed-forward neural network that takes $prey$ and $predator$ as input.

### Original training approach

In the original [article](https://arxiv.org/abs/2001.04385), the model was trained using a combination of the Adam optimizer and the BFGS algorithm. Specifically, Adam with a fixed learning rate was initially run for 500 iterations, followed by a maximum of 1000 iterations using BFGS. The issue with Adam here is that the objective value started to oscillate once a sufficiently small objective values was reached.

## SIR

This example mimics the scenario described in this [paper](https://www.sciencedirect.com/science/article/pii/S2666389920301938). An example script for evaluating training loss and gradient is available in the `sir` folder.

Briefly the model is given by:

$$\frac{\mathrm{d}S}{\mathrm{d}t} = -\frac{\beta * S * I}{N}$$

$$\frac{\mathrm{d}I}{\mathrm{d}t} = \frac{\beta * S * I}{N} - (\gamma + Q) * I$$

$$\frac{\mathrm{d}R}{\mathrm{d}t} = \gamma I + \delta T$$

$$\frac{\mathrm{d}T}{\mathrm{d}t} = Q * I - \delta * T$$

$$Q = NN\left(S, I, R\right)$$

In this model, $S$ represents the susceptible population, $I$ denotes the infected individuals, $R$ is for the recovered group, and $T$ indicates the quarantine population. The term $Q$, which represents the quarantine strength, is derived from a data-driven model because according to the authors it does not naturally arise from first principles.

### Original training approach

In the original paper for countries with a high recovery rate the model was trained by running Adam with a fixed learning rate for $40\,000$ to $100\,000$ iterations. For countries with a low recovery rate they used a two-stage approach. First they ran Adam with $40\,000$ iterations. In the second state they removed $R$ from the loss function, fixed $\gamma$ and $\delta$, and ran Adam for $30\,000$ to $100\,000$ iterations.

## Reaction–diffusion

This example tests training strategies on a high-dimensional system, adapted from [APHYNITY](https://arxiv.org/abs/2010.04456). The code is in `src/model_examples/reaction_diffusion`.

The ground truth is the FitzHugh–Nagumo model for an activator $u$ and an inhibitor $v$, in its oscillatory spiral-wave regime:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - \frac{u^3}{3} - v$$

$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \varepsilon\,(u + a - b v)$$

with $(D_u, D_v) = (1, 1)$, $(a, b) = (0.1, 1)$, and $\varepsilon = 0.005$. It is discretised on a $64\times64$ grid (domain side length 100, five-point Laplacian, Neumann boundaries), giving $2 \times 64^2 = 8192$ states. A gradient initial condition creates a single phase defect, around which a spiral wave rotates with a period of about 258 time units. The data are noise-free and fully observed. Two rotations (36 time points) are used for training and the next rotation (18 time points) for validation.

Two model variants are provided:

* **UDE**: the diffusion term is kept mechanistic with unknown coefficients $(D_u, D_v)$, and a convolutional network learns the reaction term,

$$\frac{\partial}{\partial t}\begin{pmatrix} u \\ v \end{pmatrix} = \begin{pmatrix} D_u \nabla^2 u \\ D_v \nabla^2 v \end{pmatrix} + NN(u, v).$$

* **NODE**: the convolutional network models the full right-hand side.

In both, $NN$ is a three-layer $3\times3$ ConvNet with 16 hidden channels and ReLU activations, with padding matched to the boundary condition. Unlike APHYNITY, there is no batch normalisation.

### Running the benchmark

1. Activate the configuration: copy `src/model_training/training_configuration_reactionDiffusion.py` over `src/model_training/training_configuration.py`. Per-method hyperparameters (learning rate, continuity penalty, number of curriculum stages) are pinned in `RD_METHOD_CONFIG` in `src/generate_run_configs.py`.
2. Generate the data with `src/model_examples/reaction_diffusion/generate_data.py`.
3. Train. Each run takes roughly 9–40 hours on 8 CPU cores, so the 800 runs of the paper's benchmark (2 models × 4 strategies × 100 multistarts) need about $1.5\times10^5$ CPU-hours. Launch with `WANDB_MODE=online` so long runs can be monitored while they train.
4. Make the figure: `export_final_for_r.py` writes a tidy results table, and `rd_final_figures.R` draws the paper's reaction–diffusion figure from it (both in `src/model_examples/reaction_diffusion/plots`).
