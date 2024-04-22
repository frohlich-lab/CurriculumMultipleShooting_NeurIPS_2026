# About

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