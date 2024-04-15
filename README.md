# About

This repository provides example models for training Universal Differential Equations (UDEs) where the ODE vector field integrates both a mechanistic component and a universal approximation element, such as a neural network. Overall, this project has three aims:

1. Assess both [diffrax](https://github.com/patrick-kidger/diffrax) and [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) in terms of runtime and gradient accuracy. This is relevant as both software have fundamentally different approach to handling gradient computations.
2. Evaluate various training schemes for UDE models. Lessons learned from this will likely aid in training other models that incorporate both mechanistic and data-driven components.
3. Develop a PEtab extension to support UDEs, compatible with Diffrax, PEtab.jl, and AMICI.

## Models

### Lotka volterra

This example replicates the scenario described in the UDE [paper](https://arxiv.org/abs/2001.04385). An example script for evaluating training loss and gradient, as well as validation loss, is available in the `lotka_volterra` folder.
