# Physics-Informed Neural Network (PINN) for Inviscid Burgers' Equation

This repository contains a PyTorch implementation of a Physics-Informed Neural Network (PINN) designed to solve the **inviscid Burgers' equation** ($u_t + u u_x = 0$).

Capturing non-differentiable shock discontinuities using neural networks with smooth activation functions (e.g., `Tanh`) is notoriously difficult due to **spectral bias**, gradient explosion, and optimization stagnancy. This project presents a multi-stage hybrid optimization pipeline that recovers sharp, vertical shock fronts without introducing artificial numerical dispersion or high-frequency wiggles.

---

## Key Technical Features

1. **Artificial Viscosity Annealing (Continuation Method):** Dynamically decays viscosity ($\nu: 0.05 \to 0.0$) during early training to guide the network away from poor local minima.
2. **Dynamic Spatial-Temporal Resampling:** Randomly resamples collocation points $(x, t)$ every epoch in the Adam phase to ensure complete domain coverage and prevent localized memorization.
3. **Safe First-Order PDE Formulation:** Bypasses second-derivative calculations when $\nu = 0$ to prevent floating-point $0 \times \infty = \text{NaN}$ errors during automatic differentiation.
4. **Two-Stage Hybrid Optimization:** Combines stochastic first-order search (Adam) for global topology discovery with deterministic second-order optimization (L-BFGS with Strong Wolfe line search) for sharpening the shock boundary.
5. **Temporal Domain Weighting:** Splits early-time ($t < 0.4$) and late-time ($t \ge 0.4$) loss terms to prevent the extreme gradients of the shock from corrupting smooth pre-shock curves.
6. **L2 Regularization (Weight Decay):** Introduces structural weight tension ($10^{-4} \cdot \mathcal{L}_{\text{L2}}$) to eliminate Gibbs phenomenon wiggles in pre-shock regions.

---

## Problem Statement

The governing equation is the 1D inviscid Burgers' equation:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0, \quad x \in [-1, 1], \quad t \in [0, 0.99]$$

Subject to the initial and boundary conditions:

$$u(x, 0) = -\sin(\pi x)$$
$$u(-1, t) = u(1, t) = 0$$

As time advances ($t > 1/\pi \approx 0.318$), the smooth initial sine wave steepens into a non-differentiable shock wave at $x \approx 0$.

---
## File Structure 
* **Viscous_Burgers_PINN :** Implementation of PINN to solve Viscous Burgers Equation with viscosity value 0.01 
* **Inviscid_Burgers_PINN :** Viscosity set to 0 to visualize the failure of PINN to capture the shock
* **Inv_Burgers_Adam_LBFGS_AVA :** Resolution of shock discontinuity using Aritificial Viscosity Annealing strategy, Adam optimizer and L-BFGS optimizer

---

## Pipeline Overview
Stage 1: Adam Phase       

1. Dynamic Collocation Points
2. Viscosity Annealing (nu: 0.05 -> 0.0)
3. Learns Macro Topology                  

Stage 2: Balanced L-BFGS Phase
 
1. Fixed Collocation Points
2. Pure Inviscid Residual Formulation
3. Uses Hessian Curvature to Sharpen Shock

 ## Results

* **Viscous Benchmark ($\nu = 0.01/\pi$):** Achieved an **$R^2$ Score of 99.39%** against Raissi's reference dataset using viscosity continuation.
* **Inviscid Benchmark ($\nu = 0.0$):** Captured a sharp vertical shock front at $t = 0.95$ while maintaining smooth pre-shock sine wave behavior without numerical ringing.

---
## Contact 
Kenneth Steve Diyya  
kennethsteve.d@gmail.com

