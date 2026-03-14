# Autonomous Rocket Landing with Model Predictive Control

This directory contains the implementation of a comprehensive Model Predictive Control (MPC) stack to autonomously land a 3D rocket under thrust. Developed as the final project for the **ME-425 Model Predictive Control** course at EPFL.

The project progresses from basic linear stabilization to handling nonlinear dynamics, unmodeled disturbances, and safety-critical robust constraints.

## 🚀 Control Strategies Implemented

1. **Linear MPC with Terminal Components (Deliverable 3):**
    - Decoupled the 12-state nonlinear rocket dynamics into 4 independent linear subsystems ($x, y, z, \text{roll}$).
    - Ensured recursive feasibility and stability by computing infinite-horizon LQR terminal costs ($P$) and Maximum Output Admissible Sets (MOAS, $\mathcal{X}_f$) using Polyhedral manipulations.
    - Implemented a cascaded PI-MPC architecture for global spatial navigation with safe velocity saturation clamping.

2. **Offset-Free Tracking & State Estimation (Deliverable 5):**
    - Implemented a discrete Luenberger observer to estimate unmodeled disturbances in real-time (e.g., constant thrust loss and time-varying mass depletion due to fuel burn).
    - Augmented the MPC to achieve offset-free tracking by dynamically shifting the steady-state target.

3. **Robust Tube MPC (Deliverable 6):**
    - Designed a Robust Tube MPC for the safety-critical vertical descent phase to strictly guarantee ground collision avoidance ($z \ge 0$) against worst-case stochastic wind disturbances.
    - Computed the Minimal Robust Positively Invariant (mRPI) set ($\mathcal{E}$) to tighten nominal state and input constraints, pairing a nominal MPC planner with an ancillary LQR feedback law.

4. **Nonlinear MPC (Deliverable 7):**
    - Formulated a centralized Nonlinear MPC (NMPC) to replace the decoupled linear controllers.
    - Utilized **CasADi** and a symbolic Runge-Kutta 4 (RK4) integrator to explicitly account for cross-coupling effects and nonlinear actuator characteristics during aggressive maneuvers (e.g., correcting a 30° roll offset).

## 📂 Codebase Structure

The project is organized sequentially by deliverable. Each `Deliverable_X` folder contains a Jupyter Notebook to run the simulation and generate visualizations for that specific control architecture.

- `Deliverable_3_*` to `Deliverable_5_*`/ : Notebooks covering Linear MPC, tracking, cascaded PI loops, and offset-free observers.
- `Deliverable_6_1&2`/ : Implementation of the Robust Tube MPC and the merged landing controller.
- `Deliverable_7`/ : Implementation of the full Nonlinear MPC.
- `LinearMPC`/ : Core object-oriented implementation of the Linear MPC using **CVXPY** and **OSQP**. Contains base classes and specific subsystem definitions.
- `PIControl`/ : Outer-loop proportional-integral controllers for position tracking.
- `src`/ : Core rocket dynamics, physical parameters, and 3D visualization utilities.

## 🛠️ Tech Stack

- **Languages:** Python
- **Optimization & Solvers:** CVXPY, OSQP, CasADi, IPOPT
- **Control & Math:** SciPy, `mpt4py` (Multiparametric Toolbox for Polyhedral geometry)
- **Visualization:** Matplotlib, ipympl

## 🏃‍♂️ How to Run

To test a specific controller, navigate to the respective `Deliverable` folder and run the provided Jupyter Notebook. The notebooks will initialize the rocket environment, trim/linearize the system (if applicable), solve the optimization problem, and output both 2D state/input plots and a 3D animation of the landing sequence.
