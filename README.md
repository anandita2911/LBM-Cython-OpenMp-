# Lid-Driven Cavity Simulation using Lattice Boltzmann Method

Welcome to the Lid-Driven Cavity Simulation project! This project implements a numerical simulation of fluid flow in a lid-driven cavity using the Lattice Boltzmann Method (LBM). The simulation is optimized for performance using a combination of Python, Cython, and parallel computing with OpenMP.

## Overview

The lid-driven cavity is a classic benchmark problem in computational fluid dynamics. The simulation considers a square cavity with a moving lid, where fluid dynamics phenomena such as vortex formation and boundary layer development can be observed.

## Features

- **Lattice Boltzmann Method (LBM):** The LBM is a powerful method for simulating fluid dynamics using a discrete kinetic model. It offers advantages in simulating complex fluid behaviors.
- **Cython Optimization:** The code has been optimized using Cython, a tool that allows writing C extensions for Python, enhancing the simulation's computational efficiency.
- **Parallel Computing with OpenMP:** OpenMP API has been used to parallelize the computations, significantly reducing the simulation time.
- **Visualization:** The simulation results can be visualized, showing the evolution of fluid flow and identifying key flow patterns.

## Usage

1. Clone this repository to your local machine.
2. Ensure you have Python and Cython installed.

