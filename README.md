# Helios

Helios is an open-source, GPU-accelerated multiphysics simulation platform, starting with computational fluid dynamics. Its goal is to make advanced simulation technology accessible to everyone, providing inventors with a virtual environment to experiment, innovate and bring their ideas to life.


Vision of Helios:
1) Helios aims to make advanced simulation accessible to everyone, regardless of their technical background 

2) Testing technology in real life is extremely expensive and often takes alot of manforce and skills to pull it off. So the vision 
of Helios is to give every single inventor a world of their own to experiment with. 

3) Helios should be a multi-physics, coupled or isolated, simulation tool, that is completely free to use for any inventor out in 
the world. Things like elctromagnetism, fluid dynamics, turbulence modelling, rigid body physics, deformable body physics and much 
more will come together in Helios.

4) Lastly, one of the most important of all in Helios should be speed. So this project should be as GPU accelerated as possible
and where that is not important, it should be coupled to the other parts that are GPU accelerated.

-------------------------------Currect status of helios----------------------------- 

Helios is very much in it's developmental phase right now and the code is incredibly messy.

Helios foundation and dependencies-
1) C++ orchestrated and openGL based GPU control. (.comp kernels)
2) SSBO based data storage.
3) FFmpeg based video generation.
4) GLFW and GLAD
5. Object-oriented C++ architecture for modular solver and simulation management.

Helios capabilities right now-
1) **Turbulence modelling**
   - Multiple RANS turbulence models: k-epsilon, k-omega and SST.
   - Wall treatment implementation for turbulence models.
   - Turbulence field initialization based on reference velocity, turbulence intensity and turbulence length scale.
   - Calculation of turbulent and effective viscosity.
    **Fluid dynamics**
   - Three-dimensional incompressible fluid flow simulation.
   - Finite-volume-based spatial discretization.
   - Staggered Cartesian grid arrangement for velocity and pressure-related fields.
   - Pressure-velocity coupling through a pressure projection approach.
   - Separate momentum and pressure solvers.
    **Numerical methods**
   - First-order upwind convection scheme.
   - Second-order deferred convection scheme.
   - Flux limiters: Minmod, Van Leer and Superbee.
   - Second-order central discretization for diffusion.
   - Backward Euler and BDF2 temporal discretization.
   - Seven-point stencil-based linear system assembly.

4) **Linear solvers and multigrid**
   - Multigrid-based solution of discretized equations.
   - Restriction and prolongation operations for transferring information between grid levels.
   - Support for arbitrary structured grid dimensions, including non-power-of-two resolutions.
   - Semi-coarsening to handle grid anisotropy.
   - Shared solver infrastructure and memory-conscious GPU buffer allocation.

5) **GPU acceleration**
   - GPU-based computation through OpenGL compute shaders.
   - GLSL `.comp` kernels for numerical operations.
   - Shader Storage Buffer Objects (SSBOs) for GPU-side field and coefficient storage.
   - GPU-based discretization, coefficient assembly and solver operations.
   - CPU-side orchestration of simulation execution and GPU workloads.

6) **Simulation visualization and output**
   - FFmpeg integration for simulation video generation.
   - Visualization of fluid quantities such as velocity magnitude and turbulent kinetic energy.
   - Time-dependent simulation output for post-processing and analysis.

---

## Current development status

Helios is still under active development. Its numerical methods, solver architecture, memory management and user interface are continuously evolving.

Some implementations are experimental, incomplete or undergoing validation. The current repository represents the actual development state of the project rather than a polished or production-ready release.

--------------------------------Road Map Of Helios---------------------------------
1. CFD Solver Validation and Refinement

Thoroughly validate existing k-epsilon, k-omega and SST turbulence models.

Validate solver accuracy, stability and convergence against established benchmarks.

Improve multigrid performance and numerical robustness.

Refine convection schemes, temporal discretization and boundary treatments.

Optimize GPU memory management and computational performance.

2. Rigid Body Physics

Develop a GPU-accelerated rigid body dynamics solver.

Implement collision detection and collision response.

Introduce contact mechanics and constraint handling.

Develop interaction between rigid bodies and the existing simulation infrastructure.

3. Multiphysics Coupling

Couple rigid body dynamics with fluid simulations.

Develop fluid-structure interaction capabilities.

Establish a unified architecture for exchanging physical quantities between independent solvers.

Expand towards more complex coupled physical systems.

4. Rendering and Development Tools

Significantly improve the simulation renderer and visualization pipeline.

Develop advanced debugging and diagnostic tools.

Introduce interactive visualization of simulation fields and physical quantities.

Improve simulation inspection, profiling and post-processing workflows.

5. Expansion into Additional Physics

Deformable body mechanics and structural physics.

Electromagnetic field simulation.

Heat transfer and thermal-fluid coupling.

Additional physical models and multiphysics interactions.

6. Unified Simulation Environment

Develop an intuitive and accessible graphical interface.

Integrate simulation setup, execution, visualization and debugging into a unified workflow.

Improve scalability, extensibility and GPU utilization across all physics modules.

Work towards making advanced multiphysics simulation accessible to inventors, researchers and engineers everywhere.

This roadmap represents the long-term direction of Helios. Development priorities and implementation details may evolve as the project progresses.


