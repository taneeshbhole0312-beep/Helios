# Contributing to Helios

First of all, thank you for your interest in contributing to Helios.

Helios is an open-source, GPU-accelerated multiphysics simulation project. It is currently under active development, and contributions from developers, engineers, researchers, students and enthusiasts are welcome.

You do not need to be an expert in every area of simulation to contribute. There are many ways to help the project grow.

## Ways to Contribute

Contributions are welcome in several areas:

**1. Numerical Methods and Physics**
- Improving existing CFD solvers and turbulence models.
- Developing and implementing numerical methods.
- Improving boundary conditions and physical models.
- Developing new physics modules.

**2. GPU Computing and Performance**
- Optimizing GLSL compute shaders.
- Improving GPU memory management.
- Identifying computational bottlenecks.
- Improving computational performance and scalability.

**3. Testing and Validation**
- Testing simulation results across different resolutions.
- Comparing results against analytical solutions and established benchmarks.
- Identifying numerical instabilities and convergence problems.
- Testing compatibility across different hardware configurations.

**4. Rendering and Visualization**
- Improving the simulation renderer.
- Developing visualization tools for physical quantities.
- Improving debugging and diagnostic capabilities.
- Developing post-processing tools.

**5. Documentation and General Improvements**
- Improving project documentation.
- Writing tutorials and usage guides.
- Reporting bugs and suggesting new features.
- Improving code readability and maintainability.

## Getting Started

1. Fork the Helios repository on GitHub.
2. Clone your fork to your local machine.
3. Create a separate branch for your changes.
4. Make your modifications and test them.
5. Commit your changes with a descriptive commit message.
6. Push your branch to your fork.
7. Open a Pull Request against the main Helios repository.

Please avoid making unrelated changes in a single pull request. Smaller, focused contributions are easier to review and integrate.

## Numerical Correctness and Validation

Helios is a scientific simulation project. Numerical correctness and physical validity are fundamental requirements.

Contributions involving numerical methods, discretization schemes, physical models or solver modifications should include appropriate testing and validation wherever possible.

Simply compiling successfully or producing visually plausible simulation results is not sufficient evidence of numerical correctness.

Where applicable, contributions should include:
- A description of the numerical or physical approach.
- Relevant analytical or benchmark comparisons.
- Residual convergence and stability observations.
- Conservation checks.
- Performance measurements for GPU-related optimizations.
- Hardware and configuration details required to reproduce the results.

Changes that affect numerical behaviour may require additional review before being accepted.

## Code Contributions

When submitting code:
- Follow the existing project structure and coding conventions where practical.
- Keep changes focused and avoid unnecessary modifications to unrelated components.
- Avoid introducing unnecessary dependencies.
- Document non-obvious algorithms and implementation decisions.
- Ensure that existing functionality is not unintentionally broken.
- Do not commit compiled binaries, temporary files, generated simulation outputs or personal configuration files.

For experimental implementations, clearly identify their experimental status.

## Pull Requests

Every pull request should ideally include:

- A clear description of what was changed.
- The reason for the change.
- Relevant testing performed.
- Screenshots, plots or benchmark results where applicable.
- Any known limitations or unresolved issues.

All pull requests are subject to review. Contributions may be requested to undergo modifications before being merged.

Submitting a pull request does not guarantee that it will be accepted.

## Bug Reports and Feature Requests

If you encounter a bug or have an idea for a new feature, please open a GitHub Issue.

For bug reports, include:
- A description of the problem.
- Steps to reproduce it.
- Expected and actual behaviour.
- Relevant logs or error messages.
- Hardware and simulation configuration where relevant.

For feature requests, explain the intended functionality and why it would be useful.

## Development Status

Helios is still in its early development stages. Its architecture, numerical implementations and development workflows are continuously evolving.

Some components are experimental or undergoing validation. Contributors should expect certain interfaces and implementations to change as the project develops.

Thank you for helping build Helios.

**Every contribution, whether it is a numerical improvement, a bug report, a benchmark or a documentation correction, helps move the project forward.**