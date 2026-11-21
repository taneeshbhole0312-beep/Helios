# k–ε NaN investigation (wall functions OFF)

Date: 2026-10-13 · Intel HD 630 / GL 4.3 · production code unmodified · probe: `audit/nan_probe.cpp`

Configuration: `main.cpp` values (64³, 200 m, dt = 1/60, ρ = 1.225, μ = 1.8e-5, coarsest level 8, 2 pressure V-cycles),
U_ref = 80, L = 0.1. The probe follows `Orchestrator::run_rans_k_eps` call for call up to the first k/ε solve:
- initialisation;
- initial projection;
- save φⁿ, store iterate;
- momentum assembly;
- u/v/w `multigrid_solve_system(8, 1e-3, 1e-4, 3)`;
- projection;
- `construct_k_epsilon_system`.

Momentum, k–ε and coarse wall functions are all commented out in the current sources (RANS.cpp:519-521, 717, 732, 747,
774; k_TurbulenceModel.cpp:492, 601).

Build (from the project root):

```
g++ -std=c++17 -O2 -I. audit/nan_probe.cpp shader_manager/shader_manager.cpp Generic_multigrid/generic_multigrid.cpp Linear_systems/LinearSystem.cpp Turbulence_models/RANS.cpp Turbulence_models/k_TurbulenceModel.cpp Incompressible_flow/incompressible_flow.cpp C:/GLAD/src/glad.c -IC:/GLAD/include -IC:/GLFW-3.5.1/glfw-3.5.1.bin.WIN64/include -LC:/GLFW-3.5.1/glfw-3.5.1.bin.WIN64/lib-mingw-w64 -lglfw3 -lopengl32 -lgdi32 -o nan.exe
nan.exe probe I=5      # validation + instrumented V-cycle + production solve + mu_eff probe
nan.exe pin   I=5      # causal test: pin the empty rows in the probe's buffers only, then run the production solver
```

## Root cause (confirmed)

With the k–ε wall treatment disabled, **every boundary cell of the k and ε systems is an all-zero row:
aP = 0, all aₙᵦ = 0, RHS = 0**. That is 23,816 of 23,816 boundary cells on the fine level, plus 5,768 / 1,352 / 296
rows on coarse levels 1 / 2 / 3. The first Gauss-Seidel update of such a row computes (0 − 0)/0 = **NaN**.

Why the rows are empty: every k/ε assembly kernel skips boundary cells with the comment "handled by the wall treatment":
- `assemble_time.comp:122` (`topology == 0 && boundary_cell` → return, both fine and coarse)
- `assemble_advection.comp:216`
- `assemble_k_eps_diffusion_contribution.comp:86`
- `assemble_k_destruction_contribution.comp:51`
- `assemble_Pk_contribution.comp:83`
- `assemble_eps_source_contribution.comp:103`

`K_eps_wall_treatment.comp` is the only code that writes those rows: aP = 1 and RHS = wall value on the fine level, and
the 1e20 penalty on coarse levels. Commenting its calls out leaves the rows empty.

## Evidence

**Pre-solve validation (I = 5):**
- k, ε, μ_eff, RHS and all coefficients are finite.
- k RHS range [0, 5.38e8]; ε RHS range [0, 4.33e11].
- Fine rows with aP == 0: 23,816. All of them are boundary cells, all are all-zero rows, and all have RHS = 0.
- 22,328 interior rows have a non-zero coefficient pointing at one of those rows.
- Production initial RMS: k 6.88656e9, ε 1.06437e13 (your reported values).

CPU evaluation of the smoother update at cell (0,0,0): aP = 0, RHS = 0 → (RHS − Σ)/aP = **nan**.

**Instrumented V-cycle.** Identical for k and ε; non-finite count after each stage, out of 262,144 fine values:

| Stage | Non-finite | Notes |
|---|---|---|
| start | 0 | |
| fine pre-smooth sweep 1 **RED** | **11,908** | all boundary, all zero-diagonal, first at (0,0,0) ← **first NaN** |
| sweep 1 BLACK | 34,980 | 23,816 boundary + 11,164 interior (propagation through a_nb) |
| sweeps 2, 3 | 56,588 → 113,108 | |
| fine residual | 129,340 | |
| restriction → L1 | 18,944 / 32,768 | |
| L1 smoothing → L2 → L3 coarse solve | 100 % | |
| prolongation back to the fine level | 262,144 / 262,144 | |

The residual, restriction, prolongation and correction stages only *carry* NaN; none of them *creates* one. The coarse
smoothers would also create NaN on their own empty rows.

**Production solver from the same state:**
- RMS_initial = 6.88656e9, then RMS after the 2 probe cycles = nan.
- Status 4 (`DIVERGING_FROM_ANSWER`) for both k and ε; both fields 100 % NaN.

**Causal test** (`pin`): set aP = 1, RHS = current value on the empty fine rows and aP = 1 on the empty coarse rows, in the
probe's buffers only, then run the unmodified production solver.

| I | k | ε | status |
|---|---|---|---|
| 5 | RMS 6.89e9 → 72.6, range [1.67e4, 2.4e5] | RMS 1.06e13 → 3.4e4, range [7.2e6, 1.93e8] | 1 (converged, relative) |
| 0.05 | RMS 6.86e3 → 0.023, range [21.2, 24] | RMS 1.06e5 → 0.108, range [154, 193] | 1 |

No NaN in either run. Removing the empty rows removes the failure, and nothing else was changed.

**One-setting-at-a-time comparison** (only I changes):

| I | Empty rows | Initial RMS (k / ε) | Production result |
|---|---|---|---|
| 0 | 23,816 | 0 / 0 | returns at the absolute check → no NaN **(bypassed)**; forced V-cycle → 11,908 NaN at sweep 1 RED |
| 0.05 | 23,816 | 6.86e3 / 1.06e5 | NaN after the probe cycles, status 4 |
| 5 | 23,816 | 6.89e9 / 1.06e13 | NaN after the probe cycles, status 4 |

The momentum wall functions do not touch the k/ε operators. Your "momentum WF ON, I = 0, no NaN" result is the same
bypass as the I = 0 row above.

## Why u, v, w stay finite (verified, and it is masking the failure)

`K_eps_mu_eff.comp` computes `max(k, 0)` and `max(eps, 0) + FLT_MIN`. On this driver, GLSL `max(NaN, 0.0)` returns 0.
The GLSL spec leaves this undefined, so it depends on the driver.

Measured μ_eff for controlled inputs (finite partner value k = 24 or ε = 193):

| Input | μ_eff |
|---|---|
| k NaN | 1.8e-5 (μ) |
| k NaN and ε NaN | 1.8e-5 |
| k −Inf, or k negative | 1.8e-5 |
| ε +Inf | 1.8e-5 |
| **ε NaN** (k finite) | **inf** |
| **ε = 0 or ε < 0** (k > 0) | **inf** |
| **k +Inf, or k = 1e20** | **inf** |

After the failed solve both fields are NaN, so μ_eff becomes exactly μ everywhere. The momentum equations then continue
silently as **laminar** flow, and the orchestrator only prints status 4 and carries on. If ε alone goes NaN or ≤ 0 while
k is finite, μ_eff = inf, and the momentum operators would be poisoned too.

## Confirmed vs hypothesis

**Confirmed (measured):**
- Empty boundary rows are the sole origin of the NaN.
- The first NaN appears at the fine pre-smoothing RED pass on the red boundary cells.
- Pinning those rows eliminates it.
- I = 0 is a bypass, not a fix.
- μ_eff silently maps NaN k/ε to laminar μ on this driver, and maps ε ≤ 0 or NaN (with finite k) to inf.

**Not a defect:** the generic multigrid and the RBGS smoother behave correctly for the operator they are given. A
zero-diagonal row is not a valid equation.

**Observation:** I is used as a fraction, so typing 5 means 500 % intensity (k₀ = 240,000 m²/s², u′ = 400 m/s).
Your reported initial RMS values (6.88e9 / 1.06e13) match I = 5 exactly, not I = 0.05.

## Fix options (not implemented)

Any boundary condition for the k/ε boundary cells removes the empty rows when wall functions are off:
- **Dirichlet:** aP = 1, RHS = prescribed value.
- **Zero gradient:** aP = 1, coefficient to the interior neighbour = −1, RHS = 0.
- **Assemble the boundary cells:** include them in the regular kernels, with wall-face fluxes = 0.

The coarse levels need the matching homogeneous rows (aP = 1 or the penalty), or the coarse smoother creates NaN too.

Cheap guards worth adding:
- An assertion before each solve that every aP > 0 and is finite.
- A NaN/Inf check on the solved fields that stops the time step on status 4 instead of continuing.
- A guard in `K_eps_mu_eff.comp` for ε ≤ 0 or non-finite values, not only for k.
