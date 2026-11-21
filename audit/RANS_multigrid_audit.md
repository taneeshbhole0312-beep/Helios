# Helios RANS / Generic Multigrid — Evidence-Driven Audit

Date: 2026-10-10 · GPU: Intel HD Graphics 630, OpenGL 4.3.0 (Build 31.0.101.2140) · Compiler: MSYS2 g++ (ucrt64), `-O2`

No implementation file was modified. All experiments link the **unmodified** project sources; test-only access to
private members uses `#define private public` inside the harness. Everything needed to repeat the tests is in this folder.

---

## 0. Reproduction

Code revision (no git repository, so files are identified by SHA-256 prefix + modification time):

| SHA-256 (12) | modified | file |
|---|---|---|
| E44766208187 | 2026-10-10 19:54 | orchestrator/orchestrator.cpp |
| A34A6139DAEE | 2026-10-10 19:17 | Generic_multigrid/generic_multigrid.cpp |
| 10F9B72D00FA | 2026-10-10 19:49 | Turbulence_models/RANS.cpp |
| 486F19C0105E | 2026-10-10 15:30 | Turbulence_models/k_TurbulenceModel.cpp |
| 32FDD40372CA | 2026-10-10 15:52 | Linear_systems/LinearSystem.cpp |
| 2EBB2F385418 | 2026-10-07 20:13 | Kernels/multigrid_residual_generic_restriction.comp |
| FA18E361E0F0 | 2026-10-10 19:44 | Kernels/Generic_multigrid_prolongation.comp |
| 4D08908F312F | 2026-09-13 17:07 | Kernels/GenericRBGSsmoother.comp |
| E15746CA262E | 2026-09-11 23:47 | Kernels/generic_residual_cal.comp |
| 4F7DF17F5A91 | 2026-10-05 20:25 | Kernels/residual_sq_summation_L2.comp |
| 090C48165BB5 | 2026-09-20 13:32 | Kernels/assemble_time.comp |
| BDD3FFFE72D1 | 2026-10-10 16:19 | Kernels/assemble_k_destruction_contribution.comp |
| C107FBEBDFDD | 2026-10-10 14:33 | Kernels/momentum_wall_contribution.comp |

Build (run from the project root, so `Kernels/` resolves):

```
g++ -std=c++17 -O2 -I. audit/audit_harness.cpp shader_manager/shader_manager.cpp Generic_multigrid/generic_multigrid.cpp Linear_systems/LinearSystem.cpp Turbulence_models/RANS.cpp Turbulence_models/k_TurbulenceModel.cpp Incompressible_flow/incompressible_flow.cpp C:/GLAD/src/glad.c -IC:/GLAD/include -IC:/GLFW-3.5.1/glfw-3.5.1.bin.WIN64/include -LC:/GLFW-3.5.1/glfw-3.5.1.bin.WIN64/lib-mingw-w64 -lglfw3 -lopengl32 -lgdi32 -o audit.exe
```

`orch_run.cpp` (runs the real `Orchestrator::run_rans_k_eps` for N steps) additionally links
`orchestrator/orchestrator.cpp explicit_sim/explicit_sim.cpp Debuggers/debuggers.cpp Renderer/visualization.cpp -lwinmm`.
Usage: `orch.exe U I L steps [dt]`.

Harness modes (`audit.exe <mode> [N=.. dt=.. U=.. I=.. L=..]`): `loc`, `l2`, `ops`, `zero`, `cyc<s> <n>`, `known<s>`,
`prod<s>`, `refine<s>`, `twogrid<s>`, `realcmp<s>`, `poisson0|poisson1`, `wallmag`, `orchrep`; `<s>` ∈ `u v w k e`.

Default configuration = `main.cpp`: 32³ cells, 200 m cube (h = 6.25 m), ρ = 1.225, μ = 1.8e-5, dt = 1/60 s,
coarsest level 8, Spalding E = 9.8. The k-ε inputs are typed at runtime in `main.cpp`, so I used **U=10, I=0.05, L=1**
unless stated otherwise; U=50, I=0.1, L=1 was used to reproduce the reported symptom.

---

## 7. Executive summary (read first)

**What is established**

1. **The multigrid solver is not failing on these systems.** One V-cycle takes the U system from 26 % solution error to
   **5.1e-8 relative error** (float32 machine precision, T-03, T-04). Iterative refinement with a double-precision
   residual drives the same GPU multigrid to a relative residual of **1e-16** (T-05). Residual evaluation, the L2/RMS
   reduction and the convergence-factor formula all agree with independent double-precision computations (T-01, T-07).
2. **The stagnation is the float32 floor, and the stopping tolerance is below it.** The solvers stop changing x
   (bit-identical iterates) at an RMS residual of ≈ 0.4 × (unit round-off × row magnitude): **0.054 for U**, **3.5e-4 for k**,
   **5.5e-5 for ε** at U=10, and ~0.08 (U) / ~0.055 (k) at U=50. The orchestrator asks for an **absolute RMS of 1e-5**
   (`10e-6`), which is 2–4 orders of magnitude below what float32 can represent for these magnitudes (|b| ~ 1e6 per row).
3. **The "thousands of cycles" come from the adaptive controller.** At the floor the cycle-to-cycle ratio is noise
   (exactly 1.0, or e.g. 0.999397 / 0.999986). A ratio just below 1 gives `ln(tol/RMS)/ln(cf)` = **14,910** and
   **659,608** cycles, which `multigrid_solve_system` executes inside **one** `Generic_multigrid_solve` call before it
   ever compares against `MAX_V_CYCLES = 50` (T-09, reproduced with the real orchestrator). This explains the reported
   0.997 factors, the 40,000-cycle runs, the "max 50 not enforced", and why dt only changes the predicted count.
4. **The nonlinear loop never terminates by convergence because of W.** u, v, k, ε converge to round-off
   (δ ≈ 1.5e-7) by iteration 7, but δ_w = ‖Δw‖/‖w‖ stays ≈ 0.14 forever because w is pure solver noise
   (max|w|/max|u| from 1.5e-3 down to 2.6e-5). The loop always runs to the 100-iteration cap (T-14, T-08).
5. **Face-centred residual restriction is under-scaled by 2×** (Galerkin scale 0.495 for U vs 0.987/0.985 for k/ε).
   Doubling it restores textbook factors (0.03–0.06) on a diffusion-dominated face problem at 32³ and 64³, against
   0.60–0.73 in production (T-12). At dt = 1/60 this does **not** matter, because the time term dominates and
   6 smoothing sweeps alone solve the system (T-10). It matters at large dt, or when diffusion dominates.
6. **W is genuinely converged at its early exit:** b_w = 0 and w = 0 exactly (T-06).

**What remains unknown**

- Why the large-dt U operator has 992 rows that are not diagonally dominant (min aP/Σ|a_nb| = 0.51). The leading
  hypothesis (H-3) is a non-zero net face flux ΣF from the inadmissible initial field; it is untested.
- Whether float32 accumulation limits are acceptable for the intended physics. That is a design question, not a defect.
- Reported RMS_k ≈ 2300 / RMS_ε ≈ 11000 could not be matched exactly without your k-ε inputs. They are consistent with
  **initial** residuals, which scale with the inputs (mine: 965 and 461 at U=10).

**Strongest current lead / root cause of the reported symptoms:** absolute float32-unreachable tolerance (F-01) plus an
uncapped predictor (F-02) for the linear solves, and the w-normalisation trap (F-04) for the nonlinear loop. Together
they account for every reported linear-solver symptom. A full time step at U=10 costs 70 s, almost all of it in V-cycles
performed after the solution had already reached machine precision.

---

## 1. Repository map (RANS path)

```
main.cpp ──> Orchestrator::run_rans_k_eps            orchestrator/orchestrator.cpp:355-484
  objects: RANS (u/v/w LinearSystem, mu_eff, velocity + mu_eff hierarchies)              Turbulence_models/RANS.cpp:20-85
           k_TurbulenceModels (KSystem) + ::epsilon (eps_system, k/eps hierarchies)       Turbulence_models/k_TurbulenceModel.cpp
           5 × Generic_multigrid (each owns RHS/correction/residual buffers per level + L2 reduction chain)
           incompressible_flow + specialized_pressure_Multigrid (explicit-solver pressure projection)
  init:    RANS::initialize_velocity_fields (RANS.cpp:91)  ->  epsilon::initialize_k_epsilon_RANS_fields (k0, eps0, wall cells, mu_eff)
  per physical step:
    save_previous_time_step_field × 5                       (LinearSystem.cpp:293, GPU copy field -> previous_time_step_field)
    nonlinear loop (cap 100):
      RANS::store_nonlinear_iterate                         (RANS.cpp:854)
      RANS::construct_momentum_equations                    (RANS.cpp:477)
         clear -> add_time_contribution (assemble_time.comp, phi_old = previous_time_step_field)
               -> add_advection_contribution (assemble_advection.comp, upwind)
               -> add_viscosity_contribution_momentum (assemble_viscosity_LHS/RHS_momentum.comp)
               -> [fine wall function COMMENTED OUT, RANS.cpp:513-515]
               -> build_coefficients_hierarchy (restrict_vels, restrict_mu_eff; per level: time, viscosity, advection, WALL FUNCTION)
      u/v/w: Generic_multigrid::multigrid_solve_system(8, 1e-5)
      incompressible_flow::divergence_cal -> Multigrid_pressure_solve(dt, rho, 2 V-cycles, warm start) -> project_pressure
      RANS::construct_k_epsilon_system                      (RANS.cpp:759)
         K fine (time, advection, diffusion, implicit destruction on aP, P_k on RHS), eps fine (… + production/destruction),
         fine wall treatment (pins wall rows), K hierarchy, eps hierarchy (+ coarse wall penalty for both)
      k/eps: multigrid_solve_system(8, 1e-5)
      epsilon::update_mu_eff
      RANS::calculate_nonlinear_convergence  -> break if < 1e-5
```

**One linear solve.** `multigrid_solve_system` (generic_multigrid.cpp:589-717) →
`calculate_l2_norm` (720-822: residual kernel → `residual_sq_summation_L2.comp` chain → 1-float readback) →
`Generic_multigrid_solve(n)` (93-587), which runs n V-cycles. Each V-cycle:

1. Zero the coarse corrections.
2. Finest level: 3 RBGS sweeps on `field.data`, then residual (`generic_residual_cal.comp`), then restrict
   (`multigrid_residual_generic_restriction.comp`) into `RHS[1]`.
3. Each intermediate level: 3 sweeps on its correction, residual, restrict.
4. Coarsest level: 30 sweeps.
5. Going up: prolongate (`Generic_multigrid_prolongation.comp`, trilinear, adds) and 3 sweeps per level.
6. Final: prolongate into `field.data`, then 3 sweeps.

**Terminology used below.**
- **sweep:** one red pass plus one black pass.
- **V-cycle (cycle):** the sequence above. Your "iterations" are V-cycles.
- **probe:** the first 2 cycles.
- **batch:** the `additional_cycles` executed in one call.
- **`MAX_V_CYCLES`:** counts cycles across batches, but is only checked *between* batches.

**Coarse operators** are rediscretised (not Galerkin): every term is re-assembled with coarse h and restricted
velocities, k, ε and μ_eff. Volumes, areas and spacings are therefore consistently volume-integrated on every level.

---

## 2. Findings ledger

Classification key: **CD** confirmed defect · **LD** likely defect · **PH** plausible hypothesis · **VC** verified
correct · **UQ** unresolved question · **OBS** observation.

### F-01 · CD · Stopping tolerance is unreachable in float32
- **Location:** orchestrator.cpp:422, 425, 428, 443, 446 (`10e-6`); generic_multigrid.cpp:600-615, 640, 675, 707
  (absolute RMS comparison).
- **Evidence:**
  - U system: the x-iterate is bit-identical from cycle 3 on, with RMS 0.0535 against a unit-roundoff scale of 0.129
    (T-03). The production tolerance is 7.7e-5 of that scale (T-07).
  - Iterative refinement in double goes below the floor at once (T-05). The floor is therefore representational,
    not algorithmic.
- **Consequence:** the convergence test can never pass. Every solve ends on the cycle cap (or an uncapped batch, F-02).
  The floor scales with aP·|x|, so smaller dt (larger ρV/dt) raises it (U floor ≈ 0.44 at dt = 1/360, T-09b).
- **Explains:** the stagnation, "L2 unchanged after ~40,000 cycles", "dt barely changes L2".
- **Confidence:** very high.
- **Smallest test:** T-03/T-05, or replace the test with a relative criterion (‖r‖/‖r₀‖ or ‖r‖/‖b‖ ≤ 1e-5) and observe
  2–4-cycle solves.

### F-02 · CD · Adaptive controller ignores the cycle cap inside a batch
- **Location:** generic_multigrid.cpp:684-696 (`additional_cycles` executed at once), cap checked at 710 *after* the
  batch. Branches 1 and 2 (627-682) use `>` while the batch path uses `>=` (minor inconsistency).
- **Evidence:** with the real orchestrator at U=50, the U system printed `convergence factor = 0.999397` →
  `additional cycles = 14910`, and later `0.999986` → **`659608`** (T-09). `SYSTEM DIDN'T LINEARLY CONVERGE AFTER 50`
  is printed after those batches complete.
- **Worst case:** ⌈ln(tol/RMS)/ln(cf)⌉ with cf at most the largest float below 1 (ln cf ≈ −6e-8). That is up to
  ~5e8 V-cycles in a single call.
- **Secondary waste:** when the floor ratio is exactly 1.0 (T-03), branch 1 runs single cycles until total > 50. Every
  solve therefore spends ~47 cycles after reaching machine precision (T-07, T-08: 70 s per time step).
- **Explains:** "max 50 not enforced", the huge predicted counts, and their dt sensitivity (prediction = noise).
- **Confidence:** very high.
- **Smallest test:** cap with `std::min(additional_cycles, MAX_V_CYCLES - total_cycles_performed)` and re-run `orch.exe 50 0.1 1 1`.

### F-03 · CD (regime-dependent) · Face-centred residual restriction is half the consistent scale
- **Location:** multigrid_residual_generic_restriction.comp:105-157 (XFACE/YFACE/ZFACE sum 1×2×2 fine values, with
  weight 4). Cell-centred sums 2×2×2 (weight 8) and is consistent.
- **Evidence:**
  - Galerkin consistency for smooth e_c: R·A_f·P ≈ **0.495**·A_c (U), 0.987 (k), 0.985 (ε) (T-10).
  - Face diffusion problem: production factors rise 0.16 → 0.60 by cycle 14 at 32³, and to 0.73 at 64³ (grid-dependent).
    With ×2 they are 0.03–0.06, grid-independent (T-12).
  - Control: ×2 on the cell problem diverges (T-12c), confirming the cell scale is right.
  - Code: odd normal-direction faces (2X±1) are never seen by the coarse grid. The transverse sum is correct; the
    normal direction uses injection, not full weighting.
- **Consequence:** the coarse-grid correction removes only ~½ of smooth error for u, v, w. Impact is nil at
  dt = 1/60 (smoother-dominated, T-10). It is moderate at dt = 5–50 (T-13: 14-cycle RMS 0.31 → 0.20 at dt=5,
  2.45 → 1.36 at dt=50), and large for diffusion-dominated systems.
- **Explains:** the "0.048 → 0.3 → worse" pattern *when diffusion matters*. It does not explain the dt = 1/60 stall.
- **Confidence:** high.
- **Smallest test:** T-12 (`poisson1`).

### F-04 · CD · Nonlinear metric cannot converge while w ≈ 0
- **Location:** RANS.cpp:882-926 (`δ_i = ‖Δφ_i‖/‖φ_i‖` per field, max over fields); orchestrator.cpp:452-467.
- **Evidence (T-14):**
  - u, v, k, ε reach δ ≈ 1.3–2.6e-7 by iteration 7–8.
  - δ_w = 1.0, 8.1, 15.0, 2.7, 0.19, then 0.13–0.15 indefinitely.
  - max|w| = 0.165 → 0.034 → 0.0028 (relative to max|u| ≈ 110). w is produced by the partially converged pressure
    projection (2 V-cycles, warm start) on a z-invariant flow.
  - The production metric equals the CPU max exactly.
- **Explains:** "nonlinear check never satisfied", and the 100-iteration cap on every step (T-08).
- **Confidence:** very high.
- **Smallest test:** compute δ for the velocity as ‖Δ(u,v,w)‖/‖(u,v,w)‖ (one combined norm) and re-run `orchrep`.

### F-05 · CD (physics) / OBS (convergence) · Fine wall function disabled, coarse wall function active
- **Location:** RANS.cpp:513-515 (fine calls commented out) vs 711, 726, 741 (coarse calls active).
  `assemble_viscosity_LHS_momentum.comp` adds nothing on wall faces.
- **Evidence (T-15):** the fine wall term would add 2.6 on average (max 6.6) to diagonals of ~2e4, at most **3.0e-4**
  of aP. The coarse term is at most 1.5e-4 of aP.
- **Consequence:**
  - **Convergence:** negligible.
  - **Physics:** on the fine level the momentum walls currently have **zero wall shear** (slip walls). The operator
    used for the solution differs from the one multigrid's coarse levels assume.
- **Confidence:** high.
- **Smallest test:** T-15 (`wallmag`).

### F-06 · CD · Solve status is not observable
- **Location:** generic_multigrid.hpp:34 (`void multigrid_solve_system`); orchestrator.cpp:422-446 ignore the outcome.
  The prints do not name the system.
- **Consequence:**
  - Failed, stalled or exploded solves are indistinguishable from converged ones.
  - W's early return prints nothing.
  - The "AFTER 50" message is printed even after 659,608 cycles.
- **Smallest test:** return a struct `{converged, cycles, rms0, rms_final, reason}` and log it per system.

### F-07 · OBS · Tolerance literals
- `10e-6` is 1e-5, in both the linear and nonlinear checks. Confirm that this is intended.

### F-08 · CD (initial condition) · Initial velocity violates no-penetration; φ_old is saved before any projection
- **Location:** RANS.cpp:105-131 (u = −0.85(y−H/2) on all x-faces including x = 0, N; v likewise on the y-walls);
  orchestrator.cpp:406-410.
- **Evidence (T-15, T-14):**
  - |u| = 82.3 m/s on the x-walls in the saved old-time field.
  - Every first-step nonlinear iteration's projection changes u by **38.7 %** of ‖u_projected‖.
  - That is the same magnitude as the analytic projection of rigid rotation in a square: √(1 − J/I_p) ≈ 0.40 of the
    original norm, or ≈ 0.43 of the projected norm (J = torsion constant 0.1406a⁴, I_p = a⁴/6).
  - Exact agreement is not expected, because the momentum field already has its wall-normal faces zeroed. So the
    projection behaves plausibly, and the initial field is not admissible.
- **Consequence:**
  - **Step 1:** effectively a projection of the initial condition.
  - **Assembly:** the first advection operator sees non-zero wall-normal fluxes, which is a candidate cause of the
    non-dominant rows (H-3).
  - **Solver:** not a linear-solver defect.
- **Smallest test:** project (or zero the wall-normal faces of) the initial field before `save_previous_time_step_field`,
  then re-run `ops dt=50`.

### F-09 · CD (minor) · K coarse operators lack the implicit destruction diagonal
- **Location:** fine K adds ρ(ε/k)V on aP (k_TurbulenceModel.cpp:135, kernel line 74). The K hierarchy
  (k_TurbulenceModel.cpp:180-209) does not.
- **Magnitude:** ε/k·dt ≈ 0.1 × 1/60 ≈ 0.2 % of the diagonal here. Galerkin check for k: 0.987 (T-10).
- **Consequence:** negligible now; it grows with dt·ε/k.
- **Related:** the kernel divides by k[id] without a guard.

### F-10 · VC · Residual and L2/RMS reduction
- `calculate_l2_norm` equals the CPU double residual L2 to 6–7 digits on U, V, W, k, ε and on a random field (T-01).
- The element-wise residual differs from the CPU only by round-off (max 0.36 vs a unit-roundoff scale of 0.12).
- No coefficient mass sits on out-of-range neighbours.
- **Conventions verified:**
  - **Sign:** r = b − Ax.
  - **Buffers:** b = `RHS_main`, x = `field.data`, r = `levels_residual[0]`.
  - **RMS:** RMS = L2/√(all DOFs, including Dirichlet rows). This is an absolute, dimensional quantity, not relative.
  - **Reduction chain:** the `divisor = 8` fix is in place.

### F-11 · VC · Convergence-factor formula
- cf = (RMS_now/RMS_prev)^(1/n), with n = cycles in the last batch. Both measurements come from equivalent points
  (after complete V-cycles).
- Hand check: (0.0874191 / 221,399)^(1/2) = 6.284e-4 vs printed 6.2837e-4 (T-07).
- **Caveat:** at the floor the ratio is noise, so cf is meaningless there (see F-02).

### F-12 · VC · V-cycle mathematics and transfers
- The replica V-cycle reproduces production **bit-for-bit** (T-12, T-13).
- The cell-centred diffusion problem converges at 0.0015/0.028/0.037/0.051 per cycle on 16³, 32³ and 64³
  (grid-independent textbook behaviour, T-11).
- **Verified correct:**
  - Corrections zeroed per cycle.
  - Prolongation adds trilinear values with constant extrapolation at boundaries.
  - Recursion levels: 32 → 16 → 8.
  - Coarse dimensions from `GetLevelDimension`.

### F-13 · VC · W early exit is correct
- In the momentum assembly of step 1, b_w = 0 and w = 0 exactly (z-invariant flow, w₀ = 0, cross-stress terms vanish).
  RMS = 0 ≤ tol at the first check (T-06).

### F-14 · VC · Assembled operators are well posed at dt = 1/60
- **All 5 systems, all levels:** aP > 0, no non-finite values, no positive off-diagonals.
- **Diagonal dominance:** min aP/Σ|a_nb| = 2.74 (U, V, W fine), 4.69 (k), 4.70 (ε); coarse levels 5–21 (T-02).
- **Symmetry:** operators are nonsymmetric (upwind), as intended. RBGS is a valid smoother for them.

### F-15 · LD (latent) · First residual restriction sets uniforms through another program's locations
- **Location:** generic_multigrid.cpp:217-225. `multigrid_residual_generic_restriction_program` is bound, but
  `generic_restriction_*_loc` (the **physical-field** program's locations) are used.
- **Evidence:** the locations coincide on this driver (fineDimensions 2, coarseDimensions 0, topology 1 for both, T-00).
  It works by accident and could silently break on another driver or after a kernel edit.

### F-16 · PH · Large-dt U operator has non-dominant rows
- At dt = 5 and 50: 992 / 31,744 rows have aP < Σ|a_nb| (min 0.51). Factors are 0.4–0.6 per cycle (T-13).
- **Hypothesis:** net outflow ΣF ≠ 0 in wall-adjacent CVs, from F-08's non-zero wall-normal velocities.
- **Status:** not yet tested.

### F-17 · UQ · Missing `GL_BUFFER_UPDATE_BARRIER_BIT` before buffer copies and readbacks
- **Where:** `LinearSystem::save_previous_time_step_field` (copy after shader writes) and `calculate_l2_norm`
  (`glGetBufferSubData` after `GL_SHADER_STORAGE_BARRIER_BIT` only).
- **Status:** no corruption observed; every readback matched CPU recomputation. This is a spec-level portability risk,
  not a cause of the symptoms.

### F-18 · OBS · Large reported RMS values are absolute, input-scaled residuals
- **Initial RMS** at U=10: U 2.2e5, k 965, ε 461. **Floors:** U 0.054, k 3.5e-4, ε 5.5e-5. At U=50 the k floor is about 0.055.
- RMS_k ≈ 2300 and RMS_ε ≈ 11000 are consistent with initial residuals for larger k-ε inputs. They are not evidence of
  instability.
- The k and ε solves also reach their floor in 1–4 cycles (T-05, T-13, T-08).
- Their "deteriorating" factors (0.89–0.99 at U=50) are floor noise feeding F-02.

### F-19 · OBS · Repeated work
- `restrict_vels` / `restrict_mu_eff` run three times per nonlinear iteration (momentum, K and ε hierarchies).
- Cost only; no state hazard found.

### F-20 · OBS · Loop bookkeeping
- `for (i < non_linear_iters)` with `non_linear_iters++` inside behaves as `while(true)` capped at 100.
- The printed "non linear iteration number" is i + 2 (off by one).
- `initialize_velocity_fields` is also called explicitly at orchestrator.cpp:388 (harmless).

### Symptom → finding map

| Reported symptom | Accounted for by |
|---|---|
| U/V factor 0.048 → 0.3 → 0.997, then stagnation | F-01 (floor), F-11 (ratio is noise at the floor); F-03 adds a 0.3–0.6 regime when diffusion matters |
| L2 unchanged after ~40,000 cycles | F-01 (bit-identical iterate), F-02 (batch of thousands) |
| Max 50 cycles not enforced | F-02 (cap checked only between batches; reproduced 14,910 and 659,608) |
| W exits early | F-13 (correct: x = 0 is exact) |
| k/ε RMS 2300 / 11000; factors deteriorate | F-18 (absolute, input-scaled RMS), F-01/F-02 (floor noise) |
| dt changes predicted count but not L2 | F-01 (floor ∝ ρV/dt·|x|), F-02 (prediction = noise) |
| Nonlinear check never reached / never satisfied | F-02 (time spent in linear batches); F-04 (w trap: always 100 iterations) |

---

## 3. Test log

All runs: code revision as in §0. Defaults are 32³, dt = 1/60, coarsest level 8, k-ε inputs U=10, I=0.05, L=1, unless noted.

| ID | Purpose | Setup | Result | Interpretation / remaining uncertainty |
|---|---|---|---|---|
| T-00 | D: uniform-location aliasing | `loc` | physical & residual restriction locations identical (2, 0, 1) | F-15 latent, not active on this driver |
| T-01 | A.2: production L2 vs double residual | `l2` (after 4 momentum cycles + projection + k-ε assembly) | ratio 1.000000 for u, v, w, k, ε and a random field; elementwise max diff ≈ round-off | Residual and reduction verified. Rules out "wrong diagnostic" for the residual |
| T-02 | C: operator health | `ops` | all levels finite, aP>0, off-diag ≤ 0, strictly dominant (min 2.74 fine momentum, 4.69 k) | Systems are easy at dt = 1/60. Rules out ill-conditioning as the cause |
| T-03 | A.1/A.3: per-cycle tracking, U | `cycu 25` | L2 4.07e7 → **9.90 after 1 cycle**, error 0.26 → **5.1e-8**; cycles 3–25 bit-identical, RMS 0.0535 (round-off scale 0.129) | Converges in 1 cycle, then float32 floor. Rules out a solver defect at this dt |
| T-04 | E.1: known solution, U | `knownu` | error 1.0 → **4.7e-8** after 1 cycle, then fixed point (RMS 0.0047) | Same conclusion with an exact x* |
| T-05 | Floor vs solver failure | `refineu`, `refinek` (double residual, GPU solves A e = r) | U RMS 0.0535 → 3.0e-9 → 9.9e-11 (rel 1.2e-16); k 3.5e-4 → 1.9e-11 → 6.6e-13 | Stagnation is representational (float32), not algorithmic |
| T-06 | E.5: zero field, W | `zero` | max\|b_w\| = 0, max\|w\| = 0, L2 = 0 on CPU and GPU | W early exit correct |
| T-07 | A.1: production controller, U, fresh system | `produ` | probe cf 6.2837e-4 (hand 6.284e-4), batch of 2, RMS 0.0874, then ratio 1.0 → single cycles to 51; 0.20 s | Controller wastes ~47 cycles per solve at the floor |
| T-08 | B: real orchestrator, 1 step, U=10 I=0.05 L=1 | `orch.exe 10 0.05 1 1` (log: logs/orch_U10.log) | 100 nonlinear iterations (cap), 69.6 s, max batch 7; momentum floor RMS ≈ 0.075, k 5e-4, ε 7e-5 | Every solve hits the floor; the nonlinear loop never converges (see T-14) |
| T-09 | B: reproduce reported symptom | `orch.exe 50 0.1 1 1` (logs/orch_U50.log) | U at floor RMS 0.08; cf 0.999397 → **14,910 cycles** (iteration 15); cf 0.999986 → **659,608 cycles** (iteration 30); killed | Reported symptom reproduced; F-02 confirmed at runtime |
| T-09b | dt sensitivity | `orch.exe 50 0.1 1 1 0.0027778`, killed at 150 s (logs/orch_U50_dt360.log) | momentum floor RMS ≈ 0.44 (6× higher); k/ε cf 0.89–0.994 → batches 85–1,837 | Floor scales with ρV/dt; predicted counts are noise |
| T-10 | A.4/A.5/E.3: smoothing vs CGC, Galerkin scale | `twogridu/k/e` | 6 RBGS sweeps alone: U 221,399 → 0.0547, k 965 → 3.5e-4, ε 461 → 5.5e-5; Galerkin scale U **0.495**, k 0.987, ε 0.985; CGC on the pre-smoothed residual 0.83 (U) | At dt = 1/60 the smoother does all the work; face restriction under-scaled |
| T-11 | E.4: cell diffusion problem, grid sweep | `poisson0 N=16/32/64 dt=1e4` (μ=0.5, Dirichlet half-cell walls, random x*) | factors 0.0015, 0.028, 0.037, 0.051 at every N; floor reached in 4–5 cycles | Cell-centred multigrid healthy and grid-independent |
| T-12 | E.4/E.7: face diffusion problem, ×1 vs ×2 restriction | `poisson1 N=32/64 dt=1e4` | production 0.004, 0.16, 0.30, 0.39 … 0.60 (32³), → 0.73 (64³); replica ×1 identical to production; **×2: 0.0013, 0.028, 0.046, 0.06** (both N) | F-03 is causal for slow face-topology convergence |
| T-12c | Control | `poisson0` with ×2 | diverges (factor → 1.87) | Cell scale is correct; ×2 is not a generic fix |
| T-13 | E.7: real U/k at large dt | `realcmpu dt=5`, `dt=50`, `realcmpk dt=50` | U: 992 non-dominant rows (min 0.51); 14-cycle RMS 0.314 (prod) vs 0.201 (×2) at dt=5, 2.45 vs 1.36 at dt=50; k: 0.004–0.01/cycle | Large-dt U convergence limited by non-dominant rows (H-3) and F-03 |
| T-14 | B.2: nonlinear loop replica | `orchrep` (production solvers, identical call order, CPU-recomputed δ) | δ_u,v ≈ 1.4e-7 and δ_k,ε ≈ 1.8–2.6e-7 by iteration 8; δ_w 0.13–0.15 forever; prod metric = CPU max; projection changes u by 38.7 % each iteration; div RMS after projection 1e-2 → 3e-6 | F-04 confirmed; projection magnitude consistent with the analytic rigid-rotation projection (0.40–0.43) |
| T-15 | C: wall-function magnitude, IC admissibility | `wallmag` (also U=50 I=0.1) | fine wall term would be ≤ 3.0e-4 of aP; coarse ≤ 1.5e-4; old-time u = 82.3 m/s on the x-walls | F-05 negligible for convergence; F-08 confirmed |

**Negative results, i.e. hypotheses ruled out:**
- Wrong residual/L2 (T-01).
- Wrong convergence-factor arithmetic (T-07).
- Ill-conditioned or indefinite operators at dt = 1/60 (T-02).
- Broken V-cycle or transfer operators in the cell topology (T-11).
- Buffer aliasing between systems (T-01/T-02, consistent per-system results).
- W failing to converge (T-06).
- Coarse-level wall-function inconsistency as a cause of stagnation (T-15).

---

## 4. Orchestrator audit (`Orchestrator::run_rans_k_eps`)

**Confirmed issues**

1. **Absolute, unreachable linear tolerance** (`10e-6` RMS) for all five systems (F-01).
2. **Solve outcomes ignored** (F-06). A stalled or diverged solve is indistinguishable from a converged one. The time
   step is accepted unconditionally, even after the 100-iteration "stalled" message.
3. **Nonlinear metric dominated by w noise** (F-04). The loop always runs 100 iterations.
4. **Initial condition saved as φⁿ before any projection** (F-08). It violates no-penetration (82 m/s wall-normal).
5. **Bookkeeping:** the loop acts as `while(true)` with a 100 cap; the printed iteration number is off by one (F-20).
6. **Logging:** linear-solver prints carry no system name; the nonlinear δ (and which field produced it) is never
   printed; NaN/Inf is not detected anywhere.

**Verified correct**
- **Object wiring:** each system has its own `Generic_multigrid` with its own level buffers. Every solve is called
  with the right system. Staggered dimensions come from `GetLevelDimension` everywhere.
- **Time level:** φⁿ is saved once per physical step (`save_previous_time_step_field` ×5). Assembly uses
  `previous_time_step_field`, so the old time level stays fixed across nonlinear iterations (verified: δ → 1e-7 with
  constant φⁿ). This closes my earlier review's issue #12.
- **Iterates:** `store_nonlinear_iterate` runs at the start of each iteration. `calculate_nonlinear_convergence` runs
  after the k/ε solves and the `mu_eff` update, so it includes the projection.
- **Order:** momentum (old μ_eff) → projection → k-ε (projected velocity) → μ_eff → check. This is a consistent
  segregated Picard order.
- **Coefficient refresh:** all coefficients are refreshed every nonlinear iteration (fine and coarse).
- **Pressure solve:** warm-started; the divergence RMS after projection falls 1e-2 → 1e-3 → 3e-6 across iterations
  (T-14).

**Possible issues (not shown to cause symptoms)**
- The momentum predictor has no −∇pⁿ (non-incremental projection). This is a design choice with O(dt) splitting
  error, and it fixes 2 pressure V-cycles per iteration.
- The k-ε turbulence pressure −⅔ρk is absent (expected to be absorbed into p later).
- Hierarchy restriction is repeated three times per iteration (F-19).

**Suggested diagnostics**
- Return and log `{system, converged, cycles, rms0, rms_final, reason}` per solve.
- Log the per-field δ and the field that set the max.
- Run a NaN/Inf scan on fields after each solve (one reduction).
- Report linear convergence relative to `rms0` and to the float32 floor estimate, √mean((2⁻²⁴·Σ|terms|)²).

---

## 5. Ranked hypotheses (remaining)

| Rank | Hypothesis | Evidence | Explanatory power | Cost to test |
|---|---|---|---|---|
| H-1 | Linear "stagnation" is the float32 floor; the solver is fine | **Confirmed** (T-03, T-04, T-05, T-07, T-09) | All linear symptoms at dt = 1/60 | — |
| H-2 | Nonlinear non-termination is the w normalisation | **Confirmed** (T-14) | Always-100-iterations behaviour | — |
| H-3 | Large-dt slow convergence comes from non-dominant rows caused by ΣF ≠ 0 (inadmissible initial field) plus F-03 | F-16 measured; ΣF link untested | Factors 0.4–0.6 at dt ≥ 5 | Low: project the IC, re-run `ops`/`realcmpu dt=50` |
| H-4 | Floor-level ratio noise (0.9994 vs exactly 1.0) depends on the system state (U=50 vs U=10) | Observed in both regimes | Only affects the size of uncapped batches | Irrelevant once F-01/F-02 are fixed |
| H-5 | Missing buffer-update barriers corrupt readbacks on other drivers | No corruption observed | None of the current symptoms | Medium: needs another driver |

---

## 6. Next-experiment plan (most informative first)

**X-1 · Relative tolerance plus a capped batch (test-only patch of `multigrid_solve_system`).**
- **Change:** stop when RMS ≤ max(τ_rel·RMS₀, c·floor) with τ_rel = 1e-6, and cap
  `additional_cycles ≤ MAX − total`.
- **Possible outcomes:**
  - (a) Solves take 1–4 cycles and a time step falls from ~70 s to a few seconds. This confirms F-01/F-02 as the whole
    linear story.
  - (b) Some system still needs > 10 cycles at dt = 1/60. That points to a real operator problem in that system:
    run `twogrid` on it.
- **Next:** X-2.

**X-2 · Combined-velocity nonlinear metric.**
- **Change:** δ_vel = ‖Δ(u,v,w)‖/‖(u,v,w)‖ alongside δ_k and δ_ε in `orchrep`.
- **Possible outcomes:**
  - (a) Convergence in ~6–8 iterations (expected from T-14). This confirms F-04.
  - (b) δ_vel stalls above 1e-5. The pressure projection's incomplete solve is then the limiter: raise the pressure
    cycles (`orchrep30` mode exists) and compare.

**X-3 · Face-topology restriction with consistent weight.**
- **Change:** in a copy of the restriction kernel, use full weighting in the normal direction (½, 1, ½ on faces
  2X−1, 2X, 2X+1 over the 2×2 transverse faces, total weight 8). Run `poisson1 N=32/64` and `realcmpu dt=5/50`.
- **Possible outcomes:**
  - (a) Factors ≈ 0.03–0.06 grid-independent, and the large-dt U improves. Adopt it.
  - (b) Only ×2 helps, not full weighting. Investigate the boundary rows (wall-normal Dirichlet faces at X = 0, Nc).

**X-4 · Admissible initial condition (tests H-3).**
- **Change:** project the initial velocity (or zero the wall-normal faces) before `save_previous_time_step_field`.
  Re-run `ops dt=50` and `realcmpu dt=50`.
- **Possible outcomes:**
  - (a) The non-dominant rows disappear and the factors improve. That confirms H-3.
  - (b) They persist. Then dump ΣF per row to locate their origin (advection kernel, wall-normal fluxes).

**X-5 · Re-enable the fine momentum wall function and repeat T-08 with X-1/X-2 applied.**
- **Expectation:** convergence unchanged (the term is ≤ 3e-4 of aP), but the physics becomes consistent between
  levels.
- **If convergence changes:** the wall rows deserve a dedicated `twogrid` run.

---

## Appendix: method notes

- **Floor estimate:** per row, s_i = |b_i| + Σ|a·x| over the stencil. The "unit-roundoff RMS scale" is
  √mean((2⁻²⁴ s_i)²). Observed floors are 0.2–0.5× this scale, consistent with float32 evaluation of
  b − Ax and of the GS update.
- **Reference solutions:** double-precision lexicographic Gauss-Seidel with the kernels' neighbour convention
  (out-of-range neighbour = centre value), solved to a relative residual ≤ 1.4e-15.
- **Orchestrator replica (`orchrep`):** calls the same functions in the same order as `run_rans_k_eps`. The only
  addition is CPU readbacks for the δ, projection and divergence diagnostics. Production prints are silenced during
  solves.
