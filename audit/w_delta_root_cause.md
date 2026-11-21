# Root cause: `DELTA OF 2` (W nonlinear delta) does not settle

Date: 2026-10-14 · Intel HD 630 / GL 4.3 · 64³, 200 m, dt = 1/60, U = 80, I = 0.05, L = 0.1 · all wall functions ON
(current sources). Probe: `audit/w_delta_probe.cpp`, which links the unmodified production code. No production file was
modified.

## Confirmed root cause

`RANS::calculate_nonlinear_convergence` (`Turbulence_models/RANS.cpp:907-933`) normalises each velocity component
**by its own norm**:

δ_w = ‖Δw‖ / ‖w‖

In this flow, w is a genuine but tiny secondary flow (‖w‖/‖U‖ ≈ 5×10⁻⁶). Its iteration-to-iteration change, however, is
set by float32 rounding of the u/v solution, which lives at the scale of ‖U‖. So δ_w has a floor:

δ_w,floor ≈ (½ · 2⁻²⁴ · ‖U‖) / ‖w‖ ≈ (0.5 × 6e-8) / 5e-6 ≈ 6e-3

The measured value is 5.1–6.0e-3. That is above the 1e-4 tolerance, so the loop runs to its 100-iteration cap. The
metric arithmetic itself is correct; the defect is the per-component normalisation of a small component.

## Evidence

**1. Data path: no stale data, aliasing or reduction defect.**
- `previous_iterate_w` is **bit-identical** to `w_system.field.data` at every `store_nonlinear_iterate`, iterations 1–20
  (`RANS.cpp:862-888`, size Nx·Ny·(Nz+1)).
- W is not modified between the projection and the convergence check (bit-identical).
- Production δ₂ equals a CPU double recomputation to 5 significant digits in every iteration
  (e.g. 5.2674e-3 vs 5.2674e-3). That confirms `difference_sq_summation_L2.comp`, the reduction chain, slots 2/7 and the
  readback.
- The `projectionZ.comp` update replayed on the CPU from the production pressure matches to ≤ 5e-8.

**2. Every write to W, in order, for one nonlinear iteration** (`orchestrator.cpp:424-459`):

| # | Writer | Effect |
|---|---|---|
| 1 | `initialize_velocity_fields` (`RANS.cpp:140-163`) | w = 0 exactly |
| 2 | initial `divergence_cal` → `Multigrid_pressure_solve` (2 V-cycles, cold start) → `project_pressure` (`orchestrator.cpp:406-408`) | the divergence is exactly z-invariant (column range 0), but the 2-cycle pressure has a z-column range of 218 and a z-checkerboard of 9.4 (RBGS colouring by x+y+z) → max\|w\| = **0.206** |
| 3 | `save_previous_time_step_field` | wⁿ = that w (max 0.206) |
| 4 | W multigrid solve | b_w = ρV/dt·wⁿ ≈ 460, so w ≈ 0.18 is re-created every iteration |
| 5 | `project_pressure` (`projectionZ.comp:73-81`) | w → max 5.88e-3; ‖w‖ = 0.127 |

**3. w is physical, not an artefact.**
- The EXACT (double-precision CG) pressure projection of the same divergence gives max\|w\| = 5.883e-3, matching
  production at 5.883e-3.
- With 30 pressure V-cycles (one variable changed), the initial seed falls to 6.8e-5, yet the projected w is still
  5.4e-3, again matched by the exact projection.
- The momentum wall functions slow u near the z-walls: u(z=0) = −0.96694 vs u(mid) = −0.96737. That drives the
  secondary circulation expected for a rotating flow over stationary end walls.
- w is **z-antisymmetric**: antisymmetric energy 1.27e-2 vs symmetric 1.6e-7.

**4. The remaining ΔW is float32 noise inherited from u/v** (30 pressure cycles, iterations 7–20):

| Quantity | Value |
|---|---|
| ‖ΔU‖ | plateaus at 1.25–1.32e-3, ≈ 0.95 × 2⁻²⁴‖U‖ (1.37e-3): u is at its float32 floor |
| ‖ΔW‖ | plateaus at 6.2–6.8e-4 with no geometric decay; ‖ΔW‖/‖ΔU‖ = 0.50–0.52 in every iteration |
| ‖ΔW‖/‖U‖ | 2.7–2.9e-8 |
| δ₂ = ‖ΔW‖/‖W‖ | 5.5–6.0e-3 |

The same floor appears with production's 2 pressure cycles: δ₂ = 5.27e-3, 5.14e-3, 5.08e-3 at iterations 8–10.

## Ruled out

| Hypothesis | Evidence against |
|---|---|
| Stale or aliased `previous_iterate_w` | bit-identical every iteration |
| Reduction, indexing, slot or readback bug | GPU = CPU to 5 digits |
| W changed after the projection | bit-identical |
| `projectionZ` kernel error | CPU replay matches to 5e-8 |
| Incomplete pressure solve as the cause of the floor | 30 V-cycles → same floor; exact CG projection reproduces w |
| Coarse-grid / boundary treatment of the W system | the W solve output is ≈ wⁿ every iteration; the change happens in the projection and is bounded by u/v rounding |

The 2-cycle initial projection does seed a larger wⁿ (0.206 vs 6.8e-5), but it doesn't change the floor.

## Smallest justified fix (not applied)

Normalise the velocity change as one vector field; keep k and ε separate. In `RANS::calculate_nonlinear_convergence`,
replace the per-component loop over slots 0–2:

```cpp
// velocity: one relative change for (u, v, w) together
double num_vel = std::max((double)sums[0] + sums[1] + sums[2], 0.0);
double den_vel = std::max((double)sums[5] + sums[6] + sums[7], denominator_floor);
double max_delta = std::sqrt(num_vel / den_vel);

// k and eps: unchanged, per field
for (int i = 3; i < 5; i++)
{
    double numerator   = std::max((double)sums[i], 0.0);
    double denominator = std::max((double)sums[5 + i], denominator_floor);
    max_delta = std::max(max_delta, std::sqrt(numerator / denominator));
}
```

No new reductions or kernels are needed; the 10 existing slots are reused.

## Regression test

`audit/w_delta_probe.cpp` computes the proposed metric from the **same production reduction slots** (the "proposed
metric" line). Run it with the production configuration (2 pressure V-cycles):

| iteration | δ_vel | δ_k | δ_ε | proposed max | production max (incl. δ₂) |
|---|---|---|---|---|---|
| 4 | 6.0e-6 | 7.2e-5 | 1.3e-4 | 1.3e-4 (fails 1e-4) | 8.3e-2 |
| 5 | 3.3e-7 | 5.8e-6 | 1.0e-5 | **1.0e-5 (passes)** | 8.2e-3 |
| 10 | 5.9e-8 | 5.9e-8 | 6.9e-8 | 6.9e-8 | 5.1e-3 |

**Pass criteria for the regression test:**
1. The proposed max drops below 1e-4 within 6 iterations and settles at ≤ 1e-6, which is the float32 floor.
2. Production δ₂ keeps equalling the CPU recomputation, so the reduction path stays verified.

**Optional, separate:** use more pressure V-cycles for the *initial* projection only (`orchestrator.cpp:407`). It
reduces the spurious wⁿ seed (0.206 → 6.8e-5). It is not needed for convergence.
