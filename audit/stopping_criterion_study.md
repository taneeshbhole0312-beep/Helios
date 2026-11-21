# Stopping-criterion study for `Generic_multigrid`

Date: 2026-10-11 · Same hardware, build and code revision as `RANS_multigrid_audit.md` §0. No production code was modified.
The harness is `audit/audit_harness.cpp`, modes `crit<sys><scenario>` (scenario `f` fresh, `k` known solution, `w` warm
start), `critpoisson0/1`, `critzero`, `critref<sys>`.

## Criteria compared

All are computed in double on the CPU from the float32 iterate stored on the GPU. Here r = b − Ax, and
s_i = |b_i| + Σ_j |a_ij x_j| is the row magnitude.

| Name | Definition | Notes |
|---|---|---|
| RMS | ‖r‖₂/√n | current production criterion (absolute) |
| r/‖b‖ | ‖r‖₂/‖b‖₂ | classic relative residual |
| r/r₀ | ‖r‖₂/‖r₀‖₂ | residual-reduction criterion |
| **η_F** (proposed) | ‖r‖₂ / (‖A‖_F‖x‖₂ + ‖b‖₂) | normwise backward error, Frobenius norm |
| η_S | ‖r‖₂ / ‖ \|A\|\|x\| + \|b\| ‖₂ = ‖r‖₂/‖s‖₂ | row-scaled normwise backward error |
| ω | maxᵢ \|rᵢ\| / sᵢ | Oettli–Prager componentwise backward error |

"gpu" columns use the production float32 residual buffer (`levels_residual[0]`) instead of the double residual.
Forward error ‖x − x*‖/‖x*‖ uses a double-precision reference x* (relative residual ≤ 1e-14).

## Results

### Floor values: where each criterion settles once x is bit-identical (float32 stagnation)

| Case | RMS | r/‖b‖ | r/r₀ | η_F (cpu / gpu) | η_S | ω | fwd err |
|---|---|---|---|---|---|---|---|
| U fresh, dt 1/60 (T-03 setup) | 4.9e-2 | 5.7e-8 | 2.2e-7 | 2.6e-10 / 4.2e-10 | **2.3e-8** | 6.1e-8 | 4.6e-8 |
| U fresh, dt 1/360 | 2.5e-1 | 4.9e-8 | 1.9e-7 | 2.6e-10 / 4.2e-10 | **2.3e-8** | 7.0e-8 | 4.7e-8 |
| U known solution, x₀ = 0 (T-04) | 4.7e-3 | 5.4e-8 | 5.4e-8 | 2.4e-10 / 3.8e-10 | **2.2e-8** | 8.5e-8 | 4.5e-8 |
| U warm start x₀ = fl(x*) | 4.9e-2 | 5.7e-8 | **1.80 (never < 1)** | 2.6e-10 / 4.2e-10 | **2.3e-8** | 6.1e-8 | 4.6e-8 |
| k fresh (U=10) | 3.5e-4 | 5.6e-8 | 3.6e-7 | **1.4e-11** / 2.0e-11 | **2.3e-8** | 8.9e-8 | 2.3e-9 |
| ε fresh (U=10) | 5.5e-5 | 8.6e-8 | 1.2e-7 | **3.5e-12** / 4.6e-12 | **2.6e-8** | 9.6e-8 | 5.8e-10 |
| ε fresh (U=50, I=0.1) | 3.4e-2 | 5.5e-8 | 1.9e-6 | 2.8e-10 / 4.4e-10 | **2.4e-8** | 7.5e-8 | 4.7e-8 |
| cell diffusion, dt 1e4 | 3.7e-7 | 8.7e-6 | 8.7e-6 | 2.8e-10 / 3.2e-10 | **2.8e-8** | 1.5e-7 | 2e-7 |
| W zero system (b = 0, x = 0) | 0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | — |

How far ‖A‖_F‖x‖ overestimates ‖Ax‖ (this sets η_F's floor):

| Case | ratio | √n |
|---|---|---|
| U | 224 | 184 |
| U, dt 1/360 | 190 | — |
| U, dt 50 | 4,060 | — |
| k (U=10) | 4,048 | — |
| ε (U=10) | 24,317 | — |
| ε (U=50) | 197 | — |
| diffusion problems | 31,000 | — |

### Iterative refinement (T-05 revisited, x kept in double)

Every criterion falls together:

| System | η_F | η_S | ω | fwd err |
|---|---|---|---|---|
| U | 2.6e-10 → 5.4e-19 | 2.3e-8 → 4.8e-17 | 6.1e-8 → 2.2e-16 | 4.6e-8 → 1.5e-16 |
| k | 1.4e-11 → 2.7e-20 | 2.3e-8 → 4.4e-17 | 8.9e-8 → 2.1e-16 | 2.3e-9 → 7.9e-18 |

So each criterion reflects real solution quality and is not an artefact. Only η_S and ω land on the same value in units
of machine precision (≈ 0.4–1.6 u, with u = 2⁻²⁴ ≈ 6e-8) for every system.

### Slow-converging case: face diffusion problem, dt 1e4, factor ≈ 0.64 per cycle (F-03)

The question is how wrong the solution still is when each criterion first passes a given threshold:

| Threshold first met | cycle | forward error |
|---|---|---|
| η_F ≤ 1e-4 | 1 | **64 %** |
| η_F ≤ 1e-6 | 9 | **1.8 %** |
| η_F ≤ 1e-7 | 14 | 0.20 % |
| η_S ≤ 1e-4 | 9 | 1.8 % |
| η_S ≤ 1e-6 | 19 | 0.022 % |
| r/‖b‖ ≤ 1e-4 | 22 | 0.0057 % |

### Large dt (U, dt = 50): advection-dominated, b small relative to |A||x|
- At cycle 0, r/‖b‖ = 158 even though the forward error is 40 (x₀ is far off), because ‖b‖ ≪ ‖s‖ (‖s‖/‖b‖ ≈ 40).
  The r/‖b‖ floor in this regime is therefore ≈ 40 × 2.5e-8 ≈ 1e-6, so any r/‖b‖ tolerance below 1e-6 is unreachable
  again.
- At cycle 8, η_F = 7.1e-5 while the forward error is still 15 %.

## Assessment of the proposed η_F

**Strengths (confirmed)**
- Scale-invariant: rescaling A and b does not change it. It is insensitive to dt; the U floor is identical at
  dt = 1/60 and 1/360.
- Reachable in float32: it never asks for less than round-off, so it fixes F-01's unreachable-tolerance problem.
- Well defined for a warm start (unlike r/r₀). It drops to ~1e-17–1e-20 under iterative refinement, so it tracks real
  improvement.

**Weaknesses (confirmed)**
1. **No system-independent threshold.** Its float32 floor ranges from 3.5e-12 (ε, U=10) through 1.4e-11 (k) to
   2.6e-10 (U), a 75× spread that also moves with the k-ε inputs (ε: 3.5e-12 at U=10 vs 2.8e-10 at U=50). The cause
   is that ‖A‖_F‖x‖₂ overestimates ‖Ax‖ by 190–31,000×:
   - √n alone contributes ~180×, because the Frobenius norm sums all rows.
   - In k/ε, the wall-pinned rows (aP = 1) carry the large x values while the interior rows carry the large aP, and the
     product of the two norms pairs them falsely.
   - Smooth solutions of diffusion operators give |Ax| ≪ |A||x|.
2. **Lenient enough to stop with large forward error.** η_F = 8.4e-5 with a 64 % forward error; η_F = 1e-6 with 1.8 %
   (diffusion case); η_F = 7.1e-5 with 15 % (U, dt = 50). A threshold tight enough for the slow case (≈ 1e-8 or
   below) sits only ~40× above U's floor, but ~3,000× above ε's, so it is either too strict or too loose depending on
   the system. In practice the threshold has to be tuned per system and per regime.
3. **GPU evaluation inflates it.** The float32 residual reads ~1.6× higher than the double residual (also true for the
   other criteria).
4. **The zero system is 0/0.** It needs an explicit guard (b = 0 and x = 0 means converged).
5. **Cost:** ‖A‖_F needs one extra 7-array reduction per assembly, and ‖x‖ one reduction per check. Small, but not free.

**Conclusion:** η_F is a correct backward error and much better than the absolute RMS. Its normwise ‖A‖_F‖x‖
denominator is too pessimistic for these FV systems to support one threshold, so used alone it either stalls (threshold
too low for some system) or stops early (threshold too high for the slow regimes).

## Comparison summary

| Criterion | Reachable in float32 | System/dt-independent threshold | Warm start | b = 0 | Detects early stop in slow case | GPU cost |
|---|---|---|---|---|---|---|
| Absolute RMS (production) | ✗ (floor spans 3.7e-7 … 0.25) | ✗ | ✓ | ✓ | n/a | existing |
| r/‖b‖ | ✓ at small dt; ✗ at large dt (floor ~1e-6) | ✗ | ✓ | ✗ (0/0) | good | +1 reduction/assembly |
| r/r₀ | ✓ from a cold start only | ✓ | **✗ (1.8, never < 1)** | ✗ | good | free |
| η_F (proposed) | ✓ | ✗ (floor 3.5e-12 … 2.8e-10) | ✓ | needs guard | **weak** (64 % / 1.8 % errors) | +1/assembly, +1/check |
| **η_S** | ✓ | **✓ (2.2–2.8e-8 in every case)** | ✓ | needs guard | moderate | +1 output in residual kernel, +1/check |
| ω (componentwise) | ✓ | ✓ (6e-8 … 1.5e-7) | ✓ | needs guard | good, but noisy (one row reached 2.4e-6 while η_S was at floor) | max-reduction |

## Recommendation

Use a **two-sided criterion with explicit stagnation handling**:

```
s_i = |b_i| + Σ_j |a_ij x_j|                  (computed in the existing residual kernel, second output)
η_S = ||r||_2 / ||s||_2

converged if  ||r||_2 <= τ_rel · ||r_0||_2               (accuracy the outer Picard loop actually needs)
          or  η_S     <= c_fl · u,   u = 2^-24, c_fl ≈ 4  (≈ 2.4e-7: "as accurate as float32 allows")
          or  ||s||_2 == 0                               (b = 0, x = 0: exact)

stagnated if  ratio over the last cycle > 0.9  and  η_S <= 1e-5   -> stop, report STAGNATED (not FAILED)
hard limit:   never schedule more cycles than (MAX_V_CYCLES - cycles_done); return a status struct
```

**Why this combination**
- **η_S floor:** η_S has the same float32 floor (2.2–2.8e-8, about 0.4u, or ~0.7u with the GPU residual) for U, k and
  ε, across dt = 1/360 to 50, U = 10/50 inputs, cold, warm and known-solution starts, and both diffusion problems.
  `c_fl ≈ 4` gives a ~3–5× margin above the GPU-evaluated floor with no per-system tuning. It is the quantity the
  float32 rounding of b − Ax is actually bounded by (γ_k·(|b| + |A||x|)), so the threshold has a theoretical meaning,
  unlike the Frobenius bound.
- **τ_rel term:** it covers the slow and ill-conditioned regime, where any backward error (η_S included) can be small
  while the forward error is not (η_S = 1e-4 with 1.8 % error in the diffusion case). For a Picard outer loop,
  τ_rel = 1e-3 … 1e-4 is usually enough. Choose 1e-6 if the linear solve must be tight.
- **Warm starts:** when the initial guess is already at the floor (later nonlinear iterations), r/r₀ can never pass,
  but the η_S branch stops after one cycle.
- **Stagnation rule and cap:** these replace the current predictor/batch logic (F-02). A stagnated solve at
  η_S ≈ 1e-7 is a success in float32 and should be reported as such.

If you prefer to keep the proposed η_F, use it only with a per-system threshold calibrated from its measured floor
(≈ 10× the floor values above) and pair it with τ_rel. That is strictly more tuning than η_S and buys no extra safety.

**Expected effect on the current symptoms (from T-03/T-07/T-08 data, not yet run):** U, V, k and ε at dt = 1/60 would
stop after 1–2 cycles (η_S drops from ~0.1 to 2.3e-8 in the first cycle) instead of ~51 or up to 659,608. W stops at
the first check via the ‖s‖ = 0 rule.

## Test log additions

| ID | Mode | Purpose | Key result |
|---|---|---|---|
| T-16 | `crituf` | criteria on T-03 setup | η_F floor 2.6e-10, η_S 2.3e-8, RMS 4.9e-2; all flat from cycle 2 |
| T-17 | `crituk` | criteria on T-04 known solution | identical floors; ω noisy at cycle 1 (2.4e-6) |
| T-18 | `crituw` | warm start at fl(x*) | r/r₀ rises to 1.80 and never passes; η_S/η_F unaffected |
| T-19 | `critkf`, `critef` | k, ε | η_F floors 1.4e-11 / 3.5e-12 (‖A‖_F‖x‖/‖Ax‖ = 4,048 / 24,317); η_S 2.3e-8 / 2.6e-8 |
| T-20 | `critef U=50 I=0.1` | input dependence | η_F floor moves to 2.8e-10 (ratio 197); η_S 2.4e-8 |
| T-21 | `crituf dt=0.0027778` | dt dependence | RMS floor 5× higher (0.25); η_F, η_S unchanged |
| T-22 | `crituf dt=50` | large dt | r/‖b‖ = 158 at start; η_F = 7.1e-5 with 15 % forward error at cycle 8 |
| T-23 | `critpoisson1/0 dt=1e4` | slow vs fast convergence | η_F = 1e-6 at 1.8 % error (face case); cell case reaches all floors in 5 cycles |
| T-24 | `critzero` | b = 0, x = 0 | all ratio criteria are 0/0; RMS = 0 |
| T-25 | `critrefu`, `critrefk` | T-05 revisited | all criteria fall to 1e-16–1e-20 under double refinement |
