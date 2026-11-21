// Helios RANS / Generic multigrid audit harness.
// Links the UNMODIFIED project sources. Test-only access to private members via #define private public.
// Every experiment re-builds the pipeline from scratch so experiments are independent.
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <chrono>
#include <map>
#include <memory>
#include <random>
#include <limits>
#define private public
#include "Turbulence_models/RANS.hpp"
#include "Turbulence_models/k_TurbulenceModels.hpp"
#include "Generic_multigrid/generic_multigrid.hpp"
#include "Incompressible_flow/incompressible_flow.hpp"
#undef private

// ------------------------------------------------------------------ configuration (main.cpp values)
static int   N        = 32;
static float DOMAIN   = 200.0f;
static float RHO      = 1.225f;
static float MU       = 1.8e-5f;
static float DT       = 1.0f / 60.0f;
static int   LAST     = 8;
static float E_WALL   = 9.8f;            // orchestrator passes 9.8
static float U_REF    = 10.0f;           // turbulence inputs are typed by the user at runtime: configurable here
static float T_INT    = 0.05f;
static float T_LEN    = 1.0f;

// ------------------------------------------------------------------ GL helpers
static std::vector<float> rd(GLuint b, size_t n)
{
    glMemoryBarrier(GL_ALL_BARRIER_BITS); glFinish();
    std::vector<float> v(n);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * sizeof(float), v.data());
    return v;
}
static void wr(GLuint b, const std::vector<float> &v)
{
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, v.size() * sizeof(float), v.data());
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}
static std::vector<double> todbl(const std::vector<float> &v) { return std::vector<double>(v.begin(), v.end()); }
static std::vector<float>  toflt(const std::vector<double> &v) { return std::vector<float>(v.begin(), v.end()); }

// ------------------------------------------------------------------ host copy of one stencil level
struct HostSys
{
    int nx = 0, ny = 0, nz = 0; size_t n = 0;
    std::vector<double> aP, aW, aE, aS, aN, aB, aT, b;
};
static HostSys read_op(const StencilOperator &op, int nx, int ny, int nz, GLuint rhs)
{
    HostSys h; h.nx = nx; h.ny = ny; h.nz = nz; h.n = (size_t)nx * ny * nz;
    h.aP = todbl(rd(op.aP, h.n)); h.aW = todbl(rd(op.aW, h.n)); h.aE = todbl(rd(op.aE, h.n));
    h.aS = todbl(rd(op.aS, h.n)); h.aN = todbl(rd(op.aN, h.n)); h.aB = todbl(rd(op.aB, h.n)); h.aT = todbl(rd(op.aT, h.n));
    h.b = rhs ? todbl(rd(rhs, h.n)) : std::vector<double>(h.n, 0.0);
    return h;
}
static HostSys read_fine(LinearSystem &L) { return read_op(L.coefficients, L.x_elements, L.y_elements, L.z_elements, L.RHS_main); }

// A*x with EXACTLY the kernel's neighbour convention (out-of-range neighbour -> centre value)
static void apply(const HostSys &A, const std::vector<double> &x, std::vector<double> &Ax, double *oob_coeff = nullptr, std::vector<double> *scale = nullptr)
{
    Ax.assign(A.n, 0.0); if (scale) scale->assign(A.n, 0.0);
    double oob = 0;
    const int nx = A.nx, ny = A.ny, nz = A.nz; const size_t ls = (size_t)nx * ny;
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int xx = 0; xx < nx; xx++)
    {
        size_t i = z * ls + (size_t)y * nx + xx; double c = x[i];
        double w = xx > 0 ? x[i - 1] : c, e = xx + 1 < nx ? x[i + 1] : c;
        double nn = y > 0 ? x[i - nx] : c, s = y + 1 < ny ? x[i + nx] : c;
        double t = z > 0 ? x[i - ls] : c, bb = z + 1 < nz ? x[i + ls] : c;
        if (xx == 0) oob += std::fabs(A.aW[i]); if (xx + 1 == nx) oob += std::fabs(A.aE[i]);
        if (y == 0) oob += std::fabs(A.aN[i]); if (y + 1 == ny) oob += std::fabs(A.aS[i]);
        if (z == 0) oob += std::fabs(A.aT[i]); if (z + 1 == nz) oob += std::fabs(A.aB[i]);
        Ax[i] = A.aP[i] * c + A.aW[i] * w + A.aE[i] * e + A.aS[i] * s + A.aN[i] * nn + A.aB[i] * bb + A.aT[i] * t;
        if (scale) (*scale)[i] = std::fabs(A.b[i]) + std::fabs(A.aP[i] * c) + std::fabs(A.aW[i] * w) + std::fabs(A.aE[i] * e) + std::fabs(A.aS[i] * s) + std::fabs(A.aN[i] * nn) + std::fabs(A.aB[i] * bb) + std::fabs(A.aT[i] * t);
    }
    if (oob_coeff) *oob_coeff = oob;
}
struct Res { double l2, rms, maxabs, floor_rms, bnorm; };
// double-precision residual of a host system; floor_rms = RMS of (unit roundoff * sum of |terms| of the row)
static Res residual(const HostSys &A, const std::vector<double> &x)
{
    std::vector<double> Ax, sc; apply(A, x, Ax, nullptr, &sc);
    double s2 = 0, mx = 0, f2 = 0, b2 = 0; const double u = std::ldexp(1.0, -24);
    for (size_t i = 0; i < A.n; i++) { double r = A.b[i] - Ax[i]; s2 += r * r; mx = std::max(mx, std::fabs(r)); f2 += (u * sc[i]) * (u * sc[i]); b2 += A.b[i] * A.b[i]; }
    return { std::sqrt(s2), std::sqrt(s2 / A.n), mx, std::sqrt(f2 / A.n), std::sqrt(b2) };
}
// double-precision Gauss-Seidel reference solve (lexicographic), same neighbour convention
static std::vector<double> cpu_solve(const HostSys &A, std::vector<double> x, double rel_tol, int maxit, int *its = nullptr, double *final_rel = nullptr)
{
    const int nx = A.nx, ny = A.ny, nz = A.nz; const size_t ls = (size_t)nx * ny;
    Res r0 = residual(A, x); double target = rel_tol * std::max(r0.bnorm, 1e-300);
    int it = 0; Res r = r0;
    for (; it < maxit; it++)
    {
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int xx = 0; xx < nx; xx++)
        {
            size_t i = z * ls + (size_t)y * nx + xx; double c = x[i];
            double w = xx > 0 ? x[i - 1] : c, e = xx + 1 < nx ? x[i + 1] : c;
            double nn = y > 0 ? x[i - nx] : c, s = y + 1 < ny ? x[i + nx] : c;
            double t = z > 0 ? x[i - ls] : c, bb = z + 1 < nz ? x[i + ls] : c;
            double nb = A.aW[i] * w + A.aE[i] * e + A.aS[i] * s + A.aN[i] * nn + A.aB[i] * bb + A.aT[i] * t;
            x[i] = (A.b[i] - nb) / A.aP[i];
        }
        if (it % 20 == 19) { r = residual(A, x); if (r.l2 <= target) { it++; break; } }
    }
    r = residual(A, x);
    if (its) *its = it; if (final_rel) *final_rel = r.l2 / std::max(r0.bnorm, 1e-300);
    return x;
}
static double norm2(const std::vector<double> &a) { double s = 0; for (double v : a) s += v * v; return std::sqrt(s); }
static double diffnorm(const std::vector<double> &a, const std::vector<double> &b) { double s = 0; for (size_t i = 0; i < a.size(); i++) s += (a[i] - b[i]) * (a[i] - b[i]); return std::sqrt(s); }

// operator diagnostics
static void op_report(const char *name, const HostSys &A)
{
    size_t nonfinite = 0, nonpos = 0, posoff = 0, notdd = 0, rows_nb = 0, dirichlet = 0;
    double minP = 1e300, maxP = -1e300, minratio = 1e300, maxoff = 0, maxb = 0, asym = 0;
    const int nx = A.nx, ny = A.ny, nz = A.nz; const size_t ls = (size_t)nx * ny;
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)
    {
        size_t i = z * ls + (size_t)y * nx + x;
        double nb[6] = { A.aW[i], A.aE[i], A.aS[i], A.aN[i], A.aB[i], A.aT[i] };
        double sabs = 0; for (double v : nb) { if (!std::isfinite(v)) nonfinite++; if (v > 0) posoff++; sabs += std::fabs(v); maxoff = std::max(maxoff, std::fabs(v)); }
        if (!std::isfinite(A.aP[i]) || !std::isfinite(A.b[i])) nonfinite++;
        if (A.aP[i] <= 0) nonpos++;
        minP = std::min(minP, A.aP[i]); maxP = std::max(maxP, A.aP[i]); maxb = std::max(maxb, std::fabs(A.b[i]));
        if (sabs == 0) dirichlet++; else { rows_nb++; double r = A.aP[i] / sabs; minratio = std::min(minratio, r); if (r < 1.0) notdd++; }
        // symmetry: aE(i) vs aW(i+1)
        if (x + 1 < nx) { double a = A.aE[i], b = A.aW[i + 1]; double m = std::max(std::fabs(a), std::fabs(b)); if (m > 0) asym = std::max(asym, std::fabs(a - b) / m); }
    }
    printf("  [%s] %dx%dx%d  aP in [%.4g, %.4g]  max|a_nb| %.4g  max|b| %.4g\n", name, nx, ny, nz, minP, maxP, maxoff, maxb);
    printf("        rows w/o neighbours (Dirichlet-like) %zu, aP<=0 %zu, non-finite %zu, positive off-diag %zu,\n", dirichlet, nonpos, nonfinite, posoff);
    printf("        min aP/sum|a_nb| = %.4f, rows NOT diagonally dominant %zu of %zu, max rel asym aE(i) vs aW(i+1) = %.3g\n", minratio, notdd, rows_nb, asym);
}

// ------------------------------------------------------------------ world (mirrors Orchestrator::run_rans_k_eps)
struct World
{
    ShaderManager &sm;
    RANS rans;
    k_TurbulenceModels km;
    k_TurbulenceModels::epsilon em;
    Generic_multigrid mgu, mgv, mgw;
    incompressible_flow inc;
    incompressible_flow::specialized_pressure_Multigrid ps;
    Generic_multigrid mgk, mge;
    World(ShaderManager &s)
    : sm(s), rans(N, N, N, LAST, s, DOMAIN, DOMAIN, DOMAIN, RHO, MU), km(rans, LAST, false), em(km, LAST),
      mgu(LAST, s, rans.u_system), mgv(LAST, s, rans.v_system), mgw(LAST, s, rans.w_system),
      inc(N, N, N, DOMAIN / N, DOMAIN / N, DOMAIN / N, s), ps(LAST, inc), mgk(LAST, s, km.KSystem), mge(LAST, s, em.eps_system) {}

    void init_and_build_momentum()
    {
        rans.initialize_velocity_fields();
        em.initialize_k_epsilon_RANS_fields(U_REF, T_INT, T_LEN, E_WALL);
        rans.u_system.save_previous_time_step_field(); rans.v_system.save_previous_time_step_field(); rans.w_system.save_previous_time_step_field();
        km.KSystem.save_previous_time_step_field(); em.eps_system.save_previous_time_step_field();
        rans.store_nonlinear_iterate(km, em);
        rans.construct_momentum_equations(DT, LAST, E_WALL);
        glFinish();
    }
    // momentum solved with a FIXED number of V-cycles (deviation from the orchestrator, documented) then projection + k-eps assembly
    void build_keps(int momentum_cycles)
    {
        init_and_build_momentum();
        mgu.Generic_multigrid_solve(LAST, momentum_cycles);
        mgv.Generic_multigrid_solve(LAST, momentum_cycles);
        mgw.Generic_multigrid_solve(LAST, momentum_cycles);
        inc.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        ps.Multigrid_pressure_solve(DT, RHO, 2);
        inc.project_pressure(DT, RHO, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        rans.construct_k_epsilon_system(km, em, DT, LAST, E_WALL);
        glFinish();
    }
    LinearSystem &sys(char c) { return c == 'u' ? rans.u_system : c == 'v' ? rans.v_system : c == 'w' ? rans.w_system : c == 'k' ? km.KSystem : em.eps_system; }
    Generic_multigrid &mg(char c) { return c == 'u' ? mgu : c == 'v' ? mgv : c == 'w' ? mgw : c == 'k' ? mgk : mge; }
};

// ------------------------------------------------------------------ GPU kernel wrappers identical to the production bindings
static void gpu_smooth(ShaderManager &sm, const StencilOperator &A, GLuint rhs, GLuint sol, int nx, int ny, int nz, int sweeps)
{
    glUseProgram(sm.multigrid_generic_smoother_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, A.aP); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, A.aW); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, A.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, A.aS); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, A.aN); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, A.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, A.aT); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, rhs); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, sol);
    glUniform1i(sm.Generic_smoother_cells_x_loc, nx); glUniform1i(sm.Generic_smoother_cells_y_loc, ny); glUniform1i(sm.Generic_smoother_cells_z_loc, nz);
    for (int s = 0; s < sweeps; s++) for (int c = 0; c < 2; c++)
    {
        glUniform1i(sm.Generic_smoother_color_loc, c);
        glDispatchCompute((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}
static void gpu_residual(ShaderManager &sm, const StencilOperator &A, GLuint rhs, GLuint sol, GLuint res, int nx, int ny, int nz)
{
    glUseProgram(sm.multigrid_generic_residual_calculation_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, A.aP); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, A.aW); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, A.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, A.aS); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, A.aN); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, A.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, A.aT); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, rhs); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, sol);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, res);
    glUniform1i(sm.generic_residual_current_cells_x_loc, nx); glUniform1i(sm.generic_residual_current_cells_y_loc, ny); glUniform1i(sm.generic_residual_current_cells_z_loc, nz);
    glDispatchCompute((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static void gpu_restrict_residual(ShaderManager &sm, GLuint fine, GLuint coarse, std::vector<int> fd, std::vector<int> cd, int topo)
{
    glUseProgram(sm.multigrid_residual_generic_restriction_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, fine); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coarse);
    glUniform3i(sm.residual_generic_restriction_fine_dimensions_loc, fd[0], fd[1], fd[2]);
    glUniform3i(sm.residual_generic_restriction_coarse_dimensions_loc, cd[0], cd[1], cd[2]);
    glUniform1i(sm.residual_generic_restriction_topology_loc, topo);
    glDispatchCompute((cd[0] + 7) / 8, (cd[1] + 7) / 8, (cd[2] + 7) / 8);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static void gpu_prolong(ShaderManager &sm, GLuint coarse, GLuint fine, std::vector<int> fd, std::vector<int> cd, int topo)
{
    glUseProgram(sm.multigrid_generic_prolongation_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coarse); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fine);
    glUniform3i(sm.generic_prolongation_fine_dimensions_loc, fd[0], fd[1], fd[2]);
    glUniform3i(sm.generic_prolongation_coarse_dimensions_loc, cd[0], cd[1], cd[2]);
    glUniform1i(sm.generic_prolongation_topology_loc, topo);
    glDispatchCompute((fd[0] + 7) / 8, (fd[1] + 7) / 8, (fd[2] + 7) / 8);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static GLuint newbuf(size_t n) { GLuint b; glGenBuffers(1, &b); glBindBuffer(GL_SHADER_STORAGE_BUFFER, b); std::vector<float> z(n, 0.f); glBufferData(GL_SHADER_STORAGE_BUFFER, n * 4, z.data(), GL_DYNAMIC_COPY); return b; }

// ================================================================== EXPERIMENTS

// D.1: uniform-location aliasing in the first residual restriction (production uses the PHYSICAL-field program's locations)
static void exp_locations(ShaderManager &sm)
{
    printf("\n=== D1: uniform locations (residual-restriction program vs physical-field restriction program) ===\n");
    printf("  physical : fineDimensions %d coarseDimensions %d topology %d\n", sm.generic_restriction_fine_dimensions_loc, sm.generic_restriction_coarse_dimensions_loc, sm.generic_restriction_topology_loc);
    printf("  residual : fineDimensions %d coarseDimensions %d topology %d\n", sm.residual_generic_restriction_fine_dimensions_loc, sm.residual_generic_restriction_coarse_dimensions_loc, sm.residual_generic_restriction_topology_loc);
}

// A.2: production L2 vs independent double residual, on assembled systems and on random fields
static void exp_l2(ShaderManager &sm)
{
    printf("\n=== A2/E2: production calculate_l2_norm() vs independent double-precision residual ===\n");
    World w(sm); w.build_keps(4);
    for (char c : std::string("uvwke"))
    {
        LinearSystem &L = w.sys(c); HostSys A = read_fine(L);
        auto x = todbl(rd(L.field.data, A.n));
        double oob; std::vector<double> Ax; apply(A, x, Ax, &oob);
        Res r = residual(A, x);
        float gpu = w.mg(c).calculate_l2_norm();
        // also check production residual buffer elementwise
        auto rg = rd(w.mg(c).levels_residual_SSBOs_arr[0], A.n); double maxdiff = 0;
        for (size_t i = 0; i < A.n; i++) maxdiff = std::max(maxdiff, std::fabs((double)rg[i] - (A.b[i] - Ax[i])));
        printf("  %c: n=%zu  GPU L2 %.6g | CPU(double) L2 %.6g  ratio %.6f | RMS gpu %.6g | max|r_gpu - r_cpu| %.3g (unit-roundoff RMS scale %.3g) | coeff mass on out-of-range neighbours %.3g\n",
               c, A.n, gpu, r.l2, gpu / r.l2, gpu / std::sqrt((double)A.n), maxdiff, r.floor_rms, oob);
    }
    // random field on the U system: reduction correctness independent of the solver state
    {
        LinearSystem &L = w.sys('u'); HostSys A = read_fine(L);
        std::mt19937 g(7); std::uniform_real_distribution<double> U(-1, 1);
        std::vector<double> x(A.n); for (auto &v : x) v = U(g);
        auto xf = toflt(x); wr(L.field.data, xf);
        Res r = residual(A, todbl(xf)); float gpu = w.mg('u').calculate_l2_norm();
        printf("  u random x: GPU L2 %.6g vs CPU %.6g ratio %.6f\n", gpu, r.l2, gpu / r.l2);
    }
}

// operator / RHS audit (Audit C)
static void exp_ops(ShaderManager &sm)
{
    printf("\n=== C: assembled operators (fine and coarse) ===\n");
    World w(sm); w.build_keps(4);
    for (char c : std::string("uvwke"))
    {
        LinearSystem &L = w.sys(c);
        char nm[64]; snprintf(nm, 64, "%c fine", c); op_report(nm, read_fine(L));
        for (size_t l = 0; l < L.multigrid_levels_coeffiecients_arr.size(); l++)
        {
            auto &op = L.multigrid_levels_coeffiecients_arr[l];
            snprintf(nm, 64, "%c coarse L%zu", c, l + 1); op_report(nm, read_op(op, op.level_x_elements, op.level_y_elements, op.level_z_elements, 0));
        }
    }
    // fine vs coarse wall-term consistency for U: aP of wall-adjacent vs interior rows, fine and level 1
    {
        LinearSystem &L = w.sys('u');
        HostSys F = read_fine(L); auto &op = L.multigrid_levels_coeffiecients_arr[0];
        HostSys C = read_op(op, op.level_x_elements, op.level_y_elements, op.level_z_elements, 0);
        auto row = [](const HostSys &A, int x, int y, int z) { return (size_t)z * A.nx * A.ny + (size_t)y * A.nx + x; };
        auto sumnb = [](const HostSys &A, size_t i) { return -(A.aW[i] + A.aE[i] + A.aS[i] + A.aN[i] + A.aB[i] + A.aT[i]); };
        int fx = N / 2, fy0 = 0, fyi = N / 4, fz = N / 2; int cx = N / 4, cyi = N / 8, cz = N / 4;
        size_t fi_w = row(F, fx, fy0, fz), fi_i = row(F, fx, fyi, fz), ci_w = row(C, cx, 0, cz), ci_i = row(C, cx, cyi, cz);
        printf("  U fine   wall row aP %.6g (sum a_nb %.6g)  interior row aP %.6g (sum a_nb %.6g)  -> wall excess over interior: %.6g\n", F.aP[fi_w], sumnb(F, fi_w), F.aP[fi_i], sumnb(F, fi_i), (F.aP[fi_w] - sumnb(F, fi_w)) - (F.aP[fi_i] - sumnb(F, fi_i)));
        printf("  U coarse wall row aP %.6g (sum a_nb %.6g)  interior row aP %.6g (sum a_nb %.6g)  -> wall excess over interior: %.6g\n", C.aP[ci_w], sumnb(C, ci_w), C.aP[ci_i], sumnb(C, ci_i), (C.aP[ci_w] - sumnb(C, ci_w)) - (C.aP[ci_i] - sumnb(C, ci_i)));
    }
}

// A.1 + A.3 + E.1: one V-cycle at a time, independent residual / factor / error
static void exp_cycles(ShaderManager &sm, char c, int ncyc, bool known_solution)
{
    printf("\n=== %s: system %c, %d single V-cycles, independent tracking ===\n", known_solution ? "E1 known solution" : "A1/A3 cycle tracking", c, ncyc);
    World w(sm);
    if (c == 'k' || c == 'e') w.build_keps(30); else w.init_and_build_momentum();
    LinearSystem &L = w.sys(c); Generic_multigrid &mg = w.mg(c);
    HostSys A = read_fine(L);
    std::vector<double> xstar;
    if (known_solution)
    {
        // x* = smooth + small random, b = A x* (double, rounded to float), x0 = 0
        std::mt19937 g(11); std::uniform_real_distribution<double> U(-1, 1);
        xstar.resize(A.n);
        for (int z = 0; z < A.nz; z++) for (int y = 0; y < A.ny; y++) for (int x = 0; x < A.nx; x++)
        {
            size_t i = (size_t)z * A.nx * A.ny + (size_t)y * A.nx + x;
            xstar[i] = 10.0 * std::sin(3.0 * x / A.nx) * std::cos(2.0 * y / A.ny) + 0.5 * z / A.nz + 0.1 * U(g);
        }
        // Dirichlet-like rows (no neighbours) keep x* consistent automatically through b = A x*
        std::vector<double> Ax; apply(A, xstar, Ax);
        A.b = Ax; wr(L.RHS_main, toflt(A.b)); A.b = todbl(rd(L.RHS_main, A.n));   // use the float-rounded b everywhere
        wr(L.field.data, std::vector<float>(A.n, 0.0f));
        xstar = cpu_solve(A, xstar, 1e-15, 200);                                  // exact solution of the float-rounded b
    }
    else
    {
        int its; double rel; auto x0 = todbl(rd(L.field.data, A.n));
        xstar = cpu_solve(A, x0, 1e-14, 20000, &its, &rel);
        printf("  reference x* (double GS): %d sweeps, final relative residual %.3e\n", its, rel);
    }
    auto x = todbl(rd(L.field.data, A.n));
    Res r0 = residual(A, x);
    float g0 = mg.calculate_l2_norm();
    printf("  |b| = %.6g  x*: max|x*| = %.6g\n", r0.bnorm, [&] { double m = 0; for (double v : xstar) m = std::max(m, std::fabs(v)); return m; }());
    printf("  cyc | GPU L2 (prod)  | CPU L2 (double) | ratio/cycle (CPU) | prod-style factor | ||x-x*||/||x*|| | unit-roundoff RMS scale | RMS (CPU)\n");
    printf("  %3d | %14.6g | %15.6g |                   |                   | %15.3e | %23.3g | %.4g\n", 0, g0, r0.l2, diffnorm(x, xstar) / std::max(norm2(xstar), 1e-300), r0.floor_rms, r0.rms);
    double prev = r0.l2;
    for (int k = 1; k <= ncyc; k++)
    {
        mg.Generic_multigrid_solve(LAST, 1);
        float g = mg.calculate_l2_norm();
        x = todbl(rd(L.field.data, A.n)); Res r = residual(A, x);
        printf("  %3d | %14.6g | %15.6g | %17.4f | %17.4f | %15.3e | %23.3g | %.4g\n", k, g, r.l2, r.l2 / prev, (double)g / (double)g0, diffnorm(x, xstar) / std::max(norm2(xstar), 1e-300), r.floor_rms, r.rms);
        prev = r.l2; g0 = g;
    }
}

// A.1: production controller on the real system, with wall-clock and the controller's own prints
static void exp_production(ShaderManager &sm, char c, float tol)
{
    printf("\n=== A1 production multigrid_solve_system(%d, %g) on system %c ===\n", LAST, tol, c);
    World w(sm);
    if (c == 'k' || c == 'e') w.build_keps(30); else w.init_and_build_momentum();
    LinearSystem &L = w.sys(c); HostSys A = read_fine(L);
    Res r0 = residual(A, todbl(rd(L.field.data, A.n)));
    printf("  initial CPU RMS %.6g (tolerance %g is %.2e of the unit-roundoff RMS scale %.3g)\n", r0.rms, tol, tol / r0.floor_rms, r0.floor_rms);
    fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    w.mg(c).multigrid_solve_system(LAST, tol);
    glFinish();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    Res r1 = residual(A, todbl(rd(L.field.data, A.n)));
    printf("  returned after %.2f s; final CPU RMS %.6g\n", secs, r1.rms);
}

// E: iterative refinement in double: is the stagnation a float32 floor or a solver failure?
static void exp_refine(ShaderManager &sm, char c, int inner_cycles, int outer)
{
    printf("\n=== E3b iterative refinement (double residual on CPU, GPU V-cycles solve A e = r), system %c ===\n", c);
    World w(sm);
    if (c == 'k' || c == 'e') w.build_keps(30); else w.init_and_build_momentum();
    LinearSystem &L = w.sys(c); Generic_multigrid &mg = w.mg(c); HostSys A = read_fine(L);
    auto b_orig = rd(L.RHS_main, A.n);
    // stage 1: plain GPU solve to its floor
    mg.Generic_multigrid_solve(LAST, 40);
    std::vector<double> x = todbl(rd(L.field.data, A.n));
    Res r = residual(A, x);
    printf("  after 40 plain V-cycles: CPU RMS %.6g  (unit-roundoff RMS scale %.3g)\n", r.rms, r.floor_rms);
    for (int o = 0; o < outer; o++)
    {
        std::vector<double> Ax; apply(A, x, Ax);
        std::vector<double> rr(A.n); for (size_t i = 0; i < A.n; i++) rr[i] = A.b[i] - Ax[i];
        wr(L.RHS_main, toflt(rr)); wr(L.field.data, std::vector<float>(A.n, 0.0f));
        mg.Generic_multigrid_solve(LAST, inner_cycles);
        auto e = rd(L.field.data, A.n); for (size_t i = 0; i < A.n; i++) x[i] += e[i];
        r = residual(A, x);
        printf("  refinement %d: CPU RMS %.6g  (relative to |b| RMS: %.3e)\n", o + 1, r.rms, r.rms / (r.bnorm / std::sqrt((double)A.n)));
    }
    wr(L.RHS_main, b_orig);
}

// A.4/A.5/E.3: smoothing alone, coarse-grid correction alone, Galerkin consistency
static void exp_twogrid(ShaderManager &sm, char c)
{
    printf("\n=== A4/A5/E3 two-grid analysis, system %c ===\n", c);
    World w(sm);
    if (c == 'k' || c == 'e') w.build_keps(30); else w.init_and_build_momentum();
    LinearSystem &L = w.sys(c); Generic_multigrid &mg = w.mg(c); HostSys A = read_fine(L);
    int topo = (int)L.field.topology;
    auto fd = L.GetLevelDimension(0), cd = L.GetLevelDimension(1);
    size_t nf = A.n, nc = (size_t)cd[0] * cd[1] * cd[2];
    auto &C1 = L.multigrid_levels_coeffiecients_arr[0];
    GLuint ec = newbuf(nc), ef = newbuf(nf), zf = newbuf(nf), zc = newbuf(nc), rf = newbuf(nf), rc = newbuf(nc), rc2 = newbuf(nc);

    // (1) Galerkin consistency: for SMOOTH coarse e_c compare R(A_f P e_c) with A_c e_c
    {
        std::vector<float> e(nc);
        for (int z = 0; z < cd[2]; z++) for (int y = 0; y < cd[1]; y++) for (int x = 0; x < cd[0]; x++)
            e[(size_t)z * cd[0] * cd[1] + (size_t)y * cd[0] + x] = (float)(std::sin(3.14159265358979 * (x + 0.5) / cd[0]) * std::sin(3.14159265358979 * (y + 0.5) / cd[1]) * std::sin(3.14159265358979 * (z + 0.5) / cd[2]));
        wr(ec, e); wr(ef, std::vector<float>(nf, 0.f));
        gpu_prolong(sm, ec, ef, fd, cd, topo);                                   // e_f = P e_c
        gpu_residual(sm, L.coefficients, zf, ef, rf, fd[0], fd[1], fd[2]);       // rf = -A_f e_f
        gpu_restrict_residual(sm, rf, rc, fd, cd, topo);                         // rc = -R A_f P e_c
        gpu_residual(sm, C1, zc, ec, rc2, cd[0], cd[1], cd[2]);                  // rc2 = -A_c e_c
        auto a = rd(rc, nc), b = rd(rc2, nc); double num = 0, den = 0, sa = 0, sb = 0; int cnt = 0;
        for (int z = 2; z < cd[2] - 2; z++) for (int y = 2; y < cd[1] - 2; y++) for (int x = 2; x < cd[0] - 2; x++)
        { size_t i = (size_t)z * cd[0] * cd[1] + (size_t)y * cd[0] + x; num += a[i] * b[i]; den += b[i] * b[i]; sa += a[i]; sb += b[i]; cnt++; }
        printf("  Galerkin check (interior coarse rows, smooth e_c): best-fit scale  R A_f P  ~=  %.4f x A_c   (sum ratio %.4f)\n", num / den, sa / sb);
        printf("      -> 1.0 means the restricted fine residual and the rediscretised coarse operator are consistently scaled\n");
    }

    // (2) smoothing-only and coarse-grid-correction-only factors from the SAME starting iterate
    auto x0 = rd(L.field.data, nf);
    auto cpu_rms = [&](const std::vector<float> &xf) { return residual(A, todbl(xf)).rms; };
    double R0 = cpu_rms(x0);
    // smoothing alone: 3 + 3 RBGS sweeps (what one V-cycle does on the finest level)
    gpu_smooth(sm, L.coefficients, L.RHS_main, L.field.data, fd[0], fd[1], fd[2], 6);
    double Rs = cpu_rms(rd(L.field.data, nf));
    printf("  start RMS %.6g | 6 fine RBGS sweeps only: RMS %.6g (factor %.4f)\n", R0, Rs, Rs / R0);
    // pre-smooth 3, then CGC with an (almost) exact coarse solve, without post-smoothing
    for (double scale : { 1.0, 2.0 })
    {
        wr(L.field.data, x0);
        gpu_smooth(sm, L.coefficients, L.RHS_main, L.field.data, fd[0], fd[1], fd[2], 3);
        double Rpre = cpu_rms(rd(L.field.data, nf));
        gpu_residual(sm, L.coefficients, L.RHS_main, L.field.data, rf, fd[0], fd[1], fd[2]);
        gpu_restrict_residual(sm, rf, rc, fd, cd, topo);
        if (scale != 1.0) { auto t = rd(rc, nc); for (auto &v : t) v *= (float)scale; wr(rc, t); }
        wr(ec, std::vector<float>(nc, 0.f));
        gpu_smooth(sm, C1, rc, ec, cd[0], cd[1], cd[2], 400);                    // near-exact coarse solve (level 1)
        gpu_residual(sm, C1, rc, ec, rc2, cd[0], cd[1], cd[2]);
        auto cr = rd(rc2, nc), cb = rd(rc, nc); double n1 = 0, n2 = 0; for (size_t i = 0; i < nc; i++) { n1 += cr[i] * cr[i]; n2 += cb[i] * cb[i]; }
        gpu_prolong(sm, ec, L.field.data, fd, cd, topo);
        double Rcgc = cpu_rms(rd(L.field.data, nf));
        gpu_smooth(sm, L.coefficients, L.RHS_main, L.field.data, fd[0], fd[1], fd[2], 3);
        double Rpost = cpu_rms(rd(L.field.data, nf));
        printf("  two-grid (restricted residual x%.0f): pre-smooth %.6g -> after CGC %.6g (CGC factor %.4f) -> post-smooth %.6g | two-grid factor %.4f | coarse solve rel.res %.1e\n",
               scale, Rpre, Rcgc, Rcgc / Rpre, Rpost, Rpost / R0, std::sqrt(n1 / std::max(n2, 1e-300)));
    }
    // production V-cycle from the same start
    wr(L.field.data, x0);
    mg.Generic_multigrid_solve(LAST, 1);
    double Rv = cpu_rms(rd(L.field.data, nf));
    printf("  one production V-cycle from the same start: RMS %.6g (factor %.4f)\n", Rv, Rv / R0);
}

// Test-only replica of Generic_multigrid_solve (one V-cycle) with ONE switch: the restricted residual of every level is
// multiplied by `rscale` (1.0 reproduces production exactly; verified against production below).
static void my_vcycle(ShaderManager &sm, LinearSystem &L, Generic_multigrid &mg, float rscale)
{
    int passes = (int)L.multigrid_levels_coeffiecients_arr.size();
    int topo = (int)L.field.topology;
    auto scale_buf = [&](GLuint b, size_t n) { if (rscale == 1.0f) return; auto t = rd(b, n); for (auto &v : t) v *= rscale; wr(b, t); };
    for (int l = 0; l <= passes; l++) { auto d = L.GetLevelDimension(l); wr(mg.levels_correction_SSBOs_arr[l], std::vector<float>((size_t)d[0] * d[1] * d[2], 0.f)); }
    auto d0 = L.GetLevelDimension(0);
    gpu_smooth(sm, L.coefficients, L.RHS_main, L.field.data, d0[0], d0[1], d0[2], 3);
    gpu_residual(sm, L.coefficients, L.RHS_main, L.field.data, mg.levels_residual_SSBOs_arr[0], d0[0], d0[1], d0[2]);
    { auto d1 = L.GetLevelDimension(1); gpu_restrict_residual(sm, mg.levels_residual_SSBOs_arr[0], mg.levels_RHS_SSBOs_arr[1], d0, d1, topo); scale_buf(mg.levels_RHS_SSBOs_arr[1], (size_t)d1[0] * d1[1] * d1[2]); }
    for (int i = 1; i < passes; i++)
    {
        auto d = L.GetLevelDimension(i), dc = L.GetLevelDimension(i + 1); auto &A = L.multigrid_levels_coeffiecients_arr[i - 1];
        gpu_smooth(sm, A, mg.levels_RHS_SSBOs_arr[i], mg.levels_correction_SSBOs_arr[i], d[0], d[1], d[2], 3);
        gpu_residual(sm, A, mg.levels_RHS_SSBOs_arr[i], mg.levels_correction_SSBOs_arr[i], mg.levels_residual_SSBOs_arr[i], d[0], d[1], d[2]);
        gpu_restrict_residual(sm, mg.levels_residual_SSBOs_arr[i], mg.levels_RHS_SSBOs_arr[i + 1], d, dc, topo); scale_buf(mg.levels_RHS_SSBOs_arr[i + 1], (size_t)dc[0] * dc[1] * dc[2]);
    }
    { auto d = L.GetLevelDimension(passes); gpu_smooth(sm, L.multigrid_levels_coeffiecients_arr[passes - 1], mg.levels_RHS_SSBOs_arr[passes], mg.levels_correction_SSBOs_arr[passes], d[0], d[1], d[2], 30); }
    for (int i = 0; i < passes - 1; i++)
    {
        int c = passes - i, f = passes - i - 1; auto df = L.GetLevelDimension(f), dc = L.GetLevelDimension(c);
        gpu_prolong(sm, mg.levels_correction_SSBOs_arr[c], mg.levels_correction_SSBOs_arr[f], df, dc, topo);
        gpu_smooth(sm, L.multigrid_levels_coeffiecients_arr[f - 1], mg.levels_RHS_SSBOs_arr[f], mg.levels_correction_SSBOs_arr[f], df[0], df[1], df[2], 3);
    }
    gpu_prolong(sm, mg.levels_correction_SSBOs_arr[1], L.field.data, d0, L.GetLevelDimension(1), topo);
    gpu_smooth(sm, L.coefficients, L.RHS_main, L.field.data, d0[0], d0[1], d0[2], 3);
}

// E.7: the REAL assembled system at the configured dt: production vs replica x1 vs replica x2, same start
static void exp_realcmp(ShaderManager &sm, char c)
{
    printf("\n=== E7 real system %c at dt=%g: production vs replica (x1) vs replica (restricted residual x2) ===\n", c, DT);
    World w(sm);
    if (c == 'k' || c == 'e') w.build_keps(30); else w.init_and_build_momentum();
    LinearSystem &L = w.sys(c); Generic_multigrid &mg = w.mg(c); HostSys A = read_fine(L);
    auto x0 = rd(L.field.data, A.n);
    op_report("fine", A);
    const int K = 14; std::vector<std::vector<double>> hist(3, std::vector<double>(K + 1));
    for (int v = 0; v < 3; v++)
    {
        wr(L.field.data, x0); hist[v][0] = residual(A, todbl(x0)).rms;
        for (int k = 1; k <= K; k++) { if (v == 0) mg.Generic_multigrid_solve(LAST, 1); else my_vcycle(sm, L, mg, v == 1 ? 1.0f : 2.0f); hist[v][k] = residual(A, todbl(rd(L.field.data, A.n))).rms; }
    }
    Res r = residual(A, todbl(rd(L.field.data, A.n)));
    printf("  cyc | production RMS (factor) | replica x1 RMS (factor) | replica x2 RMS (factor)\n");
    for (int k = 1; k <= K; k++) printf("  %3d | %.4e (%.4f)     | %.4e (%.4f)     | %.4e (%.4f)\n", k, hist[0][k], hist[0][k] / hist[0][k - 1], hist[1][k], hist[1][k] / hist[1][k - 1], hist[2][k], hist[2][k] / hist[2][k - 1]);
    printf("  start RMS %.4e, unit-roundoff RMS scale %.3g\n", hist[0][0], r.floor_rms);
}

// C: magnitude of the momentum wall-function diagonal that is active on coarse levels but commented out on the fine level,
// plus the initial-field divergence that the first projection has to remove
static void exp_wallmag(ShaderManager &sm)
{
    printf("\n=== C wall-function magnitude (fine call is commented out in construct_momentum_equations) ===\n");
    World w(sm); w.init_and_build_momentum();
    LinearSystem &L = w.sys('u'); HostSys F = read_fine(L);
    // fine: what the commented-out call WOULD add
    std::vector<float> keep = rd(L.coefficients.aP, F.n);
    wr(L.coefficients.aP, std::vector<float>(F.n, 0.f));
    w.rans.add_wall_function_contribution_momentum(L, E_WALL); glFinish();
    auto wf = rd(L.coefficients.aP, F.n); wr(L.coefficients.aP, keep);
    double mx = 0, mean = 0; int cnt = 0, rel_cnt = 0; double maxrel = 0;
    for (size_t i = 0; i < F.n; i++) if (wf[i] != 0) { mx = std::max(mx, (double)wf[i]); mean += wf[i]; cnt++; maxrel = std::max(maxrel, (double)wf[i] / F.aP[i]); }
    printf("  fine u: %d wall rows would get aP += [mean %.4g, max %.4g]; max (wall term / current aP) = %.3e\n", cnt, mean / std::max(cnt, 1), mx, maxrel);
    auto &op = L.multigrid_levels_coeffiecients_arr[0];
    size_t nc = (size_t)op.level_x_elements * op.level_y_elements * op.level_z_elements;
    auto cur = rd(op.aP, nc);
    w.rans.clear_coefficients(op);
    w.rans.add_wall_function_contribution_momentum(N / 2, N / 2, N / 2, op, L.field.topology, w.rans.u_vel_hierarcy[0], w.rans.v_vel_hierarcy[0], w.rans.w_vel_hierarcy[0], E_WALL); glFinish();
    auto wc = rd(op.aP, nc); mx = 0; mean = 0; cnt = 0; maxrel = 0;
    for (size_t i = 0; i < nc; i++) if (wc[i] != 0) { mx = std::max(mx, (double)wc[i]); mean += wc[i]; cnt++; maxrel = std::max(maxrel, (double)wc[i] / cur[i]); }
    printf("  coarse L1 u: %d wall rows have aP += [mean %.4g, max %.4g]; max (wall term / coarse aP) = %.3e\n", cnt, mean / std::max(cnt, 1), mx, maxrel);
    // initial divergence (time level saved by the orchestrator BEFORE any projection)
    w.inc.divergence_cal(w.rans.u_system.previous_time_step_field, w.rans.v_system.previous_time_step_field, w.rans.w_system.previous_time_step_field);
    auto dv = rd(w.inc.divergence, (size_t)N * N * N); double d2 = 0, dm = 0; for (float q : dv) { d2 += (double)q * q; dm = std::max(dm, (double)std::fabs(q)); }
    auto u0 = rd(w.rans.u_system.previous_time_step_field, F.n); double wallmax = 0;
    for (int z = 0; z < N; z++) for (int y = 0; y < N; y++) for (int x : {0, N}) wallmax = std::max(wallmax, (double)std::fabs(u0[(size_t)z * (N + 1) * N + (size_t)y * (N + 1) + x]));
    printf("  saved old-time u: max |u| on the x-walls (must be 0 for no-penetration) = %.4g ; divergence of the saved field: RMS %.4g, max %.4g\n", wallmax, std::sqrt(d2 / dv.size()), dm);
}

// E.4 / E.7: simplified diffusion (+ time) problem with consistent coefficients built on the CPU for every level
static void exp_poisson(ShaderManager &sm, int n, int topo_i, double dt, int ncyc)
{
    fieldTopology topo = (fieldTopology)topo_i;
    printf("\n=== E4 simplified diffusion problem: %d^3, topology %d, dt %g (no advection, no walls functions) ===\n", n, topo_i, dt);
    LinearSystem L(n, n, n, LAST, sm, topo);
    double h = DOMAIN / n, mu = 0.5;           // turbulent-like viscosity so diffusion is not negligible
    auto fill = [&](StencilOperator &op, int nx, int ny, int nz, double hh, bool fine)
    {
        size_t m = (size_t)nx * ny * nz; std::vector<float> P(m), W(m), Eo(m), S(m), No(m), B(m), T(m);
        double D = mu * hh * hh / hh, tc = RHO * hh * hh * hh / dt;
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)
        {
            size_t i = (size_t)z * nx * ny + (size_t)y * nx + x; double p = tc;
            bool dir = (topo == fieldTopology::XFACE_CENTERED && (x == 0 || x == nx - 1)) || (topo == fieldTopology::YFACE_CENTERED && (y == 0 || y == ny - 1)) || (topo == fieldTopology::ZFACE_CENTERED && (z == 0 || z == nz - 1));
            if (dir) { P[i] = (float)tc; continue; }
            auto face = [&](int has, float &a) { if (has) { a = (float)-D; p += D; } else p += 2 * D; };   // wall: half-cell Dirichlet 0
            face(x > 0, W[i]); face(x + 1 < nx, Eo[i]); face(y > 0, No[i]); face(y + 1 < ny, S[i]); face(z > 0, T[i]); face(z + 1 < nz, B[i]);
            P[i] = (float)p;
        }
        wr(op.aP, P); wr(op.aW, W); wr(op.aE, Eo); wr(op.aS, S); wr(op.aN, No); wr(op.aB, B); wr(op.aT, T);
    };
    fill(L.coefficients, L.x_elements, L.y_elements, L.z_elements, h, true);
    for (size_t l = 0; l < L.multigrid_levels_coeffiecients_arr.size(); l++) { auto &op = L.multigrid_levels_coeffiecients_arr[l]; fill(op, op.level_x_elements, op.level_y_elements, op.level_z_elements, h * (2 << l), false); }
    HostSys A = read_fine(L);
    std::mt19937 g(5); std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> xs(A.n); for (auto &v : xs) v = U(g);
    std::vector<double> Ax; apply(A, xs, Ax); wr(L.RHS_main, toflt(Ax)); A.b = todbl(rd(L.RHS_main, A.n));
    Generic_multigrid mg(LAST, sm, L);
    const int K = 14;
    std::vector<std::vector<double>> hist(3, std::vector<double>(K + 1));
    for (int variant = 0; variant < 3; variant++)
    {
        wr(L.field.data, std::vector<float>(A.n, 0.f));
        hist[variant][0] = residual(A, std::vector<double>(A.n, 0.0)).rms;
        for (int k = 1; k <= K; k++)
        {
            if (variant == 0) mg.Generic_multigrid_solve(LAST, 1);
            else my_vcycle(sm, L, mg, variant == 1 ? 1.0f : 2.0f);
            hist[variant][k] = residual(A, todbl(rd(L.field.data, A.n))).rms;
        }
    }
    Res r = residual(A, todbl(rd(L.field.data, A.n)));
    printf("  cyc | production RMS (factor) | replica x1 RMS (factor) | replica, restricted residual x2 RMS (factor)\n");
    for (int k = 1; k <= K; k++)
        printf("  %3d | %.4e (%.4f)     | %.4e (%.4f)     | %.4e (%.4f)\n", k, hist[0][k], hist[0][k] / hist[0][k - 1], hist[1][k], hist[1][k] / hist[1][k - 1], hist[2][k], hist[2][k] / hist[2][k - 1]);
    printf("  unit-roundoff RMS scale %.3g\n", r.floor_rms);
}

// E.5: zero-field W system
static void exp_zero(ShaderManager &sm)
{
    printf("\n=== E5 W system (x = 0 expected exact) ===\n");
    World w(sm); w.init_and_build_momentum();
    LinearSystem &L = w.sys('w'); HostSys A = read_fine(L);
    auto x = todbl(rd(L.field.data, A.n));
    double mb = 0, mx = 0; for (size_t i = 0; i < A.n; i++) { mb = std::max(mb, std::fabs(A.b[i])); mx = std::max(mx, std::fabs(x[i])); }
    Res r = residual(A, x); float g = w.mgw.calculate_l2_norm();
    printf("  max|b_w| = %.3g, max|w| = %.3g, CPU L2 %.3g, GPU L2 %.3g -> production returns at the first check: %s\n", mb, mx, r.l2, g, (g / std::sqrt((double)A.n) <= 1e-5) ? "yes" : "no");
    op_report("w fine", A);
}

// B.2: replica of Orchestrator::run_rans_k_eps nonlinear loop (same calls, same order), instrumented
static void exp_orchrep(ShaderManager &sm, int iters, bool fixed_pressure_cycles_many)
{
    printf("\n=== B2 orchestrator replica (1 time step, %d nonlinear iterations, production solvers%s) ===\n", iters, fixed_pressure_cycles_many ? ", pressure 30 V-cycles" : "");
    World w(sm);
    w.rans.initialize_velocity_fields();
    w.em.initialize_k_epsilon_RANS_fields(U_REF, T_INT, T_LEN, E_WALL);
    for (char c : std::string("uvwke")) w.sys(c).save_previous_time_step_field();
    auto sizes = [&](char c) { LinearSystem &L = w.sys(c); return (size_t)L.x_elements * L.y_elements * L.z_elements; };
    printf("  it |  delta_u    delta_v    delta_w    delta_k    delta_e  | prod max | proj |du| / |u| | div RMS after proj | k min      eps min\n");
    for (int it = 1; it <= iters; it++)
    {
        std::map<char, std::vector<float>> before; for (char c : std::string("uvwke")) before[c] = rd(w.sys(c).field.data, sizes(c));
        w.rans.store_nonlinear_iterate(w.km, w.em);
        w.rans.construct_momentum_equations(DT, LAST, E_WALL);
        std::cout.setstate(std::ios::failbit);
        w.mgu.multigrid_solve_system(LAST, 10e-6f); w.mgv.multigrid_solve_system(LAST, 10e-6f); w.mgw.multigrid_solve_system(LAST, 10e-6f);
        std::cout.clear();
        auto u_mom = rd(w.rans.u_system.field.data, sizes('u'));
        w.inc.divergence_cal(w.rans.u_system.field.data, w.rans.v_system.field.data, w.rans.w_system.field.data);
        w.ps.Multigrid_pressure_solve(DT, RHO, fixed_pressure_cycles_many ? 30 : 2);
        w.inc.project_pressure(DT, RHO, w.rans.u_system.field.data, w.rans.v_system.field.data, w.rans.w_system.field.data);
        auto u_proj = rd(w.rans.u_system.field.data, sizes('u'));
        w.inc.divergence_cal(w.rans.u_system.field.data, w.rans.v_system.field.data, w.rans.w_system.field.data);
        auto dv = rd(w.inc.divergence, (size_t)N * N * N); double d2 = 0; for (float q : dv) d2 += (double)q * q;
        w.rans.construct_k_epsilon_system(w.km, w.em, DT, LAST, E_WALL);
        std::cout.setstate(std::ios::failbit);
        w.mgk.multigrid_solve_system(LAST, 10e-6f); w.mge.multigrid_solve_system(LAST, 10e-6f);
        std::cout.clear();
        w.em.update_mu_eff();
        float prod = w.rans.calculate_nonlinear_convergence(w.km, w.em);
        double del[5]; int j = 0; double kmin = 1e30, emin = 1e30;
        for (char c : std::string("uvwke"))
        {
            auto a = rd(w.sys(c).field.data, sizes(c)); auto &b = before[c]; double n = 0, d = 0;
            for (size_t i = 0; i < a.size(); i++) { n += ((double)a[i] - b[i]) * ((double)a[i] - b[i]); d += (double)a[i] * a[i]; }
            del[j++] = std::sqrt(n / std::max(d, 1e-30));
            if (c == 'w' && (it <= 3 || it == iters)) { double mw = 0, mu = 0; for (float q : a) mw = std::max(mw, (double)std::fabs(q)); for (float q : rd(w.sys('u').field.data, sizes('u'))) mu = std::max(mu, (double)std::fabs(q)); printf("       [it %d] max|w| = %.3e, max|u| = %.3e, ratio %.2e, sum w^2 = %.3e\n", it, mw, mu, mw / mu, d); }
            if (c == 'k') for (float q : a) kmin = std::min(kmin, (double)q);
            if (c == 'e') for (float q : a) emin = std::min(emin, (double)q);
        }
        double pu = diffnorm(todbl(u_proj), todbl(u_mom)) / std::max(norm2(todbl(u_proj)), 1e-30);
        printf("  %2d | %.3e %.3e %.3e %.3e %.3e | %.3e | %.3e | %.3e | %.3e %.3e\n", it, del[0], del[1], del[2], del[3], del[4], prod, pu, std::sqrt(d2 / dv.size()), kmin, emin);
    }
}

// ================================================================== STOPPING-CRITERION STUDY
// For an iterate x (float, as stored on the GPU) report, in double:
//   RMS        = ||r||_2 / sqrt(n)                      (production criterion)
//   rel_b      = ||r||_2 / ||b||_2
//   rel_r0     = ||r||_2 / ||r_0||_2
//   eta_F      = ||r||_2 / (||A||_F ||x||_2 + ||b||_2)  (proposed normwise backward error)
//   eta_s      = ||r||_2 / || |A||x| + |b| ||_2         (row-scaled normwise backward error)
//   omega      = max_i |r_i| / (|A||x| + |b|)_i          (Oettli-Prager componentwise backward error)
//   fwd        = ||x - x*||_2 / ||x*||_2                (only when x* is known)
// "gpu" variants use the production float32 residual buffer instead of the double residual.
struct Crit { double rms, rel_b, rel_r0, etaF, etaS, omega, fwd, etaF_gpu, rms_gpu; };
static double frob(const HostSys &A)
{
    double s = 0;
    for (size_t i = 0; i < A.n; i++) s += A.aP[i] * A.aP[i] + A.aW[i] * A.aW[i] + A.aE[i] * A.aE[i] + A.aS[i] * A.aS[i] + A.aN[i] * A.aN[i] + A.aB[i] * A.aB[i] + A.aT[i] * A.aT[i];
    return std::sqrt(s);
}
static Crit crit(const HostSys &A, double AF, const std::vector<double> &x, double r0, const std::vector<double> *xs, const std::vector<float> *rgpu)
{
    std::vector<double> Ax, sc; apply(A, x, Ax, nullptr, &sc);       // sc_i = |b_i| + sum |a_ij x_j|
    double r2 = 0, s2 = 0, b2 = 0, x2 = 0, om = 0, g2 = 0;
    for (size_t i = 0; i < A.n; i++)
    {
        double r = A.b[i] - Ax[i]; r2 += r * r; s2 += sc[i] * sc[i]; b2 += A.b[i] * A.b[i]; x2 += x[i] * x[i];
        if (sc[i] > 0) om = std::max(om, std::fabs(r) / sc[i]); else if (r != 0) om = INFINITY;
        if (rgpu) g2 += (double)(*rgpu)[i] * (*rgpu)[i];
    }
    double rn = std::sqrt(r2), dF = AF * std::sqrt(x2) + std::sqrt(b2);
    auto q = [](double a, double b) { return b > 0 ? a / b : (a == 0 ? 0.0 : INFINITY); };
    Crit c; c.rms = rn / std::sqrt((double)A.n); c.rel_b = q(rn, std::sqrt(b2)); c.rel_r0 = q(rn, r0); c.etaF = q(rn, dF); c.etaS = q(rn, std::sqrt(s2)); c.omega = om;
    c.fwd = xs ? diffnorm(x, *xs) / std::max(norm2(*xs), 1e-300) : NAN;
    c.etaF_gpu = rgpu ? q(std::sqrt(g2), dF) : NAN; c.rms_gpu = rgpu ? std::sqrt(g2 / A.n) : NAN;
    return c;
}
static void crit_header() { printf("  cyc |   RMS(cpu)   RMS(gpu) |  r/|b|      r/r0     | etaF(cpu)  etaF(gpu) |  etaS      omega     |  fwd err\n"); }
static void crit_row(int k, const Crit &c) { printf("  %3d | %9.3e  %9.3e | %9.3e  %9.3e | %9.3e  %9.3e | %9.3e  %9.3e | %9.3e\n", k, c.rms, c.rms_gpu, c.rel_b, c.rel_r0, c.etaF, c.etaF_gpu, c.etaS, c.omega, c.fwd); }

// scenario: 'f' fresh production start, 'k' known solution from x=0, 'w' warm start at x = fl(x*)
static void exp_crit(ShaderManager &sm, char c, char scen, int ncyc)
{
    printf("\n=== CRIT system %c, scenario %s, dt=%g ===\n", c, scen == 'f' ? "fresh (production initial guess)" : scen == 'k' ? "known solution, x0 = 0" : "warm start x0 = fl(x*)", DT);
    World w(sm);
    if (c == 'k' || c == 'e') w.build_keps(30); else w.init_and_build_momentum();
    LinearSystem &L = w.sys(c); Generic_multigrid &mg = w.mg(c); HostSys A = read_fine(L);
    std::vector<double> xs;
    if (scen == 'k')
    {
        std::mt19937 g(11); std::uniform_real_distribution<double> U(-1, 1); xs.resize(A.n);
        for (int z = 0; z < A.nz; z++) for (int y = 0; y < A.ny; y++) for (int x = 0; x < A.nx; x++)
            xs[(size_t)z * A.nx * A.ny + (size_t)y * A.nx + x] = 10.0 * std::sin(3.0 * x / A.nx) * std::cos(2.0 * y / A.ny) + 0.5 * z / A.nz + 0.1 * U(g);
        std::vector<double> Ax; apply(A, xs, Ax); A.b = Ax; wr(L.RHS_main, toflt(A.b)); A.b = todbl(rd(L.RHS_main, A.n));
        wr(L.field.data, std::vector<float>(A.n, 0.f));
        xs = cpu_solve(A, xs, 1e-15, 200);
    }
    else
    {
        xs = cpu_solve(A, todbl(rd(L.field.data, A.n)), 1e-14, 20000);
        if (scen == 'w') wr(L.field.data, toflt(xs));
    }
    double AF = frob(A);
    std::vector<double> Ax0; auto x0 = todbl(rd(L.field.data, A.n)); apply(A, x0, Ax0);
    double r0 = 0; for (size_t i = 0; i < A.n; i++) r0 += (A.b[i] - Ax0[i]) * (A.b[i] - Ax0[i]); r0 = std::sqrt(r0);
    // how pessimistic is ||A||_F ||x|| relative to the quantity it bounds?
    { std::vector<double> Ax; apply(A, xs, Ax); printf("  ||A||_F = %.4g, ||A||_F ||x*|| / ||A x*|| = %.2f (sqrt(n) = %.1f), ||b|| = %.4g\n", AF, AF * norm2(xs) / std::max(norm2(Ax), 1e-300), std::sqrt((double)A.n), norm2(A.b)); }
    crit_header();
    mg.calculate_l2_norm(); crit_row(0, crit(A, AF, x0, r0, &xs, nullptr));
    for (int k = 1; k <= ncyc; k++)
    {
        mg.Generic_multigrid_solve(LAST, 1);
        mg.calculate_l2_norm(); auto rg = rd(mg.levels_residual_SSBOs_arr[0], A.n);
        crit_row(k, crit(A, AF, todbl(rd(L.field.data, A.n)), r0, &xs, &rg));
    }
}

// slow-convergence case (face diffusion problem, dt large): does each criterion stop too early (large forward error)?
static void exp_crit_poisson(ShaderManager &sm, int topo_i, double dt, int ncyc)
{
    fieldTopology topo = (fieldTopology)topo_i;
    printf("\n=== CRIT slow-convergence case: diffusion problem %d^3 topology %d dt=%g (production V-cycles) ===\n", N, topo_i, dt);
    LinearSystem L(N, N, N, LAST, sm, topo);
    double h = DOMAIN / N, mu = 0.5;
    auto fill = [&](StencilOperator &op, int nx, int ny, int nz, double hh)
    {
        size_t m = (size_t)nx * ny * nz; std::vector<float> P(m), W(m), Eo(m), S(m), No(m), B(m), T(m);
        double D = mu * hh, tc = RHO * hh * hh * hh / dt;
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)
        {
            size_t i = (size_t)z * nx * ny + (size_t)y * nx + x; double p = tc;
            bool dir = (topo == fieldTopology::XFACE_CENTERED && (x == 0 || x == nx - 1));
            if (dir) { P[i] = (float)tc; continue; }
            auto face = [&](int has, float &a) { if (has) { a = (float)-D; p += D; } else p += 2 * D; };
            face(x > 0, W[i]); face(x + 1 < nx, Eo[i]); face(y > 0, No[i]); face(y + 1 < ny, S[i]); face(z > 0, T[i]); face(z + 1 < nz, B[i]);
            P[i] = (float)p;
        }
        wr(op.aP, P); wr(op.aW, W); wr(op.aE, Eo); wr(op.aS, S); wr(op.aN, No); wr(op.aB, B); wr(op.aT, T);
    };
    fill(L.coefficients, L.x_elements, L.y_elements, L.z_elements, h);
    for (size_t l = 0; l < L.multigrid_levels_coeffiecients_arr.size(); l++) { auto &op = L.multigrid_levels_coeffiecients_arr[l]; fill(op, op.level_x_elements, op.level_y_elements, op.level_z_elements, h * (2 << l)); }
    HostSys A = read_fine(L);
    // smooth x* (worst case for a slowly converging smooth mode)
    std::vector<double> xs(A.n);
    for (int z = 0; z < A.nz; z++) for (int y = 0; y < A.ny; y++) for (int x = 0; x < A.nx; x++)
        xs[(size_t)z * A.nx * A.ny + (size_t)y * A.nx + x] = std::sin(3.14159265 * (x + 0.5) / A.nx) * std::sin(3.14159265 * (y + 0.5) / A.ny) * std::sin(3.14159265 * (z + 0.5) / A.nz);
    std::vector<double> Ax; apply(A, xs, Ax); wr(L.RHS_main, toflt(Ax)); A.b = todbl(rd(L.RHS_main, A.n));
    xs = cpu_solve(A, xs, 1e-15, 4000);
    wr(L.field.data, std::vector<float>(A.n, 0.f));
    Generic_multigrid mg(LAST, sm, L);
    double AF = frob(A), r0 = norm2(A.b);
    printf("  ||A||_F ||x*|| / ||A x*|| = %.2f\n", AF * norm2(xs) / norm2(A.b));
    crit_header(); crit_row(0, crit(A, AF, std::vector<double>(A.n, 0.0), r0, &xs, nullptr));
    for (int k = 1; k <= ncyc; k++)
    {
        mg.Generic_multigrid_solve(LAST, 1); mg.calculate_l2_norm(); auto rg = rd(mg.levels_residual_SSBOs_arr[0], A.n);
        crit_row(k, crit(A, AF, todbl(rd(L.field.data, A.n)), r0, &xs, &rg));
    }
}

// zero system (W): what does each criterion do with b = 0, x = 0?
static void exp_crit_zero(ShaderManager &sm)
{
    printf("\n=== CRIT zero system (W: b = 0, x = 0) ===\n");
    World w(sm); w.init_and_build_momentum(); HostSys A = read_fine(w.sys('w'));
    auto x = todbl(rd(w.sys('w').field.data, A.n));
    crit_header(); crit_row(0, crit(A, frob(A), x, 0.0, nullptr, nullptr));
    printf("  (q(a,b) convention: 0/0 -> 0, a/0 -> inf; production RMS check returns immediately)\n");
}

// T-05 revisited: iterative refinement in double, criteria on the double iterate
static void exp_crit_refine(ShaderManager &sm, char c)
{
    printf("\n=== CRIT iterative refinement, system %c (x kept in double on the CPU) ===\n", c);
    World w(sm);
    if (c == 'k' || c == 'e') w.build_keps(30); else w.init_and_build_momentum();
    LinearSystem &L = w.sys(c); Generic_multigrid &mg = w.mg(c); HostSys A = read_fine(L);
    double AF = frob(A); auto x0 = todbl(rd(L.field.data, A.n));
    std::vector<double> Ax; apply(A, x0, Ax); double r0 = 0; for (size_t i = 0; i < A.n; i++) r0 += (A.b[i] - Ax[i]) * (A.b[i] - Ax[i]); r0 = std::sqrt(r0);
    auto xs = cpu_solve(A, x0, 1e-15, 20000);
    auto b_orig = rd(L.RHS_main, A.n);
    mg.Generic_multigrid_solve(LAST, 40);
    std::vector<double> x = todbl(rd(L.field.data, A.n));
    crit_header(); crit_row(0, crit(A, AF, x, r0, &xs, nullptr));
    for (int o = 1; o <= 3; o++)
    {
        apply(A, x, Ax); std::vector<double> rr(A.n); for (size_t i = 0; i < A.n; i++) rr[i] = A.b[i] - Ax[i];
        wr(L.RHS_main, toflt(rr)); wr(L.field.data, std::vector<float>(A.n, 0.f));
        mg.Generic_multigrid_solve(LAST, 10);
        auto e = rd(L.field.data, A.n); for (size_t i = 0; i < A.n; i++) x[i] += e[i];
        crit_row(o, crit(A, AF, x, r0, &xs, nullptr));
    }
    wr(L.RHS_main, b_orig);
}

int main(int argc, char **argv)
{
    std::string mode = argc > 1 ? argv[1] : "all";
    for (int i = 2; i < argc; i++)
    {
        std::string a = argv[i];
        auto val = [&](const char *k) { return a.rfind(k, 0) == 0 ? a.substr(strlen(k)) : std::string(); };
        if (!val("N=").empty()) N = atoi(val("N=").c_str());
        if (!val("dt=").empty()) DT = (float)atof(val("dt=").c_str());
        if (!val("U=").empty()) U_REF = (float)atof(val("U=").c_str());
        if (!val("I=").empty()) T_INT = (float)atof(val("I=").c_str());
        if (!val("L=").empty()) T_LEN = (float)atof(val("L=").c_str());
    }
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(64, 64, "audit", nullptr, nullptr);
    glfwMakeContextCurrent(win); gladLoadGL();
    printf("GL: %s | %s\n", (const char *)glGetString(GL_RENDERER), (const char *)glGetString(GL_VERSION));
    ShaderManager sm;
    printf("CONFIG: N=%d domain=%g rho=%g mu=%g dt=%g last_level=%d E=%g | k-eps inputs U=%g I=%g L=%g\n", N, DOMAIN, RHO, MU, DT, LAST, E_WALL, U_REF, T_INT, T_LEN);

    if (mode == "loc") exp_locations(sm);
    else if (mode == "l2") exp_l2(sm);
    else if (mode == "ops") exp_ops(sm);
    else if (mode == "zero") exp_zero(sm);
    else if (mode.rfind("cyc", 0) == 0) exp_cycles(sm, mode[3], argc > 2 && isdigit(argv[2][0]) ? atoi(argv[2]) : 30, false);
    else if (mode.rfind("known", 0) == 0) exp_cycles(sm, mode[5], 30, true);
    else if (mode.rfind("prod", 0) == 0) exp_production(sm, mode[4], 10e-6f);
    else if (mode.rfind("refine", 0) == 0) exp_refine(sm, mode[6], 10, 4);
    else if (mode.rfind("twogrid", 0) == 0) exp_twogrid(sm, mode[7]);
    else if (mode.rfind("realcmp", 0) == 0) exp_realcmp(sm, mode[7]);
    else if (mode == "wallmag") exp_wallmag(sm);
    else if (mode.rfind("crit", 0) == 0 && mode.size() == 6) exp_crit(sm, mode[4], mode[5], 8);     // critUf, critUk, critUw ...
    else if (mode == "critpoisson1") exp_crit_poisson(sm, 1, DT, 24);
    else if (mode == "critpoisson0") exp_crit_poisson(sm, 0, DT, 12);
    else if (mode == "critzero") exp_crit_zero(sm);
    else if (mode.rfind("critref", 0) == 0) exp_crit_refine(sm, mode[7]);
    else if (mode == "orchrep") exp_orchrep(sm, 25, false);
    else if (mode == "orchrep30") exp_orchrep(sm, 25, true);
    else if (mode.rfind("poisson", 0) == 0) { int t = mode.size() > 7 ? mode[7] - '0' : 0; exp_poisson(sm, N, t, DT, 30); }
    printf("glGetError at exit: 0x%x\n", glGetError());
    glfwTerminate();
}
