// k-eps NaN investigation. Links the UNMODIFIED project sources; test-only access via #define private public.
// Mirrors Orchestrator::run_rans_k_eps up to the first k / eps solve, then instruments every stage.
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
#include <sstream>
#include <functional>
#include <map>
#include <memory>
#include <limits>
#define private public
#include "Turbulence_models/RANS.hpp"
#include "Turbulence_models/k_TurbulenceModels.hpp"
#include "Generic_multigrid/generic_multigrid.hpp"
#include "Incompressible_flow/incompressible_flow.hpp"
#undef private

static int N = 64; static float DOMAIN = 200.f, RHO = 1.225f, MU = 1.8e-5f, DT = 1.0f / 60.0f, E_WALL = 9.8f;
static int LAST = 8, VCYC = 2;
static float U_REF = 80.f, T_INT = 5.f, T_LEN = 0.1f;

static std::vector<float> rd(GLuint b, size_t n) { glMemoryBarrier(GL_ALL_BARRIER_BITS); glFinish(); std::vector<float> v(n); glBindBuffer(GL_SHADER_STORAGE_BUFFER, b); glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 4, v.data()); return v; }
static void wr(GLuint b, const std::vector<float> &v) { glMemoryBarrier(GL_ALL_BARRIER_BITS); glBindBuffer(GL_SHADER_STORAGE_BUFFER, b); glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, v.size() * 4, v.data()); glMemoryBarrier(GL_ALL_BARRIER_BITS); }
static size_t nnf(const std::vector<float> &v) { size_t c = 0; for (float q : v) if (!std::isfinite(q)) c++; return c; }
static std::string stats(const std::vector<float> &v)
{
    size_t nan = 0, inf = 0; double mn = 1e300, mx = -1e300;
    for (float q : v) { if (std::isnan(q)) nan++; else if (std::isinf(q)) inf++; else { mn = std::min(mn, (double)q); mx = std::max(mx, (double)q); } }
    char b[200]; snprintf(b, 200, "NaN %zu, Inf %zu, finite range [%.4g, %.4g]", nan, inf, mn, mx); return b;
}
static bool is_boundary(size_t i, int nx, int ny, int nz) { int x = i % nx, y = (i / nx) % ny, z = (int)(i / ((size_t)nx * ny)); return x == 0 || y == 0 || z == 0 || x == nx - 1 || y == ny - 1 || z == nz - 1; }

struct World
{
    ShaderManager &sm; RANS rans; k_TurbulenceModels km; k_TurbulenceModels::epsilon em;
    Generic_multigrid mgu, mgv, mgw; incompressible_flow inc; incompressible_flow::specialized_pressure_Multigrid ps; Generic_multigrid mgk, mge;
    World(ShaderManager &s) : sm(s), rans(N, N, N, LAST, s, DOMAIN, DOMAIN, DOMAIN, RHO, MU), km(rans, LAST, false), em(km, LAST),
        mgu(LAST, s, rans.u_system), mgv(LAST, s, rans.v_system), mgw(LAST, s, rans.w_system),
        inc(N, N, N, DOMAIN / N, DOMAIN / N, DOMAIN / N, s), ps(LAST, inc), mgk(LAST, s, km.KSystem), mge(LAST, s, em.eps_system) {}
    // exact orchestrator order (orchestrator.cpp:390-447), one time step, first nonlinear iteration
    void to_keps_solve(bool quiet = true)
    {
        rans.initialize_velocity_fields();
        em.initialize_k_epsilon_RANS_fields(U_REF, T_INT, T_LEN, E_WALL);
        inc.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        ps.Multigrid_pressure_solve(DT, RHO, VCYC);
        inc.project_pressure(DT, RHO, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        rans.u_system.save_previous_time_step_field(); rans.v_system.save_previous_time_step_field(); rans.w_system.save_previous_time_step_field();
        km.KSystem.save_previous_time_step_field(); em.eps_system.save_previous_time_step_field();
        rans.store_nonlinear_iterate(km, em);
        rans.construct_momentum_equations(DT, LAST, E_WALL);
        if (quiet) std::cout.setstate(std::ios::failbit);
        int su = mgu.multigrid_solve_system(LAST, 1e-3f, 1e-4f, 3), sv = mgv.multigrid_solve_system(LAST, 1e-3f, 1e-4f, 3), sw = mgw.multigrid_solve_system(LAST, 1e-3f, 1e-4f, 3);
        std::cout.clear();
        printf("  momentum statuses u %d v %d w %d\n", su, sv, sw);
        inc.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        ps.Multigrid_pressure_solve(DT, RHO, VCYC);
        inc.project_pressure(DT, RHO, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        rans.construct_k_epsilon_system(km, em, DT, LAST, E_WALL);
        glFinish();
    }
};

// ------------------------------------------------------------------ GPU stage wrappers (same bindings as production)
static void smooth_color(ShaderManager &sm, const StencilOperator &A, GLuint rhs, GLuint sol, int nx, int ny, int nz, int color)
{
    glUseProgram(sm.multigrid_generic_smoother_program);
    GLuint b[9] = { A.aP, A.aW, A.aE, A.aS, A.aN, A.aB, A.aT, rhs, sol }; for (int i = 0; i < 9; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, b[i]);
    glUniform1i(sm.Generic_smoother_cells_x_loc, nx); glUniform1i(sm.Generic_smoother_cells_y_loc, ny); glUniform1i(sm.Generic_smoother_cells_z_loc, nz);
    glUniform1i(sm.Generic_smoother_color_loc, color);
    glDispatchCompute((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8); glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static void residual(ShaderManager &sm, const StencilOperator &A, GLuint rhs, GLuint sol, GLuint res, int nx, int ny, int nz)
{
    glUseProgram(sm.multigrid_generic_residual_calculation_program);
    GLuint b[10] = { A.aP, A.aW, A.aE, A.aS, A.aN, A.aB, A.aT, rhs, sol, res }; for (int i = 0; i < 10; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, b[i]);
    glUniform1i(sm.generic_residual_current_cells_x_loc, nx); glUniform1i(sm.generic_residual_current_cells_y_loc, ny); glUniform1i(sm.generic_residual_current_cells_z_loc, nz);
    glDispatchCompute((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8); glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static void restrict_res(ShaderManager &sm, GLuint f, GLuint c, std::vector<int> fd, std::vector<int> cd)
{
    glUseProgram(sm.multigrid_residual_generic_restriction_program); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, f); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, c);
    glUniform3i(sm.residual_generic_restriction_fine_dimensions_loc, fd[0], fd[1], fd[2]); glUniform3i(sm.residual_generic_restriction_coarse_dimensions_loc, cd[0], cd[1], cd[2]);
    glUniform1i(sm.residual_generic_restriction_topology_loc, 0);
    glDispatchCompute((cd[0] + 7) / 8, (cd[1] + 7) / 8, (cd[2] + 7) / 8); glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static void prolong(ShaderManager &sm, GLuint c, GLuint f, std::vector<int> fd, std::vector<int> cd)
{
    glUseProgram(sm.multigrid_generic_prolongation_program); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, c); glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, f);
    glUniform3i(sm.generic_prolongation_fine_dimensions_loc, fd[0], fd[1], fd[2]); glUniform3i(sm.generic_prolongation_coarse_dimensions_loc, cd[0], cd[1], cd[2]);
    glUniform1i(sm.generic_prolongation_topology_loc, 0);
    glDispatchCompute((fd[0] + 7) / 8, (fd[1] + 7) / 8, (fd[2] + 7) / 8); glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static size_t sz(LinearSystem &L, int l) { auto d = L.GetLevelDimension(l); return (size_t)d[0] * d[1] * d[2]; }

// -------------------------------------------------------------- 2. pre-solve validation of one cell-centred system
static void validate(const char *name, LinearSystem &L)
{
    size_t n = sz(L, 0); int nx = L.x_elements, ny = L.y_elements, nz = L.z_elements;
    auto P = rd(L.coefficients.aP, n); auto b = rd(L.RHS_main, n); auto x = rd(L.field.data, n);
    GLuint nbb[6] = { L.coefficients.aW, L.coefficients.aE, L.coefficients.aS, L.coefficients.aN, L.coefficients.aB, L.coefficients.aT };
    std::vector<std::vector<float>> nb; for (GLuint g : nbb) nb.push_back(rd(g, n));
    printf("  [%s fine] field: %s\n", name, stats(x).c_str());
    printf("  [%s fine] RHS  : %s\n", name, stats(b).c_str());
    printf("  [%s fine] aP   : %s\n", name, stats(P).c_str());
    size_t nbnf = 0; for (auto &a : nb) nbnf += nnf(a);
    size_t zero = 0, zero_bnd = 0, zero_allzero = 0, zero_b0 = 0, int_rows_touching_zero = 0;
    std::vector<char> zd(n, 0);
    for (size_t i = 0; i < n; i++) if (P[i] == 0.f)
    {
        zero++; zd[i] = 1; if (is_boundary(i, nx, ny, nz)) zero_bnd++;
        bool all0 = true; for (auto &a : nb) if (a[i] != 0.f) all0 = false; if (all0) zero_allzero++; if (b[i] == 0.f) zero_b0++;
    }
    const long ls = (long)nx * ny; const long off[6] = { -1, 1, nx, -nx, ls, -ls };     // W E S N B T
    for (size_t i = 0; i < n; i++) if (!zd[i]) for (int k = 0; k < 6; k++) { long j = (long)i + off[k]; if (j >= 0 && j < (long)n && zd[j] && nb[k][i] != 0.f) { int_rows_touching_zero++; break; } }
    printf("  [%s fine] off-diagonal non-finite: %zu | rows with aP == 0: %zu (boundary %zu of %zu boundary cells; all-zero row %zu; RHS == 0 %zu)\n",
           name, nbnf, zero, zero_bnd, n - (size_t)(nx - 2) * (ny - 2) * (nz - 2), zero_allzero, zero_b0);
    printf("  [%s fine] non-zero-diagonal rows with a NON-ZERO coefficient pointing at a zero-diagonal row: %zu\n", name, int_rows_touching_zero);
    for (size_t l = 0; l < L.multigrid_levels_coeffiecients_arr.size(); l++)
    {
        auto &op = L.multigrid_levels_coeffiecients_arr[l]; size_t m = (size_t)op.level_x_elements * op.level_y_elements * op.level_z_elements;
        auto Pc = rd(op.aP, m); size_t z0 = 0; for (float q : Pc) if (q == 0.f) z0++;
        printf("  [%s coarse L%zu %dx%dx%d] aP: %s, rows with aP == 0: %zu\n", name, l + 1, op.level_x_elements, op.level_y_elements, op.level_z_elements, stats(Pc).c_str(), z0);
    }
}

// -------------------------------------------------------------- 3/4. instrumented V-cycle (same sequence as Generic_multigrid_solve)
static void instrumented_vcycle(ShaderManager &sm, const char *name, LinearSystem &L, Generic_multigrid &mg)
{
    int passes = (int)L.multigrid_levels_coeffiecients_arr.size();
    auto d0 = L.GetLevelDimension(0); size_t n0 = sz(L, 0);
    auto P = rd(L.coefficients.aP, n0);
    auto rep = [&](const char *stage, GLuint buf, size_t n, int nx = 0, int ny = 0, int nz = 0, const std::vector<float> *diag = nullptr)
    {
        auto v = rd(buf, n); size_t c = nnf(v);
        printf("    %-46s non-finite %7zu / %zu", stage, c, n);
        if (c && nx)
        {
            size_t bnd = 0, zd = 0; long first = -1; for (size_t i = 0; i < n; i++) if (!std::isfinite(v[i])) { if (first < 0) first = (long)i; if (is_boundary(i, nx, ny, nz)) bnd++; if (diag && (*diag)[i] == 0.f) zd++; }
            int fx = first % nx, fy = (first / nx) % ny, fz = (int)(first / ((long)nx * ny));
            printf("  (boundary cells %zu, zero-diagonal rows %zu, first at (%d,%d,%d))", bnd, zd, fx, fy, fz);
        }
        printf("\n"); return c;
    };
    printf("  --- instrumented V-cycle on %s (every red / black pass checked) ---\n", name);
    for (int l = 0; l <= passes; l++) wr(mg.levels_correction_SSBOs_arr[l], std::vector<float>(sz(L, l), 0.f));
    rep("start: field", L.field.data, n0, d0[0], d0[1], d0[2], &P);
    char s[96];
    for (int it = 0; it < 3; it++) for (int c = 0; c < 2; c++)
    {
        smooth_color(sm, L.coefficients, L.RHS_main, L.field.data, d0[0], d0[1], d0[2], c);
        snprintf(s, 96, "fine pre-smooth sweep %d %s: field", it + 1, c ? "BLACK" : "RED  "); rep(s, L.field.data, n0, d0[0], d0[1], d0[2], &P);
    }
    residual(sm, L.coefficients, L.RHS_main, L.field.data, mg.levels_residual_SSBOs_arr[0], d0[0], d0[1], d0[2]);
    rep("fine residual", mg.levels_residual_SSBOs_arr[0], n0, d0[0], d0[1], d0[2], &P);
    restrict_res(sm, mg.levels_residual_SSBOs_arr[0], mg.levels_RHS_SSBOs_arr[1], d0, L.GetLevelDimension(1));
    { auto d = L.GetLevelDimension(1); rep("restriction -> RHS L1", mg.levels_RHS_SSBOs_arr[1], sz(L, 1), d[0], d[1], d[2]); }
    for (int i = 1; i < passes; i++)
    {
        auto d = L.GetLevelDimension(i); auto &A = L.multigrid_levels_coeffiecients_arr[i - 1]; auto Pc = rd(A.aP, sz(L, i));
        for (int it = 0; it < 3; it++) for (int c = 0; c < 2; c++) { smooth_color(sm, A, mg.levels_RHS_SSBOs_arr[i], mg.levels_correction_SSBOs_arr[i], d[0], d[1], d[2], c); }
        snprintf(s, 96, "L%d smoothing (3 sweeps): correction", i); rep(s, mg.levels_correction_SSBOs_arr[i], sz(L, i), d[0], d[1], d[2], &Pc);
        residual(sm, A, mg.levels_RHS_SSBOs_arr[i], mg.levels_correction_SSBOs_arr[i], mg.levels_residual_SSBOs_arr[i], d[0], d[1], d[2]);
        snprintf(s, 96, "L%d residual", i); rep(s, mg.levels_residual_SSBOs_arr[i], sz(L, i));
        restrict_res(sm, mg.levels_residual_SSBOs_arr[i], mg.levels_RHS_SSBOs_arr[i + 1], d, L.GetLevelDimension(i + 1));
        snprintf(s, 96, "restriction -> RHS L%d", i + 1); rep(s, mg.levels_RHS_SSBOs_arr[i + 1], sz(L, i + 1));
    }
    { auto d = L.GetLevelDimension(passes); auto &A = L.multigrid_levels_coeffiecients_arr[passes - 1]; auto Pc = rd(A.aP, sz(L, passes));
      for (int it = 0; it < 30; it++) for (int c = 0; c < 2; c++) smooth_color(sm, A, mg.levels_RHS_SSBOs_arr[passes], mg.levels_correction_SSBOs_arr[passes], d[0], d[1], d[2], c);
      snprintf(s, 96, "coarsest L%d solve (30 sweeps): correction", passes); rep(s, mg.levels_correction_SSBOs_arr[passes], sz(L, passes), d[0], d[1], d[2], &Pc); }
    for (int i = 0; i < passes - 1; i++)
    {
        int c = passes - i, f = passes - i - 1; auto df = L.GetLevelDimension(f), dc = L.GetLevelDimension(c);
        prolong(sm, mg.levels_correction_SSBOs_arr[c], mg.levels_correction_SSBOs_arr[f], df, dc);
        snprintf(s, 96, "prolong L%d -> L%d: correction", c, f); rep(s, mg.levels_correction_SSBOs_arr[f], sz(L, f));
        for (int it = 0; it < 3; it++) for (int cc = 0; cc < 2; cc++) smooth_color(sm, L.multigrid_levels_coeffiecients_arr[f - 1], mg.levels_RHS_SSBOs_arr[f], mg.levels_correction_SSBOs_arr[f], df[0], df[1], df[2], cc);
        snprintf(s, 96, "L%d post-smoothing: correction", f); rep(s, mg.levels_correction_SSBOs_arr[f], sz(L, f));
    }
    prolong(sm, mg.levels_correction_SSBOs_arr[1], L.field.data, d0, L.GetLevelDimension(1));
    rep("final prolongation/correction -> field", L.field.data, n0, d0[0], d0[1], d0[2], &P);
    for (int it = 0; it < 3; it++) for (int c = 0; c < 2; c++) smooth_color(sm, L.coefficients, L.RHS_main, L.field.data, d0[0], d0[1], d0[2], c);
    rep("fine post-smooth: field", L.field.data, n0, d0[0], d0[1], d0[2], &P);
}

// -------------------------------------------------------------- 4b. one zero-diagonal row in isolation: what the smoother computes
static void single_row(LinearSystem &L)
{
    size_t n = sz(L, 0); auto P = rd(L.coefficients.aP, n), b = rd(L.RHS_main, n), x = rd(L.field.data, n);
    size_t i = 0;                                            // corner cell (0,0,0): RED (x+y+z even)
    printf("  row (0,0,0): aP %.6g  RHS %.6g  x %.6g  -> GS update (RHS - sum)/aP evaluated on CPU: %.6g\n", P[i], b[i], x[i], (double)((b[i] - 0.f) / P[i]));
}

// -------------------------------------------------------------- 5. eddy-viscosity kernel with controlled bad inputs
static void mu_eff_probe(World &w)
{
    printf("\n=== 5. K_eps_mu_eff.comp with controlled non-finite / negative inputs ===\n");
    size_t n = (size_t)N * N * N; const float qn = std::numeric_limits<float>::quiet_NaN(), inf = INFINITY;
    std::vector<float> k(n, 24.f), e(n, 193.f);
    struct C { const char *d; float k, e; } cases[] = { {"k NaN", qn, 193.f}, {"eps NaN", 24.f, qn}, {"k NaN, eps NaN", qn, qn}, {"k +Inf", inf, 193.f}, {"eps +Inf", 24.f, inf},
        {"k -Inf", -inf, 193.f}, {"k negative", -5.f, 193.f}, {"eps 0", 24.f, 0.f}, {"eps negative", 24.f, -1.f}, {"k=0, eps=0", 0.f, 0.f}, {"k huge 1e20", 1e20f, 193.f} };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) { k[c] = cases[c].k; e[c] = cases[c].e; }
    wr(w.km.KSystem.field.data, k); wr(w.em.eps_system.field.data, e);
    w.em.update_mu_eff();
    auto m = rd(w.rans.mu_eff, n);
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) printf("    %-16s -> mu_eff = %g\n", cases[c].d, m[c]);
}

int main(int argc, char **argv)
{
    std::string mode = argc > 1 ? argv[1] : "probe";
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a.rfind("I=", 0) == 0) T_INT = (float)atof(a.c_str() + 2); if (a.rfind("U=", 0) == 0) U_REF = (float)atof(a.c_str() + 2); if (a.rfind("L=", 0) == 0) T_LEN = (float)atof(a.c_str() + 2); if (a.rfind("N=", 0) == 0) N = atoi(a.c_str() + 2); }
    glfwInit(); glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(64, 64, "nan", nullptr, nullptr); glfwMakeContextCurrent(win); gladLoadGL();
    ShaderManager sm;
    printf("CONFIG N=%d domain=%g dt=%g | U=%g I=%g L=%g | k0 = %.6g eps0 = %.6g\n", N, DOMAIN, DT, U_REF, T_INT, T_LEN, 1.5 * pow(U_REF * T_INT, 2), pow(0.09, 0.75) * pow(1.5 * pow(U_REF * T_INT, 2), 1.5) / T_LEN);
    World w(sm);
    w.to_keps_solve();
    if (mode == "probe")
    {
        printf("\n=== 2. pre-solve validation ===\n");
        auto mu = rd(w.rans.mu_eff, (size_t)N * N * N); printf("  mu_eff: %s\n", stats(mu).c_str());
        validate("k", w.km.KSystem); validate("eps", w.em.eps_system);
        printf("  production initial RMS: k %.6g, eps %.6g\n", w.mgk.calculate_l2_norm() / std::sqrt((double)N * N * N), w.mge.calculate_l2_norm() / std::sqrt((double)N * N * N));
        single_row(w.km.KSystem);
        auto k0 = rd(w.km.KSystem.field.data, (size_t)N * N * N), e0 = rd(w.em.eps_system.field.data, (size_t)N * N * N);
        printf("\n=== 3/4. instrumented V-cycle ===\n");
        instrumented_vcycle(sm, "k", w.km.KSystem, w.mgk);
        instrumented_vcycle(sm, "eps", w.em.eps_system, w.mge);
        // production path from the same state
        wr(w.km.KSystem.field.data, k0); wr(w.em.eps_system.field.data, e0);
        printf("\n=== production multigrid_solve_system on k and eps (same state) ===\n");
        int sk = w.mgk.multigrid_solve_system(LAST, 1e-3f, 1e-5f, 3); int se = w.mge.multigrid_solve_system(LAST, 1e-3f, 1e-5f, 3);
        printf("  status k %d, eps %d (0 abs,1 rel,2 max,3 stag,4 diverging per enum order)\n", sk, se);
        auto kf = rd(w.km.KSystem.field.data, (size_t)N * N * N), ef = rd(w.em.eps_system.field.data, (size_t)N * N * N);
        printf("  k after: %s\n  eps after: %s\n", stats(kf).c_str(), stats(ef).c_str());
        w.em.update_mu_eff(); auto m2 = rd(w.rans.mu_eff, (size_t)N * N * N); printf("  mu_eff after update_mu_eff: %s\n", stats(m2).c_str());
        mu_eff_probe(w);
    }
    else if (mode == "pin")    // causal test: pin the zero rows (test-only, buffers only), then run the production solver
    {
        auto pin = [&](LinearSystem &L) {
            size_t n = sz(L, 0); auto P = rd(L.coefficients.aP, n), b = rd(L.RHS_main, n), x = rd(L.field.data, n); size_t c = 0;
            for (size_t i = 0; i < n; i++) if (P[i] == 0.f) { P[i] = 1.f; b[i] = x[i]; c++; }
            wr(L.coefficients.aP, P); wr(L.RHS_main, b);
            size_t cc = 0; for (auto &op : L.multigrid_levels_coeffiecients_arr) { size_t m = (size_t)op.level_x_elements * op.level_y_elements * op.level_z_elements; auto Pc = rd(op.aP, m); for (auto &q : Pc) if (q == 0.f) { q = 1.f; cc++; } wr(op.aP, Pc); }
            printf("  pinned %zu fine rows (aP=1, RHS=current value) and %zu coarse rows (aP=1)\n", c, cc); };
        pin(w.km.KSystem); pin(w.em.eps_system);
        int sk = w.mgk.multigrid_solve_system(LAST, 1e-3f, 1e-5f, 3); int se = w.mge.multigrid_solve_system(LAST, 1e-3f, 1e-5f, 3);
        printf("  status k %d, eps %d\n  k after: %s\n  eps after: %s\n", sk, se, stats(rd(w.km.KSystem.field.data, (size_t)N * N * N)).c_str(), stats(rd(w.em.eps_system.field.data, (size_t)N * N * N)).c_str());
    }
    printf("glGetError 0x%x\n", glGetError());
    glfwTerminate();
}
