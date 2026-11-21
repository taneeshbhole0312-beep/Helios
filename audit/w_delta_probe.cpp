// "DELTA OF 2" (W nonlinear delta) root-cause probe. Links the UNMODIFIED production sources; test-only access via #define private public.
// Replays Orchestrator::run_rans_k_eps (current order) and measures W before/after every operation that can write it.
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
static int LAST = 8, PCYC = 2, ITERS = 10;
static float U_REF = 80.f, T_INT = 0.05f, T_LEN = 0.1f;

static std::vector<float> rd(GLuint b, size_t n) { glMemoryBarrier(GL_ALL_BARRIER_BITS); glFinish(); std::vector<float> v(n); glBindBuffer(GL_SHADER_STORAGE_BUFFER, b); glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 4, v.data()); return v; }
static double rms(const std::vector<float> &a) { double s = 0; for (float q : a) s += (double)q * q; return std::sqrt(s / a.size()); }
static double l2(const std::vector<float> &a) { double s = 0; for (float q : a) s += (double)q * q; return std::sqrt(s); }
static double maxabs(const std::vector<float> &a) { double m = 0; for (float q : a) m = std::max(m, (double)std::fabs(q)); return m; }
static double l2d(const std::vector<float> &a, const std::vector<float> &b) { double s = 0; for (size_t i = 0; i < a.size(); i++) { double d = (double)a[i] - b[i]; s += d * d; } return std::sqrt(s); }
static double maxd(const std::vector<float> &a, const std::vector<float> &b) { double m = 0; for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs((double)a[i] - b[i])); return m; }
static bool bitequal(const std::vector<float> &a, const std::vector<float> &b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0; }

// z-variation of a cell-centred field: max over (x,y) columns of (max_z - min_z), and the z-checkerboard amplitude
static void zvar(const std::vector<float> &p, double &colrange, double &checker, double &range)
{
    colrange = 0; checker = 0; double mn = 1e300, mx = -1e300;
    for (int y = 0; y < N; y++) for (int x = 0; x < N; x++)
    {
        double lo = 1e300, hi = -1e300, se = 0, so = 0;
        for (int z = 0; z < N; z++) { double v = p[(size_t)z * N * N + (size_t)y * N + x]; lo = std::min(lo, v); hi = std::max(hi, v); (z & 1 ? so : se) += v; mn = std::min(mn, v); mx = std::max(mx, v); }
        colrange = std::max(colrange, hi - lo); checker = std::max(checker, std::fabs(se - so) / (N / 2));
    }
    range = mx - mn;
}

// exact (double CG) solution of the pressure equation the production smoother discretises:
//   sum over EXISTING neighbours (p_nb - p)/h^2 = rho*div/dt  (missing neighbour -> centre value, i.e. zero-gradient wall)
static std::vector<double> exact_pressure(const std::vector<float> &div)
{
    size_t n = (size_t)N * N * N; double h = DOMAIN / N, ih2 = 1.0 / (h * h);
    std::vector<double> b(n); double mean = 0;
    for (size_t i = 0; i < n; i++) { b[i] = -(double)RHO * div[i] / DT; mean += b[i]; }
    mean /= n; for (auto &v : b) v -= mean;                                    // compatibility (Neumann)
    auto A = [&](const std::vector<double> &p, std::vector<double> &out)      // out = -L p  (SPD on the zero-mean subspace)
    {
        for (int z = 0; z < N; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++)
        {
            size_t i = (size_t)z * N * N + (size_t)y * N + x; double c = p[i], s = 0;
            if (x > 0) s += c - p[i - 1]; if (x < N - 1) s += c - p[i + 1];
            if (y > 0) s += c - p[i - N]; if (y < N - 1) s += c - p[i + N];
            if (z > 0) s += c - p[i - (size_t)N * N]; if (z < N - 1) s += c - p[i + (size_t)N * N];
            out[i] = s * ih2;
        }
    };
    std::vector<double> p(n, 0.0), r = b, d = b, Ad(n); double rr = 0; for (double v : r) rr += v * v; double r0 = std::sqrt(rr);
    int it = 0;
    for (; it < 4000 && std::sqrt(rr) > 1e-13 * r0; it++)
    {
        A(d, Ad); double dAd = 0; for (size_t i = 0; i < n; i++) dAd += d[i] * Ad[i];
        double a = rr / dAd; double rr2 = 0;
        for (size_t i = 0; i < n; i++) { p[i] += a * d[i]; r[i] -= a * Ad[i]; rr2 += r[i] * r[i]; }
        double bta = rr2 / rr; rr = rr2; for (size_t i = 0; i < n; i++) d[i] = r[i] + bta * d[i];
    }
    printf("      [exact pressure: CG %d iterations, relative residual %.2e]\n", it, std::sqrt(rr) / r0);
    return p;
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) { std::string a = argv[i]; auto v = [&](const char *k) { return a.rfind(k, 0) == 0 ? atof(a.c_str() + strlen(k)) : NAN; };
        if (!std::isnan(v("PCYC="))) PCYC = (int)v("PCYC="); if (!std::isnan(v("ITERS="))) ITERS = (int)v("ITERS="); if (!std::isnan(v("N="))) N = (int)v("N="); }
    glfwInit(); glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(64, 64, "w", nullptr, nullptr); glfwMakeContextCurrent(win); gladLoadGL();
    ShaderManager sm;
    printf("CONFIG N=%d domain=%g dt=%g U=%g I=%g L=%g | pressure V-cycles per solve = %d (production: config.V_cycles_count = 2)\n", N, DOMAIN, DT, U_REF, T_INT, T_LEN, PCYC);

    RANS rans(N, N, N, LAST, sm, DOMAIN, DOMAIN, DOMAIN, RHO, MU);
    k_TurbulenceModels km(rans, LAST, false); k_TurbulenceModels::epsilon em(km, LAST);
    Generic_multigrid mgu(LAST, sm, rans.u_system), mgv(LAST, sm, rans.v_system), mgw(LAST, sm, rans.w_system);
    incompressible_flow inc(N, N, N, DOMAIN / N, DOMAIN / N, DOMAIN / N, sm); incompressible_flow::specialized_pressure_Multigrid ps(LAST, inc);
    Generic_multigrid mgk(LAST, sm, km.KSystem), mge(LAST, sm, em.eps_system);
    const size_t nu = (size_t)(N + 1) * N * N, nw = (size_t)N * N * (N + 1), nc = (size_t)N * N * N;
    auto W = [&] { return rd(rans.w_system.field.data, nw); };
    auto U = [&] { return rd(rans.u_system.field.data, nu); };

    // ---------------- orchestrator.cpp:390-408
    rans.initialize_velocity_fields();
    em.initialize_k_epsilon_RANS_fields(U_REF, T_INT, T_LEN, E_WALL);
    printf("\n[init] w after initialize_velocity_fields: max|w| = %.3e\n", maxabs(W()));
    inc.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
    { double cr, ch, rg; auto d = rd(inc.divergence, nc); zvar(d, cr, ch, rg); printf("[init] divergence: max|div| %.3e, z-column range max %.3e (div is z-invariant if 0)\n", maxabs(d), cr); }
    ps.Multigrid_pressure_solve(DT, RHO, PCYC);
    { double cr, ch, rg; auto p = rd(inc.pressure, nc); zvar(p, cr, ch, rg); printf("[init] production pressure: range %.4e, max z-column range %.4e, max z-checkerboard %.4e\n", rg, cr, ch); }
    inc.project_pressure(DT, RHO, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
    printf("[init] w after initial projection: max|w| = %.3e, rms %.3e (max|u| %.3e)\n", maxabs(W()), rms(W()), maxabs(U()));

    // ---------------- one time step (orchestrator.cpp:414-480)
    rans.u_system.save_previous_time_step_field(); rans.v_system.save_previous_time_step_field(); rans.w_system.save_previous_time_step_field();
    km.KSystem.save_previous_time_step_field(); em.eps_system.save_previous_time_step_field();
    printf("[step] w^n saved (previous_time_step_field): max|w^n| = %.3e\n", maxabs(rd(rans.w_system.previous_time_step_field, nw)));

    printf("\nit | prev_iter==field | b_w max   | w after Wsolve | w after proj (max / rms) | dW solve->proj max | ||dW||_2   | ||W||_2    | DELTA2 cpu | DELTA2 prod | ||dW||/||U|| | p z-col range / p range | p z-check\n");
    for (int it = 1; it <= ITERS; it++)
    {
        rans.store_nonlinear_iterate(km, em);
        auto w0 = W(); auto prevw = rd(rans.previous_iterate_w, nw);
        bool same = bitequal(w0, prevw);
        rans.construct_momentum_equations(DT, LAST, E_WALL);
        auto bw = rd(rans.w_system.RHS_main, nw);
        std::cout.setstate(std::ios::failbit);
        mgu.multigrid_solve_system(LAST, 1e-3f, 1e-6f, 3); mgv.multigrid_solve_system(LAST, 1e-3f, 1e-6f, 3); mgw.multigrid_solve_system(LAST, 1e-3f, 1e-6f, 3);
        std::cout.clear();
        auto w1 = W();
        inc.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        auto divv = rd(inc.divergence, nc);
        ps.Multigrid_pressure_solve(DT, RHO, PCYC);
        auto p = rd(inc.pressure, nc);
        inc.project_pressure(DT, RHO, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        auto w2 = W(); auto u2 = U();
        rans.construct_k_epsilon_system(km, em, DT, LAST, E_WALL);
        std::cout.setstate(std::ios::failbit);
        mgk.multigrid_solve_system(LAST, 1e-3f, 1e-6f, 3); mge.multigrid_solve_system(LAST, 1e-3f, 1e-6f, 3);
        em.update_mu_eff();
        float prod = rans.calculate_nonlinear_convergence(km, em);
        std::cout.clear();
        float sums[10]; glBindBuffer(GL_SHADER_STORAGE_BUFFER, rans.nonlinear_reduction_results); glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(sums), sums);
        auto w3 = W();
        double d_cpu = l2d(w3, prevw) / std::max(l2(w3), 1e-300);
        double d_prod = std::sqrt(std::max((double)sums[2], 0.0) / std::max((double)sums[7], 1e-35));
        double cr, ch, rg; zvar(p, cr, ch, rg);
        printf("%2d | %-16s | %.3e | %.3e      | %.3e / %.3e    | %.3e          | %.3e  | %.3e  | %.4e | %.4e  | %.3e   | %.3e / %.3e | %.3e\n",
               it, same ? "bit-identical" : "DIFFERENT", maxabs(bw), maxabs(w1), maxabs(w2), rms(w2), maxd(w2, w1), l2d(w3, prevw), l2(w3), d_cpu, d_prod, l2d(w3, prevw) / l2(u2), cr, rg, ch);
        if (!bitequal(w3, w2)) printf("    !! w changed between projection and the convergence check\n");
        {
            // REGRESSION CHECK for the proposed fix, computed from the SAME production reduction slots:
            // delta_vel = sqrt(sum_c ||d c||^2 / sum_c ||c||^2) over c in {u,v,w}
            double num = (double)sums[0] + sums[1] + sums[2], den = (double)sums[5] + sums[6] + sums[7];
            double dk = std::sqrt(std::max((double)sums[3], 0.0) / std::max((double)sums[8], 1e-35)), de = std::sqrt(std::max((double)sums[4], 0.0) / std::max((double)sums[9], 1e-35));
            double dvel = std::sqrt(num / den);
            printf("    proposed metric: delta_vel = %.3e | delta_k = %.3e | delta_eps = %.3e | max = %.3e  (production max incl. DELTA2 = %.3e)  -> %s 1e-4\n",
                   dvel, dk, de, std::max(dvel, std::max(dk, de)), prod, std::max(dvel, std::max(dk, de)) < 1e-4 ? "PASSES" : "fails");
        }
        {
            auto pu = rd(rans.previous_iterate_u, nu), pv = rd(rans.previous_iterate_v, (size_t)N * (N + 1) * N), v3 = rd(rans.v_system.field.data, (size_t)N * (N + 1) * N);
            double dU = l2d(u2, pu), dV = l2d(v3, pv);
            printf("    abs changes: ||dU|| %.3e (rel %.2e)  ||dV|| %.3e (rel %.2e)  ||dW|| %.3e (rel %.2e) | ||dW||/||dU|| %.3f | float floor estimate 2^-24*||U|| = %.3e\n",
                   dU, dU / l2(u2), dV, dV / l2(v3), l2d(w3, prevw), d_cpu, l2d(w3, prevw) / dU, std::ldexp(1.0, -24) * l2(u2));
            if (it == 1 || it == ITERS)
            {
                // structure: u z-profile at a mid column, and w z-(anti)symmetry  (Ekman-type secondary flow is antisymmetric about mid-depth)
                int x = N / 4, y = N / 2; auto uat = [&](int z) { return u2[(size_t)z * (N + 1) * N + (size_t)y * (N + 1) + x]; };
                printf("    u(x=N/4,y=N/2,z): z=0 %.5g  z=1 %.5g  z=N/4 %.5g  z=N/2 %.5g  z=N-1 %.5g\n", uat(0), uat(1), uat(N / 4), uat(N / 2), uat(N - 1));
                double sym = 0, asym = 0;
                for (int z = 0; z <= N; z++) for (int yy = 0; yy < N; yy++) for (int xx = 0; xx < N; xx++)
                { double a = w3[(size_t)z * N * N + (size_t)yy * N + xx], b = w3[(size_t)(N - z) * N * N + (size_t)yy * N + xx]; sym += 0.25 * (a + b) * (a + b); asym += 0.25 * (a - b) * (a - b); }
                printf("    w energy: z-antisymmetric part %.3e, z-symmetric part %.3e\n", asym, sym);
            }
        }
        if (it == 1 || it == ITERS)
        {
            // where does w come from? compare the production projection with the EXACT pressure projection of the same divergence
            auto pe = exact_pressure(divv);
            double ecr = 0; { std::vector<float> pf(pe.begin(), pe.end()); double a, b2; zvar(pf, ecr, a, b2); }
            // w implied by projectionZ with exact p: w_exact = w1 - dt/rho * dp/dz (interior faces), walls forced 0
            double h = DOMAIN / N, mw = 0, mw_prod_check = 0;
            for (int z = 1; z < N; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++)
            {
                size_t f = (size_t)z * N * N + (size_t)y * N + x, cb = f, ct = f - (size_t)N * N;
                double we = w1[f] - DT / RHO * (pe[cb] - pe[ct]) / h; mw = std::max(mw, std::fabs(we));
                double wp = w1[f] - DT / RHO * ((double)p[cb] - p[ct]) / h; mw_prod_check = std::max(mw_prod_check, std::fabs(wp - w2[f]));
            }
            printf("    it %d: exact-pressure z-column range %.3e (production %.3e) -> max|w| with EXACT projection = %.3e (production %.3e)\n", it, ecr, cr, mw, maxabs(w2));
            printf("    it %d: projectionZ reproduced on CPU from production p: max mismatch %.3e (kernel check)\n", it, mw_prod_check);
        }
    }
    printf("glGetError 0x%x\n", glGetError());
    glfwTerminate();
}
