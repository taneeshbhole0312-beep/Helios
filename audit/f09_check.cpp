// F-09 verification: is rho*(eps/k)*V present in the K coarse operators, and is it built from CURRENT restricted k/eps?
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#define private public
#include "Turbulence_models/RANS.hpp"
#include "Turbulence_models/k_TurbulenceModels.hpp"
#undef private
static std::vector<float> rd(GLuint b, size_t n) { glMemoryBarrier(GL_ALL_BARRIER_BITS); glFinish(); std::vector<float> v(n); glBindBuffer(GL_SHADER_STORAGE_BUFFER, b); glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, n * 4, v.data()); return v; }
int main()
{
    glfwInit(); glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(64, 64, "f09", nullptr, nullptr); glfwMakeContextCurrent(win); gladLoadGL();
    ShaderManager sm; const int N = 32; const float D = 200.f, rho = 1.225f, mu = 1.8e-5f, dt = 1.f / 60.f;
    RANS rans(N, N, N, 8, sm, D, D, D, rho, mu); k_TurbulenceModels km(rans, 8, false); k_TurbulenceModels::epsilon em(km, 8);
    rans.initialize_velocity_fields(); em.initialize_k_epsilon_RANS_fields(10.f, 0.05f, 1.f, 9.8f);   // k0 = 0.375, eps0 uniform
    // NOTE: k/eps are uniform non-zero on the fine grid
    for (int pass = 1; pass <= 2; pass++)
    {
        rans.construct_k_epsilon_system(km, em, dt, 8, 9.8f);
        glFinish();
        auto &op = km.KSystem.multigrid_levels_coeffiecients_arr[0]; size_t n = (size_t)op.level_x_elements * op.level_y_elements * op.level_z_elements;
        auto aP_prod = rd(op.aP, n);
        auto kc = rd(km.k_hierarchy[0], n), ec = rd(em.eps_hierarchy[0], n);                   // state AFTER the whole construct
        // rebuild the K coarse operator from scratch with hierarchies now guaranteed current (restricted from this iterate)
        em.restrict_k_eps(km); glMemoryBarrier(GL_ALL_BARRIER_BITS);
        km.construct_K_equation_hierarchy(dt, false, em.eps_hierarchy, 8); glFinish();
        auto aP_ref = rd(op.aP, n);
        auto kc2 = rd(km.k_hierarchy[0], n), ec2 = rd(em.eps_hierarchy[0], n);
        int C = op.level_x_elements; double Vc = std::pow(D / C, 3); double maxrel = 0, maxabs = 0; int cnt = 0;
        double exp_destr = 0, got_destr = 0;
        for (int z = 1; z < C - 1; z++) for (int y = 1; y < C - 1; y++) for (int x = 1; x < C - 1; x++)
        {
            size_t i = (size_t)z * C * C + (size_t)y * C + x; double d = aP_ref[i] - aP_prod[i];
            maxabs = std::max(maxabs, std::fabs(d)); maxrel = std::max(maxrel, std::fabs(d) / aP_ref[i]); cnt++;
            if (i == (size_t)(C / 2) * C * C + (size_t)(C / 2) * C + C / 2) { exp_destr = rho * ec2[i] / (kc2[i] + 1e-5) * Vc; got_destr = aP_prod[i]; }
        }
        size_t ic = (size_t)(C / 2) * C * C + (size_t)(C / 2) * C + C / 2;
        printf("pass %d: coarse L1 %d^3, %d interior rows\n", pass, C, cnt);
        printf("   hierarchy state seen by the K build : k_c = %.6g, eps_c = %.6g at centre (current restricted: k_c = %.6g, eps_c = %.6g)\n", kc[ic], ec[ic], kc2[ic], ec2[ic]);
        printf("   expected destruction rho*eps_c/(k_c+1e-5)*Vc at centre = %.6g\n", exp_destr);
        printf("   production aP vs aP rebuilt from current restricted fields: max |diff| = %.6g, max rel = %.3e\n", maxabs, maxrel);
        printf("   -> %s\n", maxrel < 1e-5 ? "consistent" : "STALE / INCONSISTENT");
        // before pass 2 emulate a solve: change the fine k (x2) and eps (x3); pass 2's K hierarchy should then see the NEW values
        std::vector<float> kf((size_t)N * N * N, 0.75f), ef((size_t)N * N * N, 3.f * 0.0f + 100.f);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, km.KSystem.field.data); glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, kf.size() * 4, kf.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, em.eps_system.field.data); glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ef.size() * 4, ef.data());
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }
    glfwTerminate();
}
