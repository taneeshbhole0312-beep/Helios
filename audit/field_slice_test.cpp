// Test of renderer_functions::render_normalized_field_slice against an independent CPU computation.
// For each topology: random field with NEGATIVE and positive values, one planted extreme (negative) value, then compare
// the GPU image (ray-march texture) texel by texel with the CPU: |value| (face pairs averaged), / max|field|, heat map.
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#define private public
#include "Renderer/visualization.hpp"
#undef private

static void heat(float t, float c[3])
{
    t = std::min(std::max(t, 0.f), 1.f);
    const float s[5][3] = { {0.02f,0.02f,0.30f}, {0.00f,0.75f,0.95f}, {0.10f,0.85f,0.20f}, {0.98f,0.90f,0.10f}, {0.90f,0.05f,0.05f} };
    int k = std::min((int)(t / 0.25f), 3); float f = (t - 0.25f * k) / 0.25f;
    for (int i = 0; i < 3; i++) c[i] = s[k][i] + (s[k + 1][i] - s[k][i]) * f;
}

int main()
{
    glfwInit(); glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(256, 192, "t", nullptr, nullptr); glfwMakeContextCurrent(win); gladLoadGL();
    ShaderManager sm;
    int sizes[3][3] = { {20, 24, 16}, {64, 64, 64}, {8, 8, 8} };
    const int W = 256, H = 192;
    int fails = 0;

    for (auto &sz : sizes)
    {
        int cx = sz[0], cy = sz[1], cz = sz[2];
        camera cam(60.f, 0, 0, 0, 0, 0, W, H);
        renderer_functions r(cx, cy, cz, 1, 1, 1, (float)cx, (float)cy, (float)cz, 0.002f, 0.001f, 0.25f, W, H, cam);
        r.create_fullscreen_quad();
        const fieldTopology tops[4] = { fieldTopology::CELL_CENTERED, fieldTopology::XFACE_CENTERED, fieldTopology::YFACE_CENTERED, fieldTopology::ZFACE_CENTERED };
        const char *names[4] = { "CELL", "XFACE", "YFACE", "ZFACE" };

        for (int t = 0; t < 4; t++)
        {
            int nx = cx + (t == 1), ny = cy + (t == 2), nz = cz + (t == 3);
            size_t n = (size_t)nx * ny * nz;
            std::mt19937 g(100 + t); std::uniform_real_distribution<float> U(-3.f, 3.f);
            std::vector<float> f(n); for (auto &v : f) v = U(g);
            // planted extreme NEGATIVE value: the max magnitude must come from it
            size_t where = n / 3 + 7; f[where] = -250.f;

            GLuint ssbo; glGenBuffers(1, &ssbo); glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo); glBufferData(GL_SHADER_STORAGE_BUFFER, n * 4, f.data(), GL_DYNAMIC_COPY);
            int slice = cz / 2;
            r.render_normalized_field_slice(sm, ssbo, tops[t], slice, W, H);
            glFinish();

            // GPU result: the ray-march texture, and the default framebuffer
            std::vector<float> tex((size_t)W * H * 4); glBindTexture(GL_TEXTURE_2D, r.image_texture); glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, tex.data());
            std::vector<unsigned char> fbpx((size_t)W * H * 3); glPixelStorei(GL_PACK_ALIGNMENT, 1); glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, fbpx.data());

            // CPU reference
            double m = 0; for (float v : f) m = std::max(m, (double)std::fabs(v));
            double maxdiff = 0, maxdiff_fb = 0; size_t nonred = 0;
            for (int py = 0; py < H; py++) for (int px = 0; px < W; px++)
            {
                int x = std::min(std::max((int)((px + 0.5f) / W * cx), 0), cx - 1), y = std::min(std::max((int)((py + 0.5f) / H * cy), 0), cy - 1), z = slice;
                auto at = [&](int xx, int yy, int zz) { return std::fabs((double)f[((size_t)zz * ny + yy) * nx + xx]); };
                double val = (t == 0) ? at(x, y, z) : (t == 1) ? 0.5 * (at(x, y, z) + at(x + 1, y, z)) : (t == 2) ? 0.5 * (at(x, y, z) + at(x, y + 1, z)) : 0.5 * (at(x, y, z) + at(x, y, z + 1));
                float c[3]; heat((float)(val / m), c);
                for (int k = 0; k < 3; k++)
                {
                    maxdiff = std::max(maxdiff, (double)std::fabs(c[k] - tex[((size_t)py * W + px) * 4 + k]));
                    // framebuffer: bottom-up rows; the quad maps texture v=0 to the bottom
                    maxdiff_fb = std::max(maxdiff_fb, (double)std::fabs(c[k] * 255.0 - fbpx[((size_t)py * W + px) * 3 + k]));
                }
                if (tex[((size_t)py * W + px) * 4] > 0.5f && tex[((size_t)py * W + px) * 4 + 1] < 0.2f) nonred++;
            }
            bool ok = maxdiff < 2e-6;
            if (!ok) fails++;
            printf("%2dx%2dx%2d %-5s: max|field| (planted -250) -> CPU %.1f | GPU texel vs CPU max diff %.2e | framebuffer vs CPU(0..255) max diff %.2f | red-ish texels %zu  %s\n", cx, cy, cz, names[t], m, maxdiff, maxdiff_fb, nonred, ok ? "PASS" : "FAIL");
        }
    }

    // zero field: must not divide by zero
    { camera cam(60.f, 0, 0, 0, 0, 0, W, H); renderer_functions r(8, 8, 8, 1, 1, 1, 8, 8, 8, 0.002f, 0.001f, 0.25f, W, H, cam); r.create_fullscreen_quad();
      std::vector<float> z(512, 0.f); GLuint s; glGenBuffers(1, &s); glBindBuffer(GL_SHADER_STORAGE_BUFFER, s); glBufferData(GL_SHADER_STORAGE_BUFFER, 2048, z.data(), GL_DYNAMIC_COPY);
      r.render_normalized_field_slice(sm, s, fieldTopology::CELL_CENTERED, 4, W, H); glFinish();
      std::vector<float> tex((size_t)W * H * 4); glBindTexture(GL_TEXTURE_2D, r.image_texture); glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, tex.data());
      bool finite = true; for (float v : tex) if (!std::isfinite(v)) finite = false;
      printf("zero field: all texels finite %s, texel0 = (%.2f,%.2f,%.2f)  %s\n", finite ? "yes" : "NO", tex[0], tex[1], tex[2], finite && std::fabs(tex[2] - 0.30f) < 1e-5 ? "PASS" : "FAIL"); if (!finite) fails++; }

    printf("glGetError 0x%x | failures %d\n", glGetError(), fails);
    glfwTerminate();
    return fails;
}
