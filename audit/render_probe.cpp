// Why is the RANS video colourless?  Replays the renderer setup EXACTLY as Orchestrator::run_rans_k_eps does it,
// then adds the two things main.cpp does for its own renderer, one at a time.
//   A: orchestrator-style      (renderer + camera + light constructed; NO create_fullscreen_quad, NO camera update)
//   B: A + create_fullscreen_quad()
//   C: B + camera_controls()/camera_update  (what main.cpp does for its camera, main.cpp:188-190, 202)
// Uses the unmodified production renderer / shaders.  Reports the glReadPixels result of the DEFAULT framebuffer.
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>
#define private public
#include "Renderer/visualization.hpp"
#undef private

static void stats(const char *tag, const std::vector<uint8_t> &p, int w, int h)
{
    int mn[3] = {255, 255, 255}, mx[3] = {0, 0, 0}; double mean[3] = {0, 0, 0}; size_t distinct = 0;
    std::vector<bool> seen(1 << 24, false);
    for (size_t i = 0; i < (size_t)w * h; i++)
    {
        for (int c = 0; c < 3; c++) { int v = p[i * 3 + c]; mn[c] = std::min(mn[c], v); mx[c] = std::max(mx[c], v); mean[c] += v; }
        size_t key = (p[i * 3] << 16) | (p[i * 3 + 1] << 8) | p[i * 3 + 2]; if (!seen[key]) { seen[key] = true; distinct++; }
    }
    size_t n = (size_t)w * h;
    printf("  %-34s RGB min (%3d,%3d,%3d) max (%3d,%3d,%3d) mean (%.2f,%.2f,%.2f) distinct colours %zu\n", tag, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], mean[0] / n, mean[1] / n, mean[2] / n, distinct);
}

int main(int argc, char **argv)
{
    float D = argc > 1 ? (float)atof(argv[1]) : 200.f;       // domain edge (user ran 200; main.cpp now says 600)
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(800, 800, "render", nullptr, nullptr);       // same window hints as main.cpp (no core-profile request)
    glfwMakeContextCurrent(win); glfwShowWindow(win); gladLoadGL(); glfwPollEvents();
    int fb_w, fb_h; glfwGetFramebufferSize(win, &fb_w, &fb_h);
    GLint prof = 0; glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &prof);
    printf("GL %s | profile mask 0x%x (%s) | framebuffer %dx%d | domain %g\n", (const char *)glGetString(GL_VERSION), prof, (prof & GL_CONTEXT_CORE_PROFILE_BIT) ? "CORE" : "compatibility", fb_w, fb_h, D);
    glDisable(GL_DEPTH_TEST); glDisable(GL_BLEND); glClearColor(0.02f, 0.02f, 0.05f, 1.0f);       // main.cpp:211-214
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    ShaderManager sm;

    const int N = 64; float cs = D / N, W = D, H = D, De = D;
    float camx = -0.75f * W, camy = 0.5f * H, camz = 1.15f * De;                                    // orchestrator.cpp:453-457
    float raymarch_ds = 0.75f * cs, light_march = 0.5f * cs;
    float lx = 0.35f * W, ly = 0.0f, lz = 1.15f * De;

    // ---- construct the objects the way run_rans_k_eps does (orchestrator.cpp:504-508)
    camera cam(60.f, camx, camy, camz, -30.f, 0.f, fb_w, fb_h);
    renderer_functions renderer(N, N, N, cs, cs, cs, W, H, De, 0.002f, 0.001f, 0.25f, fb_w, fb_h, cam);
    light Light(lx, ly, lz, N, N, N, 1e6f, 0.002f, W, H, De, light_march, renderer);
    debuggers *nodebug = nullptr; (void)nodebug;

    printf("\ncamera basis straight after construction (what the orchestrator's camera has when it renders):\n");
    printf("  forward (%g, %g, %g)  right (%g, %g, %g)  up (%g, %g, %g)\n", cam.forward[0], cam.forward[1], cam.forward[2], cam.right[0], cam.right[1], cam.right[2], cam.up[0], cam.up[1], cam.up[2]);
    printf("  renderer.quadVAO = %u (0 = never created)\n", renderer.quadVAO);

    std::vector<uint8_t> px((size_t)fb_w * fb_h * 3);
    auto frame = [&](const char *tag)
    {
        Light.dispatchLightTransmittance(sm);
        renderer.ray_march_dispatch(sm, fb_w, fb_h, raymarch_ds, Light);
        renderer.draw_fullscreen_texture(sm, renderer.image_texture, fb_w, fb_h);
        glFinish();
        glReadPixels(0, 0, fb_w, fb_h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
        stats(tag, px, fb_w, fb_h);
        // also the intermediate ray-march texture, centre pixel + a few
        std::vector<float> tex((size_t)fb_w * fb_h * 4);
        glBindTexture(GL_TEXTURE_2D, renderer.image_texture); glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, tex.data());
        double rsum = 0, asum = 0; double rmax = 0; size_t nan = 0; for (size_t i = 0; i < (size_t)fb_w * fb_h; i++) { float r = tex[i * 4], a = tex[i * 4 + 3]; if (!std::isfinite(r) || !std::isfinite(a)) nan++; else { rsum += r; asum += a; rmax = std::max(rmax, (double)r); } }
        printf("  %-34s ray-march texture: NaN/Inf texels %zu, sum(rgb) %.4g, max(rgb) %.4g, mean alpha %.4g\n", "", nan, rsum, rmax, asum / ((double)fb_w * fb_h));
    };

    printf("\n=== A: orchestrator-style (no create_fullscreen_quad, no camera update) ===\n");
    frame("A");
    printf("\n=== B: A + create_fullscreen_quad() only (camera basis still uninitialised) ===\n");
    renderer.create_fullscreen_quad();
    glGetError();
    frame("B");
    // reset: new renderer state is not needed; the quad exists from here on.  Now isolate the camera on its own with the quad present.
    printf("\n=== C: B + camera_controls() (builds forward/right/up)  == both fixes ===\n");
    cam.camera_controls(win, cam, 0.016f);
    printf("  forward (%g, %g, %g)  right (%g, %g, %g)  up (%g, %g, %g)\n", cam.forward[0], cam.forward[1], cam.forward[2], cam.right[0], cam.right[1], cam.right[2], cam.up[0], cam.up[1], cam.up[2]);
    frame("C");
    printf("\n=== E: camera fixed, quad destroyed (VAO 0 again) -> isolates the missing quad ===\n");
    renderer.quadVAO = 0;
    glGetError();
    frame("E");
    printf("  glGetError after the VAO-0 draw: 0x%x\n", glGetError());
    printf("\n=== D: C but with the camera moved for a 600-wide domain if D != 600 (info only) ===\n");
    printf("glGetError 0x%x\n", glGetError());
    glfwTerminate();
}
