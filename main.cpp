#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include "shader_manager/shader_manager.hpp"
#include "Renderer/visualization.hpp"
#include "orchestrator/orchestrator.hpp"

//---------------------------------------------------------NOTES------------------------------------------------
//------------------------------------------------------------------------------advection kernel-

// -------------------------------------------------------------------------FLUX CALCULATION
//
// Flux = (normal face velocity) × (reconstructed transported quantity)
//
// vx equation:
//   East/West : Flux = vx_face × reconstructed_vx
//   North/South: Flux = vy_face × reconstructed_vx
//
// vy equation:
//   East/West : Flux = vx_face × reconstructed_vy
//   North/South: Flux = vy_face × reconstructed_vy
//
// The normal face velocity is obtained either:
//   - directly from the MAC face (if already located there), or
//   - by interpolation from the opposite staggered component.
//

//-------------------------------------------------------------------------SSPRK2 note
// 1) The vx_grid_orignal_SSBO and vy_grid_orignal_SSBO are the SSBO that will store the data of the orignal step of time before any process is done.
// 2) The vx_grid_SSBO and vy_grid_SSBO will store the data for the first step which will then be passed into the second and then that is used as the blending in the final blend kernel.

//-----------------------------------------------------------------------CONSTANTS------------------------------------------------------------

//---------------------basic simulation constants
const float WIDTH = 200;
const float HEIGHT = 200;
const int DEPTH = 200;
const int screen_height = 800;
const int screen_width = 800;
int cells_x = 100;
int cells_y = 100;  
int cells_z = 100;
float density = 1.225; // constant as incompressible     
float injection_strength = 2.5;
float injection_radius = 80.0f; // pixels; the energy_injection kernel needs a radius or it never injects
float dynamic_viscosity = 1.8e-5f;;
float cell_side_x = (float)WIDTH / (float)cells_x; 
float cell_side_y = (float)HEIGHT / (float)cells_y;
float cell_side_z = (float)DEPTH / (float)cells_z;


//---------------------time step constants (the CFL step limits are derived inside the Orchestrator)
float dt_max = 1.0f / 30.0f; // upper bound on the time step when the fluid is slow / at rest

//---------------------renderer constants
float FOV = 60;

float camera_x = -0.25f * WIDTH;
float camera_y = 0.5f * HEIGHT;
float camera_z = 1.5f * DEPTH;
float camera_yaw = -50;
float camera_pitch = 0;

float raymarch_ds = 0.75 * std::min(std::min(cell_side_x, cell_side_y), cell_side_z);

float light_x = 0.35f * WIDTH;         
float light_y = 0.0f * HEIGHT;         
float light_z = 1.15f * DEPTH;         
float light_intensity = std::pow(10, 6.15);
float extinction_via_scattering = 0.001;     
float extinction_total = 0.002;           //extinction_by_absorbstion is the diff of them, we don't have it as different as In main math it doesn't matter for us.
float light_march = 0.5f * std::min(std::min(cell_side_x, cell_side_y), cell_side_z);
float mean_cosine = 0.25;


// offline simulation constants
float frames_per_sec_offline_sim = 30.0;    
float simulation_needed_time = 10.0;
float next_save_time = 0.0;             //initialization

int slice_id = cells_z / 2;

//---------------------------------------------------------------------RENDERER STATE-----------------------------------------------------------
// Everything below is presentation only. The renderer reads simulation buffers and
// never writes to them.

int  g_fb_w = 0;
int  g_fb_h = 0;
bool g_framebuffer_resized = false;

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        return;         // minimised; ignore until it comes back
    }

    g_fb_w = width;
    g_fb_h = height;
    g_framebuffer_resized = true;
}

// Free-flight camera.
//   W / A / S / D : forward, left, back, right (relative to where you are looking)
//   E / Q         : straight up / straight down along world up
//   RIGHT MOUSE   : hold and drag to look around
//   LEFT SHIFT    : move faster
// Mouse look is on right-drag rather than a captured cursor so the cursor stays usable
// for the mouse-driven energy injection whenever that gets switched back on.

enum class MODE_OF_SIM
{
    EXPLICIT_SIM,
    RANS
};

int main()
{
    //-------------------------------------------window + GL context  (must exist BEFORE any gl* call)

    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    // Offline capture locks the framebuffer geometry: the ffmpeg process is started with
    // one fixed -video_size and the readback buffer is allocated once, so a mid-render
    // resize would desynchronise both and corrupt every frame from that point on.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // kept at WIDTH x HEIGHT: energy_injection.comp maps raw mouse pixels straight into
    // domain coordinates, so resizing the window would silently move the injection point
    GLFWwindow *window = glfwCreateWindow(screen_width, screen_height, "3D FVM solver", nullptr, nullptr);

    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);

    glfwShowWindow(window);

    glfwFocusWindow(window);

    gladLoadGL();

    glfwPollEvents();

    int fb_w, fb_h; 
    glfwGetFramebufferSize(window, &fb_w, &fb_h);

    // declare camera and it's constants initialization
    camera Camera(FOV, camera_x, camera_y, camera_z, camera_yaw, camera_pitch, fb_w, fb_h);

    Camera.camera_controls(window, Camera, dt_max);         // axis initialization of camera

    renderer_functions render(cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z, WIDTH, HEIGHT, DEPTH, extinction_total, extinction_via_scattering, mean_cosine, fb_w, fb_h, Camera);

    //LIGHT initialization
    light Light(light_x, light_y, light_z, cells_x, cells_y, cells_z, light_intensity, extinction_total, WIDTH, HEIGHT, DEPTH, light_march, render);

    g_fb_w = fb_w;
    g_fb_h = fb_h;

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    render.create_fullscreen_quad();   // we need a full screen quad to sample from the texture to get it to display

    glfwPollEvents();   // shader compilation is done; keep the OS message queue fresh while buffers allocate next
    

    //------------------------------------static uniform upload (values that never change; set once, never again)

    glfwPollEvents();   // buffer allocation is done; one more pump before the main loop takes over

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);             // no transparency anywhere in the renderer

    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);


    //-------------------------------------------------------------------------MODE SELECTION 
    int explicit_sim = int(MODE_OF_SIM::EXPLICIT_SIM);
    int rans_mode = int(MODE_OF_SIM::RANS);

    std::cout << "Please eneter your desired mode of fluid simulation, (explicit DNS = " << explicit_sim << " / RANS = " << rans_mode << ") : ";
    
    int mode;
    std::cin >> mode;

    //--------------------------------------------------------------------------FPS COUNTER STATE (real frames, not substeps)
    int fps_frame_count = 0;
    auto fps_last_time = std::chrono::steady_clock::now();
    double total_time = 0.0;
    auto real_start = std::chrono::high_resolution_clock::now();

    //----------------------------------------------------------------------offline rendering alloting
    std::vector <uint8_t> pixels(fb_w * fb_h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);       // for optimization and compatibility

    // -video_size and -framerate are built from the same variables the renderer uses, so
    // they can't drift out of sync with the actual capture.
    std::string ffmpeg_cmd =
        R"(".\Dependencies\ffmpeg.exe" -y -f rawvideo -pixel_format rgb24 -video_size )"
        + std::to_string(fb_w) + "x" + std::to_string(fb_h)
        + " -framerate " + std::to_string(static_cast<int>(frames_per_sec_offline_sim))
        + R"( -i pipe:0 -vf vflip -c:v libx264 -pix_fmt yuv420p .\output.mp4)";

    std::cout << "ffmpeg capture: " << fb_w << "x" << fb_h
              << " @ " << static_cast<int>(frames_per_sec_offline_sim) << " fps" << std::endl;
         
    
    FILE* ffmpeg = _popen(ffmpeg_cmd.c_str(), "wb");

    if (!ffmpeg)
    {
        std::cerr << "Failed to launch ffmpeg\n";
        return 1;
    }
    else if (ffmpeg)
    {
        std::cout << "FFmpeg pipe opened successfully.\n";
    }


    ShaderManager shadermanager;

    //------------------------------------------------------------EXPLICIT SOLVER (see orchestrator/)
    configuration sim_config;
    sim_config.cells_x = cells_x;
    sim_config.cells_y = cells_y;
    sim_config.cells_z = cells_z;
    sim_config.cell_side_x = cell_side_x;
    sim_config.cell_side_y = cell_side_y;
    sim_config.cell_side_z = cell_side_z;
    sim_config.WIDTH = WIDTH;
    sim_config.HEIGHT = HEIGHT;
    sim_config.DEPTH = DEPTH;
    sim_config.frame_buffer_w = fb_w;
    sim_config.frame_buffer_h = fb_h;
    sim_config.density = density;
    sim_config.dynamic_viscosity = dynamic_viscosity;
    sim_config.pressure_coarsest_grid = 8;
    sim_config.dt_max = dt_max;
    sim_config.simulation_needed_time = simulation_needed_time;
    sim_config.frames_per_sec_offline_sim = frames_per_sec_offline_sim;
    sim_config.temporal_scheme = TemporalScheme::BDF2;
    sim_config.advection_scheme = ConvectionScheme::SECOND_ORDER_DEFERRED;      // UPWIND or SECOND_ORDER_DEFERRED
    sim_config.flux_limiter = FluxLimiter::VAN_LEER;                            // NONE (unlimited Fromm), MINMOD, VAN_LEER, SUPERBEE

    Orchestrator orchestrator(sim_config, shadermanager);

    //------------------------------------------------------------PRESENTATION HOOKS
    // Everything below is rendering / camera / event handling / video capture. The solver loop only calls
    // back into it; it never depends on it.
    explicit_solver_hooks hooks;

    hooks.begin_frame = [&]() -> bool
    {
        if (glfwWindowShouldClose(window))
            return false;

        glfwPollEvents();

        //------------------------------------------------ camera (real wall-clock dt)
        static auto camera_prev_time = std::chrono::steady_clock::now();
        auto camera_now = std::chrono::steady_clock::now();
        float camera_frame_dt = std::chrono::duration<float>(camera_now - camera_prev_time).count();
        camera_prev_time = camera_now;

        // a long solver frame shouldn't teleport the camera across the domain
        if (camera_frame_dt > 0.1f) camera_frame_dt = 0.1f;

        Camera.camera_controls(window, Camera, camera_frame_dt);

        return true;
    };

    // smoke is a passive tracer used only by the renderer: it is saved / advected / blended alongside the solver
    hooks.substep_begin = [&]()
    {
        glBindBuffer(GL_COPY_READ_BUFFER, render.smokeMatter_SSBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, render.smokeMatter_orignal_SSBO);

        glCopyBufferSubData(
            GL_COPY_READ_BUFFER,
            GL_COPY_WRITE_BUFFER,
            0,
            0,
            cells_x * cells_y * cells_z * sizeof(float)
        );

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    hooks.stage1_done = [&](explicit_simulation &world, float stable_dt)
    {
        render.smoke_advection_dispatch(shadermanager, world.vx_grid_SSBO, world.vy_grid_SSBO, world.vz_grid_SSBO, stable_dt);
    };

    //---------------------------------------------------SMOKE MATTER ADVECT-----------------------------------------------------
    hooks.substep_end = [&](explicit_simulation &world, float stable_dt)
    {
        render.smoke_advection_dispatch(shadermanager, world.vx_grid_SSBO, world.vy_grid_SSBO, world.vz_grid_SSBO, stable_dt);

        render.final_smoke_blend(shadermanager);
    };

    hooks.frame_end = [&](explicit_simulation &world, debuggers &debug, double total_time)
    {
        double real_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - real_start).count();
        std::cout  << total_time << " and real time is " << real_time << std::endl;        

        // //-------------------------------------------------Offline video simulation generation-------------------------
        // (the camera basis was already rebuilt by camera_controls at the top of the frame)
        if(total_time >= next_save_time)
        {
        //-------------------------------------------------------SMOKE MATTER BASED RENDERING------------------------------------
            render.dual_mode_render(0, shadermanager, debug, slice_id, fb_w, fb_h, raymarch_ds, Light);
            //hard coded to mode one for now
            glFinish();

            glReadPixels(
                0,
                0,
                fb_w,
                fb_h,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                pixels.data());

            fwrite(
                pixels.data(),
                1,
                pixels.size(),
                ffmpeg);

            next_save_time += (1.0 / frames_per_sec_offline_sim);
            
        }
    };


         
    if (mode == int(MODE_OF_SIM::EXPLICIT_SIM))
    {
        orchestrator.run_explicit_simulation(shadermanager, hooks);
    }
    else if (mode == int(MODE_OF_SIM::RANS))
    {  
        //ALL THESE VALUES HAVE BEEN TAKEN FOR INITIALIZATION
        float turbulence_length_scale;
        float reference_velocity;
        float turbulence_intensity;
        int rans_model_mode;

        std::cout << "ALL VALUES ADDED HERE AFTER, PUT HIGH IMPACT ON THE INITIALIZATION OF THE SIMULATION, SO ENTER VALUES ACCURATELY" << std::endl;

        std::cout << "Please eneter the turbulence length scale: ";
        std::cin >> turbulence_length_scale;

        std::cout << "Please eneter the reference velocity: ";
        std::cin >> reference_velocity;

        std::cout << "Please eneter the turbulence intensity: ";
        std::cin >> turbulence_intensity;

        std::cout << "Please eneter the mode (0-> eps, 1 -> omega, 2 -> omega_SST): ";
        std::cin >> rans_model_mode;

        bool is_omega_based;
        bool is_SST;

        if(rans_model_mode == 0)
        {
            is_omega_based = false;
            is_SST = false;
        }

        if(rans_model_mode == 1)
        {
            is_omega_based = true;
            is_SST = false;
        }

        if(rans_model_mode == 2)
        {
            is_omega_based = true;
            is_SST = true;
        }

        int vx_MAC_total = (cells_x + 1) * cells_y * cells_z;
        int vy_MAC_total = cells_x * (cells_y + 1) * cells_z;
        int vz_MAC_total = cells_x * cells_y * (cells_z + 1);
        int CFL_max_elements_temp = std::max(vx_MAC_total, vy_MAC_total);
        int CFL_max_elements = std::max(CFL_max_elements_temp, vz_MAC_total);

        std::vector <GLuint> reduction_for_CFL_SSBOs_arr;
        {
            int n = CFL_max_elements;

            while (n > 1)
            {
                n = (n + 255) / 256;

                GLuint reduction_ssbo;
                glGenBuffers(1, &reduction_ssbo);

                glBindBuffer(GL_SHADER_STORAGE_BUFFER, reduction_ssbo);

                glBufferData(
                    GL_SHADER_STORAGE_BUFFER,
                    n * sizeof(float),
                    nullptr,
                    GL_DYNAMIC_COPY);

                reduction_for_CFL_SSBOs_arr.push_back(reduction_ssbo);
            }
        }

        //HERE WE WILL HAVE RANS ORCHESTRATION FUNCTION
        if(!is_omega_based)
        {
            orchestrator.run_rans_k_eps(9.8, reference_velocity, turbulence_intensity, turbulence_length_scale, fb_w, fb_h, reduction_for_CFL_SSBOs_arr, ffmpeg, render, Light);
        }
        else if(is_omega_based)
        {
            if(!is_SST)
                orchestrator.run_rans_k_omega(9.8, reference_velocity, turbulence_intensity, turbulence_length_scale, fb_w, fb_h, reduction_for_CFL_SSBOs_arr, ffmpeg, render, Light);
            else if(is_SST)
                orchestrator.run_rans_k_omega_SST(9.8, reference_velocity, turbulence_intensity, turbulence_length_scale, fb_w, fb_h, reduction_for_CFL_SSBOs_arr, ffmpeg, render, Light);
                
        }
    }

    _pclose(ffmpeg);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
    
}
