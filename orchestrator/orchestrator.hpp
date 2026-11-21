#pragma once

#include <functional>

#include "../shader_manager/shader_manager.hpp"
#include "../explicit_sim/explicit_sim.hpp"
#include "../Debuggers/debuggers.hpp"
#include "../Renderer/visualization.hpp"
#include "../configuration.hpp"

// Everything that is NOT the solver (rendering, camera, GLFW events, video capture, ...) plugs into
// the solver loop here. Every hook is optional: with no hooks the simulation just advances.
struct explicit_solver_hooks
{
    // start of every frame; return false to stop the run (e.g. the window was closed)
    std::function<bool()> begin_frame;

    // start of every substep, after the velocity state was saved for SSPRK2
    std::function<void()> substep_begin;

    // after SSPRK2 stage 1 (velocity projected)
    std::function<void(explicit_simulation &world, float stable_dt)> stage1_done;

    // after the final SSPRK2 blend
    std::function<void(explicit_simulation &world, float stable_dt)> substep_end;

    // end of every frame, after the simulated time was advanced
    std::function<void(explicit_simulation &world, debuggers &debug, double total_time)> frame_end;
};

class Orchestrator
{
    private:
        configuration config;
        ShaderManager &shadermanager;

    public:
        explicit Orchestrator(const configuration &config_in, ShaderManager &shadermanager);

        // Builds the explicit solver (explicit_simulation, incompressible_flow, pressure multigrid,
        // debuggers) and runs the SSPRK2 loop until simulation_needed_time or until begin_frame
        // returns false.
        void run_explicit_simulation(ShaderManager &shadermanager, const explicit_solver_hooks &hooks);

        void run_rans_k_eps(float spalding_wall_smoothness, float reference_velocity, float turbulence_intensity, float turbulent_len, int fb_w, int fb_h, std::vector <GLuint> reduction_for_CFL_SSBOs_arr, FILE* ffmpeg, renderer_functions &renderer, light &Light);   // TEMP: main.cpp's (initialised) renderer + light are passed in

        void run_rans_k_omega(float spalding_wall_smoothness, float reference_velocity, float turbulence_intensity, float turbulent_len, int fb_w, int fb_h, std::vector <GLuint> reduction_for_CFL_SSBOs_arr, FILE* ffmpeg, renderer_functions &renderer, light &Light);   // TEMP: main.cpp's (initialised) renderer + light are passed in

        void run_rans_k_omega_SST(float spalding_wall_smoothness, float reference_velocity, float turbulence_intensity, float turbulent_len, int fb_w, int fb_h, std::vector <GLuint> reduction_for_CFL_SSBOs_arr, FILE* ffmpeg, renderer_functions &renderer, light &Light);   // TEMP: main.cpp's (initialised) renderer + light are passed in
        
};
