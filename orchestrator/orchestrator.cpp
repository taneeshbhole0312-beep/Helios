#include "orchestrator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "../Incompressible_flow/incompressible_flow.hpp"
#include "../Turbulence_models/RANS.hpp"
#include "../Turbulence_models/k_TurbulenceModels.hpp"
#include "../Generic_multigrid/generic_multigrid.hpp"
#include "../Renderer/visualization.hpp"

Orchestrator::Orchestrator(const configuration &config_in, ShaderManager &shadermanager)
    : config(config_in),
      shadermanager(shadermanager)
{
}

void Orchestrator::run_explicit_simulation(ShaderManager &shadermanager, const explicit_solver_hooks &hooks)
{
    // short local names for the config, identical to the names the loop used inside main.cpp
    const int cells_x = config.cells_x;
    const int cells_y = config.cells_y;
    const int cells_z = config.cells_z;
    const float cell_side_x = config.cell_side_x;
    const float cell_side_y = config.cell_side_y;
    const float cell_side_z = config.cell_side_z;
    const float density = config.density;
    float dynamic_viscosity = config.dynamic_viscosity;
    const float dt_max = config.dt_max;

    //---------------------CFL constants
    const float max_step_per_frame_x = 1.0f * cell_side_x;
    const float max_step_per_frame_y = 1.0f * cell_side_y;
    const float max_step_per_frame_z = 1.0f * cell_side_z;

    //---------------------------------------------------------------- solver resources
    explicit_simulation world(config.cells_x, config.cells_y, config.cells_z, config.cell_side_x, config.cell_side_y, config.cell_side_z, config.WIDTH, config.HEIGHT, config.frame_buffer_w, config.frame_buffer_h, shadermanager);
    world.initialize();

    incompressible_flow incompresibility(cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z, shadermanager);
    // single source of truth for the pressure multigrid levels (declared before the solver, so it outlives it)
    MultigridHierarchy hierarchy(cells_x, cells_y, cells_z, config.WIDTH, config.HEIGHT, config.DEPTH, config.pressure_coarsest_grid);
    incompressible_flow::specialized_pressure_Multigrid pressure_solver_system(hierarchy, incompresibility);

    debuggers debug(incompresibility, shadermanager, cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z);

    float diffusive_limit_denominator = (dynamic_viscosity / density) * ((2.0 / (cell_side_x * cell_side_x)) + (2.0 / (cell_side_y * cell_side_y)) + (2.0 / (cell_side_z * cell_side_z)));


    float diffusive_limit = 0.5 * (1.0 / diffusive_limit_denominator);

    double total_time = 0.0;
    auto real_start = std::chrono::high_resolution_clock::now();

    //------------------------------------------------------------MAIN LOOP--------------------------------------------------------
    while (true)
    {
        if (hooks.begin_frame && !hooks.begin_frame())
            break;

        // 2. CFL condition: dt from the fastest face velocity (post-injection, so the impulse is accounted for)
        float max_vx = world.velocity_component_reduction(world.vx_grid_SSBO, (cells_x + 1) * cells_y * cells_z);
        float max_vy = world.velocity_component_reduction(world.vy_grid_SSBO, (cells_y + 1) * cells_x * cells_z);
        float max_vz = world.velocity_component_reduction(world.vz_grid_SSBO, (cells_z + 1) * cells_y * cells_x);

        float stable_dt_vx = dt_max;
        float stable_dt_vy = dt_max;
        float stable_dt_vz = dt_max;

        if (max_vx > 1e-6f)
        {
            stable_dt_vx = std::min(stable_dt_vx, max_step_per_frame_x / max_vx);
        }

        if (max_vy > 1e-6f)
        {
            stable_dt_vy = std::min(stable_dt_vy, max_step_per_frame_y / max_vy);
        }

        if (max_vz > 1e-6f)
        {
            stable_dt_vz = std::min(stable_dt_vz, max_step_per_frame_z / max_vz);
        }

        float stable_dt_temp = std::min(stable_dt_vy, stable_dt_vx);
        float stable_dt_advective = std::min(stable_dt_temp, stable_dt_vz);
        float stable_dt = std::min(stable_dt_advective, diffusive_limit);

        int iterations = std::ceil(dt_max / stable_dt);

        // std::cout << "iterations = " << iterations << std::endl;         //stable dt debugger

        for (int i=0; i < iterations; i++)
        {
        //COPY INTO ORIGNAL SSBO (temporary storage for first step)
        glBindBuffer(GL_COPY_READ_BUFFER, world.vx_grid_SSBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, world.vx_grid_orignal_SSBO);

        glCopyBufferSubData(
            GL_COPY_READ_BUFFER,
            GL_COPY_WRITE_BUFFER,
            0,
            0,
            (cells_x + 1) * cells_y * cells_z * sizeof(float)
        );

        glBindBuffer(GL_COPY_READ_BUFFER, world.vy_grid_SSBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, world.vy_grid_orignal_SSBO);

        glCopyBufferSubData(
            GL_COPY_READ_BUFFER,
            GL_COPY_WRITE_BUFFER,
            0,
            0,
            cells_x * (cells_y + 1) * cells_z * sizeof(float)
        );

        glBindBuffer(GL_COPY_READ_BUFFER, world.vz_grid_SSBO);
        glBindBuffer(GL_COPY_WRITE_BUFFER, world.vz_grid_orignal_SSBO);

        glCopyBufferSubData(
            GL_COPY_READ_BUFFER,
            GL_COPY_WRITE_BUFFER,
            0,
            0,
            cells_x * cells_y * (cells_z + 1) * sizeof(float)
        );


        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);



        if (hooks.substep_begin)
            hooks.substep_begin();



        //------------------------------------------------------------STAGE 1 SSPRK2-----------------------------------------------
        world.viscosity_cal(stable_dt, dynamic_viscosity, density);

        world.WENOZ_advect(stable_dt);        

        incompresibility.divergence_cal(world.vx_grid_SSBO, world.vy_grid_SSBO, world.vz_grid_SSBO);

        pressure_solver_system.solve_pressure_system(1e-6, 1e-7, 3, density, stable_dt);

        incompresibility.project_pressure(stable_dt, density, world.vx_grid_SSBO, world.vy_grid_SSBO, world.vz_grid_SSBO);


        if (hooks.stage1_done)
            hooks.stage1_done(world, stable_dt);

        //------------------------------------------------------STAGE 2 SSPRK2-----------------------------------------------------
        world.viscosity_cal(stable_dt, dynamic_viscosity, density);

        world.WENOZ_advect(stable_dt);

        incompresibility.divergence_cal(world.vx_grid_SSBO, world.vy_grid_SSBO, world.vz_grid_SSBO);

        pressure_solver_system.solve_pressure_system(1e-6, 1e-7, 3, density, stable_dt);

        incompresibility.project_pressure(stable_dt, density, world.vx_grid_SSBO, world.vy_grid_SSBO, world.vz_grid_SSBO);

        //---------------------------------------------------FINAL BLENDED STEP------------------------------------------------------

        world.SSPRK2_blend_step();

        if (hooks.substep_end)
            hooks.substep_end(world, stable_dt);

        }

        // debug.get_mean_absolute_divergence();

        //-------------------------------------------------------DEBUGGERS--------------------------------------------------

        // --------------------------------------------------kinetic energy checker
        
        // Distinguishes "energy is being CREATED" (a bug) from "energy is being
        // CONCENTRATED" (legitimate physics) when max|v| is observed climbing.
        
        // With rigid walls and no forcing, pressure does zero net work
        //     integral( u . grad p ) dV = surface integral( p u.n ) dS = 0   since u.n = 0 on every wall
        // so total KE can only ever DECREASE (viscosity) -- it can never rise.
        // max|v| on the other hand is a pointwise maximum and CAN rise while total KE
        // falls, simply by the flow concentrating its energy into a smaller region.
        
        //   KE monotonically decreasing  -> no energy bug; max|v| growth is concentration
        //   KE increasing                -> genuine spurious energy production somewhere
        
        // Set ke_check_enabled = false to silence. Costs three buffer readbacks, so it is
        // sampled every ke_check_interval-th substep rather than every one.


        // {
        //     static const bool ke_check_enabled  = true;
        //     static const int  ke_check_interval = 25;

        //     static int    ke_step     = 0;
        //     static double ke_previous = -1.0;

        //     ke_step++;

        //     if (ke_check_enabled && (ke_step % ke_check_interval) == 1)
        //     {
        //         const int nx = (cells_x + 1) * cells_y * cells_z;
        //         const int ny = cells_x * (cells_y + 1) * cells_z;
        //         const int nz = cells_x * cells_y * (cells_z + 1);

        //         static std::vector<float> ke_vx, ke_vy, ke_vz;
        //         ke_vx.resize(nx);
        //         ke_vy.resize(ny);
        //         ke_vz.resize(nz);

        //         glBindBuffer(GL_SHADER_STORAGE_BUFFER, vx_grid_SSBO);
        //         glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, nx * sizeof(float), ke_vx.data());

        //         glBindBuffer(GL_SHADER_STORAGE_BUFFER, vy_grid_SSBO);
        //         glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ny * sizeof(float), ke_vy.data());

        //         glBindBuffer(GL_SHADER_STORAGE_BUFFER, vz_grid_SSBO);
        //         glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, nz * sizeof(float), ke_vz.data());

        //         // accumulate in double: at 32^3 the sum spans many orders of magnitude
        //         double sum_sq = 0.0;
        //         double peak   = 0.0;

        //         for (int i = 0; i < nx; i++)
        //         {
        //             double v = ke_vx[i];
        //             sum_sq += v * v;
        //             peak = std::max(peak, std::abs(v));
        //         }
        //         for (int i = 0; i < ny; i++)
        //         {
        //             double v = ke_vy[i];
        //             sum_sq += v * v;
        //             peak = std::max(peak, std::abs(v));
        //         }
        //         for (int i = 0; i < nz; i++)
        //         {
        //             double v = ke_vz[i];
        //             sum_sq += v * v;
        //             peak = std::max(peak, std::abs(v));
        //         }

        //         double cell_volume = (double)cell_side_x * cell_side_y * cell_side_z;
        //         double kinetic_energy = 0.5 * sum_sq * cell_volume;

        //         const char *trend = "  (first sample)";
        //         double percent_change = 0.0;

        //         if (ke_previous > 0.0)
        //         {
        //             percent_change = 100.0 * (kinetic_energy - ke_previous) / ke_previous;
        //             trend = (kinetic_energy > ke_previous) ? "  <== KE ROSE (energy being created)" : "  KE fell (ok)";
        //         }

        //         std::cout << "[KE] substep " << ke_step
        //                   << "  totalKE = " << std::setprecision(8) << kinetic_energy
        //                   << "  change = " << std::setprecision(4) << percent_change << " %"
        //                   << "  max|v| = " << std::setprecision(6) << peak
        //                   << trend << std::endl;

        //         ke_previous = kinetic_energy;
        //     }
        // }

        //--------------------------------------------------face vels checker
        // glBindBuffer(GL_SHADER_STORAGE_BUFFER, vx_grid_SSBO);

        // glGetBufferSubData(
        //     GL_SHADER_STORAGE_BUFFER,
        //     0,
        //     MAC_vx.vx.size() * sizeof(float),
        //     MAC_vx.vx.data()
        // );

        // auto max_velx = std::max_element(MAC_vx.vx.begin(), MAC_vx.vx.end());

        // std::cout << "max velx = " << *max_velx <<  std::endl;

        // for (int i=0; i < (cells_x + 1) * cells_y; i++)
        // {
        //     if (i > (cells_x + 1) / 2)
        //     {
        //         std::cout << "Face vx " << i << " = " << MAC_vertical.vx[i] << std::endl;
        //     }
        // }

        // glBindBuffer(GL_SHADER_STORAGE_BUFFER, vy_grid_SSBO);

        // glGetBufferSubData(
        //     GL_SHADER_STORAGE_BUFFER,
        //     0,
        //     MAC_vy.vy.size() * sizeof(float),
        //     MAC_vy.vy.data()
        // );

        // auto max_vely = std::max_element(MAC_vy.vy.begin(), MAC_vy.vy.end());

        // std::cout << "max vely = " << *max_vely <<  std::endl;

        // for (int i=0; i < (cells_y + 1) * cells_x; i++)
        // {
        //     if (i % 10 == 0)
        //     {
        //         std::cout << "Face vy" << i << " = " << MAC_horizontal.vy[i] << std::endl;
        //     }
        // }

        // glBindBuffer(GL_SHADER_STORAGE_BUFFER, vz_grid_SSBO);

        // glGetBufferSubData(
        //     GL_SHADER_STORAGE_BUFFER,
        //     0,
        //     MAC_vz.vz.size() * sizeof(float),
        //     MAC_vz.vz.data()
        // );

        // auto max_velz = std::max_element(MAC_vz.vz.begin(), MAC_vz.vz.end());

        // std::cout << "max velz = " << *max_velz <<  std::endl;

        // for (int i=0; i < (cells_y + 1) * cells_x; i++)
        // {
        //     if (i % 10 == 0)
        //     {
        //         std::cout << "Face vy" << i << " = " << MAC_horizontal.vy[i] << std::endl;
        //     }
        // }

        //------------------------------------------- Total simulated time
        total_time += iterations * stable_dt;

        if (hooks.frame_end)
            hooks.frame_end(world, debug, total_time);

        if(total_time >= config.simulation_needed_time)
            break;
    }
}

//-----------TEMPORARY SETUP-----------------------
void flat_groups_to_3d(int totalGroups, int &gx, int &gy, int &gz)
{
    const int maxPerAxis = 65535;

    gx = std::min(totalGroups, maxPerAxis);
    int remaining = (totalGroups + gx - 1) / gx;

    gy = std::min(remaining, maxPerAxis);
    gz = (remaining + gy - 1) / gy;
}

float velocity_component_reduction(GLuint velocity, int total_elements, ShaderManager &shadermanager, std::vector <GLuint> reduction_for_CFL_SSBOs_arr)
{
    glUseProgram(shadermanager.CFl_comp_program);       //reduction program

    GLuint input = velocity;
    int n = total_elements;
    int level = 0;

    while (n > 1)
    {
        int groups = (n + 255) / 256;

        int gx, gy, gz;
        flat_groups_to_3d(groups, gx, gy, gz);

        glUniform1ui(shadermanager.cfl_elements_count_loc, (GLuint)n);
        glUniform1ui(shadermanager.cfl_total_groups_loc, (GLuint)groups);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            0,
            input);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            1,
            reduction_for_CFL_SSBOs_arr[level]);

        glDispatchCompute(gx, gy, gz);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        input = reduction_for_CFL_SSBOs_arr[level];
        n = groups;
        level++;
    }

    // shader writes -> CPU read back
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    float max_value = 0.0f;

    glBindBuffer(
        GL_SHADER_STORAGE_BUFFER,
        input);

    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER,
        0,
        sizeof(float),
        &max_value);

    return max_value;
}


void Orchestrator::run_rans_k_eps(float spalding_wall_smoothness, float reference_velocity, float turbulence_intensity, float turbulent_len, int fb_w, int fb_h, std::vector <GLuint> reduction_for_CFL_SSBOs_arr, FILE* ffmpeg, renderer_functions &renderer, light &Light)
{
    // short local names for the config, identical to the names the loop used inside main.cpp
    const int cells_x = config.cells_x;
    const int cells_y = config.cells_y;
    const int cells_z = config.cells_z;
    const float WIDTH = config.WIDTH;
    const float HEIGHT = config.HEIGHT;
    const float DEPTH = config.DEPTH;
    const float cell_side_x = config.cell_side_x;
    const float cell_side_y = config.cell_side_y;
    const float cell_side_z = config.cell_side_z;
    const float density = config.density;
    float dynamic_viscosity = config.dynamic_viscosity;
    const float dt_max = config.dt_max;
    const float frames_per_sec_offline_sim = config.frames_per_sec_offline_sim;

    //TEMPORARY SETUP FOR FIRST VISUAL RUN
    //---------------------CFL constants
    const float max_step_per_frame_x = 1.0f * cell_side_x;
    const float max_step_per_frame_y = 1.0f * cell_side_y;
    const float max_step_per_frame_z = 1.0f * cell_side_z;
    //---------------------renderer consta
    float raymarch_ds = 0.75 * std::min(std::min(cell_side_x, cell_side_y), cell_side_z);


    // offline simulation constants
    float next_save_time = 0.0;             //initialization
    float diffusive_limit_denominator = (dynamic_viscosity / density) * ((2.0 / (cell_side_x * cell_side_x)) + (2.0 / (cell_side_y * cell_side_y)) + (2.0 / (cell_side_z * cell_side_z)));
    float diffusive_limit = 0.5 * (1.0 / diffusive_limit_denominator);

    std::vector <uint8_t> pixels(fb_w * fb_h * 3);
    

    // std::cout << "STARTING BUFFER ALLOCATION FOR RANS" << std::endl; 

    // single source of truth for every multigrid level of every system (declared first, so it outlives everything that references it)
    MultigridHierarchy hierarchy(cells_x, cells_y, cells_z, WIDTH, HEIGHT, DEPTH, 8);

    RANS rans(hierarchy, shadermanager, config);
    k_TurbulenceModels k_model(rans, false, false);
    k_TurbulenceModels::destruction_variable destruction_variable_model(k_model);

    Generic_multigrid u_solver(shadermanager, rans.u_system, rans.coefficients_stencil);
    Generic_multigrid v_solver(shadermanager, rans.v_system, rans.coefficients_stencil);
    Generic_multigrid w_solver(shadermanager, rans.w_system, rans.coefficients_stencil);

    incompressible_flow incompressibility(cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z, shadermanager);
    incompressible_flow::specialized_pressure_Multigrid pressure_solver(hierarchy, incompressibility);

    Generic_multigrid k_solver(shadermanager, k_model.KSystem, rans.coefficients_stencil);
    Generic_multigrid eps_solver(shadermanager, destruction_variable_model.destruction_variable_system, rans.coefficients_stencil);

    debuggers debug(incompressibility, shadermanager, cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z);

    // TEMP: renderer / camera / light come from main.cpp (parameters). main.cpp already builds the camera basis
    // (Camera.camera_controls, main.cpp:190) and the fullscreen quad (render.create_fullscreen_quad, main.cpp:202).

    rans.initialize_velocity_fields();
    bool check = destruction_variable_model.initialize_k_epsilon_RANS_fields(reference_velocity, turbulence_intensity, turbulent_len, spalding_wall_smoothness);

    if(!check)
    {  
        std::cout << "Initialization failed for k epsilon" << std::endl; 
        return;
    }

    double total_time = 0.0;
    auto real_start = std::chrono::high_resolution_clock::now();
    int counter = 0;        // REMOVE LATER

    // std::cout << "BUFFER ALLOCATION FOR RANS COMPLETE, NOW ENTERING MAIN LOOP" << std::endl; 


    //Initial incompressibility treatment
    incompressibility.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
    pressure_solver.solve_pressure_system(1e-6, 1e-7, 3, density, dt_max);
    incompressibility.project_pressure(dt_max, density, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);

    // time history of the very first step = the initial (post initial projection) fields, so BDF2 starts from phi^(n-1) = phi^(n-2) = phi^0
    rans.u_system.seed_time_history();
    rans.v_system.seed_time_history();
    rans.w_system.seed_time_history();
    k_model.KSystem.seed_time_history();
    destruction_variable_model.destruction_variable_system.seed_time_history();

    double RMS_of_u_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_v_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_w_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_k_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_eps_system_residual_each_non_linear_iter = 0.0;
    int number_of_non_linear_iters_performed_throughout_the_run = 0;
    int total_time_steps = 0;

    double summation_of_divergence_of_every_non_linear_iter = 0.0;
    double max_recorded_divergence_of_all_non_linear_iters = 0.0;

    std::ofstream energy_file("energy_history.csv");                           // INSTRUMENTED
    std::ofstream avgs_file("solver_parameters_avg.csv");                           // INSTRUMENTED

    energy_file << "time,mean_flow_ke,turbulent_ke,total_ke\n";
    avgs_file << "avg_divergence,max_divergence,avg_non_linear_iters,avg_k_residual_rms,avg_eps_residual_rms,avg_u_residual_rms,avg_v_residual_rms,avg_w_residual_rms\n";

    // Initial

    if (!energy_file.is_open())
    {
        throw std::runtime_error("Could not open energy CSV");
    }

    if (!avgs_file.is_open())
    {
        throw std::runtime_error("Could not open avgs CSV");
    }

    energy_file << std::setprecision(15);
    avgs_file << std::setprecision(15);

    double mean_flow_KE = debug.get_mean_flow_kinetic_energy(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, density);
    double turbulent_KE = debug.get_turbulent_kinetic_energy(k_model.k_field(), density);

    energy_file << total_time << ',' << mean_flow_KE << ',' << turbulent_KE << ',' << turbulent_KE + mean_flow_KE << '\n';

    while(true)
    {
        int non_linear_iters = 1;

        rans.u_system.save_previous_time_step_field();
        rans.v_system.save_previous_time_step_field();
        rans.w_system.save_previous_time_step_field();
        k_model.KSystem.save_previous_time_step_field();
        destruction_variable_model.destruction_variable_system.save_previous_time_step_field();

        // std::cout << "SAVED FIELD FROM LAST TIME STEP" << std::endl;

        for(int i=0; i < non_linear_iters; i++)
        {
            rans.store_nonlinear_iterate(k_model, destruction_variable_model);

            rans.construct_u_momentum_equation(dt_max, spalding_wall_smoothness); 
            int status_u = u_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "COMPLETED U SYSTEM LINEAR ITERATIONS WITH STATUS " << status_u << std::endl;

            rans.construct_v_momentum_equation(dt_max, spalding_wall_smoothness);
            int status_v = v_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "COMPLETED V SYSTEM LINEAR ITERATIONS WITH STATUS "<< status_v << std::endl;
            
            rans.construct_w_momentum_equation(dt_max, spalding_wall_smoothness);
            int status_w = w_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "COMPLETED W SYSTEM LINEAR ITERATIONS WITH STATUS "<< status_w << std::endl;
            
            incompressibility.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            
           pressure_solver.solve_pressure_system(1e-4, 1e-5, 3, density, dt_max);
            
            incompressibility.project_pressure(dt_max, density, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);

            // std::cout << "PRESSURE PROJECTION COMPLETE" << std::endl;

            //RESTRICTING THE PHYSICAL FIELDS TO CONNSTRUCT COARSER LEVELS
            destruction_variable_model.restrict_k_and_destruction_variable();

            k_model.construct_K_equation(dt_max, destruction_variable_model.destruction_variable_system.field.data, spalding_wall_smoothness);
            k_model.construct_K_equation_hierarchy(dt_max, destruction_variable_model.destruction_variable_hierarchy);
            int status_k = k_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "SOLVED K SYSTEM WITH STATUS " << status_k << std::endl;
            
            destruction_variable_model.construct_destruction_variable_equation(dt_max, spalding_wall_smoothness);
            destruction_variable_model.construct_destruction_variable_equation_hierarchy(dt_max);
            int status_eps = eps_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "SOLVED EPS SYSTEM WITH STATUS " << status_eps << std::endl;
            
            destruction_variable_model.update_mu_eff_k_eps();
            // std::cout << "UPDATED MU EFF" << std::endl;

            float max_non_linear_change = rans.calculate_nonlinear_convergence(k_model, destruction_variable_model);

            float u_RMS_residual = u_solver.calculate_l2_norm() / sqrt((cells_x + 1) * cells_y * cells_z);
            float v_RMS_residual = v_solver.calculate_l2_norm() / sqrt((cells_y + 1) * cells_x * cells_z);
            float w_RMS_residual = w_solver.calculate_l2_norm() / sqrt((cells_z + 1) * cells_y * cells_x);
            float k_RMS_residual = k_solver.calculate_l2_norm() / sqrt(cells_x * cells_y * cells_z);
            float eps_RMS_residual = eps_solver.calculate_l2_norm() / sqrt(cells_x * cells_y * cells_z);

            RMS_of_u_system_residual_each_non_linear_iter += u_RMS_residual;
            RMS_of_v_system_residual_each_non_linear_iter += v_RMS_residual;
            RMS_of_w_system_residual_each_non_linear_iter += w_RMS_residual;
            RMS_of_k_system_residual_each_non_linear_iter += k_RMS_residual;
            RMS_of_eps_system_residual_each_non_linear_iter += eps_RMS_residual;

            float mean_div = debug.get_mean_absolute_divergence(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            float max_div = debug.get_max_absolute_divergence(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            // std::cout << "mean absolute divergence = " << mean_div << std::endl;

            summation_of_divergence_of_every_non_linear_iter += mean_div;
            max_recorded_divergence_of_all_non_linear_iters = std::max((double)max_div, max_recorded_divergence_of_all_non_linear_iters);
            number_of_non_linear_iters_performed_throughout_the_run += 1;

            if(max_non_linear_change < 1e-4)
            {
                break;
            }
            else
            {
                non_linear_iters++;
            }

            if(non_linear_iters > 10000)
            {
                std::cout << "The non linear convergence has stalled, or is far too slow (performed 10000+ iters)" << std::endl;
                break;
            }

            // std::cout << "----------------------------------------------non linear iteration number = " << non_linear_iters << std::endl;
        }

        total_time_steps += 1;

        // 2. CFL condition: dt from the fastest face velocity (post-injection, so the impulse is accounted for)
        float max_vx = velocity_component_reduction(rans.u_system.field.data, (cells_x + 1) * cells_y * cells_z, shadermanager, reduction_for_CFL_SSBOs_arr);
        float max_vy = velocity_component_reduction(rans.v_system.field.data, (cells_y + 1) * cells_x * cells_z, shadermanager, reduction_for_CFL_SSBOs_arr);
        float max_vz = velocity_component_reduction(rans.w_system.field.data, (cells_z + 1) * cells_y * cells_x, shadermanager, reduction_for_CFL_SSBOs_arr);

        float stable_dt_vx = dt_max;
        float stable_dt_vy = dt_max;
        float stable_dt_vz = dt_max;

        if (max_vx > 1e-6f)
        {
            stable_dt_vx = std::min(stable_dt_vx, max_step_per_frame_x / max_vx);
        }

        if (max_vy > 1e-6f)
        {
            stable_dt_vy = std::min(stable_dt_vy, max_step_per_frame_y / max_vy);
        }

        if (max_vz > 1e-6f)
        {
            stable_dt_vz = std::min(stable_dt_vz, max_step_per_frame_z / max_vz);
        }

        float stable_dt_temp = std::min(stable_dt_vy, stable_dt_vx);
        float stable_dt_advective = std::min(stable_dt_temp, stable_dt_vz);
        float stable_dt = std::min(stable_dt_advective, diffusive_limit);

        int iterations = std::ceil(dt_max / stable_dt);

        // smoke-advection validation print (advection itself is unchanged): the smoke kernel is explicit, so the
        // CFL number of each sub-step (max|v| * dt / cell) must stay <= 1; sub-steps cover iterations*stable_dt >= dt_max
        {
            float cfl_per_substep = std::max(std::max(max_vx / cell_side_x, max_vy / cell_side_y), max_vz / cell_side_z) * stable_dt;
            // std::cout << "[smoke] RANS dt_max = " << dt_max << ", smoke sub-steps = " << iterations << " x " << stable_dt
                    //   << ", CFL per sub-step = " << cfl_per_substep << (cfl_per_substep > 1.0f ? "  <-- EXCEEDS 1" : "") << std::endl;
        }

        for(int i=0; i < iterations; i++)
        {
            glBindBuffer(GL_COPY_READ_BUFFER, renderer.smokeMatter_SSBO);
            glBindBuffer(GL_COPY_WRITE_BUFFER, renderer.smokeMatter_orignal_SSBO);

            glCopyBufferSubData(
                GL_COPY_READ_BUFFER,
                GL_COPY_WRITE_BUFFER,
                0,
                0,
                cells_x * cells_y * cells_z * sizeof(float)
            );

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.smoke_advection_dispatch(shadermanager, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, stable_dt);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.smoke_advection_dispatch(shadermanager, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, stable_dt);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.final_smoke_blend(shadermanager);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        if(total_time >= next_save_time)
        {
        //-------------------------------------------------------SMOKE MATTER BASED RENDERING------------------------------------
            renderer.dual_mode_render(0, shadermanager, debug, 0, fb_w, fb_h, raymarch_ds, Light);
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


        double real_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - real_start).count();
    
        total_time += dt_max;
        
        counter += 1;
        
        if(counter % 10 == 0)
        {
            double mean_flow_KE = debug.get_mean_flow_kinetic_energy(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, density);
            double turbulent_KE = debug.get_turbulent_kinetic_energy(k_model.k_field(), density);

            energy_file << total_time << ',' << mean_flow_KE << ',' << turbulent_KE << ',' << turbulent_KE + mean_flow_KE << '\n';
        }

        std::cout  << total_time << " and real time is " << real_time << std::endl;

        // std::cout << "--------------------------------TIME STEP COMPLETE---------------------------------" << std::endl;

        if(total_time >= config.simulation_needed_time)
            break;

    }

    avgs_file   << summation_of_divergence_of_every_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << max_recorded_divergence_of_all_non_linear_iters << ','
                << static_cast <double>(number_of_non_linear_iters_performed_throughout_the_run) / total_time_steps << ','
                << RMS_of_k_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_eps_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_u_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_v_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_w_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << '\n';

}

void Orchestrator::run_rans_k_omega(float spalding_wall_smoothness, float reference_velocity, float turbulence_intensity, float turbulent_len, int fb_w, int fb_h, std::vector <GLuint> reduction_for_CFL_SSBOs_arr, FILE* ffmpeg, renderer_functions &renderer, light &Light)
{
    // short local names for the config, identical to the names the loop used inside main.cpp
    const int cells_x = config.cells_x;
    const int cells_y = config.cells_y;
    const int cells_z = config.cells_z;
    const float WIDTH = config.WIDTH;
    const float HEIGHT = config.HEIGHT;
    const float DEPTH = config.DEPTH;
    const float cell_side_x = config.cell_side_x;
    const float cell_side_y = config.cell_side_y;
    const float cell_side_z = config.cell_side_z;
    const float density = config.density;
    float dynamic_viscosity = config.dynamic_viscosity;
    const float dt_max = config.dt_max;
    const float frames_per_sec_offline_sim = config.frames_per_sec_offline_sim;

    //TEMPORARY SETUP FOR FIRST VISUAL RUN
    //---------------------CFL constants
    const float max_step_per_frame_x = 1.0f * cell_side_x;
    const float max_step_per_frame_y = 1.0f * cell_side_y;
    const float max_step_per_frame_z = 1.0f * cell_side_z;
    //---------------------renderer consta
    float raymarch_ds = 0.75 * std::min(std::min(cell_side_x, cell_side_y), cell_side_z);


    // offline simulation constants
    float next_save_time = 0.0;             //initialization
    float diffusive_limit_denominator = (dynamic_viscosity / density) * ((2.0 / (cell_side_x * cell_side_x)) + (2.0 / (cell_side_y * cell_side_y)) + (2.0 / (cell_side_z * cell_side_z)));
    float diffusive_limit = 0.5 * (1.0 / diffusive_limit_denominator);

    std::vector <uint8_t> pixels(fb_w * fb_h * 3);
    

    // std::cout << "STARTING BUFFER ALLOCATION FOR RANS" << std::endl; 

    // single source of truth for every multigrid level of every system (declared first, so it outlives everything that references it)
    MultigridHierarchy hierarchy(cells_x, cells_y, cells_z, WIDTH, HEIGHT, DEPTH, 8);

    RANS rans(hierarchy, shadermanager, config);
    k_TurbulenceModels k_model(rans, true, false);
    k_TurbulenceModels::destruction_variable destruction_variable_model(k_model);

    Generic_multigrid u_solver(shadermanager, rans.u_system, rans.coefficients_stencil);
    Generic_multigrid v_solver(shadermanager, rans.v_system, rans.coefficients_stencil);
    Generic_multigrid w_solver(shadermanager, rans.w_system, rans.coefficients_stencil);

    incompressible_flow incompressibility(cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z, shadermanager);
    incompressible_flow::specialized_pressure_Multigrid pressure_solver(hierarchy, incompressibility);

    Generic_multigrid k_solver(shadermanager, k_model.KSystem, rans.coefficients_stencil);
    Generic_multigrid omega_solver(shadermanager, destruction_variable_model.destruction_variable_system, rans.coefficients_stencil);

    debuggers debug(incompressibility, shadermanager, cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z);

    // TEMP: renderer / camera / light come from main.cpp (parameters). main.cpp already builds the camera basis
    // (Camera.camera_controls, main.cpp:190) and the fullscreen quad (render.create_fullscreen_quad, main.cpp:202).

    rans.initialize_velocity_fields();
    bool check = destruction_variable_model.initialize_k_omega_RANS_fields(reference_velocity, turbulence_intensity, turbulent_len, spalding_wall_smoothness, false);

    if(!check)
    {  
        std::cout << "Initialization failed for k omega" << std::endl; 
        return;
    }

    double total_time = 0.0;
    auto real_start = std::chrono::high_resolution_clock::now();
    int counter = 0;        // REMOVE LATER

    std::cout << "BUFFER ALLOCATION FOR RANS COMPLETE, NOW ENTERING MAIN LOOP" << std::endl; 


    //Initial incompressibility treatment
    incompressibility.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
    pressure_solver.solve_pressure_system(1e-6, 1e-7, 3, density, dt_max);
    incompressibility.project_pressure(dt_max, density, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);

    // time history of the very first step = the initial (post initial projection) fields, so BDF2 starts from phi^(n-1) = phi^(n-2) = phi^0
    rans.u_system.seed_time_history();
    rans.v_system.seed_time_history();
    rans.w_system.seed_time_history();
    k_model.KSystem.seed_time_history();
    destruction_variable_model.destruction_variable_system.seed_time_history();

    double RMS_of_u_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_v_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_w_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_k_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_omega_system_residual_each_non_linear_iter = 0.0;
    int number_of_non_linear_iters_performed_throughout_the_run = 0;
    int total_time_steps = 0;

    double summation_of_divergence_of_every_non_linear_iter = 0.0;
    double max_recorded_divergence_of_all_non_linear_iters = 0.0;

    std::ofstream energy_file("energy_history.csv");                           // INSTRUMENTED
    std::ofstream avgs_file("solver_parameters_avg.csv");                           // INSTRUMENTED

    energy_file << "time,mean_flow_ke,turbulent_ke,total_ke\n";
    avgs_file << "avg_divergence,max_divergence,avg_non_linear_iters,avg_k_residual_rms,avg_omega_residual_rms,avg_u_residual_rms,avg_v_residual_rms,avg_w_residual_rms\n";

    // Initial

    if (!energy_file.is_open())
    {
        throw std::runtime_error("Could not open energy CSV");
    }

    if (!avgs_file.is_open())
    {
        throw std::runtime_error("Could not open avgs CSV");
    }

    energy_file << std::setprecision(15);
    avgs_file << std::setprecision(15);

    double mean_flow_KE = debug.get_mean_flow_kinetic_energy(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, density);
    double turbulent_KE = debug.get_turbulent_kinetic_energy(k_model.k_field(), density);

    energy_file << total_time << ',' << mean_flow_KE << ',' << turbulent_KE << ',' << turbulent_KE + mean_flow_KE << '\n';



    while(true)
    {
        int non_linear_iters = 1;

        rans.u_system.save_previous_time_step_field();
        rans.v_system.save_previous_time_step_field();
        rans.w_system.save_previous_time_step_field();
        k_model.KSystem.save_previous_time_step_field();
        destruction_variable_model.destruction_variable_system.save_previous_time_step_field();

        // std::cout << "SAVED FIELD FROM LAST TIME STEP" << std::endl;

        for(int i=0; i < non_linear_iters; i++)
        {
            rans.store_nonlinear_iterate(k_model, destruction_variable_model);

            rans.construct_u_momentum_equation(dt_max, spalding_wall_smoothness);
            int status_u = u_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "COMPLETED U SYSTEM LINEAR ITERATIONS WITH STATUS " << status_u << std::endl;

            rans.construct_v_momentum_equation(dt_max, spalding_wall_smoothness);
            int status_v = v_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "COMPLETED V SYSTEM LINEAR ITERATIONS WITH STATUS "<< status_v << std::endl;

            rans.construct_w_momentum_equation(dt_max, spalding_wall_smoothness);
            int status_w = w_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "COMPLETED W SYSTEM LINEAR ITERATIONS WITH STATUS "<< status_w << std::endl;

            incompressibility.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);

            pressure_solver.solve_pressure_system(1e-6, 1e-7, 3, density, dt_max);

            incompressibility.project_pressure(dt_max, density, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);

            // std::cout << "PRESSURE PROJECTION COMPLETE" << std::endl;

            //RESTRICTING THE PHYSICAL FIELDS TO CONNSTRUCT COARSER LEVELS
            destruction_variable_model.restrict_k_and_destruction_variable();

            k_model.construct_K_equation(dt_max, destruction_variable_model.destruction_variable_system.field.data, spalding_wall_smoothness);
            k_model.construct_K_equation_hierarchy(dt_max, destruction_variable_model.destruction_variable_hierarchy);
            int status_k = k_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "SOLVED K SYSTEM WITH STATUS " << status_k << std::endl;

            destruction_variable_model.construct_destruction_variable_equation(dt_max, spalding_wall_smoothness);
            destruction_variable_model.construct_destruction_variable_equation_hierarchy(dt_max);
            int status_eps = omega_solver.multigrid_solve_system(1e-3, 1e-6, 3);
            // std::cout << "SOLVED OMEGA SYSTEM WITH STATUS " << status_eps << std::endl;

            destruction_variable_model.update_mu_eff_k_omega();
            // std::cout << "UPDATED MU EFF" << std::endl;

            float max_non_linear_change = rans.calculate_nonlinear_convergence(k_model, destruction_variable_model);

            float u_RMS_residual = u_solver.calculate_l2_norm() / sqrt((cells_x + 1) * cells_y * cells_z);
            float v_RMS_residual = v_solver.calculate_l2_norm() / sqrt((cells_y + 1) * cells_x * cells_z);
            float w_RMS_residual = w_solver.calculate_l2_norm() / sqrt((cells_z + 1) * cells_y * cells_x);
            float k_RMS_residual = k_solver.calculate_l2_norm() / sqrt(cells_x * cells_y * cells_z);
            float omega_RMS_residual = omega_solver.calculate_l2_norm() / sqrt(cells_x * cells_y * cells_z);

            RMS_of_u_system_residual_each_non_linear_iter += u_RMS_residual;
            RMS_of_v_system_residual_each_non_linear_iter += v_RMS_residual;
            RMS_of_w_system_residual_each_non_linear_iter += w_RMS_residual;
            RMS_of_k_system_residual_each_non_linear_iter += k_RMS_residual;
            RMS_of_omega_system_residual_each_non_linear_iter += omega_RMS_residual;

            float mean_div = debug.get_mean_absolute_divergence(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            float max_div = debug.get_max_absolute_divergence(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            // std::cout << "mean absolute divergence = " << mean_div << std::endl;

            summation_of_divergence_of_every_non_linear_iter += mean_div;
            max_recorded_divergence_of_all_non_linear_iters = std::max((double)max_div, max_recorded_divergence_of_all_non_linear_iters);
            number_of_non_linear_iters_performed_throughout_the_run += 1;

            if(max_non_linear_change < 1e-4)
            {
                break;
            }
            else
            {
                non_linear_iters++;
            }

            if(non_linear_iters > 10000)
            {
                std::cout << "The non linear convergence has stalled, or is far too slow (performed 10000+ iters)" << std::endl;
                break;
            }

            // std::cout << "--------------------------------------------------non linear iteration number = " << non_linear_iters << std::endl;
            
            // float mean_div = debug.get_mean_absolute_divergence(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            // std::cout << "mean absolute divergence = " << mean_div << std::endl;
            
        }
        
        // float mean_div = debug.get_mean_absolute_divergence(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
        // std::cout << "mean absolute divergence = " << mean_div << std::endl;

        total_time_steps += 1;

        // 2. CFL condition: dt from the fastest face velocity (post-injection, so the impulse is accounted for)
        float max_vx = velocity_component_reduction(rans.u_system.field.data, (cells_x + 1) * cells_y * cells_z, shadermanager, reduction_for_CFL_SSBOs_arr);
        float max_vy = velocity_component_reduction(rans.v_system.field.data, (cells_y + 1) * cells_x * cells_z, shadermanager, reduction_for_CFL_SSBOs_arr);
        float max_vz = velocity_component_reduction(rans.w_system.field.data, (cells_z + 1) * cells_y * cells_x, shadermanager, reduction_for_CFL_SSBOs_arr);

        float stable_dt_vx = dt_max;
        float stable_dt_vy = dt_max;
        float stable_dt_vz = dt_max;

        if (max_vx > 1e-6f)
        {
            stable_dt_vx = std::min(stable_dt_vx, max_step_per_frame_x / max_vx);
        }

        if (max_vy > 1e-6f)
        {
            stable_dt_vy = std::min(stable_dt_vy, max_step_per_frame_y / max_vy);
        }

        if (max_vz > 1e-6f)
        {
            stable_dt_vz = std::min(stable_dt_vz, max_step_per_frame_z / max_vz);
        }

        float stable_dt_temp = std::min(stable_dt_vy, stable_dt_vx);
        float stable_dt_advective = std::min(stable_dt_temp, stable_dt_vz);
        float stable_dt = std::min(stable_dt_advective, diffusive_limit);

        int iterations = std::ceil(dt_max / stable_dt);

        // smoke-advection validation print (advection itself is unchanged): the smoke kernel is explicit, so the
        // CFL number of each sub-step (max|v| * dt / cell) must stay <= 1; sub-steps cover iterations*stable_dt >= dt_max
        {
            float cfl_per_substep = std::max(std::max(max_vx / cell_side_x, max_vy / cell_side_y), max_vz / cell_side_z) * stable_dt;
            // std::cout << "[smoke] RANS dt_max = " << dt_max << ", smoke sub-steps = " << iterations << " x " << stable_dt
                    //   << ", CFL per sub-step = " << cfl_per_substep << (cfl_per_substep > 1.0f ? "  <-- EXCEEDS 1" : "") << std::endl;
        }

        for(int i=0; i < iterations; i++)
        {
            glBindBuffer(GL_COPY_READ_BUFFER, renderer.smokeMatter_SSBO);
            glBindBuffer(GL_COPY_WRITE_BUFFER, renderer.smokeMatter_orignal_SSBO);

            glCopyBufferSubData(
                GL_COPY_READ_BUFFER,
                GL_COPY_WRITE_BUFFER,
                0,
                0,
                cells_x * cells_y * cells_z * sizeof(float)
            );

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.smoke_advection_dispatch(shadermanager, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, stable_dt);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.smoke_advection_dispatch(shadermanager, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, stable_dt);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.final_smoke_blend(shadermanager);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        if(total_time >= next_save_time)
        {
        //-------------------------------------------------------SMOKE MATTER BASED RENDERING------------------------------------
            renderer.dual_mode_render(0, shadermanager, debug, 0, fb_w, fb_h, raymarch_ds, Light);
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


        double real_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - real_start).count();
    
        total_time += dt_max;

        counter += 1;
        
        if(counter % 10 == 0)
        {
            double mean_flow_KE = debug.get_mean_flow_kinetic_energy(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, density);
            double turbulent_KE = debug.get_turbulent_kinetic_energy(k_model.k_field(), density);

            energy_file << total_time << ',' << mean_flow_KE << ',' << turbulent_KE << ',' << turbulent_KE + mean_flow_KE << '\n';
        }
        
        std::cout  << total_time << " and real time is " << real_time << std::endl;

        // std::cout << "------------------TIME STEP COMPLETE-------------------" << std::endl;

        //---------------------------DELETE LATER------------------------------------
        // float dof = cells_x * (cells_y + 1) * cells_z;
        // std::vector <float> temp(dof);
        // glBindBuffer(GL_SHADER_STORAGE_BUFFER, rans.v_system.field.data);
        // glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * dof, temp.data());

        // float sum_y_vel = 0;
        // for(int i=0; i < dof; i++)
        // {
        //     sum_y_vel += temp[i];
        // }

        // float mean_y_vel = sum_y_vel / dof;
        // std::cout << "mean Y velocity: " << mean_y_vel << std::endl;



        if(total_time >= config.simulation_needed_time)
            break;

    }

    avgs_file   << summation_of_divergence_of_every_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << max_recorded_divergence_of_all_non_linear_iters << ','
                << static_cast <double>(number_of_non_linear_iters_performed_throughout_the_run) / total_time_steps << ','
                << RMS_of_k_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_omega_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_u_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_v_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_w_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << '\n';
}

void Orchestrator::run_rans_k_omega_SST(float spalding_wall_smoothness, float reference_velocity, float turbulence_intensity, float turbulent_len, int fb_w, int fb_h, std::vector <GLuint> reduction_for_CFL_SSBOs_arr, FILE* ffmpeg, renderer_functions &renderer, light &Light)
{
// short local names for the config, identical to the names the loop used inside main.cpp
    const int cells_x = config.cells_x;
    const int cells_y = config.cells_y;
    const int cells_z = config.cells_z;
    const float WIDTH = config.WIDTH;
    const float HEIGHT = config.HEIGHT;
    const float DEPTH = config.DEPTH;
    const float cell_side_x = config.cell_side_x;
    const float cell_side_y = config.cell_side_y;
    const float cell_side_z = config.cell_side_z;
    const float density = config.density;
    float dynamic_viscosity = config.dynamic_viscosity;
    const float dt_max = config.dt_max;
    const float frames_per_sec_offline_sim = config.frames_per_sec_offline_sim;

    //TEMPORARY SETUP FOR FIRST VISUAL RUN
    //---------------------CFL constants
    const float max_step_per_frame_x = 1.0f * cell_side_x;
    const float max_step_per_frame_y = 1.0f * cell_side_y;
    const float max_step_per_frame_z = 1.0f * cell_side_z;
    //---------------------renderer consta
    float raymarch_ds = 0.75 * std::min(std::min(cell_side_x, cell_side_y), cell_side_z);


    // offline simulation constants
    float next_save_time = 0.0;             //initialization
    float diffusive_limit_denominator = (dynamic_viscosity / density) * ((2.0 / (cell_side_x * cell_side_x)) + (2.0 / (cell_side_y * cell_side_y)) + (2.0 / (cell_side_z * cell_side_z)));
    float diffusive_limit = 0.5 * (1.0 / diffusive_limit_denominator);

    std::vector <uint8_t> pixels(fb_w * fb_h * 3);
    

    // std::cout << "STARTING BUFFER ALLOCATION FOR RANS" << std::endl; 

    // single source of truth for every multigrid level of every system (declared first, so it outlives everything that references it)
    MultigridHierarchy hierarchy(cells_x, cells_y, cells_z, WIDTH, HEIGHT, DEPTH, 8);

    RANS rans(hierarchy, shadermanager, config);
    k_TurbulenceModels k_model(rans, true, true);
    k_TurbulenceModels::destruction_variable destruction_variable_model(k_model);

    Generic_multigrid u_solver(shadermanager, rans.u_system, rans.coefficients_stencil);
    Generic_multigrid v_solver(shadermanager, rans.v_system, rans.coefficients_stencil);
    Generic_multigrid w_solver(shadermanager, rans.w_system, rans.coefficients_stencil);

    incompressible_flow incompressibility(cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z, shadermanager);
    incompressible_flow::specialized_pressure_Multigrid pressure_solver(hierarchy, incompressibility);

    Generic_multigrid k_solver(shadermanager, k_model.KSystem, rans.coefficients_stencil);
    Generic_multigrid omega_solver(shadermanager, destruction_variable_model.destruction_variable_system, rans.coefficients_stencil);

    debuggers debug(incompressibility, shadermanager, cells_x, cells_y, cells_z, cell_side_x, cell_side_y, cell_side_z);

    // TEMP: renderer / camera / light come from main.cpp (parameters). main.cpp already builds the camera basis
    // (Camera.camera_controls, main.cpp:190) and the fullscreen quad (render.create_fullscreen_quad, main.cpp:202).

    rans.initialize_velocity_fields();
    k_model.sst->compute_structured_mesh_cell_min_dists(); // necessary because the initialize k omega fields function does a mu_eff update for SST bool being true
    bool check = destruction_variable_model.initialize_k_omega_RANS_fields(reference_velocity, turbulence_intensity, turbulent_len, spalding_wall_smoothness, true);

    if(!check)
    {  
        std::cout << "Initialization failed for k omega" << std::endl; 
        return;
    }

    double total_time = 0.0;
    auto real_start = std::chrono::high_resolution_clock::now();
    int counter = 0;        // REMOVE LATER

    std::cout << "BUFFER ALLOCATION FOR RANS COMPLETE, NOW ENTERING MAIN LOOP" << std::endl; 


    //Initial incompressibility treatment
    incompressibility.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
    pressure_solver.solve_pressure_system(1e-6, 1e-7, 3, density, dt_max);
    incompressibility.project_pressure(dt_max, density, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);

    // time history of the very first step = the initial (post initial projection) fields, so BDF2 starts from phi^(n-1) = phi^(n-2) = phi^0
    rans.u_system.seed_time_history();
    rans.v_system.seed_time_history();
    rans.w_system.seed_time_history();
    k_model.KSystem.seed_time_history();
    destruction_variable_model.destruction_variable_system.seed_time_history();

    double RMS_of_u_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_v_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_w_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_k_system_residual_each_non_linear_iter = 0.0;
    double RMS_of_omega_system_residual_each_non_linear_iter = 0.0;
    int number_of_non_linear_iters_performed_throughout_the_run = 0;
    int total_time_steps = 0;

    double summation_of_divergence_of_every_non_linear_iter = 0.0;
    double max_recorded_divergence_of_all_non_linear_iters = 0.0;

    int k_non_physical_values_count_all_cells = 0;
    int omega_non_physical_values_count_all_cells = 0;

    std::ofstream energy_file("energy_history.csv");                           // INSTRUMENTED
    std::ofstream avgs_file("solver_parameters_avg.csv");                           // INSTRUMENTED
    std::ofstream extra_info("extra_info.csv");                           // INSTRUMENTED

    energy_file << "time,mean_flow_ke,turbulent_ke,total_ke\n";
    avgs_file << "avg_divergence,max_divergence,avg_non_linear_iters,avg_k_residual_rms,avg_omega_residual_rms,avg_u_residual_rms,avg_v_residual_rms,avg_w_residual_rms\n";
    extra_info << "cells_with_non_physical_k_values,cells_with_non_physical_des_values\n";
    // Initial

    if (!energy_file.is_open())
    {
        throw std::runtime_error("Could not open energy CSV");
    }

    if (!avgs_file.is_open())
    {
        throw std::runtime_error("Could not open avgs CSV");
    }

    if (!extra_info.is_open())
    {
        throw std::runtime_error("Could not open extra_info.csv");
    }

    energy_file << std::setprecision(15);
    avgs_file << std::setprecision(15);

    double mean_flow_KE = debug.get_mean_flow_kinetic_energy(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, density);
    double turbulent_KE = debug.get_turbulent_kinetic_energy(k_model.k_field(), density);

    energy_file << total_time << ',' << mean_flow_KE << ',' << turbulent_KE << ',' << turbulent_KE + mean_flow_KE << '\n';



    while(true)
    {
        int non_linear_iters = 1;

        rans.u_system.save_previous_time_step_field();
        rans.v_system.save_previous_time_step_field();
        rans.w_system.save_previous_time_step_field();
        k_model.KSystem.save_previous_time_step_field();
        destruction_variable_model.destruction_variable_system.save_previous_time_step_field();

        // std::cout << "SAVED FIELD FROM LAST TIME STEP" << std::endl;

        for(int i=0; i < non_linear_iters; i++)
        {
            
            rans.store_nonlinear_iterate(k_model, destruction_variable_model);
            
            rans.construct_u_momentum_equation(dt_max, spalding_wall_smoothness);
            int status_u = u_solver.multigrid_solve_system(1e-3, 1e-4, 3);
            // std::cout << "COMPLETED U SYSTEM LINEAR ITERATIONS WITH STATUS " << status_u << std::endl;
            
            rans.construct_v_momentum_equation(dt_max, spalding_wall_smoothness);
            int status_v = v_solver.multigrid_solve_system(1e-3, 1e-4, 3);
            // std::cout << "COMPLETED V SYSTEM LINEAR ITERATIONS WITH STATUS "<< status_v << std::endl;
            
            rans.construct_w_momentum_equation(dt_max, spalding_wall_smoothness);
            int status_w = w_solver.multigrid_solve_system(1e-3, 1e-4, 3);
            // std::cout << "COMPLETED W SYSTEM LINEAR ITERATIONS WITH STATUS "<< status_w << std::endl;

            incompressibility.divergence_cal(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            
            pressure_solver.solve_pressure_system(1e-6, 1e-7, 3, density, dt_max);
            
            incompressibility.project_pressure(dt_max, density, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            
            // std::cout << "PRESSURE PROJECTION COMPLETE" << std::endl;
            
            //RESTRICTING THE PHYSICAL FIELDS TO CONNSTRUCT COARSER LEVELS
            destruction_variable_model.restrict_k_and_destruction_variable();

            k_model.sst->SST_preprocessing(destruction_variable_model.destruction_variable_field(), destruction_variable_model.destruction_variable_hierarchy);
            
            k_model.construct_K_equation(dt_max, destruction_variable_model.destruction_variable_system.field.data, spalding_wall_smoothness);
            k_model.construct_K_equation_hierarchy(dt_max, destruction_variable_model.destruction_variable_hierarchy);
            int status_k = k_solver.multigrid_solve_system(1e-3, 1e-4, 3);
            // std::cout << "SOLVED K SYSTEM WITH STATUS " << status_k << std::endl;
            destruction_variable_model.enforce_positivity();        // SST: no negative / zero k reaches the omega assembly

            destruction_variable_model.construct_destruction_variable_equation(dt_max, spalding_wall_smoothness);
            destruction_variable_model.construct_destruction_variable_equation_hierarchy(dt_max);
            int status_eps = omega_solver.multigrid_solve_system(1e-3, 1e-4, 3);
            // std::cout << "SOLVED OMEGA SYSTEM WITH STATUS " << status_eps << std::endl;
            destruction_variable_model.enforce_positivity();        // SST: no negative / zero omega reaches mu_eff, F1, the destruction term, or the coarse levels

            destruction_variable_model.update_mu_eff_k_omega_SST();
            // std::cout << "UPDATED MU EFF" << std::endl;

            float max_non_linear_change = rans.calculate_nonlinear_convergence(k_model, destruction_variable_model);

            float u_RMS_residual = u_solver.calculate_l2_norm() / sqrt((cells_x + 1) * cells_y * cells_z);
            float v_RMS_residual = v_solver.calculate_l2_norm() / sqrt((cells_y + 1) * cells_x * cells_z);
            float w_RMS_residual = w_solver.calculate_l2_norm() / sqrt((cells_z + 1) * cells_y * cells_x);
            float k_RMS_residual = k_solver.calculate_l2_norm() / sqrt(cells_x * cells_y * cells_z);
            float omega_RMS_residual = omega_solver.calculate_l2_norm() / sqrt(cells_x * cells_y * cells_z);

            RMS_of_u_system_residual_each_non_linear_iter += u_RMS_residual;
            RMS_of_v_system_residual_each_non_linear_iter += v_RMS_residual;
            RMS_of_w_system_residual_each_non_linear_iter += w_RMS_residual;
            RMS_of_k_system_residual_each_non_linear_iter += k_RMS_residual;
            RMS_of_omega_system_residual_each_non_linear_iter += omega_RMS_residual;

            float mean_div = debug.get_mean_absolute_divergence(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            float max_div = debug.get_max_absolute_divergence(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data);
            // std::cout << "mean absolute divergence = " << mean_div << std::endl;

            summation_of_divergence_of_every_non_linear_iter += mean_div;
            max_recorded_divergence_of_all_non_linear_iters = std::max((double)max_div, max_recorded_divergence_of_all_non_linear_iters);
            number_of_non_linear_iters_performed_throughout_the_run += 1;

            std::vector <int> count = debug.check_non_physical_turbulence_quantities(k_model.KSystem.field.data, destruction_variable_model.destruction_variable_system.field.data);

            k_non_physical_values_count_all_cells += count[0];
            omega_non_physical_values_count_all_cells += count[1];

            if(max_non_linear_change < 1e-4)
            {
                break;
            }
            else
            {
                non_linear_iters++;
            }

            if(non_linear_iters > 10000)
            {
                std::cout << "The non linear convergence has stalled, or is far too slow (performed 10000+ iters)" << std::endl;
                break;
            }

            // std::cout << "----------------------------------------------non linear iteration number = " << non_linear_iters << std::endl;
        }

        total_time_steps += 1;

        // 2. CFL condition: dt from the fastest face velocity (post-injection, so the impulse is accounted for)
        float max_vx = velocity_component_reduction(rans.u_system.field.data, (cells_x + 1) * cells_y * cells_z, shadermanager, reduction_for_CFL_SSBOs_arr);
        float max_vy = velocity_component_reduction(rans.v_system.field.data, (cells_y + 1) * cells_x * cells_z, shadermanager, reduction_for_CFL_SSBOs_arr);
        float max_vz = velocity_component_reduction(rans.w_system.field.data, (cells_z + 1) * cells_y * cells_x, shadermanager, reduction_for_CFL_SSBOs_arr);

        float stable_dt_vx = dt_max;
        float stable_dt_vy = dt_max;
        float stable_dt_vz = dt_max;

        if (max_vx > 1e-6f)
        {
            stable_dt_vx = std::min(stable_dt_vx, max_step_per_frame_x / max_vx);
        }

        if (max_vy > 1e-6f)
        {
            stable_dt_vy = std::min(stable_dt_vy, max_step_per_frame_y / max_vy);
        }

        if (max_vz > 1e-6f)
        {
            stable_dt_vz = std::min(stable_dt_vz, max_step_per_frame_z / max_vz);
        }

        float stable_dt_temp = std::min(stable_dt_vy, stable_dt_vx);
        float stable_dt_advective = std::min(stable_dt_temp, stable_dt_vz);
        float stable_dt = std::min(stable_dt_advective, diffusive_limit);

        int iterations = std::ceil(dt_max / stable_dt);

        // smoke-advection validation print (advection itself is unchanged): the smoke kernel is explicit, so the
        // CFL number of each sub-step (max|v| * dt / cell) must stay <= 1; sub-steps cover iterations*stable_dt >= dt_max
        {
            float cfl_per_substep = std::max(std::max(max_vx / cell_side_x, max_vy / cell_side_y), max_vz / cell_side_z) * stable_dt;
            // std::cout << "[smoke] RANS dt_max = " << dt_max << ", smoke sub-steps = " << iterations << " x " << stable_dt
                    //   << ", CFL per sub-step = " << cfl_per_substep << (cfl_per_substep > 1.0f ? "  <-- EXCEEDS 1" : "") << std::endl;
        }

        for(int i=0; i < iterations; i++)
        {
            glBindBuffer(GL_COPY_READ_BUFFER, renderer.smokeMatter_SSBO);
            glBindBuffer(GL_COPY_WRITE_BUFFER, renderer.smokeMatter_orignal_SSBO);

            glCopyBufferSubData(
                GL_COPY_READ_BUFFER,
                GL_COPY_WRITE_BUFFER,
                0,
                0,
                cells_x * cells_y * cells_z * sizeof(float)
            );

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.smoke_advection_dispatch(shadermanager, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, stable_dt);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.smoke_advection_dispatch(shadermanager, rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, stable_dt);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            renderer.final_smoke_blend(shadermanager);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        if(total_time >= next_save_time)
        {
        //-------------------------------------------------------SMOKE MATTER BASED RENDERING------------------------------------
            renderer.dual_mode_render(0, shadermanager, debug, 0, fb_w, fb_h, raymarch_ds, Light);
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


        double real_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - real_start).count();
    
        total_time += dt_max;

        counter += 1;
        
        if(counter % 10 == 0)
        {
            double mean_flow_KE = debug.get_mean_flow_kinetic_energy(rans.u_system.field.data, rans.v_system.field.data, rans.w_system.field.data, density);
            double turbulent_KE = debug.get_turbulent_kinetic_energy(k_model.k_field(), density);

            energy_file << total_time << ',' << mean_flow_KE << ',' << turbulent_KE << ',' << turbulent_KE + mean_flow_KE << '\n';
        }
        
        std::cout  << total_time << " and real time is " << real_time << std::endl;

        // std::cout << "-------------------------------------------TIME STEP COMPLETE----------------------------------------------" << std::endl;

        if(total_time >= config.simulation_needed_time)
            break;
    
    }

    avgs_file   << summation_of_divergence_of_every_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << max_recorded_divergence_of_all_non_linear_iters << ','
                << static_cast <double>(number_of_non_linear_iters_performed_throughout_the_run) / total_time_steps << ','
                << RMS_of_k_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_omega_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_u_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_v_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << ','
                << RMS_of_w_system_residual_each_non_linear_iter / number_of_non_linear_iters_performed_throughout_the_run << '\n';

    extra_info  << k_non_physical_values_count_all_cells << ','
                << omega_non_physical_values_count_all_cells << '\n';

}