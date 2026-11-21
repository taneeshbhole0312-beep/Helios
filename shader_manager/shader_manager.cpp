#include "shader_manager.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

std::string ShaderManager::loadFile(const char *path)
{
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cout << "FAILED TO OPEN: " << path << std::endl;
        exit(1);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint ShaderManager::createComputeShader(const char *path)
{
    std::string source = loadFile(path);
    const char *src = source.c_str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    std::cout << "Compiling: " << path << "\n";
    // std::cout << source << "\n";
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cout << "Shader Compile Error:\n"
                  << infoLog << std::endl;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);

    if (!linked)
    {
        char infoLog[1024];
        glGetProgramInfoLog(
            program,
            1024,
            nullptr,
            infoLog);

        std::cout << infoLog << std::endl;
        exit(0);
    }

    glDeleteShader(shader);

    return program;
}

GLuint ShaderManager::createShader(const char *vertexPath, const char *fragmentPath)
{
    std::string vertexSource = loadFile(vertexPath);
    std::string fragmentSource = loadFile(fragmentPath);

    const char *vsrc = vertexSource.c_str();
    const char *fsrc = fragmentSource.c_str();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(vertexShader, 1, &vsrc, nullptr);

    glCompileShader(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(fragmentShader, 1, &fsrc, nullptr);

    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

ShaderManager::ShaderManager()
{
    divergence_compute_program = createComputeShader("Kernels/divergence.comp");
    
    viscosityXProgram = createComputeShader("Kernels/viscosity_vx.comp");
    viscosityYProgram = createComputeShader("Kernels/viscosity_vy.comp");
    viscosityZProgram = createComputeShader("Kernels/viscosity_vz.comp");
    
    advect_vx_Program = createComputeShader("Kernels/advect_vx.comp");
    advect_vy_Program = createComputeShader("Kernels/advect_vy.comp");
    advect_vz_Program = createComputeShader("Kernels/advect_vz.comp");

    multigrid_pressure_residual_calculation_program = createComputeShader("Kernels/Pressure_residual_cal.comp");
    multigrid_pressure_smoother_program = createComputeShader("Kernels/Pressuresmoother.comp");
    
    multigrid_generic_smoother_program = createComputeShader("Kernels/GenericRBGSsmoother.comp");
    multigrid_generic_residual_calculation_program = createComputeShader("Kernels/generic_residual_cal.comp");
    physical_fields_multigrid_generic_restriction_program = createComputeShader("Kernels/physical_fields_Generic_multigrid_restriction.comp");
    multigrid_generic_prolongation_program = createComputeShader("Kernels/Generic_multigrid_prolongation.comp");

    multigrid_residual_generic_restriction_program = createComputeShader("Kernels/multigrid_residual_generic_restriction.comp");
    
    projectionXProgram = createComputeShader("Kernels/projectionX.comp");
    projectionYProgram = createComputeShader("Kernels/projectionY.comp");
    projectionZProgram = createComputeShader("Kernels/projectionZ.comp");

    
    viscosity_LHS_assemble = createComputeShader("Kernels/assemble_viscosity_LHS_momentum.comp");
    viscosity_RHS_assemble = createComputeShader("Kernels/assemble_viscosity_RHS_momentum.comp");
    momentum_wall_assemble = createComputeShader("Kernels/momentum_wall_contribution.comp");

    compute_min_wall_dists_structured_mesh = createComputeShader("Kernels/min_wall_dists.comp");
    compute_blended_coefficients_SST = createComputeShader("Kernels/blended_coefficients_SST.comp");
    k_omega_SST_mu_eff = createComputeShader("Kernels/k_omega_SST_mu_eff.comp");
    assemble_cross_diffusion_SST = createComputeShader("Kernels/assemble_cross_diffusion_term_contribution_for_SST.comp");
    diffusion_k_omega_SST_assemble = createComputeShader("Kernels/assemble_k_omega_diffusion_contribution_for_SST.comp");
    
    time_assemble = createComputeShader("Kernels/assemble_time.comp");                //Generalized kernel for all topologies
    advection_assemble = createComputeShader("Kernels/assemble_advection.comp");      //Generalized kernel for all topologies
    
    assemble_production_k_epsilon_omega = createComputeShader("Kernels/assemble_production_k_epsilon_omega_contribution.comp");
    destruction_k_and_epsilon_or_omega_assemble = createComputeShader("Kernels/assemble_k_and_epsilon_or_omega_destruction_contribution.comp");
    diffusion_k_eps_omega_assemble = createComputeShader("Kernels/assemble_k_eps_omega_diffusion_contribution.comp");

    k_wall_treatment_for_k_omega_model = createComputeShader("Kernels/k_wall_treatment_for_k_omega_model.comp");
    omega_wall_treatment_for_k_omega_model = createComputeShader("Kernels/omega_wall_treatment_for_k_omega_model.comp");
    k_omega_mu_eff = createComputeShader("Kernels/K_omega_mu_eff.comp");
    
    k_wall_treatment_for_k_eps_model = createComputeShader("Kernels/k_wall_treatment_for_k_eps_model.comp");
    eps_wall_treatment_for_k_eps_model = createComputeShader("Kernels/eps_wall_treatment_for_k_eps_model.comp");
    k_eps_mu_eff = createComputeShader("Kernels/K_eps_mu_eff.comp");
    k_eps_initialize = createComputeShader("Kernels/K_eps_initialize.comp");
    k_omega_initialize = createComputeShader("Kernels/K_omega_initialize.comp");

    L2_norm_summation_of_sq = createComputeShader("Kernels/residual_sq_summation_L2.comp");
    difference_sq_summation = createComputeShader("Kernels/difference_sq_summation_L2.comp");
    energy_reduction = createComputeShader("Kernels/energy_reduction.comp");

    CFl_comp_program = createComputeShader("Kernels/CFL_reduction.comp");
    final_blend_SSPRK2_program = createComputeShader("Kernels/SSPRK2_blend.comp");
    
    ray_march_program = createComputeShader("Kernels/ray_march.comp");
    field_normalized_slice_program = createComputeShader("Kernels/field_normalized_slice.comp");
    smoke_advection_program = createComputeShader("Kernels/smoke_advect.comp");
    transmittance_program = createComputeShader("Kernels/transmittance.comp");
    
    non_physical_turbulence_quantites_check_program = createComputeShader("Kernels/check_non_physical_turbulence_quantities.comp");
    clamp_to_floor_program = createComputeShader("Kernels/clamp_to_floor.comp");

    displayProgram = createShader("Kernels/vertex.vert", "Kernels/fragment.frag");


    // UNIFORMS LOCATIONS
    div_cells_x_loc = glGetUniformLocation(divergence_compute_program, "cells_x");
    div_cells_y_loc = glGetUniformLocation(divergence_compute_program, "cells_y");
    div_cells_z_loc = glGetUniformLocation(divergence_compute_program, "cells_z");
    div_cell_side_x_loc = glGetUniformLocation(divergence_compute_program, "cell_side_x");
    div_cell_side_y_loc = glGetUniformLocation(divergence_compute_program, "cell_side_y");
    div_cell_side_z_loc = glGetUniformLocation(divergence_compute_program, "cell_side_z");

    visc_vx_cells_x_loc = glGetUniformLocation(viscosityXProgram, "cells_x");
    visc_vx_cells_y_loc = glGetUniformLocation(viscosityXProgram, "cells_y");
    visc_vx_cells_z_loc = glGetUniformLocation(viscosityXProgram, "cells_z");
    visc_vx_viscosity_loc = glGetUniformLocation(viscosityXProgram, "dynamic_viscosity");
    visc_vx_stable_dt_loc = glGetUniformLocation(viscosityXProgram, "stable_dt");
    visc_vx_cell_side_x_loc = glGetUniformLocation(viscosityXProgram, "cell_side_x");
    visc_vx_cell_side_y_loc = glGetUniformLocation(viscosityXProgram, "cell_side_y");
    visc_vx_cell_side_z_loc = glGetUniformLocation(viscosityXProgram, "cell_side_z");
    visc_vx_density_loc = glGetUniformLocation(viscosityXProgram, "density");

    visc_vy_cells_x_loc = glGetUniformLocation(viscosityYProgram, "cells_x");
    visc_vy_cells_y_loc = glGetUniformLocation(viscosityYProgram, "cells_y");
    visc_vy_cells_z_loc = glGetUniformLocation(viscosityYProgram, "cells_z");
    visc_vy_viscosity_loc = glGetUniformLocation(viscosityYProgram, "dynamic_viscosity");
    visc_vy_stable_dt_loc = glGetUniformLocation(viscosityYProgram, "stable_dt");
    visc_vy_cell_side_x_loc = glGetUniformLocation(viscosityYProgram, "cell_side_x");
    visc_vy_cell_side_y_loc = glGetUniformLocation(viscosityYProgram, "cell_side_y");
    visc_vy_cell_side_z_loc = glGetUniformLocation(viscosityYProgram, "cell_side_z");
    visc_vy_density_loc = glGetUniformLocation(viscosityYProgram, "density");

    visc_vz_cells_x_loc = glGetUniformLocation(viscosityZProgram, "cells_x");
    visc_vz_cells_y_loc = glGetUniformLocation(viscosityZProgram, "cells_y");
    visc_vz_cells_z_loc = glGetUniformLocation(viscosityZProgram, "cells_z");
    visc_vz_viscosity_loc = glGetUniformLocation(viscosityZProgram, "dynamic_viscosity");
    visc_vz_stable_dt_loc = glGetUniformLocation(viscosityZProgram, "stable_dt");
    visc_vz_cell_side_x_loc = glGetUniformLocation(viscosityZProgram, "cell_side_x");
    visc_vz_cell_side_y_loc = glGetUniformLocation(viscosityZProgram, "cell_side_y");
    visc_vz_cell_side_z_loc = glGetUniformLocation(viscosityZProgram, "cell_side_z");
    visc_vz_density_loc = glGetUniformLocation(viscosityZProgram, "density");

    adv_vx_cells_x_loc = glGetUniformLocation(advect_vx_Program, "cells_x");
    adv_vx_cells_y_loc = glGetUniformLocation(advect_vx_Program, "cells_y");
    adv_vx_cells_z_loc = glGetUniformLocation(advect_vx_Program, "cells_z");
    adv_vx_cell_side_x_loc = glGetUniformLocation(advect_vx_Program, "cell_side_x");
    adv_vx_cell_side_y_loc = glGetUniformLocation(advect_vx_Program, "cell_side_y");
    adv_vx_cell_side_z_loc = glGetUniformLocation(advect_vx_Program, "cell_side_z");
    adv_vx_stable_dt_loc = glGetUniformLocation(advect_vx_Program, "stable_dt");

    adv_vy_cells_x_loc = glGetUniformLocation(advect_vy_Program, "cells_x");
    adv_vy_cells_y_loc = glGetUniformLocation(advect_vy_Program, "cells_y");
    adv_vy_cells_z_loc = glGetUniformLocation(advect_vy_Program, "cells_z");
    adv_vy_cell_side_x_loc = glGetUniformLocation(advect_vy_Program, "cell_side_x");
    adv_vy_cell_side_y_loc = glGetUniformLocation(advect_vy_Program, "cell_side_y");
    adv_vy_cell_side_z_loc = glGetUniformLocation(advect_vy_Program, "cell_side_z");
    adv_vy_stable_dt_loc = glGetUniformLocation(advect_vy_Program, "stable_dt");

    adv_vz_cells_x_loc = glGetUniformLocation(advect_vz_Program, "cells_x");
    adv_vz_cells_y_loc = glGetUniformLocation(advect_vz_Program, "cells_y");
    adv_vz_cells_z_loc = glGetUniformLocation(advect_vz_Program, "cells_z");
    adv_vz_cell_side_x_loc = glGetUniformLocation(advect_vz_Program, "cell_side_x");
    adv_vz_cell_side_y_loc = glGetUniformLocation(advect_vz_Program, "cell_side_y");
    adv_vz_cell_side_z_loc = glGetUniformLocation(advect_vz_Program, "cell_side_z");
    adv_vz_stable_dt_loc = glGetUniformLocation(advect_vz_Program, "stable_dt");



    residual_current_cells_x_loc = glGetUniformLocation(multigrid_pressure_residual_calculation_program, "current_cells_x");
    residual_current_cells_y_loc = glGetUniformLocation(multigrid_pressure_residual_calculation_program, "current_cells_y");
    residual_current_cells_z_loc = glGetUniformLocation(multigrid_pressure_residual_calculation_program, "current_cells_z");
    residual_dx_loc = glGetUniformLocation(multigrid_pressure_residual_calculation_program, "dx");
    residual_dy_loc = glGetUniformLocation(multigrid_pressure_residual_calculation_program, "dy");
    residual_dz_loc = glGetUniformLocation(multigrid_pressure_residual_calculation_program, "dz");
    residual_stable_dt_loc = glGetUniformLocation(multigrid_pressure_residual_calculation_program, "stable_dt");
    residual_density_loc = glGetUniformLocation(multigrid_pressure_residual_calculation_program, "density");

    generic_residual_current_cells_x_loc = glGetUniformLocation(multigrid_generic_residual_calculation_program, "current_cells_x");
    generic_residual_current_cells_y_loc = glGetUniformLocation(multigrid_generic_residual_calculation_program, "current_cells_y");
    generic_residual_current_cells_z_loc = glGetUniformLocation(multigrid_generic_residual_calculation_program, "current_cells_z");

    smoother_cells_x_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "cells_x");
    smoother_cells_y_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "cells_y");
    smoother_cells_z_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "cells_z");
    smoother_cell_side_x_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "cell_side_x");
    smoother_cell_side_y_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "cell_side_y");
    smoother_cell_side_z_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "cell_side_z");
    smoother_stable_dt_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "stable_dt");
    smoother_density_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "density");
    smoother_color_loc = glGetUniformLocation(multigrid_pressure_smoother_program, "color");

    Generic_smoother_color_loc = glGetUniformLocation(multigrid_generic_smoother_program, "color");
    Generic_smoother_cells_x_loc = glGetUniformLocation(multigrid_generic_smoother_program, "cells_x");
    Generic_smoother_cells_y_loc = glGetUniformLocation(multigrid_generic_smoother_program, "cells_y");
    Generic_smoother_cells_z_loc = glGetUniformLocation(multigrid_generic_smoother_program, "cells_z");

    generic_restriction_fine_dimensions_loc = glGetUniformLocation(physical_fields_multigrid_generic_restriction_program, "fineDimensions");
    generic_restriction_coarse_dimensions_loc = glGetUniformLocation(physical_fields_multigrid_generic_restriction_program, "coarseDimensions");
    generic_restriction_topology_loc = glGetUniformLocation(physical_fields_multigrid_generic_restriction_program, "topology");

    residual_generic_restriction_fine_dimensions_loc = glGetUniformLocation(multigrid_residual_generic_restriction_program, "fineDimensions");
    residual_generic_restriction_coarse_dimensions_loc = glGetUniformLocation(multigrid_residual_generic_restriction_program, "coarseDimensions");
    residual_generic_restriction_topology_loc = glGetUniformLocation(multigrid_residual_generic_restriction_program, "topology");

    generic_prolongation_fine_dimensions_loc = glGetUniformLocation(multigrid_generic_prolongation_program, "fineDimensions");
    generic_prolongation_coarse_dimensions_loc = glGetUniformLocation(multigrid_generic_prolongation_program, "coarseDimensions");
    generic_prolongation_topology_loc = glGetUniformLocation(multigrid_generic_prolongation_program, "topology");

    projX_cells_x_loc = glGetUniformLocation(projectionXProgram, "cells_x");
    projX_cells_y_loc = glGetUniformLocation(projectionXProgram, "cells_y");
    projX_cells_z_loc = glGetUniformLocation(projectionXProgram, "cells_z");
    projX_cell_side_x_loc = glGetUniformLocation(projectionXProgram, "cell_side_x");
    projX_stable_dt_loc = glGetUniformLocation(projectionXProgram, "stable_dt");
    projX_density_loc = glGetUniformLocation(projectionXProgram, "density");

    projY_cells_x_loc = glGetUniformLocation(projectionYProgram, "cells_x");
    projY_cells_y_loc = glGetUniformLocation(projectionYProgram, "cells_y");
    projY_cells_z_loc = glGetUniformLocation(projectionYProgram, "cells_z");
    projY_cell_side_y_loc = glGetUniformLocation(projectionYProgram, "cell_side_y");
    projY_stable_dt_loc = glGetUniformLocation(projectionYProgram, "stable_dt");
    projY_density_loc = glGetUniformLocation(projectionYProgram, "density");

    projZ_cells_x_loc = glGetUniformLocation(projectionZProgram, "cells_x");
    projZ_cells_y_loc = glGetUniformLocation(projectionZProgram, "cells_y");
    projZ_cells_z_loc = glGetUniformLocation(projectionZProgram, "cells_z");
    projZ_cell_side_z_loc = glGetUniformLocation(projectionZProgram, "cell_side_z");
    projZ_stable_dt_loc = glGetUniformLocation(projectionZProgram, "stable_dt");
    projZ_density_loc = glGetUniformLocation(projectionZProgram, "density");

    cfl_elements_count_loc = glGetUniformLocation(CFl_comp_program, "elements_count");
    cfl_total_groups_loc = glGetUniformLocation(CFl_comp_program, "total_groups");

    final_blend_SSPRK2_count_loc = glGetUniformLocation(final_blend_SSPRK2_program, "count");

    ray_march_image_height_loc = glGetUniformLocation(ray_march_program, "image_height");
    ray_march_image_width_loc = glGetUniformLocation(ray_march_program, "image_width");
    ray_march_halfWidth_loc = glGetUniformLocation(ray_march_program, "halfWidth");
    ray_march_halfHeight_loc = glGetUniformLocation(ray_march_program, "halfHeight");
    ray_march_cameraPosition_loc = glGetUniformLocation(ray_march_program, "cameraPosition");
    ray_march_cameraForward_loc = glGetUniformLocation(ray_march_program, "cameraForward");
    ray_march_cameraUp_loc = glGetUniformLocation(ray_march_program, "cameraUp");
    ray_march_cameraRight_loc = glGetUniformLocation(ray_march_program, "cameraRight");
    ray_march_volume_width_loc = glGetUniformLocation(ray_march_program, "volume_width");
    ray_march_volume_height_loc = glGetUniformLocation(ray_march_program, "volume_height");
    ray_march_volume_depth_loc = glGetUniformLocation(ray_march_program, "volume_depth");
    ray_march_ds_loc = glGetUniformLocation(ray_march_program, "ds");
    ray_march_field_stride_loc = glGetUniformLocation(ray_march_program, "field_stride");
    ray_march_field_layer_stride_loc = glGetUniformLocation(ray_march_program, "field_layer_stride");
    ray_march_field_x_elements_loc = glGetUniformLocation(ray_march_program, "field_x_elements");
    ray_march_field_y_elements_loc = glGetUniformLocation(ray_march_program, "field_y_elements");
    ray_march_field_z_elements_loc = glGetUniformLocation(ray_march_program, "field_z_elements");
    ray_march_light_x_loc = glGetUniformLocation(ray_march_program, "light_x");
    ray_march_light_y_loc = glGetUniformLocation(ray_march_program, "light_y");
    ray_march_light_z_loc = glGetUniformLocation(ray_march_program, "light_z");
    ray_march_light_intensity_loc = glGetUniformLocation(ray_march_program, "light_intensity");
    ray_march_extinction_total_loc = glGetUniformLocation(ray_march_program, "extinction_total");
    ray_march_extinction_via_scattering_loc = glGetUniformLocation(ray_march_program, "extinction_via_scattering");
    ray_march_for_light_ds_loc = glGetUniformLocation(ray_march_program, "for_light_ds");
    ray_march_mean_cosine_loc = glGetUniformLocation(ray_march_program, "mean_cosine");
    ray_march_smoke_color_min = glGetUniformLocation(ray_march_program, "smoke_color_min");
    ray_march_smoke_color_max = glGetUniformLocation(ray_march_program, "smoke_color_max");

    field_slice_image_width_loc = glGetUniformLocation(field_normalized_slice_program, "image_width");
    field_slice_image_height_loc = glGetUniformLocation(field_normalized_slice_program, "image_height");
    field_slice_cells_x_loc = glGetUniformLocation(field_normalized_slice_program, "cells_x");
    field_slice_cells_y_loc = glGetUniformLocation(field_normalized_slice_program, "cells_y");
    field_slice_cells_z_loc = glGetUniformLocation(field_normalized_slice_program, "cells_z");
    field_slice_topology_loc = glGetUniformLocation(field_normalized_slice_program, "topology");
    field_slice_slice_id_loc = glGetUniformLocation(field_normalized_slice_program, "slice_id");

    smoke_advection_cells_x_loc = glGetUniformLocation(smoke_advection_program, "cells_x");
    smoke_advection_cells_y_loc = glGetUniformLocation(smoke_advection_program, "cells_y");
    smoke_advection_cells_z_loc = glGetUniformLocation(smoke_advection_program, "cells_z");
    smoke_advection_cell_side_x_loc = glGetUniformLocation(smoke_advection_program, "dx");
    smoke_advection_cell_side_y_loc = glGetUniformLocation(smoke_advection_program, "dy");
    smoke_advection_cell_side_z_loc = glGetUniformLocation(smoke_advection_program, "dz");

    light_transmittance_volume_width_loc = glGetUniformLocation(transmittance_program, "volume_width");
    light_transmittance_volume_height_loc = glGetUniformLocation(transmittance_program, "volume_height");
    light_transmittance_volume_depth_loc = glGetUniformLocation(transmittance_program, "volume_depth");
    light_transmittance_for_light_ds_loc = glGetUniformLocation(transmittance_program, "for_light_ds");
    light_transmittance_field_stride_loc = glGetUniformLocation(transmittance_program, "field_stride");
    light_transmittance_field_layer_stride_loc = glGetUniformLocation(transmittance_program, "field_layer_stride");
    light_transmittance_field_x_elements_loc = glGetUniformLocation(transmittance_program, "field_x_elements");
    light_transmittance_field_y_elements_loc = glGetUniformLocation(transmittance_program, "field_y_elements");
    light_transmittance_field_z_elements_loc = glGetUniformLocation(transmittance_program, "field_z_elements");
    light_transmittance_light_x_loc = glGetUniformLocation(transmittance_program, "light_x");
    light_transmittance_light_y_loc = glGetUniformLocation(transmittance_program, "light_y");
    light_transmittance_light_z_loc = glGetUniformLocation(transmittance_program, "light_z");
    light_transmittance_extinction_total_loc = glGetUniformLocation(transmittance_program, "extinction_total");

    fragment_backgroundColor_loc = glGetUniformLocation(displayProgram, "backgroundColor");

    time_assemble_x_elements = glGetUniformLocation(time_assemble, "grid_x");
    time_assemble_y_elements = glGetUniformLocation(time_assemble, "grid_y");
    time_assemble_z_elements = glGetUniformLocation(time_assemble, "grid_z");
    time_assemble_dx = glGetUniformLocation(time_assemble, "dx");
    time_assemble_dy = glGetUniformLocation(time_assemble, "dy");
    time_assemble_dz = glGetUniformLocation(time_assemble, "dz");
    time_assemble_dt = glGetUniformLocation(time_assemble, "dt");
    time_assemble_rho = glGetUniformLocation(time_assemble, "density");
    time_assemble_topology = glGetUniformLocation(time_assemble, "topology");
    time_assemble_coarse_assembly = glGetUniformLocation(time_assemble, "coarse_assembly");
    time_assemble_temporal_scheme = glGetUniformLocation(time_assemble, "tempoaral_scheme");

    momentum_viscosity_LHS_assemble_cells_x = glGetUniformLocation(viscosity_LHS_assemble, "cells_x");
    momentum_viscosity_LHS_assemble_cells_y = glGetUniformLocation(viscosity_LHS_assemble, "cells_y");
    momentum_viscosity_LHS_assemble_cells_z = glGetUniformLocation(viscosity_LHS_assemble, "cells_z");
    momentum_viscosity_LHS_assemble_dx = glGetUniformLocation(viscosity_LHS_assemble, "dx");
    momentum_viscosity_LHS_assemble_dy = glGetUniformLocation(viscosity_LHS_assemble, "dy");
    momentum_viscosity_LHS_assemble_dz = glGetUniformLocation(viscosity_LHS_assemble, "dz");
    momentum_viscosity_LHS_assemble_topology = glGetUniformLocation(viscosity_LHS_assemble, "topology");
    
    momentum_viscosity_RHS_assemble_cells_x = glGetUniformLocation(viscosity_RHS_assemble, "cells_x");
    momentum_viscosity_RHS_assemble_cells_y = glGetUniformLocation(viscosity_RHS_assemble, "cells_y");
    momentum_viscosity_RHS_assemble_cells_z = glGetUniformLocation(viscosity_RHS_assemble, "cells_z");
    momentum_viscosity_RHS_assemble_dx = glGetUniformLocation(viscosity_RHS_assemble, "dx");
    momentum_viscosity_RHS_assemble_dy = glGetUniformLocation(viscosity_RHS_assemble, "dy");
    momentum_viscosity_RHS_assemble_dz = glGetUniformLocation(viscosity_RHS_assemble, "dz");
    momentum_viscosity_RHS_assemble_topology = glGetUniformLocation(viscosity_RHS_assemble, "topology");

    momentum_wall_assemble_cells_x = glGetUniformLocation(momentum_wall_assemble, "cells_x");
    momentum_wall_assemble_cells_y = glGetUniformLocation(momentum_wall_assemble, "cells_y");
    momentum_wall_assemble_cells_z = glGetUniformLocation(momentum_wall_assemble, "cells_z");
    momentum_wall_assemble_dx = glGetUniformLocation(momentum_wall_assemble, "dx");
    momentum_wall_assemble_dy = glGetUniformLocation(momentum_wall_assemble, "dy");
    momentum_wall_assemble_dz = glGetUniformLocation(momentum_wall_assemble, "dz");
    momentum_wall_assemble_dynamic_viscosity = glGetUniformLocation(momentum_wall_assemble, "dynamic_viscosity");
    momentum_wall_assemble_density = glGetUniformLocation(momentum_wall_assemble, "density");
    momentum_wall_assemble_spalding_wall_smoothness = glGetUniformLocation(momentum_wall_assemble, "spalding_wall_smoothness");
    momentum_wall_assemble_topology = glGetUniformLocation(momentum_wall_assemble, "topology");
    
    advection_assemble_cells_x = glGetUniformLocation(advection_assemble, "cells_x");
    advection_assemble_cells_y = glGetUniformLocation(advection_assemble, "cells_y");
    advection_assemble_cells_z = glGetUniformLocation(advection_assemble, "cells_z");
    advection_assemble_dx = glGetUniformLocation(advection_assemble, "dx");
    advection_assemble_dy = glGetUniformLocation(advection_assemble, "dy");
    advection_assemble_dz = glGetUniformLocation(advection_assemble, "dz");
    advection_assemble_topology = glGetUniformLocation(advection_assemble, "topology");
    advection_assemble_density = glGetUniformLocation(advection_assemble, "density");
    advection_assemble_ConvectionScheme = glGetUniformLocation(advection_assemble, "ConvectionScheme");
    advection_assemble_FluxLimiter = glGetUniformLocation(advection_assemble, "FluxLimiter");

    assemble_production_k_epsilon_omega_grid_x = glGetUniformLocation(assemble_production_k_epsilon_omega, "grid_x");
    assemble_production_k_epsilon_omega_grid_y = glGetUniformLocation(assemble_production_k_epsilon_omega, "grid_y");
    assemble_production_k_epsilon_omega_grid_z = glGetUniformLocation(assemble_production_k_epsilon_omega, "grid_z");
    assemble_production_k_epsilon_omega_dx = glGetUniformLocation(assemble_production_k_epsilon_omega, "dx");
    assemble_production_k_epsilon_omega_dy = glGetUniformLocation(assemble_production_k_epsilon_omega, "dy");
    assemble_production_k_epsilon_omega_dz = glGetUniformLocation(assemble_production_k_epsilon_omega, "dz");
    assemble_production_k_epsilon_omega_dynamic_viscosity = glGetUniformLocation(assemble_production_k_epsilon_omega, "dynamic_viscosity");
    assemble_production_k_epsilon_omega_is_omega_model = glGetUniformLocation(assemble_production_k_epsilon_omega, "is_omega_model");
    assemble_production_k_epsilon_omega_is_k = glGetUniformLocation(assemble_production_k_epsilon_omega, "is_k");
    assemble_production_k_epsilon_omega_density = glGetUniformLocation(assemble_production_k_epsilon_omega, "density");
    assemble_production_k_epsilon_omega_is_sst = glGetUniformLocation(assemble_production_k_epsilon_omega, "is_sst");

    destruction_k_and_epsilon_or_omega_assemble_grid_x = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "grid_x");
    destruction_k_and_epsilon_or_omega_assemble_grid_y = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "grid_y");
    destruction_k_and_epsilon_or_omega_assemble_grid_z = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "grid_z");
    destruction_k_and_epsilon_or_omega_assemble_dx = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "dx");
    destruction_k_and_epsilon_or_omega_assemble_dy = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "dy");
    destruction_k_and_epsilon_or_omega_assemble_dz = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "dz");
    destruction_k_and_epsilon_or_omega_assemble_density = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "density");
    destruction_k_and_epsilon_or_omega_assemble_is_omega_model = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "is_omega_model");
    destruction_k_and_epsilon_or_omega_assemble_is_k = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "is_k");
    destruction_k_and_epsilon_or_omega_assemble_is_sst = glGetUniformLocation(destruction_k_and_epsilon_or_omega_assemble, "is_sst");

    diffusion_k_eps_omega_assemble_grid_x = glGetUniformLocation(diffusion_k_eps_omega_assemble, "grid_x");
    diffusion_k_eps_omega_assemble_grid_y = glGetUniformLocation(diffusion_k_eps_omega_assemble, "grid_y");
    diffusion_k_eps_omega_assemble_grid_z = glGetUniformLocation(diffusion_k_eps_omega_assemble, "grid_z");
    diffusion_k_eps_omega_assemble_dx = glGetUniformLocation(diffusion_k_eps_omega_assemble, "dx");
    diffusion_k_eps_omega_assemble_dy = glGetUniformLocation(diffusion_k_eps_omega_assemble, "dy");
    diffusion_k_eps_omega_assemble_dz = glGetUniformLocation(diffusion_k_eps_omega_assemble, "dz");
    diffusion_k_eps_omega_assemble_is_eps = glGetUniformLocation(diffusion_k_eps_omega_assemble, "is_eps");
    diffusion_k_eps_omega_assemble_is_omega_model = glGetUniformLocation(diffusion_k_eps_omega_assemble, "is_omega_model");
    diffusion_k_eps_omega_assemble_dynamic_viscosity = glGetUniformLocation(diffusion_k_eps_omega_assemble, "dynamic_viscosity");

    k_wall_treatment_for_k_eps_model_grid_x = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "grid_x");
    k_wall_treatment_for_k_eps_model_grid_y = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "grid_y");
    k_wall_treatment_for_k_eps_model_grid_z = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "grid_z");
    k_wall_treatment_for_k_eps_model_dx = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "dx");
    k_wall_treatment_for_k_eps_model_dy = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "dy");
    k_wall_treatment_for_k_eps_model_dz = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "dz");
    k_wall_treatment_for_k_eps_model_dynamic_viscosity = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "dynamic_viscosity");
    k_wall_treatment_for_k_eps_model_density = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "density");
    k_wall_treatment_for_k_eps_model_spalding_wall_smoothness = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "spalding_wall_smoothness");
    k_wall_treatment_for_k_eps_model_coarse_assembly = glGetUniformLocation(k_wall_treatment_for_k_eps_model, "coarse_assembly");

    eps_wall_treatment_for_k_eps_model_grid_x = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "grid_x");
    eps_wall_treatment_for_k_eps_model_grid_y = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "grid_y");
    eps_wall_treatment_for_k_eps_model_grid_z = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "grid_z");
    eps_wall_treatment_for_k_eps_model_dx = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "dx");
    eps_wall_treatment_for_k_eps_model_dy = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "dy");
    eps_wall_treatment_for_k_eps_model_dz = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "dz");
    eps_wall_treatment_for_k_eps_model_dynamic_viscosity = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "dynamic_viscosity");
    eps_wall_treatment_for_k_eps_model_density = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "density");
    eps_wall_treatment_for_k_eps_model_spalding_wall_smoothness = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "spalding_wall_smoothness");
    eps_wall_treatment_for_k_eps_model_coarse_assembly = glGetUniformLocation(eps_wall_treatment_for_k_eps_model, "coarse_assembly");

    k_wall_treatment_for_k_omega_model_grid_x = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "grid_x");
    k_wall_treatment_for_k_omega_model_grid_y = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "grid_y");
    k_wall_treatment_for_k_omega_model_grid_z = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "grid_z");
    k_wall_treatment_for_k_omega_model_dx = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "dx");
    k_wall_treatment_for_k_omega_model_dy = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "dy");
    k_wall_treatment_for_k_omega_model_dz = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "dz");
    k_wall_treatment_for_k_omega_model_dynamic_viscosity = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "dynamic_viscosity");
    k_wall_treatment_for_k_omega_model_density = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "density");
    k_wall_treatment_for_k_omega_model_spalding_wall_smoothness = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "spalding_wall_smoothness");
    k_wall_treatment_for_k_omega_model_coarse_assembly = glGetUniformLocation(k_wall_treatment_for_k_omega_model, "coarse_assembly");

    omega_wall_treatment_for_k_omega_model_grid_x = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "grid_x");
    omega_wall_treatment_for_k_omega_model_grid_y = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "grid_y");
    omega_wall_treatment_for_k_omega_model_grid_z = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "grid_z");
    omega_wall_treatment_for_k_omega_model_dx = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "dx");
    omega_wall_treatment_for_k_omega_model_dy = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "dy");
    omega_wall_treatment_for_k_omega_model_dz = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "dz");
    omega_wall_treatment_for_k_omega_model_dynamic_viscosity = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "dynamic_viscosity");
    omega_wall_treatment_for_k_omega_model_density = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "density");
    omega_wall_treatment_for_k_omega_model_spalding_wall_smoothness = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "spalding_wall_smoothness");
    omega_wall_treatment_for_k_omega_model_coarse_assembly = glGetUniformLocation(omega_wall_treatment_for_k_omega_model, "coarse_assembly");


    k_eps_mu_eff_grid_x = glGetUniformLocation(k_eps_mu_eff, "grid_x");
    k_eps_mu_eff_grid_y = glGetUniformLocation(k_eps_mu_eff, "grid_y");
    k_eps_mu_eff_grid_z = glGetUniformLocation(k_eps_mu_eff, "grid_z");
    k_eps_mu_eff_dynamic_viscosity = glGetUniformLocation(k_eps_mu_eff, "dynamic_viscosity");
    k_eps_mu_eff_density = glGetUniformLocation(k_eps_mu_eff, "density");

    k_omega_mu_eff_grid_x = glGetUniformLocation(k_omega_mu_eff, "grid_x");
    k_omega_mu_eff_grid_y = glGetUniformLocation(k_omega_mu_eff, "grid_y");
    k_omega_mu_eff_grid_z = glGetUniformLocation(k_omega_mu_eff, "grid_z");
    k_omega_mu_eff_dynamic_viscosity = glGetUniformLocation(k_omega_mu_eff, "dynamic_viscosity");
    k_omega_mu_eff_density = glGetUniformLocation(k_omega_mu_eff, "density");

    k_eps_initialize_grid_x = glGetUniformLocation(k_eps_initialize, "grid_x");
    k_eps_initialize_grid_y = glGetUniformLocation(k_eps_initialize, "grid_y");
    k_eps_initialize_grid_z = glGetUniformLocation(k_eps_initialize, "grid_z");
    k_eps_initialize_reference_velocity = glGetUniformLocation(k_eps_initialize, "reference_velocity");
    k_eps_initialize_turbulence_intensity = glGetUniformLocation(k_eps_initialize, "turbulence_intensity");
    k_eps_initialize_turbulence_length_scale = glGetUniformLocation(k_eps_initialize, "turbulence_length_scale");

    k_omega_initialize_grid_x = glGetUniformLocation(k_omega_initialize, "grid_x");
    k_omega_initialize_grid_y = glGetUniformLocation(k_omega_initialize, "grid_y");
    k_omega_initialize_grid_z = glGetUniformLocation(k_omega_initialize, "grid_z");
    k_omega_initialize_reference_velocity = glGetUniformLocation(k_omega_initialize, "reference_velocity");
    k_omega_initialize_turbulence_intensity = glGetUniformLocation(k_omega_initialize, "turbulence_intensity");
    k_omega_initialize_turbulence_length_scale = glGetUniformLocation(k_omega_initialize, "turbulence_length_scale");

    l2_field_x = glGetUniformLocation(L2_norm_summation_of_sq, "field_x");
    l2_field_y = glGetUniformLocation(L2_norm_summation_of_sq, "field_y");
    l2_field_z = glGetUniformLocation(L2_norm_summation_of_sq, "field_z");
    l2_squared_input = glGetUniformLocation(L2_norm_summation_of_sq, "squared_input");

    energy_reduction_grid_x = glGetUniformLocation(energy_reduction, "grid_x");
    energy_reduction_grid_y = glGetUniformLocation(energy_reduction, "grid_y");
    energy_reduction_grid_z = glGetUniformLocation(energy_reduction, "grid_z");
    energy_reduction_mode = glGetUniformLocation(energy_reduction, "mode");

    diff_sq_field_x = glGetUniformLocation(difference_sq_summation, "field_x");
    diff_sq_field_y = glGetUniformLocation(difference_sq_summation, "field_y");
    diff_sq_field_z = glGetUniformLocation(difference_sq_summation, "field_z");

    min_wall_dists_compute_grid_x = glGetUniformLocation(compute_min_wall_dists_structured_mesh, "grid_x");
    min_wall_dists_compute_grid_y = glGetUniformLocation(compute_min_wall_dists_structured_mesh, "grid_y");
    min_wall_dists_compute_grid_z = glGetUniformLocation(compute_min_wall_dists_structured_mesh, "grid_z");
    min_wall_dists_compute_dx = glGetUniformLocation(compute_min_wall_dists_structured_mesh, "dx");
    min_wall_dists_compute_dy = glGetUniformLocation(compute_min_wall_dists_structured_mesh, "dy");
    min_wall_dists_compute_dz = glGetUniformLocation(compute_min_wall_dists_structured_mesh, "dz");

    blended_coefficients_SST_grid_x = glGetUniformLocation(compute_blended_coefficients_SST, "grid_x");
    blended_coefficients_SST_grid_y = glGetUniformLocation(compute_blended_coefficients_SST, "grid_y");
    blended_coefficients_SST_grid_z = glGetUniformLocation(compute_blended_coefficients_SST, "grid_z");
    blended_coefficients_SST_dx = glGetUniformLocation(compute_blended_coefficients_SST, "dx");
    blended_coefficients_SST_dy = glGetUniformLocation(compute_blended_coefficients_SST, "dy");
    blended_coefficients_SST_dz = glGetUniformLocation(compute_blended_coefficients_SST, "dz");
    blended_coefficients_SST_dynamic_viscosity = glGetUniformLocation(compute_blended_coefficients_SST, "dynamic_viscosity");
    blended_coefficients_SST_density = glGetUniformLocation(compute_blended_coefficients_SST, "density");
    blended_coefficients_SST_omega_min = glGetUniformLocation(compute_blended_coefficients_SST, "omega_min");

    k_omega_SST_mu_eff_grid_x = glGetUniformLocation(k_omega_SST_mu_eff, "grid_x");
    k_omega_SST_mu_eff_grid_y = glGetUniformLocation(k_omega_SST_mu_eff, "grid_y");
    k_omega_SST_mu_eff_grid_z = glGetUniformLocation(k_omega_SST_mu_eff, "grid_z");
    k_omega_SST_mu_eff_dx = glGetUniformLocation(k_omega_SST_mu_eff, "dx");
    k_omega_SST_mu_eff_dy = glGetUniformLocation(k_omega_SST_mu_eff, "dy");
    k_omega_SST_mu_eff_dz = glGetUniformLocation(k_omega_SST_mu_eff, "dz");
    k_omega_SST_mu_eff_dynamic_viscosity = glGetUniformLocation(k_omega_SST_mu_eff, "dynamic_viscosity");
    k_omega_SST_mu_eff_density = glGetUniformLocation(k_omega_SST_mu_eff, "density");

    assemble_cross_diffusion_SST_grid_x = glGetUniformLocation(assemble_cross_diffusion_SST, "grid_x");
    assemble_cross_diffusion_SST_grid_y = glGetUniformLocation(assemble_cross_diffusion_SST, "grid_y");
    assemble_cross_diffusion_SST_grid_z = glGetUniformLocation(assemble_cross_diffusion_SST, "grid_z");
    assemble_cross_diffusion_SST_dx = glGetUniformLocation(assemble_cross_diffusion_SST, "dx");
    assemble_cross_diffusion_SST_dy = glGetUniformLocation(assemble_cross_diffusion_SST, "dy");
    assemble_cross_diffusion_SST_dz = glGetUniformLocation(assemble_cross_diffusion_SST, "dz");
    assemble_cross_diffusion_SST_density = glGetUniformLocation(assemble_cross_diffusion_SST, "density");
    assemble_cross_diffusion_SST_omega_min = glGetUniformLocation(assemble_cross_diffusion_SST, "omega_min");

    diffusion_k_omega_SST_assemble_grid_x = glGetUniformLocation(diffusion_k_omega_SST_assemble, "grid_x");
    diffusion_k_omega_SST_assemble_grid_y = glGetUniformLocation(diffusion_k_omega_SST_assemble, "grid_y");
    diffusion_k_omega_SST_assemble_grid_z = glGetUniformLocation(diffusion_k_omega_SST_assemble, "grid_z");
    diffusion_k_omega_SST_assemble_dx = glGetUniformLocation(diffusion_k_omega_SST_assemble, "dx");
    diffusion_k_omega_SST_assemble_dy = glGetUniformLocation(diffusion_k_omega_SST_assemble, "dy");
    diffusion_k_omega_SST_assemble_dz = glGetUniformLocation(diffusion_k_omega_SST_assemble, "dz");
    diffusion_k_omega_SST_assemble_dynamic_viscosity = glGetUniformLocation(diffusion_k_omega_SST_assemble, "dynamic_viscosity");
    diffusion_k_omega_SST_assemble_is_omega_model = glGetUniformLocation(diffusion_k_omega_SST_assemble, "is_omega_model");
    diffusion_k_omega_SST_assemble_is_k = glGetUniformLocation(diffusion_k_omega_SST_assemble, "is_k");

    non_physical_turbulence_quantites_check_cells_x = glGetUniformLocation(non_physical_turbulence_quantites_check_program, "cells_x");
    non_physical_turbulence_quantites_check_cells_y = glGetUniformLocation(non_physical_turbulence_quantites_check_program, "cells_y");
    non_physical_turbulence_quantites_check_cells_z = glGetUniformLocation(non_physical_turbulence_quantites_check_program, "cells_z");

    clamp_to_floor_element_count = glGetUniformLocation(clamp_to_floor_program, "element_count");
    clamp_to_floor_value = glGetUniformLocation(clamp_to_floor_program, "floor_value");

}

ShaderManager::~ShaderManager() = default;