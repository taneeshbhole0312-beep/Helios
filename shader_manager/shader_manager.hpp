#pragma once

#include <glad/glad.h>
#include <string>

class ShaderManager
{
private:

    // --------------------------------------------------
    // Shader loading / creation
    // --------------------------------------------------

    std::string loadFile(const char* path);

    GLuint createComputeShader(const char* path);

    GLuint createShader(
        const char* vertexPath,
        const char* fragmentPath
    );


public:

    // --------------------------------------------------
    // Programs
    // --------------------------------------------------

    GLuint divergence_compute_program;

    GLuint viscosityXProgram;
    GLuint viscosityYProgram;
    GLuint viscosityZProgram;

    GLuint advect_vx_Program;
    GLuint advect_vy_Program;
    GLuint advect_vz_Program;

    GLuint multigrid_pressure_residual_calculation_program;
    GLuint multigrid_pressure_smoother_program;

    GLuint multigrid_generic_residual_calculation_program;
    GLuint multigrid_generic_smoother_program;
    GLuint physical_fields_multigrid_generic_restriction_program;
    GLuint multigrid_generic_prolongation_program;
    
    GLuint multigrid_residual_generic_restriction_program;
    
    GLuint viscosity_LHS_assemble;      // specialized for momentum equations
    GLuint viscosity_RHS_assemble;      // specialized for momentum equations
    GLuint momentum_wall_assemble;      // Spalding wall shear, implicit (aP only), fine AND coarse momentum operators
    
    GLuint compute_min_wall_dists_structured_mesh;
    GLuint k_omega_SST_mu_eff;                         // SST mu_eff (mu_t with the a1 limiter and F2)
    GLuint assemble_cross_diffusion_SST;              // SST cross diffusion source of the omega equation
    GLuint compute_blended_coefficients_SST;          // SST blending function F1 field
    GLuint diffusion_k_omega_SST_assemble;                  // k / omega diffusion with the F1-blended sigma_k, sigma_omega (SST model)

    GLuint time_assemble;         //Common / Generalized infra for all topologies
    GLuint advection_assemble;            //Common / Generalized infra for all topologies
    
    GLuint assemble_production_k_epsilon_omega;
    GLuint destruction_k_and_epsilon_or_omega_assemble;
    GLuint diffusion_k_eps_omega_assemble;
    
    // Spalding wall function, one kernel per transported variable (k-epsilon model)
    GLuint k_wall_treatment_for_k_eps_model;             // k in wall cells
    GLuint eps_wall_treatment_for_k_eps_model;           // epsilon in wall cells
    GLuint k_eps_mu_eff;                  // mu_eff = mu + rho C_mu k^2 / eps
    GLuint k_eps_initialize;              // uniform k0, eps0 from reference velocity / turbulence intensity / length scale
    GLuint k_omega_initialize;            // uniform k0, omega0 from reference velocity / turbulence intensity / length scale

    GLuint k_omega_mu_eff;                  // mu_eff = mu + rho * k / omega
    // Spalding wall function, one kernel per transported variable (k-omega model)
    GLuint k_wall_treatment_for_k_omega_model;           // k in wall cells
    GLuint omega_wall_treatment_for_k_omega_model;       // omega in wall cells, and mu_eff (= mu + mu_t_wall) in wall cells

    GLuint L2_norm_summation_of_sq;
    GLuint difference_sq_summation;       // first pass of sum((new - old)^2); later passes use L2_norm_summation_of_sq
    GLuint energy_reduction;              // first pass of the mean flow / turbulent kinetic energy sums (debuggers); later passes use L2_norm_summation_of_sq
    
    GLuint projectionXProgram;
    GLuint projectionYProgram;
    GLuint projectionZProgram;

    GLuint CFl_comp_program;

    GLuint ray_march_program;
    GLuint field_normalized_slice_program;      // visualises any field (cell / u / v / w) on one slice, normalised by max |field|

    GLuint final_blend_SSPRK2_program;

    GLuint smoke_advection_program;

    GLuint transmittance_program;

    GLuint displayProgram;

    GLuint non_physical_turbulence_quantites_check_program;
    GLuint clamp_to_floor_program;                    // x = max(x, floor) on a scalar field, NaN left untouched


    // --------------------------------------------------
    // Divergence uniform locations
    // --------------------------------------------------

    GLint div_cells_x_loc;
    GLint div_cells_y_loc;
    GLint div_cells_z_loc;

    GLint div_cell_side_x_loc;
    GLint div_cell_side_y_loc;
    GLint div_cell_side_z_loc;

    // --------------------------------------------------
    // Viscosity VX uniform locations
    // --------------------------------------------------

    GLint visc_vx_cells_x_loc;
    GLint visc_vx_cells_y_loc;
    GLint visc_vx_cells_z_loc;

    GLint visc_vx_viscosity_loc;
    GLint visc_vx_density_loc;
    GLint visc_vx_stable_dt_loc;

    GLint visc_vx_cell_side_x_loc;
    GLint visc_vx_cell_side_y_loc;
    GLint visc_vx_cell_side_z_loc;


    // --------------------------------------------------
    // Viscosity VY uniform locations
    // --------------------------------------------------

    GLint visc_vy_cells_x_loc;
    GLint visc_vy_cells_y_loc;
    GLint visc_vy_cells_z_loc;

    GLint visc_vy_viscosity_loc;
    GLint visc_vy_density_loc;
    GLint visc_vy_stable_dt_loc;

    GLint visc_vy_cell_side_x_loc;
    GLint visc_vy_cell_side_y_loc;
    GLint visc_vy_cell_side_z_loc;


    // --------------------------------------------------
    // Viscosity VZ uniform locations
    // --------------------------------------------------

    GLint visc_vz_cells_x_loc;
    GLint visc_vz_cells_y_loc;
    GLint visc_vz_cells_z_loc;

    GLint visc_vz_viscosity_loc;
    GLint visc_vz_density_loc;
    GLint visc_vz_stable_dt_loc;

    GLint visc_vz_cell_side_x_loc;
    GLint visc_vz_cell_side_y_loc;
    GLint visc_vz_cell_side_z_loc;


    // --------------------------------------------------
    // Advection VX uniform locations
    // --------------------------------------------------

    GLint adv_vx_cells_x_loc;
    GLint adv_vx_cells_y_loc;
    GLint adv_vx_cells_z_loc;

    GLint adv_vx_cell_side_x_loc;
    GLint adv_vx_cell_side_y_loc;
    GLint adv_vx_cell_side_z_loc;

    GLint adv_vx_stable_dt_loc;


    // --------------------------------------------------
    // Advection VY uniform locations
    // --------------------------------------------------

    GLint adv_vy_cells_x_loc;
    GLint adv_vy_cells_y_loc;
    GLint adv_vy_cells_z_loc;

    GLint adv_vy_cell_side_x_loc;
    GLint adv_vy_cell_side_y_loc;
    GLint adv_vy_cell_side_z_loc;

    GLint adv_vy_stable_dt_loc;


    // --------------------------------------------------
    // Advection VZ uniform locations
    // --------------------------------------------------

    GLint adv_vz_cells_x_loc;
    GLint adv_vz_cells_y_loc;
    GLint adv_vz_cells_z_loc;

    GLint adv_vz_cell_side_x_loc;
    GLint adv_vz_cell_side_y_loc;
    GLint adv_vz_cell_side_z_loc;

    GLint adv_vz_stable_dt_loc;



    // --------------------------------------------------
    // Multigrid residual uniform locations
    // --------------------------------------------------

    GLint residual_current_cells_x_loc;
    GLint residual_current_cells_y_loc;
    GLint residual_current_cells_z_loc;

    GLint residual_dx_loc;
    GLint residual_dy_loc;
    GLint residual_dz_loc;

    GLint residual_stable_dt_loc;
    GLint residual_density_loc;

    GLint generic_residual_current_cells_x_loc;
    GLint generic_residual_current_cells_y_loc;
    GLint generic_residual_current_cells_z_loc;

    // --------------------------------------------------
    // Multigrid smoother uniform locations
    // --------------------------------------------------

    GLint smoother_cells_x_loc;
    GLint smoother_cells_y_loc;
    GLint smoother_cells_z_loc;

    GLint smoother_cell_side_x_loc;
    GLint smoother_cell_side_y_loc;
    GLint smoother_cell_side_z_loc;

    GLint smoother_stable_dt_loc;
    GLint smoother_density_loc;
    GLint smoother_color_loc;

    GLint Generic_smoother_color_loc;
    GLint Generic_smoother_cells_x_loc;
    GLint Generic_smoother_cells_y_loc;
    GLint Generic_smoother_cells_z_loc;

    // --------------------------------------------------
    // Generic multigrid restriction uniform locations
    // --------------------------------------------------

    GLint generic_restriction_fine_dimensions_loc;
    GLint generic_restriction_coarse_dimensions_loc;
    GLint generic_restriction_topology_loc;

    // --------------------------------------------------
    // Generic multigrid restriction for residual uniform locations
    // --------------------------------------------------

    GLint residual_generic_restriction_fine_dimensions_loc;
    GLint residual_generic_restriction_coarse_dimensions_loc;
    GLint residual_generic_restriction_topology_loc;

    // --------------------------------------------------
    // Generic multigrid prolongation uniform locations
    // --------------------------------------------------

    GLint generic_prolongation_fine_dimensions_loc;
    GLint generic_prolongation_coarse_dimensions_loc;
    GLint generic_prolongation_topology_loc;

    // --------------------------------------------------
    // Pressure projection X uniform locations
    // --------------------------------------------------

    GLint projX_cells_x_loc;
    GLint projX_cells_y_loc;
    GLint projX_cells_z_loc;

    GLint projX_cell_side_x_loc;
    GLint projX_stable_dt_loc;
    GLint projX_density_loc;


    // --------------------------------------------------
    // Pressure projection Y uniform locations
    // --------------------------------------------------

    GLint projY_cells_x_loc;
    GLint projY_cells_y_loc;
    GLint projY_cells_z_loc;

    GLint projY_cell_side_y_loc;
    GLint projY_stable_dt_loc;
    GLint projY_density_loc;


    // --------------------------------------------------
    // Pressure projection Z uniform locations
    // --------------------------------------------------

    GLint projZ_cells_x_loc;
    GLint projZ_cells_y_loc;
    GLint projZ_cells_z_loc;

    GLint projZ_cell_side_z_loc;
    GLint projZ_stable_dt_loc;
    GLint projZ_density_loc;


    // --------------------------------------------------
    // CFL uniform locations
    // --------------------------------------------------

    GLint cfl_elements_count_loc;
    GLint cfl_total_groups_loc;


    // --------------------------------------------------
    // SSPRK2 uniform locations
    // --------------------------------------------------

    GLint final_blend_SSPRK2_count_loc;


    // --------------------------------------------------
    // Ray marching uniform locations
    // --------------------------------------------------

    GLint ray_march_image_height_loc;
    GLint ray_march_image_width_loc;
    GLint ray_march_halfWidth_loc;
    GLint ray_march_halfHeight_loc;
    GLint ray_march_cameraPosition_loc;
    GLint ray_march_cameraForward_loc;
    GLint ray_march_cameraUp_loc;
    GLint ray_march_cameraRight_loc;
    GLint ray_march_volume_width_loc;
    GLint ray_march_volume_height_loc;
    GLint ray_march_volume_depth_loc;
    GLint ray_march_ds_loc;
    GLint ray_march_field_stride_loc;
    GLint ray_march_field_layer_stride_loc;
    GLint ray_march_field_x_elements_loc;
    GLint ray_march_field_y_elements_loc;
    GLint ray_march_field_z_elements_loc;
    GLint ray_march_light_x_loc;
    GLint ray_march_light_y_loc;
    GLint ray_march_light_z_loc;
    GLint ray_march_light_intensity_loc;
    GLint ray_march_extinction_total_loc;
    GLint ray_march_extinction_via_scattering_loc;
    GLint ray_march_for_light_ds_loc;
    GLint ray_march_mean_cosine_loc;
    GLint ray_march_smoke_color_min;
    GLint ray_march_smoke_color_max;

    GLint field_slice_image_width_loc;
    GLint field_slice_image_height_loc;
    GLint field_slice_cells_x_loc;
    GLint field_slice_cells_y_loc;
    GLint field_slice_cells_z_loc;
    GLint field_slice_topology_loc;
    GLint field_slice_slice_id_loc;


    // --------------------------------------------------
    // Smoke advection uniform locations
    // --------------------------------------------------

    GLint smoke_advection_cells_x_loc;
    GLint smoke_advection_cells_y_loc;
    GLint smoke_advection_cells_z_loc;

    GLint smoke_advection_cell_side_x_loc;
    GLint smoke_advection_cell_side_y_loc;
    GLint smoke_advection_cell_side_z_loc;


    // --------------------------------------------------
    // Transmittance uniform locations
    // --------------------------------------------------

    GLint light_transmittance_volume_width_loc;
    GLint light_transmittance_volume_height_loc;
    GLint light_transmittance_volume_depth_loc;

    GLint light_transmittance_for_light_ds_loc;

    GLint light_transmittance_field_stride_loc;
    GLint light_transmittance_field_layer_stride_loc;

    GLint light_transmittance_field_x_elements_loc;
    GLint light_transmittance_field_y_elements_loc;
    GLint light_transmittance_field_z_elements_loc;

    GLint light_transmittance_light_x_loc;
    GLint light_transmittance_light_y_loc;
    GLint light_transmittance_light_z_loc;

    GLint light_transmittance_extinction_total_loc;

    //--------------------------------------------------
    // Fragment
    //-----------------------------------------------
    GLint fragment_backgroundColor_loc;

    //--------------------------------------------------
    // Time coefficients assemble
    //--------------------------------------------------

    GLint time_assemble_x_elements;
    GLint time_assemble_y_elements;
    GLint time_assemble_z_elements;
    GLint time_assemble_dx;
    GLint time_assemble_dy;
    GLint time_assemble_dz;
    GLint time_assemble_dt;
    GLint time_assemble_rho;
    GLint time_assemble_topology;
    GLint time_assemble_coarse_assembly;
    GLint time_assemble_temporal_scheme;

    //--------------------------------------------------
    // viscosity LHS coefficients assemble
    //--------------------------------------------------

    GLint momentum_viscosity_LHS_assemble_cells_x;
    GLint momentum_viscosity_LHS_assemble_cells_y;
    GLint momentum_viscosity_LHS_assemble_cells_z;
    GLint momentum_viscosity_LHS_assemble_dx;
    GLint momentum_viscosity_LHS_assemble_dy;
    GLint momentum_viscosity_LHS_assemble_dz;
    GLint momentum_viscosity_LHS_assemble_topology;

    //--------------------------------------------------
    // viscosity RHS assemble
    //--------------------------------------------------

    GLint momentum_viscosity_RHS_assemble_cells_x;
    GLint momentum_viscosity_RHS_assemble_cells_y;
    GLint momentum_viscosity_RHS_assemble_cells_z;
    GLint momentum_viscosity_RHS_assemble_dx;
    GLint momentum_viscosity_RHS_assemble_dy;
    GLint momentum_viscosity_RHS_assemble_dz;
    GLint momentum_viscosity_RHS_assemble_topology;

    //--------------------------------------------------
    // momentum wall function (implicit, aP only)
    //--------------------------------------------------

    GLint momentum_wall_assemble_cells_x;
    GLint momentum_wall_assemble_cells_y;
    GLint momentum_wall_assemble_cells_z;
    GLint momentum_wall_assemble_dx;
    GLint momentum_wall_assemble_dy;
    GLint momentum_wall_assemble_dz;
    GLint momentum_wall_assemble_dynamic_viscosity;
    GLint momentum_wall_assemble_density;
    GLint momentum_wall_assemble_spalding_wall_smoothness;
    GLint momentum_wall_assemble_topology;

    //--------------------------------------------------
    // advection assemble
    //--------------------------------------------------

    GLint advection_assemble_cells_x;
    GLint advection_assemble_cells_y;
    GLint advection_assemble_cells_z;
    GLint advection_assemble_dx;
    GLint advection_assemble_dy;
    GLint advection_assemble_dz;
    GLint advection_assemble_topology;
    GLint advection_assemble_density;
    GLint advection_assemble_ConvectionScheme;      // 0 = first-order upwind, 1 = second-order deferred correction (needs phi / RHS_phi bound)
    GLint advection_assemble_FluxLimiter;           // 0 NONE, 1 MINMOD, 2 VAN_LEER, 3 SUPERBEE

    //--------------------------------------------------
    // Pk assemble
    //--------------------------------------------------

    GLint assemble_production_k_epsilon_omega_grid_x;
    GLint assemble_production_k_epsilon_omega_grid_y;
    GLint assemble_production_k_epsilon_omega_grid_z;
    GLint assemble_production_k_epsilon_omega_dx;
    GLint assemble_production_k_epsilon_omega_dy;
    GLint assemble_production_k_epsilon_omega_dz;
    GLint assemble_production_k_epsilon_omega_dynamic_viscosity;
    GLint assemble_production_k_epsilon_omega_is_omega_model;
    GLint assemble_production_k_epsilon_omega_is_k;
    GLint assemble_production_k_epsilon_omega_density;
    GLint assemble_production_k_epsilon_omega_is_sst;

    //--------------------------------------------------
    // K destruction assemble
    //--------------------------------------------------

    GLint destruction_k_and_epsilon_or_omega_assemble_grid_x;
    GLint destruction_k_and_epsilon_or_omega_assemble_grid_y;
    GLint destruction_k_and_epsilon_or_omega_assemble_grid_z;
    GLint destruction_k_and_epsilon_or_omega_assemble_dx;
    GLint destruction_k_and_epsilon_or_omega_assemble_dy;
    GLint destruction_k_and_epsilon_or_omega_assemble_dz;
    GLint destruction_k_and_epsilon_or_omega_assemble_density;
    GLint destruction_k_and_epsilon_or_omega_assemble_is_omega_model;
    GLint destruction_k_and_epsilon_or_omega_assemble_is_k;
    GLint destruction_k_and_epsilon_or_omega_assemble_is_sst;
    
    //--------------------------------------------------
    // K_eps Diffusion assemble
    //--------------------------------------------------
    GLint diffusion_k_eps_omega_assemble_grid_x;
    GLint diffusion_k_eps_omega_assemble_grid_y;
    GLint diffusion_k_eps_omega_assemble_grid_z;
    GLint diffusion_k_eps_omega_assemble_dx;
    GLint diffusion_k_eps_omega_assemble_dy;
    GLint diffusion_k_eps_omega_assemble_dz;
    GLint diffusion_k_eps_omega_assemble_is_eps;
    GLint diffusion_k_eps_omega_assemble_is_omega_model;
    GLint diffusion_k_eps_omega_assemble_dynamic_viscosity;

    //--------------------------------------------------
    // k_wall_treatment_for_k_eps_model (Spalding)
    //--------------------------------------------------
    GLint k_wall_treatment_for_k_eps_model_grid_x;
    GLint k_wall_treatment_for_k_eps_model_grid_y;
    GLint k_wall_treatment_for_k_eps_model_grid_z;
    GLint k_wall_treatment_for_k_eps_model_dx;
    GLint k_wall_treatment_for_k_eps_model_dy;
    GLint k_wall_treatment_for_k_eps_model_dz;
    GLint k_wall_treatment_for_k_eps_model_dynamic_viscosity;
    GLint k_wall_treatment_for_k_eps_model_density;
    GLint k_wall_treatment_for_k_eps_model_spalding_wall_smoothness;
    GLint k_wall_treatment_for_k_eps_model_coarse_assembly;

    //--------------------------------------------------
    // eps_wall_treatment_for_k_eps_model (Spalding)
    //--------------------------------------------------
    GLint eps_wall_treatment_for_k_eps_model_grid_x;
    GLint eps_wall_treatment_for_k_eps_model_grid_y;
    GLint eps_wall_treatment_for_k_eps_model_grid_z;
    GLint eps_wall_treatment_for_k_eps_model_dx;
    GLint eps_wall_treatment_for_k_eps_model_dy;
    GLint eps_wall_treatment_for_k_eps_model_dz;
    GLint eps_wall_treatment_for_k_eps_model_dynamic_viscosity;
    GLint eps_wall_treatment_for_k_eps_model_density;
    GLint eps_wall_treatment_for_k_eps_model_spalding_wall_smoothness;
    GLint eps_wall_treatment_for_k_eps_model_coarse_assembly;

    //--------------------------------------------------
    // k_wall_treatment_for_k_omega_model (Spalding)
    //--------------------------------------------------
    GLint k_wall_treatment_for_k_omega_model_grid_x;
    GLint k_wall_treatment_for_k_omega_model_grid_y;
    GLint k_wall_treatment_for_k_omega_model_grid_z;
    GLint k_wall_treatment_for_k_omega_model_dx;
    GLint k_wall_treatment_for_k_omega_model_dy;
    GLint k_wall_treatment_for_k_omega_model_dz;
    GLint k_wall_treatment_for_k_omega_model_dynamic_viscosity;
    GLint k_wall_treatment_for_k_omega_model_density;
    GLint k_wall_treatment_for_k_omega_model_spalding_wall_smoothness;
    GLint k_wall_treatment_for_k_omega_model_coarse_assembly;

    //--------------------------------------------------
    // omega_wall_treatment_for_k_omega_model (Spalding)
    //--------------------------------------------------
    GLint omega_wall_treatment_for_k_omega_model_grid_x;
    GLint omega_wall_treatment_for_k_omega_model_grid_y;
    GLint omega_wall_treatment_for_k_omega_model_grid_z;
    GLint omega_wall_treatment_for_k_omega_model_dx;
    GLint omega_wall_treatment_for_k_omega_model_dy;
    GLint omega_wall_treatment_for_k_omega_model_dz;
    GLint omega_wall_treatment_for_k_omega_model_dynamic_viscosity;
    GLint omega_wall_treatment_for_k_omega_model_density;
    GLint omega_wall_treatment_for_k_omega_model_spalding_wall_smoothness;
    GLint omega_wall_treatment_for_k_omega_model_coarse_assembly;


    //--------------------------------------------------
    // K eps mu_eff update
    //--------------------------------------------------
    GLint k_eps_mu_eff_grid_x;
    GLint k_eps_mu_eff_grid_y;
    GLint k_eps_mu_eff_grid_z;
    GLint k_eps_mu_eff_dynamic_viscosity;
    GLint k_eps_mu_eff_density;

    //--------------------------------------------------
    // K omega mu_eff update
    //--------------------------------------------------
    GLint k_omega_mu_eff_grid_x;
    GLint k_omega_mu_eff_grid_y;
    GLint k_omega_mu_eff_grid_z;
    GLint k_omega_mu_eff_dynamic_viscosity;
    GLint k_omega_mu_eff_density;

    //--------------------------------------------------
    // K eps omega initialization
    //--------------------------------------------------
    GLint k_eps_initialize_grid_x;
    GLint k_eps_initialize_grid_y;
    GLint k_eps_initialize_grid_z;
    GLint k_eps_initialize_reference_velocity;
    GLint k_eps_initialize_turbulence_intensity;
    GLint k_eps_initialize_turbulence_length_scale;

    GLint k_omega_initialize_grid_x;
    GLint k_omega_initialize_grid_y;
    GLint k_omega_initialize_grid_z;
    GLint k_omega_initialize_reference_velocity;
    GLint k_omega_initialize_turbulence_intensity;
    GLint k_omega_initialize_turbulence_length_scale;

    //--------------------------------------------------
    // residual sq summation L2 norm
    //--------------------------------------------------
    GLint l2_field_x;
    GLint l2_field_y;
    GLint l2_field_z;
    GLint l2_squared_input;

    GLint energy_reduction_grid_x;
    GLint energy_reduction_grid_y;
    GLint energy_reduction_grid_z;
    GLint energy_reduction_mode;

    //--------------------------------------------------
    // difference sq summation (nonlinear convergence)
    //--------------------------------------------------
    GLint diff_sq_field_x;
    GLint diff_sq_field_y;
    GLint diff_sq_field_z;

    //-----------------------------------------------------------------------------
    // Minimum distance to walls for a structured mesh computation kernel uniforms
    //-----------------------------------------------------------------------------
    GLint min_wall_dists_compute_grid_x;
    GLint min_wall_dists_compute_grid_y;
    GLint min_wall_dists_compute_grid_z;
    GLint min_wall_dists_compute_dx;
    GLint min_wall_dists_compute_dy;
    GLint min_wall_dists_compute_dz;

    //--------------------------------------------------
    // SST blended coefficient (F1) field
    //--------------------------------------------------
    GLint blended_coefficients_SST_grid_x;
    GLint blended_coefficients_SST_grid_y;
    GLint blended_coefficients_SST_grid_z;
    GLint blended_coefficients_SST_dx;
    GLint blended_coefficients_SST_dy;
    GLint blended_coefficients_SST_dz;
    GLint blended_coefficients_SST_dynamic_viscosity;
    GLint blended_coefficients_SST_density;

    GLint k_omega_SST_mu_eff_grid_x;
    GLint k_omega_SST_mu_eff_grid_y;
    GLint k_omega_SST_mu_eff_grid_z;
    GLint k_omega_SST_mu_eff_dx;
    GLint k_omega_SST_mu_eff_dy;
    GLint k_omega_SST_mu_eff_dz;
    GLint k_omega_SST_mu_eff_dynamic_viscosity;
    GLint k_omega_SST_mu_eff_density;

    GLint assemble_cross_diffusion_SST_grid_x;
    GLint assemble_cross_diffusion_SST_grid_y;
    GLint assemble_cross_diffusion_SST_grid_z;
    GLint assemble_cross_diffusion_SST_dx;
    GLint assemble_cross_diffusion_SST_dy;
    GLint assemble_cross_diffusion_SST_dz;
    GLint assemble_cross_diffusion_SST_density;
    GLint assemble_cross_diffusion_SST_ap_only;

    //--------------------------------------------------
    // k / omega diffusion assembly for the SST model
    //--------------------------------------------------
    GLint diffusion_k_omega_SST_assemble_grid_x;
    GLint diffusion_k_omega_SST_assemble_grid_y;
    GLint diffusion_k_omega_SST_assemble_grid_z;
    GLint diffusion_k_omega_SST_assemble_dx;
    GLint diffusion_k_omega_SST_assemble_dy;
    GLint diffusion_k_omega_SST_assemble_dz;
    GLint diffusion_k_omega_SST_assemble_dynamic_viscosity;
    GLint diffusion_k_omega_SST_assemble_is_omega_model;
    GLint diffusion_k_omega_SST_assemble_is_k;
    
    //--------------------------------------------------
    // negative turbulence quantites check
    //--------------------------------------------------
    GLint non_physical_turbulence_quantites_check_cells_x;
    GLint non_physical_turbulence_quantites_check_cells_y;
    GLint non_physical_turbulence_quantites_check_cells_z;

    //--------------------------------------------------
    // positivity floor of a scalar field
    //--------------------------------------------------
    GLint clamp_to_floor_element_count;
    GLint clamp_to_floor_value;

    // --------------------------------------------------
    // Constructor / destructor
    // --------------------------------------------------

    ShaderManager();

    ~ShaderManager();
};