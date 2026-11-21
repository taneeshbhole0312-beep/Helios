
#include "k_TurbulenceModels.hpp"
#include "RANS.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>




// ============================================================================================================================================================
// ============================================================================================================================================================
//                                                                  K_SYSTEM CODE
// ============================================================================================================================================================
// ============================================================================================================================================================


k_TurbulenceModels::k_TurbulenceModels(RANS &rans, bool is_omega_based_system, bool is_SST_in)
:   rans(rans),
    KSystem(rans.hierarchy, rans.shadermanager, fieldTopology::CELL_CENTERED, rans.config),
    is_omega_based(is_omega_based_system),
    is_SST(is_SST_in && is_omega_based_system)          // SST only exists as an omega based model
{
    if (is_SST)
    {
        sst = std::make_unique<SST_additions>(*this);       // allocates the wall distance and F1 fields
    }

    // Same layout as eps_system.multigrid_levels_coeffiecients_arr: element [i - 1] is coarse level i (cell counts from the shared hierarchy)
    for (int i = 1; i < rans.hierarchy.level_count(); i++)
    {
        const GridLevelInfo &level_geometry = rans.hierarchy.level(i);
        std::vector <float> temp((size_t)level_geometry.cells_x * level_geometry.cells_y * level_geometry.cells_z, 0.0f);

        k_hierarchy.push_back(create_SSBO(temp));
    }
};

GLuint& k_TurbulenceModels::k_field()
{
    return KSystem.field.data;
}


void k_TurbulenceModels::assemble_diffusion_contribution_k()
{
    if (is_SST)
    {
        // SST: sigma_k is blended with F1 (the F1 field has to be up to date, see SST_additions::compute_blended_coefficient_field)
        glUseProgram(rans.shadermanager.diffusion_k_omega_SST_assemble);

        glUniform1ui(rans.shadermanager.diffusion_k_omega_SST_assemble_grid_x, KSystem.cells_x);
        glUniform1ui(rans.shadermanager.diffusion_k_omega_SST_assemble_grid_y, KSystem.cells_y);
        glUniform1ui(rans.shadermanager.diffusion_k_omega_SST_assemble_grid_z, KSystem.cells_z);
        glUniform1f(rans.shadermanager.diffusion_k_omega_SST_assemble_dx, rans.domain_size_x / KSystem.x_elements);
        glUniform1f(rans.shadermanager.diffusion_k_omega_SST_assemble_dy, rans.domain_size_y / KSystem.y_elements);
        glUniform1f(rans.shadermanager.diffusion_k_omega_SST_assemble_dz, rans.domain_size_z / KSystem.z_elements);
        glUniform1f(rans.shadermanager.diffusion_k_omega_SST_assemble_dynamic_viscosity, rans.dynamic_viscosity);
        glUniform1i(rans.shadermanager.diffusion_k_omega_SST_assemble_is_omega_model, is_omega_based);
        glUniform1i(rans.shadermanager.diffusion_k_omega_SST_assemble_is_k, GL_TRUE);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, rans.coefficients_stencil.coefficients.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rans.coefficients_stencil.coefficients.aW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, rans.coefficients_stencil.coefficients.aE);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, rans.coefficients_stencil.coefficients.aS);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, rans.coefficients_stencil.coefficients.aN);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, rans.coefficients_stencil.coefficients.aB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, rans.coefficients_stencil.coefficients.aT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, rans.mu_eff);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, sst->blended_coefficient);

        glDispatchCompute(
            (KSystem.x_elements + 7) / 8,
            (KSystem.y_elements + 7) / 8,
            (KSystem.z_elements + 7) / 8
        );

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        return;
    }

    glUseProgram(rans.shadermanager.diffusion_k_eps_omega_assemble);

    glUniform1ui(rans.shadermanager.diffusion_k_eps_omega_assemble_grid_x, KSystem.cells_x);
    glUniform1ui(rans.shadermanager.diffusion_k_eps_omega_assemble_grid_y, KSystem.cells_y);
    glUniform1ui(rans.shadermanager.diffusion_k_eps_omega_assemble_grid_z, KSystem.cells_z);
    glUniform1f(rans.shadermanager.diffusion_k_eps_omega_assemble_dx, rans.domain_size_x / KSystem.x_elements);
    glUniform1f(rans.shadermanager.diffusion_k_eps_omega_assemble_dy, rans.domain_size_y / KSystem.y_elements);
    glUniform1f(rans.shadermanager.diffusion_k_eps_omega_assemble_dz, rans.domain_size_z / KSystem.z_elements);
    glUniform1i(rans.shadermanager.diffusion_k_eps_omega_assemble_is_eps, GL_FALSE);
    glUniform1i(rans.shadermanager.diffusion_k_eps_omega_assemble_is_omega_model, is_omega_based);
    glUniform1f(rans.shadermanager.diffusion_k_eps_omega_assemble_dynamic_viscosity, rans.dynamic_viscosity);
    

    // Bind the LinearSystem coefficient buffers.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, rans.coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rans.coefficients_stencil.coefficients.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, rans.coefficients_stencil.coefficients.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, rans.coefficients_stencil.coefficients.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, rans.coefficients_stencil.coefficients.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, rans.coefficients_stencil.coefficients.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, rans.coefficients_stencil.coefficients.aT);

    // mu_eff
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, rans.mu_eff);

    glDispatchCompute(
        (KSystem.x_elements + 7) / 8,
        (KSystem.y_elements + 7) / 8,
        (KSystem.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::assemble_destruction_contribution_k(GLuint &destruction_variable)
{
    glUseProgram(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, KSystem.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, rans.coefficients_stencil.coefficients.aP);

    if(is_SST)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, sst->blended_coefficient);

    glUniform1ui(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_x, KSystem.x_elements);
    glUniform1ui(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_y, KSystem.y_elements);
    glUniform1ui(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_z, KSystem.z_elements);
    glUniform1f(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dx, float(rans.domain_size_x / KSystem.x_elements));
    glUniform1f(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dy, float(rans.domain_size_y / KSystem.y_elements));
    glUniform1f(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dz, float(rans.domain_size_z / KSystem.z_elements));
    glUniform1f(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_density, rans.density);
    glUniform1i(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_omega_model, is_omega_based);
    glUniform1i(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_k, GL_TRUE);
    glUniform1i(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_sst, is_SST);

    glDispatchCompute((KSystem.x_elements + 7) / 8, (KSystem.y_elements + 7) / 8, (KSystem.z_elements + 7) / 8);
}

void k_TurbulenceModels::assemble_production_contribution_k(GLuint &destruction_variable)
{
    glUseProgram(rans.shadermanager.assemble_production_k_epsilon_omega);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, rans.u_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rans.v_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, rans.w_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, rans.mu_eff);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, rans.coefficients_stencil.RHS_main);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, KSystem.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, destruction_variable);

    if (is_SST)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, sst->blended_coefficient);       // F1, only attached (and only read by the kernel) for SST

    glUniform1ui(rans.shadermanager.assemble_production_k_epsilon_omega_grid_x, KSystem.x_elements);
    glUniform1ui(rans.shadermanager.assemble_production_k_epsilon_omega_grid_y, KSystem.y_elements);
    glUniform1ui(rans.shadermanager.assemble_production_k_epsilon_omega_grid_z, KSystem.z_elements);
    glUniform1f(rans.shadermanager.assemble_production_k_epsilon_omega_dx, float(rans.domain_size_x / KSystem.x_elements));
    glUniform1f(rans.shadermanager.assemble_production_k_epsilon_omega_dy, float(rans.domain_size_y / KSystem.y_elements));
    glUniform1f(rans.shadermanager.assemble_production_k_epsilon_omega_dz, float(rans.domain_size_z / KSystem.z_elements));
    glUniform1f(rans.shadermanager.assemble_production_k_epsilon_omega_dynamic_viscosity, rans.dynamic_viscosity);
    glUniform1i(rans.shadermanager.assemble_production_k_epsilon_omega_is_omega_model, is_omega_based);
    glUniform1i(rans.shadermanager.assemble_production_k_epsilon_omega_is_k, GL_TRUE);
    glUniform1f(rans.shadermanager.assemble_production_k_epsilon_omega_density, rans.density);
    glUniform1i(rans.shadermanager.assemble_production_k_epsilon_omega_is_sst, is_SST);

    glDispatchCompute((KSystem.x_elements + 7) / 8, (KSystem.y_elements + 7) / 8, (KSystem.z_elements + 7) / 8);
}

void k_TurbulenceModels::construct_K_equation(float stable_dt , GLuint &destruction_variable, float spalding_wall_smoothness)
{
    rans.clear_coefficients(rans.coefficients_stencil.coefficients);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, rans.coefficients_stencil.RHS_main);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    //=======================================================
    //COEFFICIENTS
    //=======================================================

    rans.add_time_contribution(stable_dt, KSystem);                    //Shared functions (common infra)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    rans.add_advection_contribution(KSystem, false);        // false: live (post-projection) velocities                    //Shared functions (common infra)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    assemble_diffusion_contribution_k();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    assemble_destruction_contribution_k(destruction_variable);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    assemble_production_contribution_k(destruction_variable);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

     // K EPSILON ? OMEGA WALL TREATMENT (FINE, PHYSICAL) 
    if(!is_omega_based)
    {
        apply_k_wall_treatment_for_k_eps_model(spalding_wall_smoothness);
    }
    else if(is_omega_based)
    {
        apply_k_wall_treatment_for_k_omega_model(spalding_wall_smoothness);
    }

}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//OVERLOADED FUNCTION FOR THE COARSER LEVEL     (KINETIC TURBULENT ENERGY)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Same kernel and same math as the fine-level assemble_destruction_contribution_k, on the restricted old k / destruction variable of that level
void k_TurbulenceModels::assemble_destruction_contribution_k(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, GLuint k_at_coarse_level, GLuint destruction_variable_at_coarse_level)
{
    glUseProgram(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_at_coarse_level);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable_at_coarse_level);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aP);

    if(is_SST)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, sst->blended_coefficient);

    glUniform1ui(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_x, CurrentCells_x);
    glUniform1ui(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_y, CurrentCells_y);
    glUniform1ui(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_z, CurrentCells_z);
    glUniform1f(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dx, float(rans.domain_size_x / CurrentCells_x));
    glUniform1f(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dy, float(rans.domain_size_y / CurrentCells_y));
    glUniform1f(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dz, float(rans.domain_size_z / CurrentCells_z));
    glUniform1f(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_density, rans.density);
    glUniform1i(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_omega_model, is_omega_based);
    glUniform1i(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_k, GL_TRUE);
    glUniform1i(rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_sst, is_SST);

    glDispatchCompute((CurrentCells_x + 7) / 8, (CurrentCells_y + 7) / 8, (CurrentCells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::assemble_diffusion_contribution_k(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, GLuint mu_eff_at_coarse_level, GLuint blended_coefficient_at_coarse_level)
{
    if (is_SST)
    {
        // SST: sigma_k is blended with F1, which must be given on THIS level (blended_coefficient_at_coarse_level)
        glUseProgram(rans.shadermanager.diffusion_k_omega_SST_assemble);

        glUniform1ui(rans.shadermanager.diffusion_k_omega_SST_assemble_grid_x, CurrentCells_x);
        glUniform1ui(rans.shadermanager.diffusion_k_omega_SST_assemble_grid_y, CurrentCells_y);
        glUniform1ui(rans.shadermanager.diffusion_k_omega_SST_assemble_grid_z, CurrentCells_z);
        glUniform1f(rans.shadermanager.diffusion_k_omega_SST_assemble_dx, rans.domain_size_x / CurrentCells_x);
        glUniform1f(rans.shadermanager.diffusion_k_omega_SST_assemble_dy, rans.domain_size_y / CurrentCells_y);
        glUniform1f(rans.shadermanager.diffusion_k_omega_SST_assemble_dz, rans.domain_size_z / CurrentCells_z);
        glUniform1f(rans.shadermanager.diffusion_k_omega_SST_assemble_dynamic_viscosity, rans.dynamic_viscosity);
        glUniform1i(rans.shadermanager.diffusion_k_omega_SST_assemble_is_omega_model, is_omega_based);
        glUniform1i(rans.shadermanager.diffusion_k_omega_SST_assemble_is_k, GL_TRUE);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_at_coarse_level.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_at_coarse_level.aW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aE);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_at_coarse_level.aS);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aN);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_at_coarse_level.aB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_at_coarse_level.aT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, mu_eff_at_coarse_level);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, blended_coefficient_at_coarse_level);

        glDispatchCompute(
            (CurrentCells_x + 7) / 8,
            (CurrentCells_y + 7) / 8,
            (CurrentCells_z + 7) / 8
        );

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        return;
    }

    glUseProgram(rans.shadermanager.diffusion_k_eps_omega_assemble);

    glUniform1ui(rans.shadermanager.diffusion_k_eps_omega_assemble_grid_x, CurrentCells_x);
    glUniform1ui(rans.shadermanager.diffusion_k_eps_omega_assemble_grid_y, CurrentCells_y);
    glUniform1ui(rans.shadermanager.diffusion_k_eps_omega_assemble_grid_z, CurrentCells_z);
    glUniform1f(rans.shadermanager.diffusion_k_eps_omega_assemble_dx, rans.domain_size_x / CurrentCells_x);
    glUniform1f(rans.shadermanager.diffusion_k_eps_omega_assemble_dy, rans.domain_size_y / CurrentCells_y);
    glUniform1f(rans.shadermanager.diffusion_k_eps_omega_assemble_dz, rans.domain_size_z / CurrentCells_z);
    glUniform1i(rans.shadermanager.diffusion_k_eps_omega_assemble_is_eps, GL_FALSE);
    glUniform1i(rans.shadermanager.diffusion_k_eps_omega_assemble_is_omega_model, is_omega_based);
    glUniform1f(rans.shadermanager.diffusion_k_eps_omega_assemble_dynamic_viscosity, rans.dynamic_viscosity);

    // Bind the LinearSystem coefficient buffers.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_at_coarse_level.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_at_coarse_level.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_at_coarse_level.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_at_coarse_level.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_at_coarse_level.aT);

    // mu_eff
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, mu_eff_at_coarse_level);

    glDispatchCompute(
        (CurrentCells_x + 7) / 8,
        (CurrentCells_y + 7) / 8,
        (CurrentCells_z + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}


void k_TurbulenceModels::construct_K_equation_hierarchy(float stable_dt ,std::vector <GLuint> &destruction_variable_hierarchy)
{
    rans.restrict_vels(false);           // false: live (post-projection) velocities
    rans.restrict_mu_eff();             // NO RISK

    const int passes = rans.hierarchy.coarse_level_count();

    for (int level = 0; level < passes; level++)
    {
        int current_cells_x = rans.hierarchy.level(level + 1).cells_x;
        int current_cells_y = rans.hierarchy.level(level + 1).cells_y;
        int current_cells_z = rans.hierarchy.level(level + 1).cells_z;   

        rans.clear_coefficients(rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);

        rans.add_time_contribution(stable_dt, current_cells_x, current_cells_y, current_cells_z, fieldTopology::CELL_CENTERED, rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level], KSystem.levels_info[level]);

        rans.add_advection_contribution(current_cells_x, current_cells_y, current_cells_z, rans.u_vel_hierarcy[level], rans.v_vel_hierarcy[level], rans.w_vel_hierarcy[level], KSystem.field.topology, rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level], KSystem.levels_info[level]);

        assemble_diffusion_contribution_k(current_cells_x, current_cells_y, current_cells_z, rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level], rans.mu_eff_hierarchy[level], is_SST ? sst->blended_coefficient_hierarchy[level] : 0u);        

        // destruction on the restricted old k / destruction variable of this level    (NO PRODUCTION AS IT IS COARSE LEVELS)
        assemble_destruction_contribution_k(current_cells_x, current_cells_y, current_cells_z, rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level], k_hierarchy[level], destruction_variable_hierarchy[level]);

        if(is_omega_based)
        {
            apply_k_wall_treatment_for_k_omega_model(current_cells_x, current_cells_y, current_cells_z, rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);
        }
        else if(!is_omega_based)
        {
            apply_k_wall_treatment_for_k_eps_model(current_cells_x, current_cells_y, current_cells_z, rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);
        }
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}


// ============================================================================================================================================================
// ============================================================================================================================================================
//                                                                  DESTRUCTION_VARIABLE_CODE
// ============================================================================================================================================================
// ============================================================================================================================================================

k_TurbulenceModels::destruction_variable::destruction_variable(k_TurbulenceModels &k_model)
:   k_model(k_model),
    destruction_variable_system(k_model.rans.hierarchy, k_model.rans.shadermanager, fieldTopology::CELL_CENTERED, k_model.rans.config)
{
    // Same layout as eps_system.multigrid_levels_coeffiecients_arr: element [i - 1] is coarse level i (cell counts from the shared hierarchy)
    for (int i = 1; i < k_model.rans.hierarchy.level_count(); i++)
    {
        const GridLevelInfo &level_geometry = k_model.rans.hierarchy.level(i);
        std::vector <float> temp((size_t)level_geometry.cells_x * level_geometry.cells_y * level_geometry.cells_z, 0.0f);

        destruction_variable_hierarchy.push_back(k_model.create_SSBO(temp));
    }
};

GLuint k_TurbulenceModels::create_SSBO(const std::vector<float> &data)
{
    GLuint ssbo;
    glGenBuffers(1, &ssbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        data.size() * sizeof(float),
        data.data(),
        GL_DYNAMIC_COPY);

    return ssbo;
}

GLuint &k_TurbulenceModels::destruction_variable::destruction_variable_field()
{
    return destruction_variable_system.field.data;
}

// x = max(x, 1e-10) on k and on the destruction variable. 1e-10 is far below any physical value (it only has to be positive and representable in float32:
// 1/1e-10 = 1e10 is still finite), and far above the +-1e-14 / 1e-9 rounding noise the multigrid leaves in near-zero regions.
void k_TurbulenceModels::destruction_variable::enforce_positivity()
{
    constexpr float tiny_positive_floor = 1.0e-10f;

    k_model.rans.clamp_field_to_floor(k_model.KSystem.field.data, k_model.KSystem.x_elements * k_model.KSystem.y_elements * k_model.KSystem.z_elements, tiny_positive_floor);
    k_model.rans.clamp_field_to_floor(destruction_variable_system.field.data, destruction_variable_system.x_elements * destruction_variable_system.y_elements * destruction_variable_system.z_elements, tiny_positive_floor);
}

void k_TurbulenceModels::destruction_variable::restrict_k_and_destruction_variable()
{
    int passes = static_cast<int>(k_model.k_hierarchy.size());   // destruction variable and k fields have the same sizes

    glUseProgram(k_model.rans.shadermanager.physical_fields_multigrid_generic_restriction_program);
    glUniform1i(k_model.rans.shadermanager.generic_restriction_topology_loc, int(fieldTopology::CELL_CENTERED));

    for (int i = 0; i < passes; i++)
    {
        // level i -> level i + 1 ; level 0 is the fine field, level L lives at hierarchy[L - 1]
        std::vector <int> fine_dimensions = destruction_variable_system.GetLevelDimension(i);
        std::vector <int> coarse_dimensions = destruction_variable_system.GetLevelDimension(i + 1);

        glUniform3i(k_model.rans.shadermanager.generic_restriction_fine_dimensions_loc, fine_dimensions[0], fine_dimensions[1], fine_dimensions[2]);
        glUniform3i(k_model.rans.shadermanager.generic_restriction_coarse_dimensions_loc, coarse_dimensions[0], coarse_dimensions[1], coarse_dimensions[2]);

        // k
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, (i == 0) ? k_model.KSystem.field.data : k_model.k_hierarchy[i - 1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.k_hierarchy[i]);

        glDispatchCompute((coarse_dimensions[0] + 7) / 8, (coarse_dimensions[1] + 7) / 8, (coarse_dimensions[2] + 7) / 8);

        // epsilon
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, (i == 0) ? destruction_variable_system.field.data : destruction_variable_hierarchy[i - 1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable_hierarchy[i]);

        glDispatchCompute((coarse_dimensions[0] + 7) / 8, (coarse_dimensions[1] + 7) / 8, (coarse_dimensions[2] + 7) / 8);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

void k_TurbulenceModels::destruction_variable::assemble_diffusion_contribution_destruction_variable()
{
    if (k_model.is_SST)
    {
        // SST: sigma_omega is blended with F1 (the F1 field has to be up to date, see SST_additions::compute_blended_coefficient_field)
        glUseProgram(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble);

        glUniform1ui(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_grid_x, destruction_variable_system.cells_x);
        glUniform1ui(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_grid_y, destruction_variable_system.cells_y);
        glUniform1ui(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_grid_z, destruction_variable_system.cells_z);
        glUniform1f(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_dx, k_model.rans.domain_size_x / destruction_variable_system.x_elements);
        glUniform1f(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_dy, k_model.rans.domain_size_y / destruction_variable_system.y_elements);
        glUniform1f(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_dz, k_model.rans.domain_size_z / destruction_variable_system.z_elements);
        glUniform1f(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_dynamic_viscosity, k_model.rans.dynamic_viscosity);
        glUniform1i(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_is_omega_model, k_model.is_omega_based);
        glUniform1i(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_is_k, GL_FALSE);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.rans.coefficients_stencil.coefficients.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.rans.coefficients_stencil.coefficients.aW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.coefficients_stencil.coefficients.aE);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, k_model.rans.coefficients_stencil.coefficients.aS);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, k_model.rans.coefficients_stencil.coefficients.aN);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, k_model.rans.coefficients_stencil.coefficients.aB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, k_model.rans.coefficients_stencil.coefficients.aT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, k_model.rans.mu_eff);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, k_model.sst->blended_coefficient);

        glDispatchCompute(
            (destruction_variable_system.x_elements + 7) / 8,
            (destruction_variable_system.y_elements + 7) / 8,
            (destruction_variable_system.z_elements + 7) / 8
        );

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        return;
    }

    bool is_eps;

    if(k_model.is_omega_based)
        is_eps = false;
    else if(!k_model.is_omega_based)
        is_eps = true;

    glUseProgram(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble);

    glUniform1ui(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_grid_x, destruction_variable_system.cells_x);
    glUniform1ui(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_grid_y, destruction_variable_system.cells_y);
    glUniform1ui(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_grid_z, destruction_variable_system.cells_z);
    glUniform1f(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_dx, k_model.rans.domain_size_x / destruction_variable_system.x_elements);
    glUniform1f(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_dy, k_model.rans.domain_size_y / destruction_variable_system.y_elements);
    glUniform1f(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_dz, k_model.rans.domain_size_z / destruction_variable_system.z_elements);
    glUniform1i(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_is_eps, is_eps);
    glUniform1i(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_is_omega_model, k_model.is_omega_based);
    glUniform1f(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_dynamic_viscosity, k_model.rans.dynamic_viscosity);     // Gamma_eps = mu + (mu_eff - mu) / sigma_eps

    // Bind the LinearSystem coefficient buffers.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.rans.coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.rans.coefficients_stencil.coefficients.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.coefficients_stencil.coefficients.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, k_model.rans.coefficients_stencil.coefficients.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, k_model.rans.coefficients_stencil.coefficients.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, k_model.rans.coefficients_stencil.coefficients.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, k_model.rans.coefficients_stencil.coefficients.aT);

    // mu_eff
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, k_model.rans.mu_eff);

    glDispatchCompute(
        (destruction_variable_system.x_elements + 7) / 8,
        (destruction_variable_system.y_elements + 7) / 8,
        (destruction_variable_system.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::destruction_variable::assemble_destruction_contribution_destruction_variable()
{
    glUseProgram(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble);

    // k and epsilon / omega of the START of the nonlinear iteration: k has already been SOLVED by the time this equation is assembled,
    // but the destruction term must use the old k (as when both equations were assembled before either was solved)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.rans.previous_iterate_k);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.rans.previous_iterate_eps);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.coefficients_stencil.coefficients.aP);
    if (k_model.is_SST)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, k_model.sst->blended_coefficient);       // F1, only attached (and only read by the kernel) for SST; TODO(SST): the coarse-level call needs F1 restricted to its level

    glUniform1ui(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dx, float(k_model.rans.domain_size_x / destruction_variable_system.x_elements));
    glUniform1f(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dy, float(k_model.rans.domain_size_y / destruction_variable_system.y_elements));
    glUniform1f(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dz, float(k_model.rans.domain_size_z / destruction_variable_system.z_elements));
    glUniform1f(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_density, k_model.rans.density);

    //bools
    glUniform1i(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_omega_model, k_model.is_omega_based);
    glUniform1i(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_k, GL_FALSE);
    glUniform1i(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_sst, k_model.is_SST);

    glDispatchCompute((destruction_variable_system.x_elements + 7) / 8, (destruction_variable_system.y_elements + 7) / 8, (destruction_variable_system.z_elements + 7) / 8);
}

void k_TurbulenceModels::destruction_variable::assemble_production_contribution_destruction_variable()
{
    glUseProgram(k_model.rans.shadermanager.assemble_production_k_epsilon_omega);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.rans.u_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.rans.v_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.w_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, k_model.rans.mu_eff);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, k_model.rans.coefficients_stencil.RHS_main);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, k_model.rans.previous_iterate_k);  // old k (k is already solved when this equation is assembled), see the destruction function
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, k_model.rans.previous_iterate_eps);

    if (k_model.is_SST)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, k_model.sst->blended_coefficient);       // F1, only attached (and only read by the kernel) for SST

    glUniform1ui(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_dx, float(k_model.rans.domain_size_x / destruction_variable_system.x_elements));
    glUniform1f(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_dy, float(k_model.rans.domain_size_y / destruction_variable_system.y_elements));
    glUniform1f(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_dz, float(k_model.rans.domain_size_z / destruction_variable_system.z_elements));
    glUniform1f(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_dynamic_viscosity, k_model.rans.dynamic_viscosity);
    glUniform1i(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_is_omega_model, k_model.is_omega_based);
    glUniform1i(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_is_k, GL_FALSE);
    glUniform1f(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_density, k_model.rans.density);
    glUniform1i(k_model.rans.shadermanager.assemble_production_k_epsilon_omega_is_sst, k_model.is_SST);

    glDispatchCompute((destruction_variable_system.x_elements + 7) / 8, (destruction_variable_system.y_elements + 7) / 8, (destruction_variable_system.z_elements + 7) / 8);
}

void k_TurbulenceModels::destruction_variable::construct_destruction_variable_equation(float stable_dt, float spalding_wall_smoothness)
{
    k_model.rans.clear_coefficients(k_model.rans.coefficients_stencil.coefficients);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, k_model.rans.coefficients_stencil.RHS_main);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    //=======================================================
    //COEFFICIENTS
    //=======================================================

    k_model.rans.add_time_contribution(stable_dt, destruction_variable_system);           //Shared functions (common infra)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    k_model.rans.add_advection_contribution(destruction_variable_system, false);        // false: live (post-projection) velocities       //Shared functions (common infra)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    assemble_diffusion_contribution_destruction_variable();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    assemble_destruction_contribution_destruction_variable();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    assemble_production_contribution_destruction_variable();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if (k_model.is_SST) 
        k_model.sst->assemble_cross_diffusion_contribution();
    
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    // EPSILON ? OMEGA WALL TREATMENT (FINE, PHYSICAL) 
    if(!k_model.is_omega_based)
    {
        apply_eps_wall_treatment_for_k_eps_model(spalding_wall_smoothness);
    }
    else if(k_model.is_omega_based)
    {
        apply_omega_wall_treatment_for_k_omega_model(spalding_wall_smoothness);
    }
}




//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//OVERLOADED FUNCTION FOR THE COARSER LEVEL    (DESTRUCTION VARIABLE)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void k_TurbulenceModels::destruction_variable::assemble_diffusion_contribution_destruction_variable(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, GLuint mu_eff_at_coarse_level, GLuint blended_coefficient_at_coarse_level)
{
    if (k_model.is_SST)
    {
        // SST: sigma_omega is blended with F1, which must be given on THIS level (blended_coefficient_at_coarse_level)
        glUseProgram(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble);

        glUniform1ui(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_grid_x, CurrentCells_x);
        glUniform1ui(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_grid_y, CurrentCells_y);
        glUniform1ui(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_grid_z, CurrentCells_z);
        glUniform1f(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_dx, k_model.rans.domain_size_x / CurrentCells_x);
        glUniform1f(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_dy, k_model.rans.domain_size_y / CurrentCells_y);
        glUniform1f(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_dz, k_model.rans.domain_size_z / CurrentCells_z);
        glUniform1f(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_dynamic_viscosity, k_model.rans.dynamic_viscosity);
        glUniform1i(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_is_omega_model, k_model.is_omega_based);
        glUniform1i(k_model.rans.shadermanager.diffusion_k_omega_SST_assemble_is_k, GL_FALSE);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_at_coarse_level.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_at_coarse_level.aW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aE);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_at_coarse_level.aS);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aN);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_at_coarse_level.aB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_at_coarse_level.aT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, mu_eff_at_coarse_level);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, blended_coefficient_at_coarse_level);

        glDispatchCompute(
            (CurrentCells_x + 7) / 8,
            (CurrentCells_y + 7) / 8,
            (CurrentCells_z + 7) / 8
        );

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        return;
    }

    bool is_eps;

    if(k_model.is_omega_based)
        is_eps = false;
    else if(!k_model.is_omega_based)
        is_eps = true;


    glUseProgram(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble);

    glUniform1ui(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_grid_x, CurrentCells_x);
    glUniform1ui(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_grid_y, CurrentCells_y);
    glUniform1ui(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_grid_z, CurrentCells_z);
    glUniform1f(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_dx, k_model.rans.domain_size_x / CurrentCells_x);
    glUniform1f(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_dy, k_model.rans.domain_size_y / CurrentCells_y);
    glUniform1f(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_dz, k_model.rans.domain_size_z / CurrentCells_z);
    glUniform1i(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_is_eps, is_eps);
    glUniform1i(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_is_omega_model, k_model.is_omega_based);
    glUniform1f(k_model.rans.shadermanager.diffusion_k_eps_omega_assemble_dynamic_viscosity, k_model.rans.dynamic_viscosity);     // Gamma_eps = mu + (mu_eff - mu) / sigma_eps

    // Bind the LinearSystem coefficient buffers.
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_at_coarse_level.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_at_coarse_level.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_at_coarse_level.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_at_coarse_level.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_at_coarse_level.aT);

    // mu_eff
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, mu_eff_at_coarse_level);

    glDispatchCompute(
        (CurrentCells_x + 7) / 8,
        (CurrentCells_y + 7) / 8,
        (CurrentCells_z + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// Same kernel and same math as the fine-level assemble_destruction_contribution_destruction_variable, on the restricted old k / destruction variable of that level
void k_TurbulenceModels::destruction_variable::assemble_destruction_contribution_destruction_variable(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, GLuint k_at_coarse_level, GLuint destruction_variable_at_coarse_level)
{
    glUseProgram(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_at_coarse_level);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable_at_coarse_level);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aP);

    if(k_model.is_SST)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, k_model.sst->blended_coefficient);

    glUniform1ui(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_x, CurrentCells_x);
    glUniform1ui(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_y, CurrentCells_y);
    glUniform1ui(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_grid_z, CurrentCells_z);
    glUniform1f(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dx, float(k_model.rans.domain_size_x / CurrentCells_x));
    glUniform1f(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dy, float(k_model.rans.domain_size_y / CurrentCells_y));
    glUniform1f(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_dz, float(k_model.rans.domain_size_z / CurrentCells_z));
    glUniform1f(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_density, k_model.rans.density);
    glUniform1i(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_omega_model, k_model.is_omega_based);
    glUniform1i(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_k, GL_FALSE);
    glUniform1i(k_model.rans.shadermanager.destruction_k_and_epsilon_or_omega_assemble_is_sst, k_model.is_SST);

    glDispatchCompute((CurrentCells_x + 7) / 8, (CurrentCells_y + 7) / 8, (CurrentCells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::destruction_variable::construct_destruction_variable_equation_hierarchy(float stable_dt)
{
    k_model.rans.restrict_vels(false);        // false: live (post-projection) velocities
    k_model.rans.restrict_mu_eff();

    const int passes = k_model.rans.hierarchy.coarse_level_count();

    for (int level = 0; level < passes; level++)
    {
        int current_cells_x = k_model.rans.hierarchy.level(level + 1).cells_x;
        int current_cells_y = k_model.rans.hierarchy.level(level + 1).cells_y;
        int current_cells_z = k_model.rans.hierarchy.level(level + 1).cells_z;

        k_model.rans.clear_coefficients(k_model.rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);

        k_model.rans.add_time_contribution(stable_dt, current_cells_x, current_cells_y, current_cells_z, fieldTopology::CELL_CENTERED, k_model.rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level], destruction_variable_system.levels_info[level]);

        k_model.rans.add_advection_contribution(current_cells_x, current_cells_y, current_cells_z, k_model.rans.u_vel_hierarcy[level], k_model.rans.v_vel_hierarcy[level], k_model.rans.w_vel_hierarcy[level], destruction_variable_system.field.topology, k_model.rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level], destruction_variable_system.levels_info[level]);

        assemble_diffusion_contribution_destruction_variable(current_cells_x, current_cells_y, current_cells_z, k_model.rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level], k_model.rans.mu_eff_hierarchy[level], k_model.is_SST ? k_model.sst->blended_coefficient_hierarchy[level] : 0u);   

        // destruction on the restricted old k / destruction variable of this level    (NO PRODUCTION AS IT IS COARSE LEVELS)
        assemble_destruction_contribution_destruction_variable(current_cells_x, current_cells_y, current_cells_z, k_model.rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level], k_model.k_hierarchy[level], destruction_variable_hierarchy[level]);

        if(k_model.is_omega_based)
        {
            apply_omega_wall_treatment_for_k_omega_model(current_cells_x, current_cells_y, current_cells_z, k_model.rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);
        }
        else if(!k_model.is_omega_based)
        {
            apply_eps_wall_treatment_for_k_eps_model(current_cells_x, current_cells_y, current_cells_z, k_model.rans.coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);
        }

    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//WALL TREATMENT (SPALDING), ONE KERNEL PER TRANSPORTED VARIABLE
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
// FINE LEVEL: each function writes the wall cells of ITS variable only, and pins that variable's row (aP = 1, RHS = wall value) in the shared stencil set.
// COARSE LEVEL: each function writes the penalty diagonal (homogeneous correction condition e = 0) on the wall rows of ITS coarse operator only.

// ---------------- k-epsilon model: k
void k_TurbulenceModels::apply_k_wall_treatment_for_k_eps_model(float spalding_wall_smoothness)
{
    glUseProgram(rans.shadermanager.k_wall_treatment_for_k_eps_model);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, rans.u_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rans.v_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, rans.w_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, KSystem.field.data);

    // Wall row is pinned (aP = 1, RHS = wall value): this must run AFTER the fine k system is assembled
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, rans.coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, rans.coefficients_stencil.RHS_main);

    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_eps_model_grid_x, KSystem.x_elements);
    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_eps_model_grid_y, KSystem.y_elements);
    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_eps_model_grid_z, KSystem.z_elements);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_eps_model_dx, rans.domain_size_x / KSystem.x_elements);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_eps_model_dy, rans.domain_size_y / KSystem.y_elements);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_eps_model_dz, rans.domain_size_z / KSystem.z_elements);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_eps_model_dynamic_viscosity, rans.dynamic_viscosity);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_eps_model_density, rans.density);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_eps_model_spalding_wall_smoothness, spalding_wall_smoothness);
    glUniform1i(rans.shadermanager.k_wall_treatment_for_k_eps_model_coarse_assembly, GL_FALSE);

    glDispatchCompute(
        (KSystem.x_elements + 7) / 8,
        (KSystem.y_elements + 7) / 8,
        (KSystem.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::apply_k_wall_treatment_for_k_eps_model(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level)
{
    glUseProgram(rans.shadermanager.k_wall_treatment_for_k_eps_model);

    // coarse_assembly == true: only the diagonal (binding 4) is written
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aP);

    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_eps_model_grid_x, CurrentCells_x);
    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_eps_model_grid_y, CurrentCells_y);
    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_eps_model_grid_z, CurrentCells_z);
    glUniform1i(rans.shadermanager.k_wall_treatment_for_k_eps_model_coarse_assembly, GL_TRUE);

    glDispatchCompute(
        (CurrentCells_x + 7) / 8,
        (CurrentCells_y + 7) / 8,
        (CurrentCells_z + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ---------------- k-epsilon model: epsilon
void k_TurbulenceModels::destruction_variable::apply_eps_wall_treatment_for_k_eps_model(float spalding_wall_smoothness)
{
    glUseProgram(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.rans.u_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.rans.v_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.w_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, destruction_variable_system.field.data);

    // Wall row is pinned (aP = 1, RHS = wall value): this must run AFTER the fine epsilon system is assembled
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, k_model.rans.coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, k_model.rans.coefficients_stencil.RHS_main);

    glUniform1ui(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_dx, k_model.rans.domain_size_x / destruction_variable_system.x_elements);
    glUniform1f(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_dy, k_model.rans.domain_size_y / destruction_variable_system.y_elements);
    glUniform1f(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_dz, k_model.rans.domain_size_z / destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_dynamic_viscosity, k_model.rans.dynamic_viscosity);
    glUniform1f(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_density, k_model.rans.density);
    glUniform1f(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_spalding_wall_smoothness, spalding_wall_smoothness);
    glUniform1i(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_coarse_assembly, GL_FALSE);

    glDispatchCompute(
        (destruction_variable_system.x_elements + 7) / 8,
        (destruction_variable_system.y_elements + 7) / 8,
        (destruction_variable_system.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::destruction_variable::apply_eps_wall_treatment_for_k_eps_model(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level)
{
    glUseProgram(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model);

    // coarse_assembly == true: only the diagonal (binding 4) is written
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aP);

    glUniform1ui(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_grid_x, CurrentCells_x);
    glUniform1ui(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_grid_y, CurrentCells_y);
    glUniform1ui(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_grid_z, CurrentCells_z);
    glUniform1i(k_model.rans.shadermanager.eps_wall_treatment_for_k_eps_model_coarse_assembly, GL_TRUE);

    glDispatchCompute(
        (CurrentCells_x + 7) / 8,
        (CurrentCells_y + 7) / 8,
        (CurrentCells_z + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ---------------- k-omega model: k
void k_TurbulenceModels::apply_k_wall_treatment_for_k_omega_model(float spalding_wall_smoothness)
{
    glUseProgram(rans.shadermanager.k_wall_treatment_for_k_omega_model);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, rans.u_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rans.v_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, rans.w_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, KSystem.field.data);

    // Wall row is pinned (aP = 1, RHS = wall value): this must run AFTER the fine k system is assembled
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, rans.coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, rans.coefficients_stencil.RHS_main);

    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_omega_model_grid_x, KSystem.x_elements);
    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_omega_model_grid_y, KSystem.y_elements);
    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_omega_model_grid_z, KSystem.z_elements);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_omega_model_dx, rans.domain_size_x / KSystem.x_elements);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_omega_model_dy, rans.domain_size_y / KSystem.y_elements);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_omega_model_dz, rans.domain_size_z / KSystem.z_elements);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_omega_model_dynamic_viscosity, rans.dynamic_viscosity);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_omega_model_density, rans.density);
    glUniform1f(rans.shadermanager.k_wall_treatment_for_k_omega_model_spalding_wall_smoothness, spalding_wall_smoothness);
    glUniform1i(rans.shadermanager.k_wall_treatment_for_k_omega_model_coarse_assembly, GL_FALSE);

    glDispatchCompute(
        (KSystem.x_elements + 7) / 8,
        (KSystem.y_elements + 7) / 8,
        (KSystem.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::apply_k_wall_treatment_for_k_omega_model(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level)
{
    glUseProgram(rans.shadermanager.k_wall_treatment_for_k_omega_model);

    // coarse_assembly == true: only the diagonal (binding 4) is written
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aP);

    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_omega_model_grid_x, CurrentCells_x);
    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_omega_model_grid_y, CurrentCells_y);
    glUniform1ui(rans.shadermanager.k_wall_treatment_for_k_omega_model_grid_z, CurrentCells_z);
    glUniform1i(rans.shadermanager.k_wall_treatment_for_k_omega_model_coarse_assembly, GL_TRUE);

    glDispatchCompute(
        (CurrentCells_x + 7) / 8,
        (CurrentCells_y + 7) / 8,
        (CurrentCells_z + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ---------------- k-omega model: omega  (also prescribes mu_eff = mu + mu_t_wall in the wall cells)
void k_TurbulenceModels::destruction_variable::apply_omega_wall_treatment_for_k_omega_model(float spalding_wall_smoothness)
{
    glUseProgram(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.rans.u_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.rans.v_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.w_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, destruction_variable_system.field.data);

    // Wall row is pinned (aP = 1, RHS = wall value): this must run AFTER the fine omega system is assembled
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, k_model.rans.coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, k_model.rans.coefficients_stencil.RHS_main);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, k_model.rans.mu_eff);       // the omega wall kernel also prescribes mu_eff (= mu + mu_t_wall) in the wall cells

    glUniform1ui(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_dx, k_model.rans.domain_size_x / destruction_variable_system.x_elements);
    glUniform1f(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_dy, k_model.rans.domain_size_y / destruction_variable_system.y_elements);
    glUniform1f(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_dz, k_model.rans.domain_size_z / destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_dynamic_viscosity, k_model.rans.dynamic_viscosity);
    glUniform1f(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_density, k_model.rans.density);
    glUniform1f(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_spalding_wall_smoothness, spalding_wall_smoothness);
    glUniform1i(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_coarse_assembly, GL_FALSE);

    glDispatchCompute(
        (destruction_variable_system.x_elements + 7) / 8,
        (destruction_variable_system.y_elements + 7) / 8,
        (destruction_variable_system.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::destruction_variable::apply_omega_wall_treatment_for_k_omega_model(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level)
{
    glUseProgram(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model);

    // coarse_assembly == true: only the diagonal (binding 4) is written
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aP);

    glUniform1ui(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_grid_x, CurrentCells_x);
    glUniform1ui(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_grid_y, CurrentCells_y);
    glUniform1ui(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_grid_z, CurrentCells_z);
    glUniform1i(k_model.rans.shadermanager.omega_wall_treatment_for_k_omega_model_coarse_assembly, GL_TRUE);

    glDispatchCompute(
        (CurrentCells_x + 7) / 8,
        (CurrentCells_y + 7) / 8,
        (CurrentCells_z + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//INITIALIZATION      u / v / w  ->  k0, eps0  ->  mu_eff
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool k_TurbulenceModels::destruction_variable::initialize_k_epsilon_RANS_fields(float reference_velocity, float turbulence_intensity, float turbulence_length_scale, float spalding_wall_smoothness)
{
    // eps0 divides by L: refuse before any buffer is touched
    if (!std::isfinite(turbulence_length_scale) || !(turbulence_length_scale > 0.0f))
    {
        std::cerr << "k-epsilon initialization failed: the turbulence length scale must be a positive number (got " << turbulence_length_scale << ")\n";
        return false;
    }

    // 1a. k0 and eps0 into the existing k / epsilon SSBOs
    glUseProgram(k_model.rans.shadermanager.k_eps_initialize);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.KSystem.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable_system.field.data);

    glUniform1ui(k_model.rans.shadermanager.k_eps_initialize_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.k_eps_initialize_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.k_eps_initialize_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.k_eps_initialize_reference_velocity, reference_velocity);
    glUniform1f(k_model.rans.shadermanager.k_eps_initialize_turbulence_intensity, turbulence_intensity);
    glUniform1f(k_model.rans.shadermanager.k_eps_initialize_turbulence_length_scale, turbulence_length_scale);

    glDispatchCompute(
        (destruction_variable_system.x_elements + 7) / 8,
        (destruction_variable_system.y_elements + 7) / 8,
        (destruction_variable_system.z_elements + 7) / 8
    );

    // the wall kernel reads the k / epsilon this dispatch wrote
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 1b. wall-cell k / eps from Spalding (so the first mu_eff does not see the uniform k0 / eps0 in the wall cells).
    //     The kernel also writes aP = 1 / RHS on the wall rows; the next assembly clears and rewrites them.
       
    //Both of these functions add coefficients and RHS values to the common buffer, making it useless as k and destruction variable are different systems, but they are cleared anyway right after this but it is useful as we get the

    apply_eps_wall_treatment_for_k_eps_model(spalding_wall_smoothness);
    k_model.apply_k_wall_treatment_for_k_eps_model(spalding_wall_smoothness);

    // 2. mu_eff = mu + rho C_mu k^2 / eps  through the existing GPU path (ends with its own barrier)
    update_mu_eff_k_eps();

    return true;
}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//INITIALIZATION (K-OMEGA)      u / v / w  ->  k0, omega0  ->  mu_eff
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
bool k_TurbulenceModels::destruction_variable::initialize_k_omega_RANS_fields(float reference_velocity, float turbulence_intensity, float turbulence_length_scale, float spalding_wall_smoothness, bool is_SST)
{
    // omega0 divides by L: refuse before any buffer is touched
    if (!std::isfinite(turbulence_length_scale) || !(turbulence_length_scale > 0.0f))
    {
        std::cerr << "k-omega initialization failed: the turbulence length scale must be a positive number (got " << turbulence_length_scale << ")\n";
        return false;
    }

    // 1a. k0 and omega0 into the existing k / omega SSBOs
    glUseProgram(k_model.rans.shadermanager.k_omega_initialize);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.KSystem.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable_system.field.data);

    glUniform1ui(k_model.rans.shadermanager.k_omega_initialize_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.k_omega_initialize_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.k_omega_initialize_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.k_omega_initialize_reference_velocity, reference_velocity);
    glUniform1f(k_model.rans.shadermanager.k_omega_initialize_turbulence_intensity, turbulence_intensity);
    glUniform1f(k_model.rans.shadermanager.k_omega_initialize_turbulence_length_scale, turbulence_length_scale);

    glDispatchCompute(
        (destruction_variable_system.x_elements + 7) / 8,
        (destruction_variable_system.y_elements + 7) / 8,
        (destruction_variable_system.z_elements + 7) / 8
    );

    // the wall kernel reads the k / omega this dispatch wrote
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // 1b. wall-cell k / omega from Spalding (so the first mu_eff does not see the uniform k0 / omega0 in the wall cells).
    //     The kernel also writes aP = 1 / RHS on the wall rows; the next assembly clears and rewrites them.

    //Both of these functions add coefficients and RHS values to the common buffer, making it useless as k and destruction variable are different systems, but they are cleared anyway right after this but it is useful as we get the

    apply_omega_wall_treatment_for_k_omega_model(spalding_wall_smoothness);
    k_model.apply_k_wall_treatment_for_k_omega_model(spalding_wall_smoothness);

    // SST ONLY: freestream omega floor of the cross diffusion term, built from the same two inputs (U_ref, length scale) that fix k0 and omega0,
    // and the positivity floor on the freshly initialised fields (with zero turbulence intensity k0 = omega0 = 0 exactly)
    if (is_SST && k_model.sst)
    {
        k_model.sst->set_omega_floor(reference_velocity, turbulence_length_scale);
        enforce_positivity();
    }

    // 2. mu_eff = mu + rho k / omega  through the existing GPU path (ends with its own barrier)
    if(is_SST)
        update_mu_eff_k_omega_SST();
    else
        update_mu_eff_k_omega();

    return true;
}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//EFFECTIVE VISCOSITY      mu_eff = mu + rho C_mu k^2 / eps
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
void k_TurbulenceModels::destruction_variable::update_mu_eff_k_eps()
{
    glUseProgram(k_model.rans.shadermanager.k_eps_mu_eff);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.KSystem.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.mu_eff);

    glUniform1ui(k_model.rans.shadermanager.k_eps_mu_eff_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.k_eps_mu_eff_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.k_eps_mu_eff_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.k_eps_mu_eff_dynamic_viscosity, k_model.rans.dynamic_viscosity);
    glUniform1f(k_model.rans.shadermanager.k_eps_mu_eff_density, k_model.rans.density);

    glDispatchCompute(
        (destruction_variable_system.x_elements + 7) / 8,
        (destruction_variable_system.y_elements + 7) / 8,
        (destruction_variable_system.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::destruction_variable::update_mu_eff_k_omega()
{
    glUseProgram(k_model.rans.shadermanager.k_omega_mu_eff);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.KSystem.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.mu_eff);

    glUniform1ui(k_model.rans.shadermanager.k_omega_mu_eff_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.k_omega_mu_eff_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.k_omega_mu_eff_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.k_omega_mu_eff_dynamic_viscosity, k_model.rans.dynamic_viscosity);
    glUniform1f(k_model.rans.shadermanager.k_omega_mu_eff_density, k_model.rans.density);

    glDispatchCompute(
        (destruction_variable_system.x_elements + 7) / 8,
        (destruction_variable_system.y_elements + 7) / 8,
        (destruction_variable_system.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// SST: mu_eff = mu + mu_t with mu_t = rho a1 k / max(a1 omega, S F2). Needs the wall distances (sst->min_wall_dists), so only call it when is_SST
void k_TurbulenceModels::destruction_variable::update_mu_eff_k_omega_SST()
{
    glUseProgram(k_model.rans.shadermanager.k_omega_SST_mu_eff);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.rans.u_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.rans.v_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.w_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, k_model.rans.mu_eff);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, k_model.KSystem.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, destruction_variable_system.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, k_model.sst->min_wall_dists);

    glUniform1ui(k_model.rans.shadermanager.k_omega_SST_mu_eff_grid_x, destruction_variable_system.x_elements);
    glUniform1ui(k_model.rans.shadermanager.k_omega_SST_mu_eff_grid_y, destruction_variable_system.y_elements);
    glUniform1ui(k_model.rans.shadermanager.k_omega_SST_mu_eff_grid_z, destruction_variable_system.z_elements);
    glUniform1f(k_model.rans.shadermanager.k_omega_SST_mu_eff_dx, float(k_model.rans.domain_size_x / destruction_variable_system.x_elements));
    glUniform1f(k_model.rans.shadermanager.k_omega_SST_mu_eff_dy, float(k_model.rans.domain_size_y / destruction_variable_system.y_elements));
    glUniform1f(k_model.rans.shadermanager.k_omega_SST_mu_eff_dz, float(k_model.rans.domain_size_z / destruction_variable_system.z_elements));
    glUniform1f(k_model.rans.shadermanager.k_omega_SST_mu_eff_dynamic_viscosity, k_model.rans.dynamic_viscosity);
    glUniform1f(k_model.rans.shadermanager.k_omega_SST_mu_eff_density, k_model.rans.density);

    glDispatchCompute(
        (destruction_variable_system.x_elements + 7) / 8,
        (destruction_variable_system.y_elements + 7) / 8,
        (destruction_variable_system.z_elements + 7) / 8
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}



// ============================================================================================================================================================
// ============================================================================================================================================================
//                                                                      SST SPECIFICS CODE
// ============================================================================================================================================================
// ============================================================================================================================================================

k_TurbulenceModels::SST_additions::SST_additions(k_TurbulenceModels &k_model_in)
:   k_model(k_model_in)
{
    std::vector<float> temp((size_t)k_model.rans.cells_x * k_model.rans.cells_y * k_model.rans.cells_z, 0.0); 
    min_wall_dists = k_model.create_SSBO(temp);
    blended_coefficient = k_model.create_SSBO(temp);

    // Same layout as eps_system.multigrid_levels_coeffiecients_arr: element [i - 1] is coarse level i (cell counts from the shared hierarchy)
    for (int i = 1; i < k_model.rans.hierarchy.level_count(); i++)
    {
        const GridLevelInfo &level_geometry = k_model.rans.hierarchy.level(i);
        std::vector <float> temp((size_t)level_geometry.cells_x * level_geometry.cells_y * level_geometry.cells_z, 0.0f);

        min_wall_dists_hierarchy.push_back(k_model.create_SSBO(temp));
        blended_coefficient_hierarchy.push_back(k_model.create_SSBO(temp));
    }
}

// omega_min = c U_ref / L_ref.
//   * U_ref is the reference velocity of the run, L_ref = min(turbulence length scale, longest domain side) (an eddy cannot be larger than the domain).
//   * 1 / omega is the turbulence time scale, so omega_min says "no turbulent time scale longer than c flow-through times of an L_ref sized eddy". Equivalently
//     the largest eddy viscosity the source can see is k / omega_min <= (k / U_ref^2) c^-1 U_ref L_ref, i.e. of the order of (velocity scale x length scale).
//   * It is a floor on the DIVIDING omega only: omega itself (and with it mu_t, k, the production / destruction terms) is never touched by it.
void k_TurbulenceModels::SST_additions::set_omega_floor(float reference_velocity, float turbulence_length_scale)
{
    omega_min = 0.0f;

    const float longest_domain_side = std::max(k_model.rans.domain_size_x, std::max(k_model.rans.domain_size_y, k_model.rans.domain_size_z));
    const float length_ref = std::min(turbulence_length_scale, longest_domain_side);

    if (!std::isfinite(reference_velocity) || !(std::fabs(reference_velocity) > 0.0f) || !std::isfinite(length_ref) || !(length_ref > 0.0f))
    {
        std::cerr << "SST warning: the omega floor of the cross diffusion term needs a non-zero reference velocity and a positive length scale (got U_ref = "
                  << reference_velocity << ", L = " << turbulence_length_scale << "); the floor is OFF and the run can diverge where k and omega are ~0\n";
        return;
    }

    omega_min = omega_floor_factor * std::fabs(reference_velocity) / length_ref;
}

void k_TurbulenceModels::SST_additions::compute_structured_mesh_cell_min_dists()
{
    glUseProgram(k_model.rans.shadermanager.compute_min_wall_dists_structured_mesh);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, min_wall_dists);

    glUniform1ui(k_model.rans.shadermanager.min_wall_dists_compute_grid_x, k_model.rans.cells_x);
    glUniform1ui(k_model.rans.shadermanager.min_wall_dists_compute_grid_y, k_model.rans.cells_y);
    glUniform1ui(k_model.rans.shadermanager.min_wall_dists_compute_grid_z, k_model.rans.cells_z);
    glUniform1f(k_model.rans.shadermanager.min_wall_dists_compute_dx, float(k_model.rans.domain_size_x / k_model.rans.cells_x));
    glUniform1f(k_model.rans.shadermanager.min_wall_dists_compute_dy, float(k_model.rans.domain_size_y / k_model.rans.cells_y));
    glUniform1f(k_model.rans.shadermanager.min_wall_dists_compute_dz, float(k_model.rans.domain_size_z / k_model.rans.cells_z));

    glDispatchCompute((k_model.rans.cells_x + 7) / 8, (k_model.rans.cells_y + 7) / 8, (k_model.rans.cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

//FOR COARSER LEVELS WALL DIST FIELD EVALUATION
void k_TurbulenceModels::SST_additions::compute_structured_mesh_cell_min_dists(int current_cells_x, int current_cells_y, int current_cells_z, GLuint wall_dists_ssbo_for_coarse_level)
{
    glUseProgram(k_model.rans.shadermanager.compute_min_wall_dists_structured_mesh);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, wall_dists_ssbo_for_coarse_level);

    glUniform1ui(k_model.rans.shadermanager.min_wall_dists_compute_grid_x, current_cells_x);
    glUniform1ui(k_model.rans.shadermanager.min_wall_dists_compute_grid_y, current_cells_y);
    glUniform1ui(k_model.rans.shadermanager.min_wall_dists_compute_grid_z, current_cells_z);
    glUniform1f(k_model.rans.shadermanager.min_wall_dists_compute_dx, float(k_model.rans.domain_size_x / current_cells_x));
    glUniform1f(k_model.rans.shadermanager.min_wall_dists_compute_dy, float(k_model.rans.domain_size_y / current_cells_y));
    glUniform1f(k_model.rans.shadermanager.min_wall_dists_compute_dz, float(k_model.rans.domain_size_z / current_cells_z));

    glDispatchCompute((current_cells_x + 7) / 8, (current_cells_y + 7) / 8, (current_cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// Cross diffusion source 2 (1 - F1) rho sigma_omega2 / omega * grad(k).grad(omega), added explicitly to the omega RHS (omega equation only, assembled after k is solved)
void k_TurbulenceModels::SST_additions::assemble_cross_diffusion_contribution()
{
    glUseProgram(k_model.rans.shadermanager.assemble_cross_diffusion_SST);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_model.rans.coefficients_stencil.RHS_main);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k_model.rans.previous_iterate_eps);       // old omega, same as the production / destruction of the omega equation
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.rans.previous_iterate_k);         // old k
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, blended_coefficient);

    glUniform1ui(k_model.rans.shadermanager.assemble_cross_diffusion_SST_grid_x, k_model.rans.cells_x);
    glUniform1ui(k_model.rans.shadermanager.assemble_cross_diffusion_SST_grid_y, k_model.rans.cells_y);
    glUniform1ui(k_model.rans.shadermanager.assemble_cross_diffusion_SST_grid_z, k_model.rans.cells_z);
    glUniform1f(k_model.rans.shadermanager.assemble_cross_diffusion_SST_dx, float(k_model.rans.domain_size_x / k_model.rans.cells_x));
    glUniform1f(k_model.rans.shadermanager.assemble_cross_diffusion_SST_dy, float(k_model.rans.domain_size_y / k_model.rans.cells_y));
    glUniform1f(k_model.rans.shadermanager.assemble_cross_diffusion_SST_dz, float(k_model.rans.domain_size_z / k_model.rans.cells_z));
    glUniform1f(k_model.rans.shadermanager.assemble_cross_diffusion_SST_density, k_model.rans.density);
    glUniform1f(k_model.rans.shadermanager.assemble_cross_diffusion_SST_omega_min, omega_min);

    glDispatchCompute((k_model.rans.cells_x + 7) / 8, (k_model.rans.cells_y + 7) / 8, (k_model.rans.cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// F1 blending function of the SST model, written into blended_coefficient (needs omega, k and the wall distances)
void k_TurbulenceModels::SST_additions::compute_blended_coefficient_field(GLuint omega_field)
{
    glUseProgram(k_model.rans.shadermanager.compute_blended_coefficients_SST);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, min_wall_dists);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, omega_field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_model.KSystem.field.data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, blended_coefficient);

    glUniform1ui(k_model.rans.shadermanager.blended_coefficients_SST_grid_x, k_model.rans.cells_x);
    glUniform1ui(k_model.rans.shadermanager.blended_coefficients_SST_grid_y, k_model.rans.cells_y);
    glUniform1ui(k_model.rans.shadermanager.blended_coefficients_SST_grid_z, k_model.rans.cells_z);
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_dx, float(k_model.rans.domain_size_x / k_model.rans.cells_x));
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_dy, float(k_model.rans.domain_size_y / k_model.rans.cells_y));
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_dz, float(k_model.rans.domain_size_z / k_model.rans.cells_z));
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_dynamic_viscosity, k_model.rans.dynamic_viscosity);
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_density, k_model.rans.density);
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_omega_min, omega_min);

    glDispatchCompute((k_model.rans.cells_x + 7) / 8, (k_model.rans.cells_y + 7) / 8, (k_model.rans.cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void k_TurbulenceModels::SST_additions::compute_blended_coefficient_field(int current_cells_x, int current_cells_y, int current_cells_z, GLuint wall_dists_ssbo_for_coarse_level, GLuint blended_coefficient_at_coarse_level, GLuint k_field_at_coarse_level, GLuint omega_field_at_coarse_level)
{
    glUseProgram(k_model.rans.shadermanager.compute_blended_coefficients_SST);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, wall_dists_ssbo_for_coarse_level);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, omega_field_at_coarse_level);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, k_field_at_coarse_level);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, blended_coefficient_at_coarse_level);

    glUniform1ui(k_model.rans.shadermanager.blended_coefficients_SST_grid_x, current_cells_x);
    glUniform1ui(k_model.rans.shadermanager.blended_coefficients_SST_grid_y, current_cells_y);
    glUniform1ui(k_model.rans.shadermanager.blended_coefficients_SST_grid_z, current_cells_z);
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_dx, float(k_model.rans.domain_size_x / current_cells_x));
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_dy, float(k_model.rans.domain_size_y / current_cells_y));
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_dz, float(k_model.rans.domain_size_z / current_cells_z));
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_dynamic_viscosity, k_model.rans.dynamic_viscosity);
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_density, k_model.rans.density);
    glUniform1f(k_model.rans.shadermanager.blended_coefficients_SST_omega_min, omega_min);

    glDispatchCompute((current_cells_x + 7) / 8, (current_cells_y + 7) / 8, (current_cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

//Assumes a pre restricted hierarchy of omega and k values in the respective hierarchy 
void k_TurbulenceModels::SST_additions::assemble_coarse_level_hierarchy_blended_coefficient(std::vector <GLuint> &destruction_variable_hierarchy)
{
    int passes = static_cast<int>(blended_coefficient_hierarchy.size());

    for (int i = 0; i <= passes - 1; i++)
    {
        int current_cells_x = k_model.rans.hierarchy.level(i + 1).cells_x;
        int current_cells_y = k_model.rans.hierarchy.level(i + 1).cells_y;
        int current_cells_z = k_model.rans.hierarchy.level(i + 1).cells_z;  

        // IMPORTANT NOTE: Right now we have a structured mesh that doesn't change throughout the simulation, So we can get away with only computing the wall dist once before the main loop but that doesn't work if the mesh becomes changing, so we are calling the function for wall dists in here anyway even though it will result in unecessary computation, later We CAN add some bool differentiating in structured persistent mesh and structured unpersistent mesh or other types.

        compute_structured_mesh_cell_min_dists(current_cells_x, current_cells_y, current_cells_z, min_wall_dists_hierarchy[i]);
        compute_blended_coefficient_field(current_cells_x, current_cells_y, current_cells_z, min_wall_dists_hierarchy[i], blended_coefficient_hierarchy[i], k_model.k_hierarchy[i], destruction_variable_hierarchy[i]);
    }
}

void k_TurbulenceModels::SST_additions::SST_preprocessing(GLuint &omega_field, std::vector <GLuint> &omega_hierarchy)
{
    // IMPORTANT NOTE: Right now we have a structured mesh that doesn't change throughout the simulation, So we can get away with only computing the wall dist once before the main loop but that doesn't work if the mesh becomes changing, so we are calling the function for wall dists in here anyway even though it will result in unecessary computation, later We CAN add some bool differentiating in structured persistent mesh and structured unpersistent mesh or other types.

    compute_structured_mesh_cell_min_dists();

    compute_blended_coefficient_field(omega_field);
    assemble_coarse_level_hierarchy_blended_coefficient(omega_hierarchy);
}