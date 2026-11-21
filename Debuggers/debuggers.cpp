#include "debuggers.hpp"
#include <algorithm>
#include <iostream>

float debuggers::get_max_absolute_divergence(GLuint vx_grid, GLuint vy_grid, GLuint vz_grid)
{
    std::vector <float> temporary(cells_x * cells_y * cells_z);
    incompressible.divergence_cal(vx_grid, vy_grid, vz_grid);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, incompressible.divergence);

    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER,
        0,
        temporary.size() * sizeof(float),
        temporary.data()
    );

    float max_divergence = 0.0f;

    for (float divergence : temporary)
    {
        max_divergence = std::max(max_divergence, std::abs(divergence));
    }

    return max_divergence;
}

float debuggers::get_mean_absolute_divergence(GLuint vx_grid, GLuint vy_grid, GLuint vz_grid)
{
    std::vector<float> temporary(cells_x * cells_y * cells_z);

    incompressible.divergence_cal(vx_grid, vy_grid, vz_grid);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, incompressible.divergence);

    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER,
        0,
        temporary.size() * sizeof(float),
        temporary.data()
    );

    float sum_divergence = 0.0f;

    for (float divergence : temporary)
    {
        sum_divergence += std::abs(divergence);
    }

    float mean_divergence = sum_divergence / (cells_x * cells_y * cells_z);

    return mean_divergence;
}

void debuggers::allocate_curl_arr()
{
    if(curl_allocated)
        return; 
    
    glGenBuffers(1, &curl_ssbo);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, curl_ssbo);
    
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        cells_x * cells_y * cells_z * sizeof(float) * 4,            //as we use vec4 in the curl computation
        nullptr,
        GL_DYNAMIC_COPY
    );

    curl_allocated = true;
}

// ============================================================================================================================
// ENERGY DIAGNOSTICS
// ============================================================================================================================
void debuggers::allocate_energy_reduction_chain()
{
    if (!energy_reduction_SSBOs.empty())
        return;

    int groups_x = (cells_x + 7) / 8;
    int groups_y = (cells_y + 7) / 8;
    int groups_z = (cells_z + 7) / 8;

    while (true)
    {
        GLuint ssbo;
        glGenBuffers(1, &ssbo);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)groups_x * groups_y * groups_z * sizeof(float), nullptr, GL_DYNAMIC_COPY);

        energy_reduction_SSBOs.push_back(ssbo);
        energy_reduction_dims.push_back(groups_x);
        energy_reduction_dims.push_back(groups_y);
        energy_reduction_dims.push_back(groups_z);

        if (groups_x == 1 && groups_y == 1 && groups_z == 1)
            break;

        groups_x = (groups_x + 7) / 8;
        groups_y = (groups_y + 7) / 8;
        groups_z = (groups_z + 7) / 8;
    }
}


double debuggers::sum_over_cells(int mode, GLuint vx_grid, GLuint vy_grid, GLuint vz_grid, GLuint k_field)
{
    allocate_energy_reduction_chain();

    //---------------------------------------- first pass: one partial sum per work group
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);         // the fields were written by other kernels

    glUseProgram(shadermanager.energy_reduction);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vx_grid);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vy_grid);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vz_grid);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, k_field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, energy_reduction_SSBOs[0]);

    glUniform1ui(shadermanager.energy_reduction_grid_x, cells_x);
    glUniform1ui(shadermanager.energy_reduction_grid_y, cells_y);
    glUniform1ui(shadermanager.energy_reduction_grid_z, cells_z);
    glUniform1i(shadermanager.energy_reduction_mode, mode);

    glDispatchCompute(energy_reduction_dims[0], energy_reduction_dims[1], energy_reduction_dims[2]);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    //---------------------------------------- later passes: plain sums of the partial sums (existing kernel, squared_input = true)
    glUseProgram(shadermanager.L2_norm_summation_of_sq);
    glUniform1i(shadermanager.l2_squared_input, GL_TRUE);

    for (size_t level = 0; level + 1 < energy_reduction_SSBOs.size(); level++)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, energy_reduction_SSBOs[level]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, energy_reduction_SSBOs[level + 1]);

        // the previous pass wrote one value per work group at its flat group index, read back as a (groups_x, groups_y, groups_z) field
        glUniform1ui(shadermanager.l2_field_x, energy_reduction_dims[3 * level + 0]);
        glUniform1ui(shadermanager.l2_field_y, energy_reduction_dims[3 * level + 1]);
        glUniform1ui(shadermanager.l2_field_z, energy_reduction_dims[3 * level + 2]);

        glDispatchCompute(energy_reduction_dims[3 * (level + 1) + 0], energy_reduction_dims[3 * (level + 1) + 1], energy_reduction_dims[3 * (level + 1) + 2]);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    //---------------------------------------- the single remaining float
    float total = 0.0f;

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, energy_reduction_SSBOs.back());
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float), &total);

    return (double)total;
}

double debuggers::get_mean_flow_kinetic_energy(GLuint vx_grid, GLuint vy_grid, GLuint vz_grid, float density)
{
    // KE = sum_cells 1/2 rho |u_c|^2 V,  |u_c| from the face velocities interpolated to the cell centroid (kernel, mode 0)
    const double cell_volume = (double)cell_side_x * (double)cell_side_y * (double)cell_side_z;

    return (double)density * cell_volume * sum_over_cells(0, vx_grid, vy_grid, vz_grid, 0);
}

double debuggers::get_turbulent_kinetic_energy(GLuint k_field, float density)
{
    // TKE = sum_cells rho k V   (kernel, mode 1)
    const double cell_volume = (double)cell_side_x * (double)cell_side_y * (double)cell_side_z;

    return (double)density * cell_volume * sum_over_cells(1, 0, 0, 0, k_field);
}

std::vector <int> debuggers::check_non_physical_turbulence_quantities(GLuint k_field, GLuint destruction_variable_field)
{
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);

    std::vector <int> output(2, 0);

    glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(int), output.data(), GL_DYNAMIC_COPY);

    glUseProgram(shadermanager.non_physical_turbulence_quantites_check_program);

    glUniform1i(shadermanager.non_physical_turbulence_quantites_check_cells_x, cells_x);
    glUniform1i(shadermanager.non_physical_turbulence_quantites_check_cells_y, cells_y);
    glUniform1i(shadermanager.non_physical_turbulence_quantites_check_cells_z, cells_z);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, k_field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, destruction_variable_field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo);

    glDispatchCompute((cells_x + 7) / 8, (cells_y + 7) / 8, (cells_z + 7) / 8);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 2 * sizeof(int), output.data());
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &ssbo);

    return output;          // output[0] = k negative count and output[1] = destruction variable negative count
}