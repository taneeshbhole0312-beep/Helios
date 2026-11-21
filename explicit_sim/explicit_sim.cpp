#include "explicit_sim.hpp"
#include <algorithm>


void explicit_simulation::flat_groups_to_3d(int totalGroups, int &gx, int &gy, int &gz)
{
    const int maxPerAxis = 65535;

    gx = std::min(totalGroups, maxPerAxis);
    int remaining = (totalGroups + gx - 1) / gx;

    gy = std::min(remaining, maxPerAxis);
    gz = (remaining + gy - 1) / gy;
}

GLuint explicit_simulation::create_SSBO(const std::vector<float> &data)
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

float stream_func_dy(
    float x,
    float y,
    float u_Not,
    float shear_distance,
    float epsilon,
    float num_of_waves,
    float HEIGHT,
    float WIDTH)
{
    const float pi = 3.14159265358979323846f;
    const float dy = y - HEIGHT / 2.0f;
    const float wave_number = 2.0f * pi * num_of_waves / WIDTH;

    const float gaussian = std::exp(
        -(dy * dy) / (2.0f * shear_distance * shear_distance));

    const float first_term =
        -u_Not * std::tanh(dy / shear_distance);

    const float second_term =
        -epsilon * u_Not * (dy / shear_distance)
        * std::cos(wave_number * x) * gaussian;

    return first_term + second_term;
}
//DELETE LATER, FOR COOL INITIALIZATION
float stream_func_dx(
    float x,
    float y,
    float u_Not,
    float shear_distance,
    float epsilon,
    float num_of_waves,
    float HEIGHT,
    float WIDTH)
{
    const float pi = 3.14159265358979323846f;
    const float dy = y - HEIGHT / 2.0f;
    const float wave_number = 2.0f * pi * num_of_waves / WIDTH;

    const float gaussian = std::exp(
        -(dy * dy) / (2.0f * shear_distance * shear_distance));

    // v = -d(psi)/dx
    const float v =
        epsilon * u_Not * shear_distance * wave_number
        * std::sin(wave_number * x) * gaussian;

    return v;
}

//------------------------------explicit_simulation functions
explicit_simulation::MAC_vx explicit_simulation::MAC_vx_grid(int cells_x, int cells_y, int cells_z, float cell_side_y, float HEIGHT)
{
    MAC_vx grid;
    grid.vx.resize((cells_x + 1) * cells_y * cells_z);
    grid.vx_orignal.resize((cells_x + 1) * cells_y * cells_z);

    for (int i = 0; i < (cells_z); i++)
    {
        for (int j = 0; j < (cells_y); j++)
        {
            for (int k = 0; k < (cells_x + 1); k++)
            {
                int idx = i * ((cells_x + 1) * cells_y) + (j * (cells_x + 1)) + k;

                // float relative_to_center_face_y = ((float(j) + 0.5f) * cell_side_y) - (HEIGHT / 2);

                float input = 3.1415 * 2.0 * (((float)j + 0.5f) * cell_side_y) / HEIGHT;
                float y = ((float)j + 0.5f) * cell_side_y;
                float x = ((float)k) * cell_side_x;

                grid.vx[idx] = stream_func_dy(x, y, 500.0f, 45.0f, 0.02, 5.0f, HEIGHT, WIDTH);
                grid.vx_orignal[idx] = 0.0;

            }
        }
    }

    return grid;
}

explicit_simulation::MAC_vy explicit_simulation::MAC_vy_grid(int cells_x, int cells_y, int cells_z, float cell_side_x, float WIDTH)
{
    MAC_vy grid;
    grid.vy.resize((cells_y + 1) * cells_x * cells_z);
    grid.vy_orignal.resize(cells_x * (cells_y + 1) * cells_z);

    for (int i = 0; i < (cells_z); i++)
    {
        for (int j = 0; j < (cells_y + 1); j++)
        {
            for (int k = 0; k < (cells_x); k++)
            {
                int idx = i * ((cells_x) * (cells_y + 1)) + (j * (cells_x)) + k;

                // float relative_to_center_face_x = (k * cell_side_x) - (WIDTH / 2);

                // grid.vy[idx] = 0.85 * relative_to_center_face_x;
                float x = ((float)k + 0.5f) * cell_side_x;
                float y = (float)j * cell_side_y;

                grid.vy[idx] = stream_func_dx(x, y, 500.0f, 45.0f, 0.02, 5.0f, HEIGHT, WIDTH);
                grid.vy_orignal[idx] = 0.0;

            }
        }
    }

    return grid;
}

explicit_simulation::MAC_vz explicit_simulation::MAC_vz_grid(int cells_x, int cells_y, int cells_z)
{
    MAC_vz grid;
    grid.vz.resize(cells_y * cells_x * (cells_z + 1)) ;
    grid.vz_orignal.resize(cells_x * cells_y * (cells_z + 1));


    for (int i = 0; i < cells_x; i++)
    {
        for (int j = 0; j < cells_y; j++)
        {
            for (int k = 0; k < (cells_z + 1); k++)
            {
                int idx = k * (cells_x * cells_y) + j * cells_x + i;

                grid.vz[idx] = 0.0f;
                grid.vz_orignal[idx] = 0.0f;

            }
        }
    }

    return grid;
}

explicit_simulation::explicit_simulation(int& cells_x_in,int& cells_y_in,int& cells_z_in,float& cell_side_x_in,float& cell_side_y_in,float& cell_side_z_in, float &WIDTH_in, float &HEIGHT_in, int frame_buffer_w_in, int frame_buffer_h_in, ShaderManager &shadermanager_in)
    : cells_x(cells_x_in),
    cells_y(cells_y_in),
    cells_z(cells_z_in),
    cell_side_x(cell_side_x_in),
    cell_side_y(cell_side_y_in),
    cell_side_z(cell_side_z_in),
    WIDTH(WIDTH_in),
    HEIGHT(HEIGHT_in),
    frame_buffer_w(frame_buffer_w_in),
    frame_buffer_h(frame_buffer_h_in),
    shadermanager(shadermanager_in)
{

}


void explicit_simulation::initialize()
{
    //-------------------------------------------main field SSBO explicit_simulation + upload
    auto MAC_vx = MAC_vx_grid(cells_x, cells_y, cells_z, cell_side_y, HEIGHT);
    auto MAC_vy = MAC_vy_grid(cells_x, cells_y, cells_z, cell_side_x, WIDTH);
    auto MAC_vz = MAC_vz_grid(cells_x, cells_y, cells_z);

    vx_grid_SSBO = create_SSBO(MAC_vx.vx);
    Buffer_vx_grid_SSBO = create_SSBO(MAC_vx.vx); // same size, same zeros; ping-pong target

    vy_grid_SSBO = create_SSBO(MAC_vy.vy);
    Buffer_vy_grid_SSBO = create_SSBO(MAC_vy.vy); // same size, same zeros; ping-pong target

    vz_grid_SSBO = create_SSBO(MAC_vz.vz);
    Buffer_vz_grid_SSBO = create_SSBO(MAC_vz.vz); // same size, same zeros; ping-pong target

    vy_grid_orignal_SSBO = create_SSBO(MAC_vy.vy_orignal);
    vx_grid_orignal_SSBO = create_SSBO(MAC_vx.vx_orignal);
    vz_grid_orignal_SSBO = create_SSBO(MAC_vz.vz_orignal);

    //-------------------------------------------CFL max-|v| reduction buffers
    // chain of ever-smaller buffers: N -> ceil(N/256) -> ... -> 1
    int vx_MAC_total = (cells_x + 1) * cells_y * cells_z;
    int vy_MAC_total = cells_x * (cells_y + 1) * cells_z;
    int vz_MAC_total = cells_x * cells_y * (cells_z + 1);
    int CFL_max_elements_temp = std::max(vx_MAC_total, vy_MAC_total);
    int CFL_max_elements = std::max(CFL_max_elements_temp, vz_MAC_total);

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
}


//=========================================================================================================================================
//                                              EXPLICIT DNSish SOLVER OPERATORS
//=========================================================================================================================================

void explicit_simulation::viscosity_cal(float &stable_dt, float &dynamic_viscosity, float density)
{
    glUseProgram(shadermanager.viscosityXProgram);
    glUniform1i(shadermanager.visc_vx_cells_x_loc, cells_x);
    glUniform1i(shadermanager.visc_vx_cells_y_loc, cells_y);
    glUniform1i(shadermanager.visc_vx_cells_z_loc, cells_z);
    glUniform1f(shadermanager.visc_vx_viscosity_loc, dynamic_viscosity);
    glUniform1f(shadermanager.visc_vx_density_loc, density);
    glUniform1f(shadermanager.visc_vx_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.visc_vx_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.visc_vx_cell_side_z_loc, cell_side_z);

    glUseProgram(shadermanager.viscosityYProgram);
    glUniform1i(shadermanager.visc_vy_cells_x_loc, cells_x);
    glUniform1i(shadermanager.visc_vy_cells_y_loc, cells_y);
    glUniform1i(shadermanager.visc_vy_cells_z_loc, cells_z);
    glUniform1f(shadermanager.visc_vy_viscosity_loc, dynamic_viscosity);
    glUniform1f(shadermanager.visc_vy_density_loc, density);
    glUniform1f(shadermanager.visc_vy_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.visc_vy_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.visc_vy_cell_side_z_loc, cell_side_z);


    glUseProgram(shadermanager.viscosityZProgram);
    glUniform1i(shadermanager.visc_vz_cells_x_loc, cells_x);
    glUniform1i(shadermanager.visc_vz_cells_y_loc, cells_y);
    glUniform1i(shadermanager.visc_vz_cells_z_loc, cells_z);
    glUniform1f(shadermanager.visc_vz_viscosity_loc, dynamic_viscosity);
    glUniform1f(shadermanager.visc_vz_density_loc, density);
    glUniform1f(shadermanager.visc_vz_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.visc_vz_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.visc_vz_cell_side_z_loc, cell_side_z);

    // viscosity calculation
    glUseProgram(shadermanager.viscosityXProgram);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vx_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        Buffer_vx_grid_SSBO);

    glUniform1f(shadermanager.visc_vx_stable_dt_loc, stable_dt);

    glDispatchCompute(
        (cells_x + 8) / 8,
        (cells_y + 7) / 8,
        (cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(shadermanager.viscosityYProgram);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vy_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        Buffer_vy_grid_SSBO);

    glUniform1f(shadermanager.visc_vy_stable_dt_loc, stable_dt);

    glDispatchCompute(
        (cells_x + 7) / 8,
        (cells_y + 8) / 8,
        (cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(shadermanager.viscosityZProgram);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vz_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        Buffer_vz_grid_SSBO);

    glUniform1f(shadermanager.visc_vz_stable_dt_loc, stable_dt);

    glDispatchCompute(
        (cells_x + 7) / 8,
        (cells_y + 7) / 8,
        (cells_z + 8) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::swap(vx_grid_SSBO, Buffer_vx_grid_SSBO);
    std::swap(vy_grid_SSBO, Buffer_vy_grid_SSBO);
    std::swap(vz_grid_SSBO, Buffer_vz_grid_SSBO);
}

void explicit_simulation::WENOZ_advect(float &stable_dt)
{   
    glUseProgram(shadermanager.advect_vx_Program);
    glUniform1i(shadermanager.adv_vx_cells_x_loc, cells_x);
    glUniform1i(shadermanager.adv_vx_cells_y_loc, cells_y);
    glUniform1i(shadermanager.adv_vx_cells_z_loc, cells_z);
    glUniform1f(shadermanager.adv_vx_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.adv_vx_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.adv_vx_cell_side_z_loc, cell_side_z);

    glUseProgram(shadermanager.advect_vy_Program);
    glUniform1i(shadermanager.adv_vy_cells_x_loc, cells_x);
    glUniform1i(shadermanager.adv_vy_cells_y_loc, cells_y);
    glUniform1i(shadermanager.adv_vy_cells_z_loc, cells_z);
    glUniform1f(shadermanager.adv_vy_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.adv_vy_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.adv_vy_cell_side_z_loc, cell_side_z);

    glUseProgram(shadermanager.advect_vz_Program);
    glUniform1i(shadermanager.adv_vz_cells_x_loc, cells_x);
    glUniform1i(shadermanager.adv_vz_cells_y_loc, cells_y);
    glUniform1i(shadermanager.adv_vz_cells_z_loc, cells_z);
    glUniform1f(shadermanager.adv_vz_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.adv_vz_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.adv_vz_cell_side_z_loc, cell_side_z);


    //------------------------------------------------------advection shaders
    glUseProgram(shadermanager.advect_vx_Program);

    glUniform1f(shadermanager.adv_vx_stable_dt_loc, stable_dt);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vx_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vy_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        2,
        vz_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        3,
        Buffer_vx_grid_SSBO);

    // vx lattice: (cells_x + 1) columns, cells_y rows, cells_z layers   (local size 8,8,8)
    glDispatchCompute(((cells_x + 1) + 7) / 8, (cells_y + 7) / 8, (cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    //------------------------------------- VY advection

    glUseProgram(shadermanager.advect_vy_Program);

    glUniform1f(shadermanager.adv_vy_stable_dt_loc, stable_dt);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vy_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vx_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        2,
        vz_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        3,
        Buffer_vy_grid_SSBO);

    // vy lattice: cells_x columns, (cells_y + 1) rows, cells_z layers   (local size 8,8,8)
    glDispatchCompute((cells_x + 7) / 8, ((cells_y + 1) + 7) / 8, (cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    //------------------------------------- VZ advection

    glUseProgram(shadermanager.advect_vz_Program);

    glUniform1f(shadermanager.adv_vz_stable_dt_loc, stable_dt);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vz_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vx_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        2,
        vy_grid_SSBO);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        3,
        Buffer_vz_grid_SSBO);

    // vz lattice: cells_x columns, cells_y rows, (cells_z + 1) layers   (local size 8,8,8)
    glDispatchCompute((cells_x + 7) / 8, (cells_y + 7) / 8, ((cells_z + 1) + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // all three kernels read the pre-swap (old) fields, so the swaps happen together at the end
    std::swap(Buffer_vx_grid_SSBO, vx_grid_SSBO);
    std::swap(Buffer_vy_grid_SSBO, vy_grid_SSBO);
    std::swap(Buffer_vz_grid_SSBO, vz_grid_SSBO);
}

void explicit_simulation::SSPRK2_blend_step()
{
    glUseProgram(shadermanager.final_blend_SSPRK2_program);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vx_grid_orignal_SSBO
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vx_grid_SSBO
    );

    glUniform1ui(shadermanager.final_blend_SSPRK2_count_loc, (cells_x + 1) * cells_y * cells_z);

    {
        int groups = (((cells_x + 1) * cells_y * cells_z) + 255) / 256;
        int gx, gy, gz;
        flat_groups_to_3d(groups, gx, gy, gz);
        glDispatchCompute(gx, gy, gz);
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(shadermanager.final_blend_SSPRK2_program);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vy_grid_orignal_SSBO
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vy_grid_SSBO
    );

    glUniform1ui(shadermanager.final_blend_SSPRK2_count_loc, (cells_y + 1) * cells_x * cells_z);

    {
        int groups = (((cells_y + 1) * cells_x * cells_z) + 255) / 256;
        int gx, gy, gz;
        flat_groups_to_3d(groups, gx, gy, gz);
        glDispatchCompute(gx, gy, gz);
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(shadermanager.final_blend_SSPRK2_program);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        vz_grid_orignal_SSBO
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vz_grid_SSBO
    );

    glUniform1ui(shadermanager.final_blend_SSPRK2_count_loc, cells_y * cells_x * (cells_z + 1));

    {
        int groups = (cells_y * cells_x * (cells_z + 1) + 255) / 256;
        int gx, gy, gz;
        flat_groups_to_3d(groups, gx, gy, gz);
        glDispatchCompute(gx, gy, gz);
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}


float explicit_simulation::velocity_component_reduction(GLuint velocity, int total_elements)
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
