#include "LinearSystem.hpp"

GLuint LinearSystem::create_SSBO(const std::vector<float> &data)
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

LinearSystem::LinearSystem(const MultigridHierarchy &hierarchy_in, ShaderManager &shadermanager_in, fieldTopology topology, configuration &config)
:   shadermanager(shadermanager_in),
    hierarchy(hierarchy_in),
    cells_x(config.cells_x),
    cells_y(config.cells_y),
    cells_z(config.cells_z),
    config(config)
{
    field.topology = topology;

    // A staggered field has one more DOF than cells along its normal axis:
    //     CELL (Nx, Ny, Nz)   XFACE (Nx + 1, Ny, Nz)   YFACE (Nx, Ny + 1, Nz)   ZFACE (Nx, Ny, Nz + 1)
    const int extra_x = (topology == fieldTopology::XFACE_CENTERED) ? 1 : 0;
    const int extra_y = (topology == fieldTopology::YFACE_CENTERED) ? 1 : 0;
    const int extra_z = (topology == fieldTopology::ZFACE_CENTERED) ? 1 : 0;

    x_elements = cells_x + extra_x;
    y_elements = cells_y + extra_y;
    z_elements = cells_z + extra_z;

    std::vector<float> temp((size_t)x_elements * y_elements * z_elements, 0.0f);

    field.data = create_SSBO(temp);
    previous_time_step_field = create_SSBO(temp);

    if(config.temporal_scheme == TemporalScheme::BDF2)
        pre_previous_time_step_field = create_SSBO(temp);
    else
        pre_previous_time_step_field = 0; 

    // Element counts of every coarse level, from the shared hierarchy (levels_info[k] = level k + 1)
    for (int level = 1; level < hierarchy.level_count(); level++)
    {
        const GridLevelInfo &level_geometry = hierarchy.level(level);

        Levels_info coarse_level_info;

        coarse_level_info.level_x_elements = level_geometry.cells_x + extra_x;
        coarse_level_info.level_y_elements = level_geometry.cells_y + extra_y;
        coarse_level_info.level_z_elements = level_geometry.cells_z + extra_z;

        levels_info.push_back(coarse_level_info);
    }
}

std::vector<int> LinearSystem::GetLevelDimension(int level) const
{
    if (level == 0)
    {
        return {
            x_elements,
            y_elements,
            z_elements
        };
    }

    return {
        levels_info[level - 1].level_x_elements,
        levels_info[level - 1].level_y_elements,
        levels_info[level - 1].level_z_elements
    };
}

void LinearSystem::save_previous_time_step_field()
{


    //---------------------------------------------------------------
    // previous_time_step_field ------> pre_previous_time_step_field 
    //---------------------------------------------------------------
    if(config.temporal_scheme == TemporalScheme::BDF2)
    {
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        glBindBuffer(GL_COPY_READ_BUFFER, previous_time_step_field);
        glBindBuffer(GL_COPY_WRITE_BUFFER, pre_previous_time_step_field);

        glCopyBufferSubData(
            GL_COPY_READ_BUFFER,
            GL_COPY_WRITE_BUFFER,
            0,
            0,
            x_elements * y_elements * z_elements * sizeof(float)
        );

        glBindBuffer(GL_COPY_READ_BUFFER, 0);
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    }

    //--------------------------------------------------------------
    // Freshly solved solution ------> previous_time_step_field
    //--------------------------------------------------------------

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_COPY_READ_BUFFER, field.data);
    glBindBuffer(GL_COPY_WRITE_BUFFER, previous_time_step_field);

    glCopyBufferSubData(
        GL_COPY_READ_BUFFER,
        GL_COPY_WRITE_BUFFER,
        0,
        0,
        x_elements * y_elements * z_elements * sizeof(float)
    );

    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}

//We have this separate function because when we do the copying for the Very first time pre while loop we need to do the copying like
// Current ---> previous --> pre previous
//Whereas usually inside loop it is
// previous ---> pre previous      then
//  current ---> previous
void LinearSystem::seed_time_history()  
{
    const GLsizeiptr bytes = (GLsizeiptr)x_elements * y_elements * z_elements * sizeof(float);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_COPY_READ_BUFFER, field.data);

    glBindBuffer(GL_COPY_WRITE_BUFFER, previous_time_step_field);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes);

    if (config.temporal_scheme == TemporalScheme::BDF2)
    {
        glBindBuffer(GL_COPY_WRITE_BUFFER, pre_previous_time_step_field);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes);
    }

    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
}