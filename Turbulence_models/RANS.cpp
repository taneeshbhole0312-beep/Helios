#include "RANS.hpp"
#include <algorithm>

GLuint RANS::create_SSBO(const std::vector<float> &data)
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

GLuint common_stencil::create_SSBO(const std::vector<float> &data)
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


// Largest DOF count of any topology (x-, y- or z-face) on one level: the shared stencil of that level must hold every system
static size_t largest_topology_dof_count(const GridLevelInfo &level)
{
    size_t x_centered_dof = (size_t)(level.cells_x + 1) * level.cells_y * level.cells_z;
    size_t y_centered_dof = (size_t)level.cells_x * (level.cells_y + 1) * level.cells_z;
    size_t z_centered_dof = (size_t)level.cells_x * level.cells_y * (level.cells_z + 1);

    return std::max({x_centered_dof, y_centered_dof, z_centered_dof});
}

common_stencil::common_stencil(const MultigridHierarchy &hierarchy)
{
    std::vector <float> temp(largest_topology_dof_count(hierarchy.level(0)));
    coefficients.aP = create_SSBO(temp);
    coefficients.aE = create_SSBO(temp);
    coefficients.aW = create_SSBO(temp);
    coefficients.aS = create_SSBO(temp);
    coefficients.aN = create_SSBO(temp);
    coefficients.aB = create_SSBO(temp);
    coefficients.aT = create_SSBO(temp);
            
    RHS_main = create_SSBO(temp);

    for (int i = 1; i < hierarchy.level_count(); i++)
    {
        // sized for the largest topology at THIS level (x-, y- or z-face), shared by every system
        std::vector<float> coarse(largest_topology_dof_count(hierarchy.level(i)), 0.0f);

        StencilOperator coarse_operator;

        coarse_operator.aP = create_SSBO(coarse);
        coarse_operator.aW = create_SSBO(coarse);
        coarse_operator.aE = create_SSBO(coarse);
        coarse_operator.aS = create_SSBO(coarse);
        coarse_operator.aN = create_SSBO(coarse);
        coarse_operator.aB = create_SSBO(coarse);
        coarse_operator.aT = create_SSBO(coarse);

        multigrid_levels_coeffiecients_arr.push_back(coarse_operator);
    }

}


RANS::RANS(const MultigridHierarchy &hierarchy_in, ShaderManager &shadermanager_in, configuration &config)
:   hierarchy(hierarchy_in),
    u_system(hierarchy_in, shadermanager_in, fieldTopology::XFACE_CENTERED, config),
    v_system(hierarchy_in, shadermanager_in, fieldTopology::YFACE_CENTERED, config),
    w_system(hierarchy_in, shadermanager_in, fieldTopology::ZFACE_CENTERED, config),
    cells_x(config.cells_x),
    cells_y(config.cells_y),
    cells_z(config.cells_z),
    shadermanager(shadermanager_in),
    domain_size_x(hierarchy_in.domain_size_x()),
    domain_size_y(hierarchy_in.domain_size_y()),
    domain_size_z(hierarchy_in.domain_size_z()),
    density(config.density),
    dynamic_viscosity(config.dynamic_viscosity),
    coefficients_stencil(hierarchy_in),
    config(config)
{
    std::vector <float> temp((size_t)cells_x * cells_y * cells_z, dynamic_viscosity);

    mu_eff = create_SSBO(temp);

    //Vel hierarchy initialization: element counts of every coarse level come from the same LinearSystems the solvers use
    for(int level = 1; level < hierarchy.level_count(); level++)
    {
        std::vector <int> u_dims = u_system.GetLevelDimension(level);
        std::vector <int> v_dims = v_system.GetLevelDimension(level);
        std::vector <int> w_dims = w_system.GetLevelDimension(level);

        u_vel_hierarcy.push_back(create_SSBO(std::vector <float>((size_t)u_dims[0] * u_dims[1] * u_dims[2])));
        v_vel_hierarcy.push_back(create_SSBO(std::vector <float>((size_t)v_dims[0] * v_dims[1] * v_dims[2])));
        w_vel_hierarcy.push_back(create_SSBO(std::vector <float>((size_t)w_dims[0] * w_dims[1] * w_dims[2])));

        const GridLevelInfo &level_geometry = hierarchy.level(level);
        mu_eff_hierarchy.push_back(create_SSBO(std::vector <float>((size_t)level_geometry.cells_x * level_geometry.cells_y * level_geometry.cells_z)));      //Cell centered mu_eff adjusted for levels
    }

    //---------------------------------------- nonlinear convergence: previous-iterate buffers (same layouts as the unknowns)
    previous_iterate_u   = create_SSBO(std::vector<float>((size_t)u_system.x_elements * u_system.y_elements * u_system.z_elements, 0.0f));
    previous_iterate_v   = create_SSBO(std::vector<float>((size_t)v_system.x_elements * v_system.y_elements * v_system.z_elements, 0.0f));
    previous_iterate_w   = create_SSBO(std::vector<float>((size_t)w_system.x_elements * w_system.y_elements * w_system.z_elements, 0.0f));
    previous_iterate_k   = create_SSBO(std::vector<float>((size_t)cells_x * cells_y * cells_z, 0.0f));
    previous_iterate_eps = create_SSBO(std::vector<float>((size_t)cells_x * cells_y * cells_z, 0.0f));

    //---------------------------------------- nonlinear convergence: reduction chain (8 x 8 x 8 work groups -> one float per group)
    {
        int current_x = cells_x + 1;
        int current_y = cells_y + 1;
        int current_z = cells_z + 1;

        while (current_x > 1 || current_y > 1 || current_z > 1)
        {
            current_x = (current_x + 7) / 8;
            current_y = (current_y + 7) / 8;
            current_z = (current_z + 7) / 8;

            nonlinear_reduction_chain.push_back(create_SSBO(std::vector<float>((size_t)current_x * current_y * current_z, 0.0f)));
        }
    }

    nonlinear_reduction_results = create_SSBO(std::vector<float>(10, 0.0f));
}

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
// VELOCITY INITIALIZATION: identical equations, coordinates and indexing to explicit_simulation::MAC_vx_grid / MAC_vy_grid / MAC_vz_grid.
// WIDTH = domain_size_x, HEIGHT = domain_size_y, cell_side_* = domain_size_* / cells_*  (same float values the explicit solver uses)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//DELETE LATER, FOR COOL INITIALIZATION
float RANS::stream_func_dy(
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
float RANS::stream_func_dx(
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


void RANS::initialize_velocity_fields()
{
    float WIDTH = domain_size_x;
    float HEIGHT = domain_size_y;
    float cell_side_x = domain_size_x / (float)cells_x;
    float cell_side_y = domain_size_y / (float)cells_y;

    //---------------------------------------------------------------- u (vx):  (cells_x + 1) * cells_y * cells_z
    std::vector<float> vx((cells_x + 1) * cells_y * cells_z, 0.0);

    for (int i = 0; i < (cells_z); i++)
    {
        for (int j = 0; j < (cells_y); j++)
        {
            for (int k = 0; k < (cells_x + 1); k++)
            {
                if (k == 0 || k == cells_x)
                    continue;

                int idx = i * ((cells_x + 1) * cells_y) + (j * (cells_x + 1)) + k;

                // float relative_to_center_face_y = ((float(j) + 0.5f) * cell_side_y) - (HEIGHT / 2);

                // float input = 3.1415 * 2.0 * (((float)j + 0.5f) * cell_side_y) / HEIGHT;

                float y = ((float)j + 0.5f) * cell_side_y;
                float x = ((float)k) * cell_side_x;

                vx[idx] = stream_func_dy(x, y, 100.0f, 45.0f, 0.02, 20.0f, HEIGHT, WIDTH);
                // vx[idx] = 0.0;
            }
        }
    }

    //---------------------------------------------------------------- v (vy):  cells_x * (cells_y + 1) * cells_z
    std::vector<float> vy((cells_y + 1) * cells_x * cells_z, 0.0);
    const float U_ref = 10.0f;  // m/s

    for (int i = 0; i < (cells_z); i++)
    {
        for (int j = 0; j < (cells_y + 1); j++)
        {
            if(j == 0 || j == cells_y)
                continue;

            for (int k = 0; k < (cells_x); k++)
            {
                int idx = i * ((cells_x) * (cells_y + 1)) + (j * (cells_x)) + k;
                // float relative_to_center_face_x = (k * cell_side_x) - (WIDTH / 2);

                float x = ((float)k + 0.5f) * cell_side_x;
                float y = (float)j * cell_side_y;
                
                vy[idx] = stream_func_dx(x, y, 100.0f, 45.0f, 0.02, 20.0f, HEIGHT, WIDTH);

                // vy[idx] = U_ref * ((float)k + 0.5f) / (float)cells_x;
            }
        }
    }

    //---------------------------------------------------------------- w (vz):  cells_x * cells_y * (cells_z + 1)
    std::vector<float> vz(cells_y * cells_x * (cells_z + 1), 0.0);

    for (int i = 0; i < cells_x; i++)
    {
        for (int j = 0; j < cells_y; j++)
        {
            for (int k = 0; k < (cells_z + 1); k++)
            {
                int idx = k * (cells_x * cells_y) + j * cells_x + i;

                vz[idx] = 0.0f;
            }
        }
    }

    //---------------------------------------------------------------- upload into the existing RANS velocity SSBOs (same size, so no re-allocation)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, u_system.field.data);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vx.size() * sizeof(float), vx.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, v_system.field.data);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vy.size() * sizeof(float), vy.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, w_system.field.data);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, vz.size() * sizeof(float), vz.data());

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

void RANS::generic_restriction_staggered(LinearSystem &linear_system, int level, std::vector <GLuint> vel_fields)
{
    glUseProgram(shadermanager.physical_fields_multigrid_generic_restriction_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vel_fields[level - 1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_fields[level]);

    // vel_fields[level - 1] holds hierarchy level `level`, vel_fields[level] holds level `level + 1`.
    // Staggered DOF dimensions come from the LinearSystem so the stride matches what the assembly kernels read.
    std::vector <int> fine_dimensions = linear_system.GetLevelDimension(level);
    std::vector <int> coarse_dimensions = linear_system.GetLevelDimension(level + 1);

    int fine_level_dimension_x = fine_dimensions[0];
    int fine_level_dimension_y = fine_dimensions[1];
    int fine_level_dimension_z = fine_dimensions[2];

    int coarse_level_dimension_x = coarse_dimensions[0];
    int coarse_level_dimension_y = coarse_dimensions[1];
    int coarse_level_dimension_z = coarse_dimensions[2];

    glUniform3i(shadermanager.generic_restriction_fine_dimensions_loc, fine_level_dimension_x, fine_level_dimension_y, fine_level_dimension_z);
    glUniform3i(shadermanager.generic_restriction_coarse_dimensions_loc, coarse_level_dimension_x, coarse_level_dimension_y, coarse_level_dimension_z);
    glUniform1i(shadermanager.generic_restriction_topology_loc, int(linear_system.field.topology));

    glDispatchCompute((coarse_level_dimension_x + 7) / 8, (coarse_level_dimension_y + 7) / 8, (coarse_level_dimension_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);         // next level reads what this dispatch wrote
}

void RANS::restrict_vels(bool use_previous_iterate_velocities)
{
    // momentum equations are assembled from the velocities of the START of the nonlinear iteration (previous_iterate_*), because
    // the u, v and w systems are now assembled and solved one after another; k / epsilon use the live (post-projection) velocities
    const GLuint u_source = use_previous_iterate_velocities ? previous_iterate_u : u_system.field.data;
    const GLuint v_source = use_previous_iterate_velocities ? previous_iterate_v : v_system.field.data;
    const GLuint w_source = use_previous_iterate_velocities ? previous_iterate_w : w_system.field.data;

    const int passes = hierarchy.coarse_level_count();

    for(int i=0; i < passes; i++)
    {
        if(i == 0)
        {
            // U Special case
            glUseProgram(shadermanager.physical_fields_multigrid_generic_restriction_program);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, u_source);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, u_vel_hierarcy[0]);

            std::vector <int> u_dimensions = u_system.GetLevelDimension(0);
            std::vector <int> u_coarse_dimensions = u_system.GetLevelDimension(1);          // staggered: (N/2 + 1) in x, not (N + 1) / 2

            int coarse_level_dimension_x_u = u_coarse_dimensions[0];
            int coarse_level_dimension_y_u = u_coarse_dimensions[1];
            int coarse_level_dimension_z_u = u_coarse_dimensions[2];

            glUniform3i(shadermanager.generic_restriction_fine_dimensions_loc, u_dimensions[0], u_dimensions[1], u_dimensions[2]);
            glUniform3i(shadermanager.generic_restriction_coarse_dimensions_loc, coarse_level_dimension_x_u, coarse_level_dimension_y_u, coarse_level_dimension_z_u);
            glUniform1i(shadermanager.generic_restriction_topology_loc, int(u_system.field.topology));

            glDispatchCompute((coarse_level_dimension_x_u + 7) / 8, (coarse_level_dimension_y_u + 7) / 8, (coarse_level_dimension_z_u + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // V Special case
            glUseProgram(shadermanager.physical_fields_multigrid_generic_restriction_program);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, v_source);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, v_vel_hierarcy[0]);

            std::vector <int> v_dimensions = v_system.GetLevelDimension(0);
            std::vector <int> v_coarse_dimensions = v_system.GetLevelDimension(1);          // staggered: (N/2 + 1) in y

            int coarse_level_dimension_x_v = v_coarse_dimensions[0];
            int coarse_level_dimension_y_v = v_coarse_dimensions[1];
            int coarse_level_dimension_z_v = v_coarse_dimensions[2];

            glUniform3i(shadermanager.generic_restriction_fine_dimensions_loc, v_dimensions[0], v_dimensions[1], v_dimensions[2]);
            glUniform3i(shadermanager.generic_restriction_coarse_dimensions_loc, coarse_level_dimension_x_v, coarse_level_dimension_y_v, coarse_level_dimension_z_v);
            glUniform1i(shadermanager.generic_restriction_topology_loc, int(v_system.field.topology));

            glDispatchCompute((coarse_level_dimension_x_v + 7) / 8, (coarse_level_dimension_y_v + 7) / 8, (coarse_level_dimension_z_v + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            // W Special case
            glUseProgram(shadermanager.physical_fields_multigrid_generic_restriction_program);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, w_source);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, w_vel_hierarcy[0]);

            std::vector <int> w_dimensions = w_system.GetLevelDimension(0);
            std::vector <int> w_coarse_dimensions = w_system.GetLevelDimension(1);          // staggered: (N/2 + 1) in z

            int coarse_level_dimension_x_w = w_coarse_dimensions[0];
            int coarse_level_dimension_y_w = w_coarse_dimensions[1];
            int coarse_level_dimension_z_w = w_coarse_dimensions[2];

            glUniform3i(shadermanager.generic_restriction_fine_dimensions_loc, w_dimensions[0], w_dimensions[1], w_dimensions[2]);
            glUniform3i(shadermanager.generic_restriction_coarse_dimensions_loc, coarse_level_dimension_x_w, coarse_level_dimension_y_w, coarse_level_dimension_z_w);
            glUniform1i(shadermanager.generic_restriction_topology_loc, int(w_system.field.topology));

            glDispatchCompute((coarse_level_dimension_x_w + 7) / 8, (coarse_level_dimension_y_w + 7) / 8, (coarse_level_dimension_z_w + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
        else
        {
            generic_restriction_staggered(u_system, i, u_vel_hierarcy);
            generic_restriction_staggered(v_system, i, v_vel_hierarcy);
            generic_restriction_staggered(w_system, i, w_vel_hierarcy);
        }
    }
}

void RANS::restrict_mu_eff()
{
    glUseProgram(shadermanager.physical_fields_multigrid_generic_restriction_program);
    glUniform1i(shadermanager.generic_restriction_topology_loc, int(fieldTopology::CELL_CENTERED));

    // level i -> level i + 1 ; level 0 is mu_eff itself, level L lives at mu_eff_hierarchy[L - 1]
    for(int i = 0; i < hierarchy.coarse_level_count(); i++)
    {
        const GridLevelInfo &fine = hierarchy.level(i);
        const GridLevelInfo &coarse = hierarchy.level(i + 1);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, (i == 0) ? mu_eff : mu_eff_hierarchy[i - 1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mu_eff_hierarchy[i]);

        glUniform3i(shadermanager.generic_restriction_fine_dimensions_loc, fine.cells_x, fine.cells_y, fine.cells_z);
        glUniform3i(shadermanager.generic_restriction_coarse_dimensions_loc, coarse.cells_x, coarse.cells_y, coarse.cells_z);

        glDispatchCompute((coarse.cells_x + 7) / 8, (coarse.cells_y + 7) / 8, (coarse.cells_z + 7) / 8);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

// IN BACKWARD EULER, this function MUST use the previous converged non linear iteration value of phi
void RANS::add_time_contribution(float dt, LinearSystem & linear_system)                                          // COMMON MATRIX CONSTRUCTION FUNCTION  (u, v, w and k, eps)
{
    glUseProgram(shadermanager.time_assemble);

    glUniform1ui(shadermanager.time_assemble_x_elements, linear_system.x_elements);
    glUniform1ui(shadermanager.time_assemble_y_elements, linear_system.y_elements);
    glUniform1ui(shadermanager.time_assemble_z_elements, linear_system.z_elements);
    glUniform1f(shadermanager.time_assemble_dx, domain_size_x / cells_x);
    glUniform1f(shadermanager.time_assemble_dy, domain_size_y / cells_y);
    glUniform1f(shadermanager.time_assemble_dz, domain_size_z / cells_z);
    glUniform1f(shadermanager.time_assemble_dt, dt);
    glUniform1f(shadermanager.time_assemble_rho, density);
    glUniform1i(shadermanager.time_assemble_topology, int(linear_system.field.topology));
    glUniform1i(shadermanager.time_assemble_coarse_assembly, GL_FALSE);
    glUniform1i(shadermanager.time_assemble_temporal_scheme, int(config.temporal_scheme));

    if(config.temporal_scheme == TemporalScheme::BACKWARD_EULER)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, linear_system.previous_time_step_field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.coefficients.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.RHS_main);
    }
    else if(config.temporal_scheme == TemporalScheme::BDF2)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, linear_system.previous_time_step_field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, linear_system.pre_previous_time_step_field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.coefficients.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.RHS_main); 
    }

    glDispatchCompute((linear_system.x_elements + 7) / 8, (linear_system.y_elements + 7) / 8, (linear_system.z_elements + 7) / 8);
}

void RANS::add_viscosity_contribution_momentum(LinearSystem &linear_system_momentum)
{
    glUseProgram(shadermanager.viscosity_LHS_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.coefficients.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.coefficients.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.coefficients.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.coefficients.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.coefficients.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.coefficients.aT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, mu_eff);

    glUniform1i(shadermanager.momentum_viscosity_LHS_assemble_cells_x, cells_x);
    glUniform1i(shadermanager.momentum_viscosity_LHS_assemble_cells_y, cells_y);
    glUniform1i(shadermanager.momentum_viscosity_LHS_assemble_cells_z, cells_z);
    glUniform1f(shadermanager.momentum_viscosity_LHS_assemble_dx, float(domain_size_x / cells_x));
    glUniform1f(shadermanager.momentum_viscosity_LHS_assemble_dy, float(domain_size_y / cells_y));
    glUniform1f(shadermanager.momentum_viscosity_LHS_assemble_dz, float(domain_size_z / cells_z));
    glUniform1i(shadermanager.momentum_viscosity_LHS_assemble_topology, int(linear_system_momentum.field.topology));

    glDispatchCompute((linear_system_momentum.x_elements + 7) / 8, (linear_system_momentum.y_elements + 7) / 8, (linear_system_momentum.z_elements + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(shadermanager.viscosity_RHS_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.RHS_main);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, previous_iterate_u);       // velocities of the start of the nonlinear iteration (see store_nonlinear_iterate)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, previous_iterate_v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, previous_iterate_w);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mu_eff);


    glUniform1i(shadermanager.momentum_viscosity_RHS_assemble_cells_x, cells_x);
    glUniform1i(shadermanager.momentum_viscosity_RHS_assemble_cells_y, cells_y);
    glUniform1i(shadermanager.momentum_viscosity_RHS_assemble_cells_z, cells_z);
    glUniform1f(shadermanager.momentum_viscosity_RHS_assemble_dx, float(domain_size_x / cells_x));
    glUniform1f(shadermanager.momentum_viscosity_RHS_assemble_dy, float(domain_size_y / cells_y));
    glUniform1f(shadermanager.momentum_viscosity_RHS_assemble_dz, float(domain_size_z / cells_z));
    glUniform1i(shadermanager.momentum_viscosity_RHS_assemble_topology, int(linear_system_momentum.field.topology));

    glDispatchCompute((linear_system_momentum.x_elements + 7) / 8, (linear_system_momentum.y_elements + 7) / 8, (linear_system_momentum.z_elements + 7) / 8);
};


// MOMENTUM WALL FUNCTION (SPALDING), IMPLICIT:  aP += rho u_tau^2 A / |U_t|  on every tangential DOF whose CV touches a wall.
// The viscous LHS kernel adds nothing on wall faces, so this is the ONLY wall shear in the momentum operator.
void RANS::add_wall_function_contribution_momentum(LinearSystem &linear_system_momentum, float spalding_wall_smoothness)
{
    glUseProgram(shadermanager.momentum_wall_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, previous_iterate_u);       // velocities of the start of the nonlinear iteration (see store_nonlinear_iterate)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, previous_iterate_v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, previous_iterate_w);

    glUniform1i(shadermanager.momentum_wall_assemble_cells_x, cells_x);
    glUniform1i(shadermanager.momentum_wall_assemble_cells_y, cells_y);
    glUniform1i(shadermanager.momentum_wall_assemble_cells_z, cells_z);
    glUniform1f(shadermanager.momentum_wall_assemble_dx, float(domain_size_x / cells_x));
    glUniform1f(shadermanager.momentum_wall_assemble_dy, float(domain_size_y / cells_y));
    glUniform1f(shadermanager.momentum_wall_assemble_dz, float(domain_size_z / cells_z));
    glUniform1f(shadermanager.momentum_wall_assemble_dynamic_viscosity, dynamic_viscosity);
    glUniform1f(shadermanager.momentum_wall_assemble_density, density);
    glUniform1f(shadermanager.momentum_wall_assemble_spalding_wall_smoothness, spalding_wall_smoothness);
    glUniform1i(shadermanager.momentum_wall_assemble_topology, int(linear_system_momentum.field.topology));

    glDispatchCompute((linear_system_momentum.x_elements + 7) / 8, (linear_system_momentum.y_elements + 7) / 8, (linear_system_momentum.z_elements + 7) / 8);
}


void RANS::add_advection_contribution(LinearSystem &linear_system, bool use_previous_iterate_velocities)                 // COMMON MATRIX CONSTRUCTION FUNCTION  (u, v, w and k, eps)
{
    // momentum (u, v, w): advecting velocities of the START of the nonlinear iteration (previous_iterate_*), because the three systems
    // are assembled and solved one after another;  k / epsilon: the live (post-projection) velocities
    const GLuint advecting_u = use_previous_iterate_velocities ? previous_iterate_u : u_system.field.data;
    const GLuint advecting_v = use_previous_iterate_velocities ? previous_iterate_v : v_system.field.data;
    const GLuint advecting_w = use_previous_iterate_velocities ? previous_iterate_w : w_system.field.data;

    glUseProgram(shadermanager.advection_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.coefficients.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.coefficients.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.coefficients.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.coefficients.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.coefficients.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.coefficients.aT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, advecting_u);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, advecting_v);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, advecting_w);

    glUniform1i(shadermanager.advection_assemble_cells_x, cells_x);
    glUniform1i(shadermanager.advection_assemble_cells_y, cells_y);
    glUniform1i(shadermanager.advection_assemble_cells_z, cells_z);
    glUniform1f(shadermanager.advection_assemble_dx, float(domain_size_x / cells_x));
    glUniform1f(shadermanager.advection_assemble_dy, float(domain_size_y / cells_y));
    glUniform1f(shadermanager.advection_assemble_dz, float(domain_size_z / cells_z));
    glUniform1f(shadermanager.advection_assemble_density, density);
    glUniform1i(shadermanager.advection_assemble_topology, int(linear_system.field.topology));

    // The matrix is first-order upwind for every scheme. For the second-order deferred scheme the kernel also needs the unknown at its latest iterate (phi)
    // and the RHS to subtract the corrections from (RHS_phi); these two bindings are only attached in that case.
    glUniform1i(shadermanager.advection_assemble_ConvectionScheme, int(config.advection_scheme));
    glUniform1i(shadermanager.advection_assemble_FluxLimiter, int(config.flux_limiter));

    if(config.advection_scheme == ConvectionScheme::SECOND_ORDER_DEFERRED)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, linear_system.field.data);               // unknown: not yet solved in this assembly, so it holds the last iterate
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, coefficients_stencil.RHS_main);
    }


    glDispatchCompute((linear_system.x_elements + 7) / 8, (linear_system.y_elements + 7) / 8, (linear_system.z_elements + 7) / 8);
}

void RANS::clear_coefficients(StencilOperator &stencil)
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stencil.aP);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stencil.aW);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stencil.aE);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stencil.aS);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stencil.aN);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stencil.aB);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stencil.aT);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
}

void RANS::clamp_field_to_floor(GLuint field, int element_count, float floor_value)
{
    if (element_count <= 0)
        return;

    glUseProgram(shadermanager.clamp_to_floor_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, field);

    glUniform1ui(shadermanager.clamp_to_floor_element_count, static_cast<GLuint>(element_count));
    glUniform1f(shadermanager.clamp_to_floor_value, floor_value);

    // 256 elements per work group; the groups are laid out over x and y (the kernel flattens them) so no dimension passes 65535
    const unsigned int groups = (static_cast<unsigned int>(element_count) + 255u) / 256u;
    const unsigned int groups_x = std::min(groups, 65535u);
    const unsigned int groups_y = (groups + groups_x - 1u) / groups_x;

    glDispatchCompute(groups_x, groups_y, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void RANS::construct_u_momentum_equation(float stable_dt, float spalding_wall_smoothness)
{
    clear_coefficients(coefficients_stencil.coefficients);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coefficients_stencil.RHS_main);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    add_time_contribution(stable_dt, u_system);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    add_advection_contribution(u_system, true);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    add_viscosity_contribution_momentum(u_system);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // wall shear (Spalding) on the diagonal: the viscous kernel leaves wall faces empty
    add_wall_function_contribution_momentum(u_system, spalding_wall_smoothness);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    build_u_coefficients_hierarchy(stable_dt, spalding_wall_smoothness);   // FIX THIS SHIT
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
}

void RANS::construct_v_momentum_equation(float stable_dt, float spalding_wall_smoothness)
{
    clear_coefficients(coefficients_stencil.coefficients);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coefficients_stencil.RHS_main);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    add_time_contribution(stable_dt, v_system);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    add_advection_contribution(v_system, true);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    add_viscosity_contribution_momentum(v_system);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // wall shear (Spalding) on the diagonal: the viscous kernel leaves wall faces empty
    add_wall_function_contribution_momentum(v_system, spalding_wall_smoothness);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    build_v_coefficients_hierarchy(stable_dt, spalding_wall_smoothness);   // FIX THIS SHIT
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
}

void RANS::construct_w_momentum_equation(float stable_dt, float spalding_wall_smoothness)
{
    clear_coefficients(coefficients_stencil.coefficients);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coefficients_stencil.RHS_main);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

    add_time_contribution(stable_dt, w_system);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    add_advection_contribution(w_system, true);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    add_viscosity_contribution_momentum(w_system);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // wall shear (Spalding) on the diagonal: the viscous kernel leaves wall faces empty
    add_wall_function_contribution_momentum(w_system, spalding_wall_smoothness);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    build_w_coefficients_hierarchy(stable_dt, spalding_wall_smoothness);   // FIX THIS SHIT
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
}



// --------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                          OVER LOADED FUNCTIONS FOR COARSER LEVELS
// --------------------------------------------------------------------------------------------------------------------------------------------------------------


// COMMON MATRIX CONSTRUCTION FUNCTION  (u, v, w and k, eps)
void RANS::add_time_contribution(float dt, int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, fieldTopology topology, StencilOperator &coefficients_at_coarse_level, Levels_info &level_info)
{
    glUseProgram(shadermanager.time_assemble);
    glUniform1ui(shadermanager.time_assemble_x_elements, level_info.level_x_elements);
    glUniform1ui(shadermanager.time_assemble_y_elements, level_info.level_y_elements);
    glUniform1ui(shadermanager.time_assemble_z_elements, level_info.level_z_elements);
    glUniform1f(shadermanager.time_assemble_dx, domain_size_x / CurrentCells_x);
    glUniform1f(shadermanager.time_assemble_dy, domain_size_y / CurrentCells_y);
    glUniform1f(shadermanager.time_assemble_dz,domain_size_z / CurrentCells_z);
    glUniform1f(shadermanager.time_assemble_dt, dt);
    glUniform1f(shadermanager.time_assemble_rho, density);
    glUniform1i(shadermanager.time_assemble_topology, int(topology));
    glUniform1i(shadermanager.time_assemble_coarse_assembly, GL_TRUE);
    glUniform1i(shadermanager.time_assemble_temporal_scheme, int(config.temporal_scheme));

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aP);     // binding 2 = aP in assemble_time.comp (0 / 1 = phi_old / phi_older, not read on coarse levels); only the coarse diagonal is written

    glDispatchCompute((level_info.level_x_elements + 7) / 8, (level_info.level_y_elements + 7) / 8, (level_info.level_z_elements + 7) / 8);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void RANS::add_viscosity_contribution_momentum(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, fieldTopology topology, GLuint &mu_eff_at_coarse_level, Levels_info &level_info)
{
    glUseProgram(shadermanager.viscosity_LHS_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_at_coarse_level.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_at_coarse_level.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_at_coarse_level.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_at_coarse_level.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_at_coarse_level.aT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, mu_eff_at_coarse_level);       

    glUniform1i(shadermanager.momentum_viscosity_LHS_assemble_cells_x, CurrentCells_x);
    glUniform1i(shadermanager.momentum_viscosity_LHS_assemble_cells_y, CurrentCells_y);
    glUniform1i(shadermanager.momentum_viscosity_LHS_assemble_cells_z, CurrentCells_z);
    glUniform1f(shadermanager.momentum_viscosity_LHS_assemble_dx, domain_size_x / CurrentCells_x);
    glUniform1f(shadermanager.momentum_viscosity_LHS_assemble_dy, domain_size_y / CurrentCells_y);
    glUniform1f(shadermanager.momentum_viscosity_LHS_assemble_dz, domain_size_z / CurrentCells_z);
    glUniform1i(shadermanager.momentum_viscosity_LHS_assemble_topology, int(topology));
    glDispatchCompute((level_info.level_x_elements + 7) / 8, (level_info.level_y_elements + 7) / 8, (level_info.level_z_elements + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}


// COARSE MOMENTUM WALL FUNCTION: same kernel, Spalding re-solved on the restricted velocities with the coarse
// spacing (wall distance = coarse half cell, wall area = coarse face). Only the coarse aP is written.
void RANS::add_wall_function_contribution_momentum(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, fieldTopology topology, GLuint &current_level_u_field, GLuint &current_level_v_field, GLuint &current_level_w_field, float spalding_wall_smoothness, Levels_info &level_info)
{
    glUseProgram(shadermanager.momentum_wall_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_at_coarse_level.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, current_level_u_field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, current_level_v_field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, current_level_w_field);

    glUniform1i(shadermanager.momentum_wall_assemble_cells_x, CurrentCells_x);
    glUniform1i(shadermanager.momentum_wall_assemble_cells_y, CurrentCells_y);
    glUniform1i(shadermanager.momentum_wall_assemble_cells_z, CurrentCells_z);
    glUniform1f(shadermanager.momentum_wall_assemble_dx, domain_size_x / CurrentCells_x);
    glUniform1f(shadermanager.momentum_wall_assemble_dy, domain_size_y / CurrentCells_y);
    glUniform1f(shadermanager.momentum_wall_assemble_dz, domain_size_z / CurrentCells_z);
    glUniform1f(shadermanager.momentum_wall_assemble_dynamic_viscosity, dynamic_viscosity);
    glUniform1f(shadermanager.momentum_wall_assemble_density, density);
    glUniform1f(shadermanager.momentum_wall_assemble_spalding_wall_smoothness, spalding_wall_smoothness);
    glUniform1i(shadermanager.momentum_wall_assemble_topology, int(topology));

    glDispatchCompute((level_info.level_x_elements + 7) / 8, (level_info.level_y_elements + 7) / 8, (level_info.level_z_elements + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}


// COMMON MATRIX CONSTRUCTION FUNCTION  (u, v, w and k, eps)
void RANS::add_advection_contribution(int current_cells_x, int current_cells_y, int current_cells_z, GLuint &current_level_u_field, GLuint &current_level_v_field, GLuint &current_level_w_field,fieldTopology topology, StencilOperator &coefficients_at_coarse_level, Levels_info &level_info)
{
    glUseProgram(shadermanager.advection_assemble);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_at_coarse_level.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_at_coarse_level.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_at_coarse_level.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_at_coarse_level.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_at_coarse_level.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_at_coarse_level.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_at_coarse_level.aT);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, current_level_u_field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, current_level_v_field);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, current_level_w_field);

    glUniform1i(shadermanager.advection_assemble_cells_x, current_cells_x);
    glUniform1i(shadermanager.advection_assemble_cells_y, current_cells_y);
    glUniform1i(shadermanager.advection_assemble_cells_z, current_cells_z);
    glUniform1f(shadermanager.advection_assemble_dx, domain_size_x / current_cells_x);
    glUniform1f(shadermanager.advection_assemble_dy, domain_size_y / current_cells_y);
    glUniform1f(shadermanager.advection_assemble_dz, domain_size_z / current_cells_z);
    glUniform1f(shadermanager.advection_assemble_density, density);
    glUniform1i(shadermanager.advection_assemble_topology, int(topology));

    // Every multigrid level carries the SAME first-order upwind operator: the second-order accuracy lives only in the fine-level RHS
    // (deferred correction). The coarse systems are residual equations with a restricted RHS, so no correction (and no phi / RHS_phi) here.
    glUniform1i(shadermanager.advection_assemble_ConvectionScheme, int(ConvectionScheme::UPWIND));


    glDispatchCompute((level_info.level_x_elements + 7) / 8, (level_info.level_y_elements + 7) / 8, (level_info.level_z_elements + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void RANS::build_u_coefficients_hierarchy(float dt, float spalding_wall_smoothness)
{
    // ------------------------------------------------------------
    // 1. Build physical-field hierarchies
    // ------------------------------------------------------------

    restrict_vels(true);
    restrict_mu_eff();


    // ------------------------------------------------------------
    // 2. Assemble the complete coarse operator at every level
    // ------------------------------------------------------------

    const int passes = hierarchy.coarse_level_count();


    // ------------------------------------------------------------
    // 3. Build each coarse momentum operator
    // ------------------------------------------------------------

    for (int level = 0; level < passes; level++)
    {
        int current_cells_x = hierarchy.level(level + 1).cells_x;
        int current_cells_y = hierarchy.level(level + 1).cells_y;
        int current_cells_z = hierarchy.level(level + 1).cells_z;


        // --------------------------------------------------------
        // U momentum operator
        // --------------------------------------------------------

        clear_coefficients(coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);

        add_time_contribution(dt, current_cells_x, current_cells_y, current_cells_z, u_system.field.topology, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], u_system.levels_info[level]);

        add_viscosity_contribution_momentum(current_cells_x, current_cells_y, current_cells_z, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], u_system.field.topology, mu_eff_hierarchy[level], u_system.levels_info[level]);

        add_advection_contribution(current_cells_x, current_cells_y, current_cells_z, u_vel_hierarcy[level], v_vel_hierarcy[level], w_vel_hierarcy[level], u_system.field.topology, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], u_system.levels_info[level]);

        add_wall_function_contribution_momentum(current_cells_x, current_cells_y, current_cells_z, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], u_system.field.topology, u_vel_hierarcy[level], v_vel_hierarcy[level], w_vel_hierarcy[level], spalding_wall_smoothness, u_system.levels_info[level]);
    }
}

void RANS::build_v_coefficients_hierarchy(float dt, float spalding_wall_smoothness)
{
    // ------------------------------------------------------------
    // 1. Build physical-field hierarchies
    // ------------------------------------------------------------

    restrict_vels(true);
    restrict_mu_eff();


    // ------------------------------------------------------------
    // 2. Assemble the complete coarse operator at every level
    // ------------------------------------------------------------

    const int passes = hierarchy.coarse_level_count();


    // ------------------------------------------------------------
    // 3. Build each coarse momentum operator
    // ------------------------------------------------------------

    for (int level = 0; level < passes; level++)
    {
        int current_cells_x = hierarchy.level(level + 1).cells_x;
        int current_cells_y = hierarchy.level(level + 1).cells_y;
        int current_cells_z = hierarchy.level(level + 1).cells_z;


        // --------------------------------------------------------
        // U momentum operator
        // --------------------------------------------------------

        clear_coefficients(coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);

        add_time_contribution(dt, current_cells_x, current_cells_y, current_cells_z, v_system.field.topology, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], v_system.levels_info[level]);

        add_viscosity_contribution_momentum(current_cells_x, current_cells_y, current_cells_z, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], v_system.field.topology, mu_eff_hierarchy[level], v_system.levels_info[level]);

        add_advection_contribution(current_cells_x, current_cells_y, current_cells_z, u_vel_hierarcy[level], v_vel_hierarcy[level], w_vel_hierarcy[level], v_system.field.topology, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], v_system.levels_info[level]);

        add_wall_function_contribution_momentum(current_cells_x, current_cells_y, current_cells_z, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], v_system.field.topology, u_vel_hierarcy[level], v_vel_hierarcy[level], w_vel_hierarcy[level], spalding_wall_smoothness, v_system.levels_info[level]);
    }
}

void RANS::build_w_coefficients_hierarchy(float dt, float spalding_wall_smoothness)
{
    // ------------------------------------------------------------
    // 1. Build physical-field hierarchies
    // ------------------------------------------------------------

    restrict_vels(true);
    restrict_mu_eff();


    // ------------------------------------------------------------
    // 2. Assemble the complete coarse operator at every level
    // ------------------------------------------------------------

    const int passes = hierarchy.coarse_level_count();


    // ------------------------------------------------------------
    // 3. Build each coarse momentum operator
    // ------------------------------------------------------------

    for (int level = 0; level < passes; level++)
    {
        int current_cells_x = hierarchy.level(level + 1).cells_x;
        int current_cells_y = hierarchy.level(level + 1).cells_y;
        int current_cells_z = hierarchy.level(level + 1).cells_z;


        // --------------------------------------------------------
        // U momentum operator
        // --------------------------------------------------------

        clear_coefficients(coefficients_stencil.multigrid_levels_coeffiecients_arr[level]);

        add_time_contribution(dt, current_cells_x, current_cells_y, current_cells_z, w_system.field.topology, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], w_system.levels_info[level]);

        add_viscosity_contribution_momentum(current_cells_x, current_cells_y, current_cells_z, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], w_system.field.topology, mu_eff_hierarchy[level], w_system.levels_info[level]);

        add_advection_contribution(current_cells_x, current_cells_y, current_cells_z, u_vel_hierarcy[level], v_vel_hierarcy[level], w_vel_hierarcy[level], w_system.field.topology, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], w_system.levels_info[level]);

        add_wall_function_contribution_momentum(current_cells_x, current_cells_y, current_cells_z, coefficients_stencil.multigrid_levels_coeffiecients_arr[level], w_system.field.topology, u_vel_hierarcy[level], v_vel_hierarcy[level], w_vel_hierarcy[level], spalding_wall_smoothness, w_system.levels_info[level]);
    }
}


// --------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                          K - DESTRUCTION VARIABLE ORCHESTRATION (SINGLE ENTRY POINT)
// --------------------------------------------------------------------------------------------------------------------------------------------------------------

// Assembles both fine systems and both coarse hierarchies. Nothing is solved here, so K sees eps_old and
// epsilon sees k_old for every interior row. Interior assembly never reads k / eps in a wall cell, so the
// wall treatment does not change the interior rows assembled before it.
//
// ORDER MATTERS:
//   1. fine K / eps assembly       (clears aP / RHS, skips wall cells)
//   2. fine wall treatment         (Spalding k / eps in wall cells, pins those fine rows)  -> must follow 1
//   3. K hierarchy, eps hierarchy  Restriction of k and eps should be done as the first thing for the hierarchy of k and eps
// --------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                          NONLINEAR CONVERGENCE (iterates of the SAME physical time step)
// --------------------------------------------------------------------------------------------------------------------------------------------------------------

void RANS::reduce_sum_of_squares(GLuint new_field, GLuint old_field, int field_x, int field_y, int field_z, int result_slot)
{
    //---------------------------------------- first pass: squares the input  ((new - old)^2  or  new^2)
    if (old_field != 0)
    {
        glUseProgram(shadermanager.difference_sq_summation);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, new_field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, old_field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, nonlinear_reduction_chain[0]);

        glUniform1ui(shadermanager.diff_sq_field_x, field_x);
        glUniform1ui(shadermanager.diff_sq_field_y, field_y);
        glUniform1ui(shadermanager.diff_sq_field_z, field_z);
    }
    else
    {
        glUseProgram(shadermanager.L2_norm_summation_of_sq);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, new_field);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, nonlinear_reduction_chain[0]);

        glUniform1ui(shadermanager.l2_field_x, field_x);
        glUniform1ui(shadermanager.l2_field_y, field_y);
        glUniform1ui(shadermanager.l2_field_z, field_z);
        glUniform1i(shadermanager.l2_squared_input, GL_FALSE);
    }

    int current_x = (field_x + 7) / 8;
    int current_y = (field_y + 7) / 8;
    int current_z = (field_z + 7) / 8;

    glDispatchCompute(current_x, current_y, current_z);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    //---------------------------------------- later passes: plain sums of already-squared partial sums (squared_input = true)
    int level = 0;

    glUseProgram(shadermanager.L2_norm_summation_of_sq);
    glUniform1i(shadermanager.l2_squared_input, GL_TRUE);

    while (current_x > 1 || current_y > 1 || current_z > 1)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, nonlinear_reduction_chain[level]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, nonlinear_reduction_chain[level + 1]);

        // the previous pass wrote one value per work group at its flat group index: read back as a (groups_x, groups_y, groups_z) field
        glUniform1ui(shadermanager.l2_field_x, current_x);
        glUniform1ui(shadermanager.l2_field_y, current_y);
        glUniform1ui(shadermanager.l2_field_z, current_z);

        current_x = (current_x + 7) / 8;
        current_y = (current_y + 7) / 8;
        current_z = (current_z + 7) / 8;

        glDispatchCompute(current_x, current_y, current_z);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        level++;
    }

    //---------------------------------------- final scalar -> results[slot]  (GPU -> GPU, no CPU transfer here)
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_COPY_READ_BUFFER, nonlinear_reduction_chain[level]);
    glBindBuffer(GL_COPY_WRITE_BUFFER, nonlinear_reduction_results);

    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, result_slot * sizeof(float), sizeof(float));

    
}

void RANS::store_nonlinear_iterate(k_TurbulenceModels &k_model, k_TurbulenceModels::destruction_variable &destruction_variable_model)
{
    // the solver kernels that produced the current fields wrote them as shader storage
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    GLuint sources[5] = { u_system.field.data, v_system.field.data, w_system.field.data, k_model.k_field(), destruction_variable_model.destruction_variable_field() };
    GLuint targets[5] = { previous_iterate_u, previous_iterate_v, previous_iterate_w, previous_iterate_k, previous_iterate_eps };

    GLsizeiptr sizes[5] =
    {
        (GLsizeiptr)u_system.x_elements * u_system.y_elements * u_system.z_elements * (GLsizeiptr)sizeof(float),
        (GLsizeiptr)v_system.x_elements * v_system.y_elements * v_system.z_elements * (GLsizeiptr)sizeof(float),
        (GLsizeiptr)w_system.x_elements * w_system.y_elements * w_system.z_elements * (GLsizeiptr)sizeof(float),
        (GLsizeiptr)cells_x * cells_y * cells_z * (GLsizeiptr)sizeof(float),
        (GLsizeiptr)cells_x * cells_y * cells_z * (GLsizeiptr)sizeof(float)
    };

    for (int i = 0; i < 5; i++)
    {
        glBindBuffer(GL_COPY_READ_BUFFER, sources[i]);
        glBindBuffer(GL_COPY_WRITE_BUFFER, targets[i]);

        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, sizes[i]);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

float RANS::calculate_nonlinear_convergence(k_TurbulenceModels &k_model, k_TurbulenceModels::destruction_variable &destruction_variable_model)
{
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    GLuint new_fields[5] = { u_system.field.data, v_system.field.data, w_system.field.data, k_model.k_field(), destruction_variable_model.destruction_variable_field() };
    GLuint old_fields[5] = { previous_iterate_u, previous_iterate_v, previous_iterate_w, previous_iterate_k, previous_iterate_eps };

    // actual DOF dimensions: XFACE (Nx+1,Ny,Nz), YFACE (Nx,Ny+1,Nz), ZFACE (Nx,Ny,Nz+1), CELL (Nx,Ny,Nz)
    int dims[5][3] =
    {
        { u_system.x_elements, u_system.y_elements, u_system.z_elements },
        { v_system.x_elements, v_system.y_elements, v_system.z_elements },
        { w_system.x_elements, w_system.y_elements, w_system.z_elements },
        { cells_x, cells_y, cells_z },
        { cells_x, cells_y, cells_z }
    };

    for (int i = 0; i < 5; i++)
    {
        reduce_sum_of_squares(new_fields[i], old_fields[i], dims[i][0], dims[i][1], dims[i][2], i);         // numerator    sum((new - old)^2)
        reduce_sum_of_squares(new_fields[i], 0,             dims[i][0], dims[i][1], dims[i][2], 5 + i);     // denominator  sum(new^2)
    }

    // the ONLY CPU transfer: 10 floats
    float sums[10];

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, nonlinear_reduction_results);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(sums), sums);

    // A field that is (numerically) zero everywhere has no relative change: the floor turns 0 / 0 into 0, and a field
    // that collapsed to zero from a non-zero value gives a huge delta (not converged), never a division by zero.
    const double denominator_floor = 1e-35;

    double max_delta = 0.0;

    
    double numerator_u   = std::max((double)sums[0], 0.0); // U
    double denominator_u = std::max((double)sums[5 + 0], denominator_floor);
    // std::cout << "DELTA OF u is " << std::sqrt(numerator_u / denominator_u) << std::endl; 
    
    double numerator_v   = std::max((double)sums[1], 0.0); // V
    double denominator_v = std::max((double)sums[5 + 1], denominator_floor);
    // std::cout << "DELTA OF v is " << std::sqrt(numerator_v / denominator_v) << std::endl; 
    
    double numerator_w   = std::max((double)sums[2], 0.0); // U
    double denominator_w = std::max((double)sums[5 + 2], denominator_floor);
    // std::cout << "DELTA OF w is " << std::sqrt(numerator_w / denominator_w) << std::endl; 
    
    
    double cumulative_change_in_vels_numerator = std::sqrt(numerator_u + numerator_v + numerator_w);
    double cumulative_change_in_vels_denominator = std::sqrt(denominator_u + denominator_v + denominator_w);

    double cumulative_change_in_vels = cumulative_change_in_vels_numerator / cumulative_change_in_vels_denominator;
    
    max_delta = std::max(max_delta, cumulative_change_in_vels);
    
    // std::cout << "DELTA OF all vels combined " << cumulative_change_in_vels << std::endl; 

    double numerator_k   = std::max((double)sums[3], 0.0); // U
    double denominator_k = std::max((double)sums[5 + 3], denominator_floor);
    // std::cout << "DELTA OF k is " << std::sqrt(numerator_k / denominator_k) << std::endl; 
    max_delta = std::max(max_delta, std::sqrt(numerator_k / denominator_k));
    
    double numerator_destruction_variable   = std::max((double)sums[4], 0.0); // U
    double denominator_destruction_variable = std::max((double)sums[5 + 4], denominator_floor);
    // std::cout << "DELTA OF destruction variable is " << std::sqrt(numerator_destruction_variable / denominator_destruction_variable) << std::endl; 
    max_delta = std::max(max_delta, std::sqrt(numerator_destruction_variable / denominator_destruction_variable));

    return (float)max_delta;
}
