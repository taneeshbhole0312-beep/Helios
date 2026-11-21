#include "incompressible_flow.hpp"
#include <algorithm>
#include <stdexcept>

GLuint incompressible_flow::create_SSBO(const std::vector<float>& data)
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


incompressible_flow::incompressible_flow(int cells_x_in, int cells_y_in, int cells_z_in, float cell_side_x_in, float cell_side_y_in, float cell_side_z_in, ShaderManager &shadermanager)
: cells_x(cells_x_in),
  cells_y(cells_y_in),
  cells_z(cells_z_in),
  shadermanager(shadermanager),
  cell_side_x(cell_side_x_in),
  cell_side_y(cell_side_y_in),
  cell_side_z(cell_side_z_in)
{
   std::vector <float> main_grid_size(cells_x * cells_y * cells_z);

   pressure = create_SSBO(main_grid_size);
   divergence = create_SSBO(main_grid_size);
}

incompressible_flow::specialized_pressure_Multigrid::specialized_pressure_Multigrid(const MultigridHierarchy &hierarchy_in, incompressible_flow &outer) // constructor
: hierarchy(hierarchy_in),
  shadermanager(outer.shadermanager),
  outer(outer)
{
    const GridLevelInfo &finest = hierarchy.level(0);

    if (finest.cells_x != outer.cells_x || finest.cells_y != outer.cells_y || finest.cells_z != outer.cells_z)
        throw std::invalid_argument("specialized_pressure_Multigrid: level 0 of the hierarchy is not the pressure grid");

    for (int i = 0; i < hierarchy.level_count(); i++)
    {
        size_t level_x = hierarchy.level(i).cells_x;
        size_t level_y = hierarchy.level(i).cells_y;
        size_t level_z = hierarchy.level(i).cells_z;

        GLuint level_correction_ssbo = 0;
        GLuint level_RHS_ssbo = 0;

        GLuint level_residual_ssbo;
        std::vector<float> level_residual(level_x * level_y * level_z);

        glGenBuffers(1, &level_residual_ssbo);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER,level_residual_ssbo);

        glBufferData(GL_SHADER_STORAGE_BUFFER,level_residual.size() * sizeof(float),level_residual.data(),GL_DYNAMIC_COPY);

        // MEMORY: level 0 never uses a correction or RHS buffer (the finest level works on outer.pressure and
        // outer.divergence; only its residual is needed). Index 0 stays in the arrays as an empty slot (GL name 0) so
        // every level index below is unchanged.
        if (i > 0)
        {
            std::vector<float> level_correction(level_x * level_y * level_z);
            std::vector<float> level_RHS(level_x * level_y * level_z);

            glGenBuffers(1, &level_correction_ssbo);
            glGenBuffers(1, &level_RHS_ssbo);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER,level_correction_ssbo);

            glBufferData(GL_SHADER_STORAGE_BUFFER,level_correction.size() * sizeof(float),level_correction.data(),GL_DYNAMIC_COPY);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER,level_RHS_ssbo);

            glBufferData(GL_SHADER_STORAGE_BUFFER,level_RHS.size() * sizeof(float),level_RHS.data(),GL_DYNAMIC_COPY);
        }

        levels_correction_SSBOs_arr.push_back(level_correction_ssbo);
        levels_residual_SSBOs_arr.push_back(level_residual_ssbo);
        levels_RHS_SSBOs_arr.push_back(level_RHS_ssbo);
    };

    
    //---------------------------------------------- Initialization of the L2 norm reductions levels to test for convergence -----------------------------
    int current_x = outer.cells_x;
    int current_y = outer.cells_y;
    int current_z = outer.cells_z;

    while (current_x > 1 || current_y > 1 || current_z > 1)
    {
        int groups_x = (current_x + 7) / 8;
        int groups_y = (current_y + 7) / 8;
        int groups_z = (current_z + 7) / 8;

        int total_groups = groups_x * groups_y * groups_z;

        // Allocate buffer for this reduction level.
        GLuint reduction_ssbo;
        glGenBuffers(1, &reduction_ssbo);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, reduction_ssbo);

        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            total_groups * sizeof(float),
            nullptr,
            GL_DYNAMIC_COPY
        );

        pressure_residual_L2_reductions_SSBOs_arr.push_back(reduction_ssbo);

        current_x = groups_x;
        current_y = groups_y;
        current_z = groups_z;
    }
}                                   // constructor end


// Residual of level fine_level -> RHS of level fine_level + 1.
// The pressure equation is in finite-difference form (lap(p) = rho/dt div(u), per unit volume), so its residual is a DENSITY and is
// restricted as a VOLUME AVERAGE (R_c = sum V_overlap R_f / V_c). That is exactly the cell-centered branch of the physical-field
// restriction kernel; the generic multigrid's residual kernel SUMS instead, because the RANS systems are volume-integrated.
void incompressible_flow::specialized_pressure_Multigrid::restrict_residual(int fine_level)
{
    const GridLevelInfo &fine = hierarchy.level(fine_level);
    const GridLevelInfo &coarse = hierarchy.level(fine_level + 1);

    glUseProgram(shadermanager.physical_fields_multigrid_generic_restriction_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, levels_residual_SSBOs_arr[fine_level]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, levels_RHS_SSBOs_arr[fine_level + 1]);

    glUniform3i(shadermanager.generic_restriction_fine_dimensions_loc, fine.cells_x, fine.cells_y, fine.cells_z);
    glUniform3i(shadermanager.generic_restriction_coarse_dimensions_loc, coarse.cells_x, coarse.cells_y, coarse.cells_z);
    glUniform1i(shadermanager.generic_restriction_topology_loc, 0);        // CELL_CENTERED

    glDispatchCompute((coarse.cells_x + 7) / 8, (coarse.cells_y + 7) / 8, (coarse.cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// fine_target += trilinear interpolation of the level coarse_level correction (fine_target lives on level coarse_level - 1)
void incompressible_flow::specialized_pressure_Multigrid::prolongate_correction(int coarse_level, GLuint fine_target)
{
    const GridLevelInfo &fine = hierarchy.level(coarse_level - 1);
    const GridLevelInfo &coarse = hierarchy.level(coarse_level);

    glUseProgram(shadermanager.multigrid_generic_prolongation_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, levels_correction_SSBOs_arr[coarse_level]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, fine_target);

    glUniform3i(shadermanager.generic_prolongation_fine_dimensions_loc, fine.cells_x, fine.cells_y, fine.cells_z);
    glUniform3i(shadermanager.generic_prolongation_coarse_dimensions_loc, coarse.cells_x, coarse.cells_y, coarse.cells_z);
    glUniform1i(shadermanager.generic_prolongation_topology_loc, 0);       // CELL_CENTERED

    glDispatchCompute((fine.cells_x + 7) / 8, (fine.cells_y + 7) / 8, (fine.cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void incompressible_flow::specialized_pressure_Multigrid::Multigrid_pressure_solve(float stable_dt, float density, int V_cycles_count)
{
    // passes = number of restriction steps, from the shared hierarchy
    const int passes = hierarchy.coarse_level_count();

    for (int vCycles = 0; vCycles < V_cycles_count; vCycles++)
    {

        // ---- zero coarse-grid corrections each V-cycle: the error equation A e = r needs initial guess e = 0
        for (int lvl = 1; lvl <= passes; lvl++)         // MEMORY: level 0 has no correction buffer (see constructor)
        {
            glBindBuffer(
                GL_SHADER_STORAGE_BUFFER,
                levels_correction_SSBOs_arr[lvl]);

            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, nullptr);
        }

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

        // ------------------------------------initial smoother (pre-smooth finest)

        glUseProgram(shadermanager.multigrid_pressure_smoother_program);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0, outer.pressure);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1, outer.divergence);

        glUniform1f(shadermanager.smoother_stable_dt_loc, stable_dt);
        glUniform1i(shadermanager.smoother_cells_x_loc, outer.cells_x); // initial lvel
        glUniform1i(shadermanager.smoother_cells_y_loc, outer.cells_y);
        glUniform1i(shadermanager.smoother_cells_z_loc, outer.cells_z);
        glUniform1f(shadermanager.smoother_cell_side_x_loc, outer.cell_side_x); // must reset: up-loop leaves this at a coarse-level value
        glUniform1f(shadermanager.smoother_cell_side_y_loc, outer.cell_side_y); // must reset: up-loop leaves this at a coarse-level value
        glUniform1f(shadermanager.smoother_cell_side_z_loc, outer.cell_side_z); // must reset: up-loop leaves this at a coarse-level value
        glUniform1f(shadermanager.smoother_density_loc, density); 

        for (int i = 0; i < 3; i++)
        {
            /* RED CELLS */
            glUniform1i(shadermanager.smoother_color_loc, 0);

            glDispatchCompute(
                (outer.cells_x + 7) / 8,
                (outer.cells_y + 7) / 8,
                (outer.cells_z + 7) / 8
                );

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            /* BLACK CELLS */
            glUniform1i(shadermanager.smoother_color_loc, 1);

            glDispatchCompute(
                (outer.cells_x + 7) / 8,
                (outer.cells_y + 7) / 8,
                (outer.cells_z + 7) / 8
                );

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        //------------------------------------calculate the residual for the first step down
        glUseProgram(shadermanager.multigrid_pressure_residual_calculation_program);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            0,
            outer.pressure);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            1,
            outer.divergence);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            2,
            levels_residual_SSBOs_arr[0]);

        glUniform1i(shadermanager.residual_current_cells_x_loc, outer.cells_x);
        glUniform1i(shadermanager.residual_current_cells_y_loc, outer.cells_y);
        glUniform1i(shadermanager.residual_current_cells_z_loc, outer.cells_z);
        glUniform1f(shadermanager.residual_dx_loc, outer.cell_side_x);
        glUniform1f(shadermanager.residual_dy_loc, outer.cell_side_y);
        glUniform1f(shadermanager.residual_dz_loc, outer.cell_side_z);
        glUniform1f(shadermanager.residual_stable_dt_loc, stable_dt);
        glUniform1f(shadermanager.residual_density_loc, density);

        glDispatchCompute(
            (outer.cells_x + 7) / 8,
            (outer.cells_y + 7) / 8,
            (outer.cells_z + 7) / 8
            );

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        //-------------------------------------------------intial step down
        restrict_residual(0);

        // passes and the vCycles loop now open at the top of this function

        //-----------------step down phase
        for (int i = 1; i < passes; i++) // starts from one as we already do one step down before
        {
            //---------------------------------------------first few smoothing iterations of Gauss sidel
            glUseProgram(shadermanager.multigrid_pressure_smoother_program);

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                0,
                levels_correction_SSBOs_arr[i]);

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                1,
                levels_RHS_SSBOs_arr[i]);

            int currentCellsY = hierarchy.level(i).cells_y;
            int currentCellsX = hierarchy.level(i).cells_x;
            int currentCellsZ = hierarchy.level(i).cells_z;
            float cell_side_level_x = hierarchy.level(i).dx;
            float cell_side_level_y = hierarchy.level(i).dy;
            float cell_side_level_z = hierarchy.level(i).dz;

            glUniform1f(shadermanager.smoother_cell_side_x_loc, cell_side_level_x);
            glUniform1f(shadermanager.smoother_cell_side_y_loc, cell_side_level_y);
            glUniform1f(shadermanager.smoother_cell_side_z_loc, cell_side_level_z);
            glUniform1i(shadermanager.smoother_cells_x_loc, currentCellsX);
            glUniform1i(shadermanager.smoother_cells_y_loc, currentCellsY);
            glUniform1i(shadermanager.smoother_cells_z_loc, currentCellsZ);
            glUniform1f(shadermanager.smoother_stable_dt_loc, 1.0);
            glUniform1f(shadermanager.smoother_density_loc, 1.0);

            for (int iter = 0; iter < 3; iter++)
            {
                /* RED CELLS */
                glUniform1i(shadermanager.smoother_color_loc, 0);

                glDispatchCompute(
                    (currentCellsX + 7) / 8, // this is due to the fact that we have already done first iteration
                    (currentCellsY + 7) / 8, // this is due to the fact that we have already done first iteration
                    (currentCellsZ + 7) / 8);

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                /* BLACK CELLS */
                glUniform1i(shadermanager.smoother_color_loc, 1);

                glDispatchCompute(
                    (currentCellsX + 7) / 8, // this is due to the fact that we have already done first iteration
                    (currentCellsY + 7) / 8, // this is due to the fact that we have already done first iteration
                    (currentCellsZ + 7) / 8);

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            } // -- pressure iter end

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            //--------------------------------------------------------- residual calculation
            glUseProgram(shadermanager.multigrid_pressure_residual_calculation_program);

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                0,
                levels_correction_SSBOs_arr[i]);

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                1,
                levels_RHS_SSBOs_arr[i] // acts as the RHS term
            );

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                2,
                levels_residual_SSBOs_arr[i] // computed residual, restricted down to RHS[i+1] next
            );

            glUniform1i(shadermanager.residual_current_cells_x_loc, currentCellsX);
            glUniform1i(shadermanager.residual_current_cells_y_loc, currentCellsY);
            glUniform1i(shadermanager.residual_current_cells_z_loc, currentCellsZ);
            glUniform1f(shadermanager.residual_dx_loc, cell_side_level_x);
            glUniform1f(shadermanager.residual_dy_loc, cell_side_level_y);
            glUniform1f(shadermanager.residual_dz_loc, cell_side_level_z);
            glUniform1f(shadermanager.residual_density_loc, 1.0);
            glUniform1f(shadermanager.residual_stable_dt_loc, 1.0);

            glDispatchCompute(
            (currentCellsX + 7) / 8,
            (currentCellsY + 7) / 8,
            (currentCellsZ + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            //----------------------------------------------------------------- step down to coarser grid
            restrict_residual(i);

        } // down stepping phase end

        // manual smoothing of the last level correction due to the structure of code
        glUseProgram(shadermanager.multigrid_pressure_smoother_program);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            0,
            levels_correction_SSBOs_arr[passes] // as indexing starts from zero
        );

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            1,
            levels_RHS_SSBOs_arr[passes] // as indexing starts from zero
        );

        glUniform1f(shadermanager.smoother_stable_dt_loc, stable_dt);

        int currentCellsY = hierarchy.level(passes).cells_y;                   // as indexing starts from zero
        int currentCellsX = hierarchy.level(passes).cells_x;                   // as indexing starts from zero
        int currentCellsZ = hierarchy.level(passes).cells_z;                   // as indexing starts from zero
        float cell_side_level_x = hierarchy.level(passes).dx;                  // domain / cells of that level, NOT fine spacing * 2^passes
        float cell_side_level_y = hierarchy.level(passes).dy;
        float cell_side_level_z = hierarchy.level(passes).dz;

        glUniform1f(shadermanager.smoother_cell_side_x_loc, cell_side_level_x);
        glUniform1f(shadermanager.smoother_cell_side_y_loc, cell_side_level_y);
        glUniform1f(shadermanager.smoother_cell_side_z_loc, cell_side_level_z);
        glUniform1i(shadermanager.smoother_cells_x_loc, currentCellsX);
        glUniform1i(shadermanager.smoother_cells_y_loc, currentCellsY);
        glUniform1i(shadermanager.smoother_cells_z_loc, currentCellsZ);
        glUniform1f(shadermanager.smoother_stable_dt_loc, 1.0);
        glUniform1f(shadermanager.smoother_density_loc, 1.0);

        // The coarsest level is "solved" by plain red-black Gauss-Seidel. 30 sweeps were tuned for an 8^3 coarsest grid; with
        // floor coarsening the coarsest grid can have up to 2 * min_coarse_cells - 1 cells per axis (125 -> 62 -> 31 -> 15), and
        // Gauss-Seidel needs ~N^2 sweeps on this pure-Neumann Poisson problem, so the count scales with the largest coarsest axis
        // (exactly 30 for an 8-cell coarsest grid).
        const int coarsest_axis = std::max({currentCellsX, currentCellsY, currentCellsZ});
        const int coarsest_sweeps = 30 * std::max(1, (coarsest_axis * coarsest_axis + 63) / 64);

        for (int iter = 0; iter < coarsest_sweeps; iter++) //----------------critical phase: the coarsest correction must be (nearly) solved
        {
            /* RED CELLS */
            glUniform1i(shadermanager.smoother_color_loc, 0);

            glDispatchCompute(
            (currentCellsX + 7) / 8,
            (currentCellsY + 7) / 8,
            (currentCellsZ + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            /* BLACK CELLS */
            glUniform1i(shadermanager.smoother_color_loc, 1);

            glDispatchCompute(
            (currentCellsX + 7) / 8,
            (currentCellsY + 7) / 8,
            (currentCellsZ + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // step up phase

        for (int i = 0; i < passes - 1; i++)
        {
            int coarse_level = passes - i;
            int fine_level = passes - i - 1;

            prolongate_correction(coarse_level, levels_correction_SSBOs_arr[fine_level]);

            int FineCells_x = hierarchy.level(fine_level).cells_x;
            int FineCells_y = hierarchy.level(fine_level).cells_y;
            int FineCells_z = hierarchy.level(fine_level).cells_z;

            //----------------------------------------------------------post prolongation smoothening

            glUseProgram(shadermanager.multigrid_pressure_smoother_program);

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                0,
                levels_correction_SSBOs_arr[fine_level] // as indexing starts from zero
            );

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                1,
                levels_RHS_SSBOs_arr[fine_level] // as indexing starts from zero
            );

            glUniform1f(shadermanager.smoother_stable_dt_loc, stable_dt);

            float cell_side_level_x = hierarchy.level(fine_level).dx;
            float cell_side_level_y = hierarchy.level(fine_level).dy;
            float cell_side_level_z = hierarchy.level(fine_level).dz;

            currentCellsX = FineCells_x; // fix: smoother dispatch below reads currentCells*
            currentCellsY = FineCells_y;
            currentCellsZ = FineCells_z;

            glUniform1f(shadermanager.smoother_cell_side_x_loc, cell_side_level_x);
            glUniform1f(shadermanager.smoother_cell_side_y_loc, cell_side_level_y);
            glUniform1f(shadermanager.smoother_cell_side_z_loc, cell_side_level_z);
            glUniform1i(shadermanager.smoother_cells_x_loc, currentCellsX);
            glUniform1i(shadermanager.smoother_cells_y_loc, currentCellsY);
            glUniform1i(shadermanager.smoother_cells_z_loc, currentCellsZ);
            glUniform1f(shadermanager.smoother_stable_dt_loc, 1.0);
            glUniform1f(shadermanager.smoother_density_loc, 1.0);

            for (int iter = 0; iter < 3; iter++)
            {
                /* RED CELLS */
                glUniform1i(shadermanager.smoother_color_loc, 0);

                glDispatchCompute(
                (currentCellsX + 7) / 8,
                (currentCellsY + 7) / 8,
                (currentCellsZ + 7) / 8);

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                /* BLACK CELLS */
                glUniform1i(shadermanager.smoother_color_loc, 1);

                glDispatchCompute(
                (currentCellsX + 7) / 8,
                (currentCellsY + 7) / 8,
                (currentCellsZ + 7) / 8);

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        //-------------------------------------------------------------------------------- final step up into pressure SSBO
        prolongate_correction(1, outer.pressure);

        //-------------------------------------------------------------------------final smoothening
        glUseProgram(shadermanager.multigrid_pressure_smoother_program);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            0,
            outer.pressure);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            1,
            outer.divergence);

        glUniform1f(shadermanager.smoother_stable_dt_loc, stable_dt);
        glUniform1i(shadermanager.smoother_cells_x_loc, outer.cells_x); // initial lvel
        glUniform1i(shadermanager.smoother_cells_y_loc, outer.cells_y);
        glUniform1i(shadermanager.smoother_cells_z_loc, outer.cells_z);
        glUniform1f(shadermanager.smoother_cell_side_x_loc, outer.cell_side_x); // must reset: up-loop leaves this at a coarse-level value
        glUniform1f(shadermanager.smoother_cell_side_y_loc, outer.cell_side_y); // must reset: up-loop leaves this at a coarse-level value
        glUniform1f(shadermanager.smoother_cell_side_z_loc, outer.cell_side_z); // must reset: up-loop leaves this at a coarse-level value
        glUniform1f(shadermanager.smoother_stable_dt_loc, stable_dt);
        glUniform1f(shadermanager.smoother_density_loc, density);

        for (int i = 0; i < 3; i++)
        {
            /* RED CELLS */
            glUniform1i(shadermanager.smoother_color_loc, 0);

            glDispatchCompute(
                (outer.cells_x + 7) / 8,
                (outer.cells_y + 7) / 8,
                (outer.cells_z + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            /* BLACK CELLS */
            glUniform1i(shadermanager.smoother_color_loc, 1);

            glDispatchCompute(
                (outer.cells_x + 7) / 8,
                (outer.cells_y + 7) / 8,
                (outer.cells_z + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void incompressible_flow::specialized_pressure_Multigrid::solve_pressure_system(float absolute_tolerance_of_error, float relative_tolerance_of_error, int stagnated_cycles_count_tolerance, float density, float stable_dt)
{
 // ------------------------------------------------------------
    // Basic quantities
    // ------------------------------------------------------------

    const int total_dofs = outer.cells_x * outer.cells_y * outer.cells_z;

    const float sqrt_total_dofs = std::sqrt(static_cast<float>(total_dofs));

    // Initial residual
    float RMS_previous = calculate_error_l2_norm_pressure(density, stable_dt) / sqrt_total_dofs;      // this is modified later

    float RMS_initial = RMS_previous;          // We need initial
    
    // std::cout << "RMS_initial = " << RMS_initial << std::endl;

    if (RMS_previous <= absolute_tolerance_of_error)
    {
        // std::cout << "PRESSURE SOLVER REACHED ABSOLUTE TOLERANCE OF ERROR" << std::endl;
        return;
    }

    constexpr int probe_cycles = 2;
    
    //initial 
    Multigrid_pressure_solve(stable_dt, density, probe_cycles);

    int total_cycles_performed = probe_cycles;

    float RMS_current = calculate_error_l2_norm_pressure(density, stable_dt) / sqrt_total_dofs;

    float relative_residual = RMS_current / RMS_initial;

    // std::cout << "RMS_current = " << RMS_current << std::endl;
    // std::cout << "Relative residual = " << relative_residual << std::endl;

    if (RMS_current <= absolute_tolerance_of_error)
    {
        // std::cout << "PRESSURE SOLVER REACHED ABSOLUTE TOLERANCE OF ERROR" << std::endl;
        return;
    }
    
    if (relative_residual < relative_tolerance_of_error)
    {
        // std::cout << "PRESSURE SOLVER REACHED RELATIVE TOLERANCE OF ERROR" << std::endl;
        return;   
    }


    // ------------------------------------------------------------
    // Adaptive V-cycle controller
    // ------------------------------------------------------------

    int MAX_V_CYCLES = 30;
    int cycles_since_measurement = total_cycles_performed;
    int stagnation_counter = 0;

    while (true)
    {
        const float residual_ratio = RMS_current / RMS_previous;

        if (residual_ratio > 1.0f ||   
            residual_ratio <= 0.0f ||
            !std::isfinite(residual_ratio))                                 // technically impossible to be negative, but handled
        {  
        
            // std::cout << "PRESSURE SOLVER DIVERGING FROM ANSWER OR STOPPED CHANGING" << std::endl;
            return;
        }


        // --------------------------------------------------------
        // Effective per-V-cycle convergence factor
        // --------------------------------------------------------

        const float convergence_factor = std::pow(residual_ratio, 1.0f / static_cast<float>(cycles_since_measurement));

        // std::cout << "convergence factor = " << convergence_factor << std::endl;

        // --------------------------------------------------------
        // Validate convergence factor.
        // --------------------------------------------------------

        if (!std::isfinite(convergence_factor) ||
            convergence_factor <= 0.0f ||
            convergence_factor > 1.0f)
        {
            // std::cout << "PRESSURE SOLVER DIVERGING FROM ANSWER OR STOPPED CHANGING" << std::endl;
            return;
        }

        int remaining_cycles_to_limit = MAX_V_CYCLES - total_cycles_performed;

        if (remaining_cycles_to_limit <= 0)
        {
            // std::cout << "SYSTEM DIDN'T LINEARLY CONVERGE AFTER " << MAX_V_CYCLES << std::endl;
            return;
        }

        // Because the adaptive controller can execute multiple V-cycles between measurements, stagnated_cycles_count_tolerance currently counts consecutive measurements with a convergence factor above 0.95, not consecutive stagnant V-cycles. Keep that distinction in mind when choosing the threshold.


        if(convergence_factor > 0.95 && convergence_factor <= 1.00)
        {
            stagnation_counter++;

            if(stagnation_counter > stagnated_cycles_count_tolerance)             // This is expected to be a reasonable number (for eg-> 5) 
            {
                // std::cout << "PRESSURE SOLVER HAS STAGNATED FOR MORE THAN THE GIVEN TOLERANCE" << std::endl;
                return;
            }

            Multigrid_pressure_solve(stable_dt, density, 1);          // just any number

            cycles_since_measurement = 1;
            total_cycles_performed++;

            RMS_previous = RMS_current;
            float l2 = calculate_error_l2_norm_pressure(density, stable_dt);
            RMS_current = l2 / sqrt_total_dofs;

            float relative_residual = RMS_current / RMS_initial;

            if (RMS_current <= absolute_tolerance_of_error)
            {
                // std::cout << "PRESSURE SOLVER REACHED ABSOLUTE TOLERANCE OF ERROR" << std::endl;
                return;
            }

            if (relative_residual < relative_tolerance_of_error)
            {
                // std::cout << "PRESSURE SOLVER REACHED RELATIVE TOLERANCE OF ERROR" << std::endl;
                return;
            }

            if (total_cycles_performed >= MAX_V_CYCLES)
            {
                // std::cout << "SYSTEM DIDN'T LINEARLY CONVERGE AFTER " << MAX_V_CYCLES << std::endl;
                return;
            }

            continue;

        }   
        else
        {
            stagnation_counter = 0;
        }

        float predicted_additional_cycles_abs = std::log(absolute_tolerance_of_error / RMS_current) / std::log(convergence_factor);

        float relative_target = relative_tolerance_of_error * RMS_initial;
        float predicted_additional_cycles_relative = std::log(relative_target / RMS_current) / std::log(convergence_factor);

        // std::cout << "predicted_cycles_relative = " << predicted_additional_cycles_relative << std::endl;
        // std::cout << "predicted_cycles_abs = " << predicted_additional_cycles_abs << std::endl;

        predicted_additional_cycles_abs = std::min(predicted_additional_cycles_abs, float(remaining_cycles_to_limit));
        predicted_additional_cycles_relative = std::min(predicted_additional_cycles_relative, float(remaining_cycles_to_limit));

        int additional_cycles_abs = static_cast<int>(std::ceil(predicted_additional_cycles_abs));
        int additional_cycles_relative = static_cast<int>(std::ceil(predicted_additional_cycles_relative));

        additional_cycles_abs = std::max(additional_cycles_abs, 1);
        additional_cycles_relative = std::max(additional_cycles_relative, 1);

        int additional_cycles = std::min(additional_cycles_abs, additional_cycles_relative);

        // --------------------------------------------------------
        // Perform predicted cycles.
        // --------------------------------------------------------

        additional_cycles = std::min(remaining_cycles_to_limit, additional_cycles);

        Multigrid_pressure_solve(stable_dt, density, additional_cycles);
        cycles_since_measurement = additional_cycles;
        total_cycles_performed += additional_cycles;

        // Measure actual results
        RMS_previous = RMS_current;
        float l2 = calculate_error_l2_norm_pressure(density, stable_dt);
        RMS_current = l2 / sqrt_total_dofs;

        // std::cout << "L2 norm = " << l2 << std::endl;

        // std::cout << "RMS_current = " << RMS_current << std::endl;

        float relative_residual = RMS_current / RMS_initial;

        if (RMS_current <= absolute_tolerance_of_error)
        {
            // std::cout << "PRESSURE SOLVER REACHED ABSOLUTE TOLERANCE OF ERROR" << std::endl;
            return;
        }


        if (relative_residual < relative_tolerance_of_error)
        {
            // std::cout << "PRESSURE SOLVER REACHED RELATIVE TOLERANCE OF ERROR" << std::endl;
            return;   
        }
            

        if (total_cycles_performed >= MAX_V_CYCLES)
        {
            // std::cout << "SYSTEM DIDN'T LINEARLY CONVERGE AFTER " << MAX_V_CYCLES << std::endl;
            return;
        }

    }

}

float incompressible_flow::specialized_pressure_Multigrid::calculate_error_l2_norm_pressure(float density, float stable_dt)
{
    // CALCULATING RESIDUAL
    glUseProgram(shadermanager.multigrid_pressure_residual_calculation_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, outer.pressure);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outer.divergence); // acts as the RHS term;

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, levels_residual_SSBOs_arr[0]); // computed residual, restricted down to RHS[i+1] next;

    glUniform1i(shadermanager.residual_current_cells_x_loc, outer.cells_x);
    glUniform1i(shadermanager.residual_current_cells_y_loc, outer.cells_y);
    glUniform1i(shadermanager.residual_current_cells_z_loc, outer.cells_z);
    glUniform1f(shadermanager.residual_dx_loc, outer.cell_side_x);
    glUniform1f(shadermanager.residual_dy_loc, outer.cell_side_y);
    glUniform1f(shadermanager.residual_dz_loc, outer.cell_side_z);
    glUniform1f(shadermanager.residual_density_loc, density);
    glUniform1f(shadermanager.residual_stable_dt_loc, stable_dt);

    glDispatchCompute(
    (outer.cells_x + 7) / 8,
    (outer.cells_y + 7) / 8,
    (outer.cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    // INITIAL REDUCTION
    glUseProgram(shadermanager.L2_norm_summation_of_sq);
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, levels_residual_SSBOs_arr[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pressure_residual_L2_reductions_SSBOs_arr[0]);

    glUniform1ui(shadermanager.l2_field_x, outer.cells_x);
    glUniform1ui(shadermanager.l2_field_y, outer.cells_y);
    glUniform1ui(shadermanager.l2_field_z, outer.cells_z);
    glUniform1i(shadermanager.l2_squared_input, GL_FALSE);
    
    glDispatchCompute(
        (outer.cells_x + 7) / 8,
        (outer.cells_y + 7) / 8,
        (outer.cells_z + 7) / 8
    );
    
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    int levels = pressure_residual_L2_reductions_SSBOs_arr.size();

    // Local size in the summation reduction kernel is 8 per axis, so after the first reduction, the initial dimension are 1/8 of the full field      
    int divisor = 8;

    for(int i=0; i < levels - 1; i++)         // due to the indices the levels-1
    {

        int current_x_elements = (outer.cells_x + (divisor - 1)) / divisor;
        int current_y_elements = (outer.cells_y + (divisor - 1)) / divisor;
        int current_z_elements = (outer.cells_z + (divisor - 1)) / divisor;

        glUseProgram(shadermanager.L2_norm_summation_of_sq);
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pressure_residual_L2_reductions_SSBOs_arr[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pressure_residual_L2_reductions_SSBOs_arr[i + 1]);

        glUniform1ui(shadermanager.l2_field_x, current_x_elements);
        glUniform1ui(shadermanager.l2_field_y, current_y_elements);
        glUniform1ui(shadermanager.l2_field_z, current_z_elements);
        glUniform1i(shadermanager.l2_squared_input, GL_TRUE);
        
        glDispatchCompute(
            (current_x_elements + 7) / 8,
            (current_y_elements + 7) / 8,
            (current_z_elements + 7) / 8
        );
        
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);   

        divisor *= 8;         // Local size per axis in the L2 norm kernel is 8

    }

    int L2_reductions = pressure_residual_L2_reductions_SSBOs_arr.size();

    std::vector <float> summation_of_sq(1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pressure_residual_L2_reductions_SSBOs_arr[L2_reductions - 1]);

    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 
        0, 
        sizeof(float), 
        summation_of_sq.data()
    );


    float L2_norm = sqrt(summation_of_sq[0]);

    return L2_norm;
}

void incompressible_flow::project_pressure(float stable_dt, float density,  GLuint vx_grid, GLuint vy_grid, GLuint vz_grid)
{
    glUseProgram(shadermanager.projectionXProgram);
    glUniform1i(shadermanager.projX_cells_x_loc, cells_x);
    glUniform1i(shadermanager.projX_cells_y_loc, cells_y);
    glUniform1i(shadermanager.projX_cells_z_loc, cells_z);
    glUniform1f(shadermanager.projX_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.projX_density_loc, density);

    glUseProgram(shadermanager.projectionYProgram);
    glUniform1i(shadermanager.projY_cells_x_loc, cells_x);
    glUniform1i(shadermanager.projY_cells_y_loc, cells_y);
    glUniform1i(shadermanager.projY_cells_z_loc, cells_z);
    glUniform1f(shadermanager.projY_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.projY_density_loc, density);

    glUseProgram(shadermanager.projectionZProgram);
    glUniform1i(shadermanager.projZ_cells_x_loc, cells_x);
    glUniform1i(shadermanager.projZ_cells_y_loc, cells_y);
    glUniform1i(shadermanager.projZ_cells_z_loc, cells_z);
    glUniform1f(shadermanager.projZ_cell_side_z_loc, cell_side_z);
    glUniform1f(shadermanager.projZ_density_loc, density);

    glUseProgram(shadermanager.projectionXProgram);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        pressure);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vx_grid);

    glUniform1f(shadermanager.projX_stable_dt_loc, stable_dt);

    // vx lattice: (cells_x + 1) columns, cells_y rows, cells_z layers   (local size 8,8,8)
    glDispatchCompute(
        ((cells_x + 1) + 7) / 8,
        (cells_y + 7) / 8,
        (cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(shadermanager.projectionYProgram);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        pressure);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vy_grid);

    glUniform1f(shadermanager.projY_stable_dt_loc, stable_dt);

    // vy lattice: cells_x columns, (cells_y + 1) rows, cells_z layers   (local size 8,8,8)
    glDispatchCompute(
        (cells_x + 7) / 8,
        ((cells_y + 1) + 7) / 8,
        (cells_z + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glUseProgram(shadermanager.projectionZProgram);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        pressure);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vz_grid);

    glUniform1f(shadermanager.projZ_stable_dt_loc, stable_dt);

    // vz lattice: cells_x columns, cells_y rows, (cells_z + 1) layers   (local size 8,8,8)
    glDispatchCompute(
        (cells_x + 7) / 8,
        (cells_y + 7) / 8,
        ((cells_z + 1) + 7) / 8);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void incompressible_flow::divergence_cal(GLuint vx_grid, GLuint vy_grid, GLuint vz_grid)
{
    glUseProgram(shadermanager.divergence_compute_program);
    glUniform1i(shadermanager.div_cells_x_loc, cells_x);
    glUniform1i(shadermanager.div_cells_y_loc, cells_y);
    glUniform1i(shadermanager.div_cells_z_loc, cells_z);
    glUniform1f(shadermanager.div_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.div_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.div_cell_side_z_loc, cell_side_z);


    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        divergence);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        vx_grid);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        2,
        vy_grid);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        3,
        vz_grid);

    glDispatchCompute(
        (cells_x + 7) / 8,
        (cells_y + 7) / 8,
        (cells_z + 7) / 8
        );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
