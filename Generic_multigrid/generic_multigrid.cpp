#include "generic_multigrid.hpp"
#include <cmath>
#include <iostream>

//===================================================================================================================================================
//                                                          GENERIC MULTI-GRID
//===================================================================================================================================================

Generic_multigrid::Generic_multigrid(ShaderManager &shadermanager_in, LinearSystem &linearsystem, common_stencil &Common_stencil) // constructor
: shadermanager(shadermanager_in),
  linearsystem(linearsystem),
  coefficients_stencil(Common_stencil)
{

    //------------------------------------------ Levels initialization for Multigrid ---------------------------------------

    const int passes = linearsystem.hierarchy.coarse_level_count();

    for (int i = 0; i < passes + 1; i++)
    {
        std::vector <int> level_dimensions = linearsystem.GetLevelDimension(i);

        GLuint level_correction_ssbo = 0;
        GLuint level_RHS_ssbo = 0;

        GLuint level_residual_ssbo;
        std::vector<float> level_residual(level_dimensions[0] * level_dimensions[1] * level_dimensions[2]);

        glGenBuffers(1, &level_residual_ssbo);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER,level_residual_ssbo);

        glBufferData(GL_SHADER_STORAGE_BUFFER,level_residual.size() * sizeof(float),level_residual.data(),GL_DYNAMIC_COPY);

        // MEMORY: level 0 never uses a correction or RHS buffer (the finest level works on linearsystem.field.data and
        // linearsystem.RHS_main; only its residual is needed). Index 0 stays in the arrays as an empty slot (GL name 0) so
        // every level index below is unchanged.
        if (i > 0)
        {
            std::vector<float> level_correction(level_dimensions[0] * level_dimensions[1] * level_dimensions[2]);
            std::vector<float> level_RHS(level_dimensions[0] * level_dimensions[1] * level_dimensions[2]);

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
    int current_x = linearsystem.x_elements;
    int current_y = linearsystem.y_elements;
    int current_z = linearsystem.z_elements;

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

        L2_reductions_SSBOs_arr.push_back(reduction_ssbo);

        current_x = groups_x;
        current_y = groups_y;
        current_z = groups_z;
    }


}                                   // constructor end

void Generic_multigrid::Generic_multigrid_solve(int V_cycles_count)
{
    // passes = number of restriction steps, from the shared hierarchy
    const int passes = linearsystem.hierarchy.coarse_level_count();

    const int topology = static_cast<int>(linearsystem.field.topology);

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

        glUseProgram(shadermanager.multigrid_generic_smoother_program);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.coefficients.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.coefficients.aW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.coefficients.aE);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.coefficients.aS);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.coefficients.aN);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.coefficients.aB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.coefficients.aT);
        

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, coefficients_stencil.RHS_main);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, linearsystem.field.data);

        glUniform1i(shadermanager.Generic_smoother_cells_x_loc, linearsystem.x_elements); // initial lvel
        glUniform1i(shadermanager.Generic_smoother_cells_y_loc, linearsystem.y_elements);
        glUniform1i(shadermanager.Generic_smoother_cells_z_loc, linearsystem.z_elements);

        for (int i = 0; i < 3; i++)
        {
            /* RED CELLS */
            glUniform1i(shadermanager.Generic_smoother_color_loc, 0);

            glDispatchCompute(
                (linearsystem.x_elements + 7) / 8,
                (linearsystem.y_elements + 7) / 8,
                (linearsystem.z_elements + 7) / 8
                );

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            /* BLACK CELLS */
            glUniform1i(shadermanager.Generic_smoother_color_loc, 1);

            glDispatchCompute(
                (linearsystem.x_elements + 7) / 8,
                (linearsystem.y_elements + 7) / 8,
                (linearsystem.z_elements + 7) / 8
                );

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        //------------------------------------calculate the residual for the first step down
        glUseProgram(shadermanager.multigrid_generic_residual_calculation_program);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.coefficients.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.coefficients.aW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.coefficients.aE);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.coefficients.aS);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.coefficients.aN);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.coefficients.aB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.coefficients.aT);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, coefficients_stencil.RHS_main);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, linearsystem.field.data);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, levels_residual_SSBOs_arr[0]);

        glUniform1i(shadermanager.generic_residual_current_cells_x_loc, linearsystem.x_elements);
        glUniform1i(shadermanager.generic_residual_current_cells_y_loc, linearsystem.y_elements);
        glUniform1i(shadermanager.generic_residual_current_cells_z_loc, linearsystem.z_elements);

        glDispatchCompute(
            (linearsystem.x_elements + 7) / 8,
            (linearsystem.y_elements + 7) / 8,
            (linearsystem.z_elements + 7) / 8
            );

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        //-------------------------------------------------intial step down

        glUseProgram(shadermanager.multigrid_residual_generic_restriction_program);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            0,
            levels_residual_SSBOs_arr[0] // step zero resolution
        );

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            1,
            levels_RHS_SSBOs_arr[1] // the stepped down resolution
        );

        {
            // Actual DOF dimensions of both levels, not physical cell counts.
            std::vector<int> fine_dimensions = linearsystem.GetLevelDimension(0);
            std::vector<int> coarse_dimensions = linearsystem.GetLevelDimension(1);

            glUniform3i(
                shadermanager.residual_generic_restriction_fine_dimensions_loc,
                fine_dimensions[0], fine_dimensions[1], fine_dimensions[2]);

            glUniform3i(
                shadermanager.residual_generic_restriction_coarse_dimensions_loc,
                coarse_dimensions[0], coarse_dimensions[1], coarse_dimensions[2]);

            glUniform1i(shadermanager.residual_generic_restriction_topology_loc, topology);

            // one invocation per COARSE DOF: the kernel bounds-checks coarseDimensions
            glDispatchCompute(
                (coarse_dimensions[0] + 7) / 8,
                (coarse_dimensions[1] + 7) / 8,
                (coarse_dimensions[2] + 7) / 8
                );
        }

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // passes and the vCycles loop now open at the top of this function

        //-----------------step down phase
        for (int i = 1; i < passes; i++) // starts from one as we already do one step down before
        {
            std::vector <int> level_dimensions = linearsystem.GetLevelDimension(i);

            //---------------------------------------------first few smoothing iterations of Gauss sidel
            glUseProgram(shadermanager.multigrid_generic_smoother_program);

            // multigrid_levels_coeffiecients_arr holds COARSE levels only:
            // element [k] is level k + 1, so level i lives at [i - 1].
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aP);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aE);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aS);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aN);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aB);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aT);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, levels_RHS_SSBOs_arr[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, levels_correction_SSBOs_arr[i]);

            glUniform1i(shadermanager.Generic_smoother_cells_x_loc, level_dimensions[0]);
            glUniform1i(shadermanager.Generic_smoother_cells_y_loc, level_dimensions[1]);
            glUniform1i(shadermanager.Generic_smoother_cells_z_loc, level_dimensions[2]);

            for (int iter = 0; iter < 3; iter++)
            {
                /* RED CELLS */
                glUniform1i(shadermanager.Generic_smoother_color_loc, 0);

                glDispatchCompute(
                    (level_dimensions[0] + 7) / 8, // this is due to the fact that we have already done first iteration
                    (level_dimensions[1] + 7) / 8, // this is due to the fact that we have already done first iteration
                    (level_dimensions[2] + 7) / 8);

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                /* BLACK CELLS */
                glUniform1i(shadermanager.Generic_smoother_color_loc, 1);

                glDispatchCompute(
                    (level_dimensions[0] + 7) / 8, // this is due to the fact that we have already done first iteration
                    (level_dimensions[1] + 7) / 8, // this is due to the fact that we have already done first iteration
                    (level_dimensions[2] + 7) / 8);

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            } // -- pressure iter end

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            //--------------------------------------------------------- residual calculation
            glUseProgram(shadermanager.multigrid_generic_residual_calculation_program);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aP);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aE);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aS);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aN);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aB);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.multigrid_levels_coeffiecients_arr[i - 1].aT);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, levels_RHS_SSBOs_arr[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, levels_correction_SSBOs_arr[i]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, levels_residual_SSBOs_arr[i]);


            glUniform1i(shadermanager.generic_residual_current_cells_x_loc, level_dimensions[0]);
            glUniform1i(shadermanager.generic_residual_current_cells_y_loc, level_dimensions[1]);
            glUniform1i(shadermanager.generic_residual_current_cells_z_loc, level_dimensions[2]);

            glDispatchCompute(
            (level_dimensions[0] + 7) / 8,
            (level_dimensions[1] + 7) / 8,
            (level_dimensions[2] + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            //----------------------------------------------------------------- step down to coarser grid
            glUseProgram(shadermanager.multigrid_residual_generic_restriction_program);

            glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            0,
            levels_residual_SSBOs_arr[i]);

            glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            1,
            levels_RHS_SSBOs_arr[i + 1]);

            {
                // level_dimensions is level i (the fine side of this transfer)
                std::vector<int> coarse_dimensions = linearsystem.GetLevelDimension(i + 1);
                
                glUniform3i(shadermanager.residual_generic_restriction_fine_dimensions_loc, level_dimensions[0], level_dimensions[1], level_dimensions[2]);
                glUniform3i(shadermanager.residual_generic_restriction_coarse_dimensions_loc, coarse_dimensions[0], coarse_dimensions[1], coarse_dimensions[2]);
                glUniform1i(shadermanager.residual_generic_restriction_topology_loc, topology);

                // one invocation per COARSE DOF
                glDispatchCompute(
                    (coarse_dimensions[0] + 7) / 8,
                    (coarse_dimensions[1] + 7) / 8,
                    (coarse_dimensions[2] + 7) / 8);
            }

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        } // down stepping phase end

        // manual smoothing of the last level correction due to the structure of code
        glUseProgram(shadermanager.multigrid_generic_smoother_program);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.multigrid_levels_coeffiecients_arr[passes - 1].aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.multigrid_levels_coeffiecients_arr[passes - 1].aW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.multigrid_levels_coeffiecients_arr[passes - 1].aE);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.multigrid_levels_coeffiecients_arr[passes - 1].aS);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.multigrid_levels_coeffiecients_arr[passes - 1].aN);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.multigrid_levels_coeffiecients_arr[passes - 1].aB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.multigrid_levels_coeffiecients_arr[passes - 1].aT);

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            7,
            levels_RHS_SSBOs_arr[passes] // as indexing starts from zero
        );

        glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            8,
            levels_correction_SSBOs_arr[passes] // as indexing starts from zero
        );

        // actual DOF dimensions of the coarsest level, not cells >> passes
        std::vector<int> coarsest_dimensions = linearsystem.GetLevelDimension(passes);

        int currentCellsX = coarsest_dimensions[0];
        int currentCellsY = coarsest_dimensions[1];
        int currentCellsZ = coarsest_dimensions[2];

        glUniform1i(shadermanager.Generic_smoother_cells_x_loc, currentCellsX);
        glUniform1i(shadermanager.Generic_smoother_cells_y_loc, currentCellsY);
        glUniform1i(shadermanager.Generic_smoother_cells_z_loc, currentCellsZ);

        for (int iter = 0; iter < 30; iter++) //----------------30 because this is a critical phases
        {
            /* RED CELLS */
            glUniform1i(shadermanager.Generic_smoother_color_loc, 0);

            glDispatchCompute(
            (currentCellsX + 7) / 8,
            (currentCellsY + 7) / 8,
            (currentCellsZ + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            /* BLACK CELLS */
            glUniform1i(shadermanager.Generic_smoother_color_loc, 1);

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
            glUseProgram(shadermanager.multigrid_generic_prolongation_program);

            int coarse_level = passes - i;
            int fine_level = passes - i - 1;

            glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            0,
            levels_correction_SSBOs_arr[coarse_level]);

            glBindBufferBase(
            GL_SHADER_STORAGE_BUFFER,
            1,
            levels_correction_SSBOs_arr[fine_level]);

            std::vector<int> fine_dimensions = linearsystem.GetLevelDimension(fine_level);
            std::vector<int> coarse_dimensions = linearsystem.GetLevelDimension(coarse_level);

            glUniform3i(
                shadermanager.generic_prolongation_fine_dimensions_loc,
                fine_dimensions[0], fine_dimensions[1], fine_dimensions[2]);

            glUniform3i(
                shadermanager.generic_prolongation_coarse_dimensions_loc,
                coarse_dimensions[0], coarse_dimensions[1], coarse_dimensions[2]);

            glUniform1i(shadermanager.generic_prolongation_topology_loc, topology);

            // one invocation per FINE DOF: the kernel bounds-checks fineDimensions
            glDispatchCompute(
                (fine_dimensions[0] + 7) / 8,
                (fine_dimensions[1] + 7) / 8,
                (fine_dimensions[2] + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            //----------------------------------------------------------post prolongation smoothening

            glUseProgram(shadermanager.multigrid_generic_smoother_program);

            // bindings 0-6 are the stencil coefficients of this level; level
            // fine_level lives at multigrid_levels_coeffiecients_arr[fine_level - 1]
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.multigrid_levels_coeffiecients_arr[fine_level - 1].aP);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.multigrid_levels_coeffiecients_arr[fine_level - 1].aW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.multigrid_levels_coeffiecients_arr[fine_level - 1].aE);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.multigrid_levels_coeffiecients_arr[fine_level - 1].aS);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.multigrid_levels_coeffiecients_arr[fine_level - 1].aN);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.multigrid_levels_coeffiecients_arr[fine_level - 1].aB);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.multigrid_levels_coeffiecients_arr[fine_level - 1].aT);

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                7,
                levels_RHS_SSBOs_arr[fine_level] // as indexing starts from zero
            );

            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                8,
                levels_correction_SSBOs_arr[fine_level] // as indexing starts from zero
            );

            currentCellsX = fine_dimensions[0];
            currentCellsY = fine_dimensions[1];
            currentCellsZ = fine_dimensions[2];

            glUniform1i(shadermanager.Generic_smoother_cells_x_loc, currentCellsX);
            glUniform1i(shadermanager.Generic_smoother_cells_y_loc, currentCellsY);
            glUniform1i(shadermanager.Generic_smoother_cells_z_loc, currentCellsZ);

            for (int iter = 0; iter < 3; iter++)
            {
                /* RED CELLS */
                glUniform1i(shadermanager.Generic_smoother_color_loc, 0);

                glDispatchCompute(
                (currentCellsX + 7) / 8,
                (currentCellsY + 7) / 8,
                (currentCellsZ + 7) / 8);

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                /* BLACK CELLS */
                glUniform1i(shadermanager.Generic_smoother_color_loc, 1);

                glDispatchCompute(
                (currentCellsX + 7) / 8,
                (currentCellsY + 7) / 8,
                (currentCellsZ + 7) / 8);

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            }

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }

        //-------------------------------------------------------------------------------- final step up
        glUseProgram(shadermanager.multigrid_generic_prolongation_program);

        glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        levels_correction_SSBOs_arr[1]);

        glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        linearsystem.field.data);

        {
            // finest level: actual DOF dimensions, which are not cells_* for a staggered field
            std::vector<int> fine_dimensions = linearsystem.GetLevelDimension(0);
            std::vector<int> coarse_dimensions = linearsystem.GetLevelDimension(1);

            glUniform3i(
                shadermanager.generic_prolongation_fine_dimensions_loc,
                fine_dimensions[0], fine_dimensions[1], fine_dimensions[2]);

            glUniform3i(
                shadermanager.generic_prolongation_coarse_dimensions_loc,
                coarse_dimensions[0], coarse_dimensions[1], coarse_dimensions[2]);

            glUniform1i(shadermanager.generic_prolongation_topology_loc, topology);

            // one invocation per FINE DOF
            glDispatchCompute(
                (fine_dimensions[0] + 7) / 8,
                (fine_dimensions[1] + 7) / 8,
                (fine_dimensions[2] + 7) / 8);
        }

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        //-------------------------------------------------------------------------final smoothening
        glUseProgram(shadermanager.multigrid_generic_smoother_program);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.coefficients.aP);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.coefficients.aW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.coefficients.aE);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.coefficients.aS);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.coefficients.aN);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.coefficients.aB);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.coefficients.aT);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, coefficients_stencil.RHS_main);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, linearsystem.field.data);

        // back on the finest level: reset the dimensions, the up-loop left them coarse
        glUniform1i(shadermanager.Generic_smoother_cells_x_loc, linearsystem.x_elements);
        glUniform1i(shadermanager.Generic_smoother_cells_y_loc, linearsystem.y_elements);
        glUniform1i(shadermanager.Generic_smoother_cells_z_loc, linearsystem.z_elements);

        for (int i = 0; i < 3; i++)
        {
            /* RED CELLS */
            glUniform1i(shadermanager.Generic_smoother_color_loc, 0);

            glDispatchCompute(
                (linearsystem.x_elements + 7) / 8,
                (linearsystem.y_elements + 7) / 8,
                (linearsystem.z_elements + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            /* BLACK CELLS */
            glUniform1i(shadermanager.Generic_smoother_color_loc, 1);

            glDispatchCompute(
                (linearsystem.x_elements + 7) / 8,
                (linearsystem.y_elements + 7) / 8,
                (linearsystem.z_elements + 7) / 8);

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
};

int Generic_multigrid::multigrid_solve_system(float absolute_tolerance_of_error, float relative_tolerance_of_error, int stagnated_cycles_count_tolerance)
{
    // ------------------------------------------------------------
    // Basic quantities
    // ------------------------------------------------------------

    const int total_dofs = linearsystem.x_elements * linearsystem.y_elements * linearsystem.z_elements;

    const float sqrt_total_dofs = std::sqrt(static_cast<float>(total_dofs));

    // Initial residual
    float RMS_previous = calculate_l2_norm() / sqrt_total_dofs;      // this is modified later

    float RMS_initial = RMS_previous;          // We need initial
    
    // std::cout << "RMS_initial = " << RMS_initial << std::endl;

    if (RMS_previous <= absolute_tolerance_of_error)
        return int(linear_solver_status::CONVERGED_ABSOLUTE_TOLERANCE);

    constexpr int probe_cycles = 2;


    //initial 
    Generic_multigrid_solve(probe_cycles);

    int total_cycles_performed = probe_cycles;

    float RMS_current = calculate_l2_norm() / sqrt_total_dofs;

    float relative_residual = RMS_current / RMS_initial;

    // std::cout << "RMS_current = " << RMS_current << std::endl;
    // std::cout << "Relative residual = " << relative_residual << std::endl;

    if (RMS_current <= absolute_tolerance_of_error)
        return int(linear_solver_status::CONVERGED_ABSOLUTE_TOLERANCE);
    
    if (relative_residual < relative_tolerance_of_error)
        return int(linear_solver_status::CONVERGED_RELATIVE_TOLERANCE);   


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
            return int(linear_solver_status::DIVERGING_FROM_ANSWER);
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
            return int(linear_solver_status::DIVERGING_FROM_ANSWER);
        }

        int remaining_cycles_to_limit = MAX_V_CYCLES - total_cycles_performed;

        if (remaining_cycles_to_limit <= 0)
        {
            std::cout << "SYSTEM DIDN'T LINEARLY CONVERGE AFTER " << MAX_V_CYCLES << std::endl;
            return int(linear_solver_status::MAX_CYCLES_REACHED);
        }

        // Because the adaptive controller can execute multiple V-cycles between measurements, stagnated_cycles_count_tolerance currently counts consecutive measurements with a convergence factor above 0.95, not consecutive stagnant V-cycles. Keep that distinction in mind when choosing the threshold.


        if(convergence_factor > 0.95 && convergence_factor <= 1.00)
        {
            stagnation_counter++;

            if(stagnation_counter > stagnated_cycles_count_tolerance)             // This is expected to be a reasonable number (for eg-> 5) 
            {
                return int(linear_solver_status::STAGNATED);
            }

            Generic_multigrid_solve(1);          // just any number

            cycles_since_measurement = 1;
            total_cycles_performed++;

            RMS_previous = RMS_current;
            float l2 = calculate_l2_norm();
            RMS_current = l2 / sqrt_total_dofs;

            float relative_residual = RMS_current / RMS_initial;

            if (RMS_current <= absolute_tolerance_of_error)
                return int(linear_solver_status::CONVERGED_ABSOLUTE_TOLERANCE);

            if (relative_residual < relative_tolerance_of_error)
                return int(linear_solver_status::CONVERGED_RELATIVE_TOLERANCE);

            if (total_cycles_performed >= MAX_V_CYCLES)
            {
                // std::cout << "SYSTEM DIDN'T LINEARLY CONVERGE AFTER " << MAX_V_CYCLES << std::endl;
                return int(linear_solver_status::MAX_CYCLES_REACHED);
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

        Generic_multigrid_solve(additional_cycles);
        cycles_since_measurement = additional_cycles;
        total_cycles_performed += additional_cycles;

        // Measure actual results
        RMS_previous = RMS_current;
        float l2 = calculate_l2_norm();
        RMS_current = l2 / sqrt_total_dofs;

        // std::cout << "L2 norm = " << l2 << std::endl;

        // std::cout << "RMS_current = " << RMS_current << std::endl;

        float relative_residual = RMS_current / RMS_initial;

        if (RMS_current <= absolute_tolerance_of_error)
            return int(linear_solver_status::CONVERGED_ABSOLUTE_TOLERANCE);


        if (relative_residual < relative_tolerance_of_error)
            return int(linear_solver_status::CONVERGED_RELATIVE_TOLERANCE);   
            

        if (total_cycles_performed >= MAX_V_CYCLES)
        {
            std::cout << "SYSTEM DIDN'T LINEARLY CONVERGE AFTER " << MAX_V_CYCLES << std::endl;
            return int(linear_solver_status::MAX_CYCLES_REACHED);
        }

    }

}


float Generic_multigrid::calculate_l2_norm() //---------- L2 norm code
{
    // CALCULATING RESIDUAL
    glUseProgram(shadermanager.multigrid_generic_residual_calculation_program);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, coefficients_stencil.coefficients.aP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, coefficients_stencil.coefficients.aW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coefficients_stencil.coefficients.aE);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coefficients_stencil.coefficients.aS);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, coefficients_stencil.coefficients.aN);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, coefficients_stencil.coefficients.aB);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coefficients_stencil.coefficients.aT);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, coefficients_stencil.RHS_main);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, linearsystem.field.data);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, levels_residual_SSBOs_arr[0]);

    glUniform1i(shadermanager.generic_residual_current_cells_x_loc, linearsystem.x_elements);   //intentional
    glUniform1i(shadermanager.generic_residual_current_cells_y_loc, linearsystem.y_elements);   //intentional
    glUniform1i(shadermanager.generic_residual_current_cells_z_loc, linearsystem.z_elements);   //intentional

    glDispatchCompute(
        (linearsystem.x_elements + 7) / 8,
        (linearsystem.y_elements + 7) / 8,
        (linearsystem.z_elements + 7) / 8
        );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    
    // INITIAL REDUCTION
    glUseProgram(shadermanager.L2_norm_summation_of_sq);
    
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, levels_residual_SSBOs_arr[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, L2_reductions_SSBOs_arr[0]);

    glUniform1ui(shadermanager.l2_field_x, linearsystem.x_elements);
    glUniform1ui(shadermanager.l2_field_y, linearsystem.y_elements);
    glUniform1ui(shadermanager.l2_field_z, linearsystem.z_elements);
    glUniform1i(shadermanager.l2_squared_input, GL_FALSE);
    
    glDispatchCompute(
        (linearsystem.x_elements + 7) / 8,
        (linearsystem.y_elements + 7) / 8,
        (linearsystem.z_elements + 7) / 8
    );
    
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    int levels = L2_reductions_SSBOs_arr.size();

    // Local size in the summation reduction kernel is 8 per axis, so after the first reduction, the initial dimension are 1/8 of the full field      
    int divisor = 8;

    for(int i=0; i < levels - 1; i++)         // due to the indices the levels-1
    {

        int current_x_elements = (linearsystem.x_elements + (divisor - 1)) / divisor;
        int current_y_elements = (linearsystem.y_elements + (divisor - 1)) / divisor;
        int current_z_elements = (linearsystem.z_elements + (divisor - 1)) / divisor;

        glUseProgram(shadermanager.L2_norm_summation_of_sq);
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, L2_reductions_SSBOs_arr[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, L2_reductions_SSBOs_arr[i + 1]);

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

    int L2_reductions = L2_reductions_SSBOs_arr.size();

    std::vector <float> summation_of_sq(1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, L2_reductions_SSBOs_arr[L2_reductions - 1]);

    glGetBufferSubData(
        GL_SHADER_STORAGE_BUFFER, 
        0, 
        sizeof(float), 
        summation_of_sq.data()
    );


    float L2_norm = sqrt(summation_of_sq[0]);

    return L2_norm;
}
