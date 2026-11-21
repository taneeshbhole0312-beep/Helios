#pragma once

#include "../Incompressible_flow/incompressible_flow.hpp"
#include "../shader_manager/shader_manager.hpp"
#include "../explicit_sim/explicit_sim.hpp"
#include <vector>

class debuggers
{
    private:
        incompressible_flow &incompressible;
        ShaderManager &shadermanager;
        int cells_x;
        int cells_y;
        int cells_z;
        float cell_side_x;
        float cell_side_y;
        float cell_side_z;

        bool curl_allocated = false;

        void allocate_curl_arr();

        // reduction chain of the energy sums: [0] = one partial sum per 8 x 8 x 8 work group, every next level 8 x 8 x 8 times fewer, last = 1 float
        std::vector <GLuint> energy_reduction_SSBOs;
        std::vector <int> energy_reduction_dims;                    // (x, y, z) work group counts of every level, 3 ints per level
        void allocate_energy_reduction_chain();
        double sum_over_cells(int mode, GLuint vx_grid, GLuint vy_grid, GLuint vz_grid, GLuint k_field);       // kinematic sum, see energy_reduction.comp

    public:
        GLuint curl_ssbo;
        
        debuggers(incompressible_flow &incompressible_in, ShaderManager &shadermanager_in, int cells_x_in, int cells_y_in, int cells_z_in, float cell_side_x_in, float cell_side_y_in, float cell_side_z_in)
        :   incompressible(incompressible_in),
            shadermanager(shadermanager_in)
        {
            cells_x = cells_x_in;
            cells_y = cells_y_in;
            cells_z = cells_z_in;
            cell_side_x = cell_side_x_in;
            cell_side_y = cell_side_y_in;
            cell_side_z = cell_side_z_in;
        }


        float get_max_absolute_divergence(GLuint vx_grid, GLuint vy_grid, GLuint vz_grid);

        float get_mean_absolute_divergence(GLuint vx_grid, GLuint vy_grid, GLuint vz_grid);

        // Total kinetic energy [J] of the MEAN FLOW:  sum over the cells of  1/2 rho (u_c^2 + v_c^2 + w_c^2) V_cell,
        // with the staggered face velocities interpolated to the cell centroid (u_c = (u_w + u_e) / 2, ...).
        double get_mean_flow_kinetic_energy(GLuint vx_grid, GLuint vy_grid, GLuint vz_grid, float density);

        // Total turbulent kinetic energy [J]:  sum over the cells of  rho k V_cell  (k is cell centered; every cell is counted,
        // including the wall-treatment layer, with its k as stored).
        double get_turbulent_kinetic_energy(GLuint k_field, float density);

        std::vector <int> check_non_physical_turbulence_quantities(GLuint k_field, GLuint destruction_variable_field);

};