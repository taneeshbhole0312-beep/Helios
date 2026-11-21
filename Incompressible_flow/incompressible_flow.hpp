#pragma once

#include <glad/glad.h>
#include <vector>
#include "../shader_manager/shader_manager.hpp"
#include "../Linear_systems/multigrid_hierarchy.hpp"
#include <Cmath>
#include <iostream>

class incompressible_flow
{
    private:
    
        int cells_x;
        int cells_y;
        int cells_z;
        
        float cell_side_x;
        float cell_side_y;
        float cell_side_z;
        
        ShaderManager &shadermanager;
        
        GLuint create_SSBO(const std::vector<float>& data);
        
        public:
        GLuint pressure;
        GLuint divergence;
        incompressible_flow(int cells_x_in, int cells_y_in, int cells_z_in, float cell_side_x_in, float cell_side_y_in, float cell_side_z_in, ShaderManager &shadermanager);
        
        class specialized_pressure_Multigrid
        {
            private:
            
            std::vector<GLuint> levels_RHS_SSBOs_arr;
            std::vector<GLuint> levels_correction_SSBOs_arr;
            std::vector<GLuint> levels_residual_SSBOs_arr;
            std::vector<GLuint> pressure_residual_L2_reductions_SSBOs_arr;

            const MultigridHierarchy &hierarchy;     // shared level geometry; level 0 must be the grid of `outer`

            ShaderManager& shadermanager;
            incompressible_flow &outer;

            float calculate_error_l2_norm_pressure(float density, float stable_dt);

            // level transfers (generic kernels, CELL_CENTERED), dimensions from the hierarchy
            void restrict_residual(int fine_level);
            void prolongate_correction(int coarse_level, GLuint fine_target);
            
            public:
            
            specialized_pressure_Multigrid(const MultigridHierarchy &hierarchy_in, incompressible_flow &outer);
            
            void Multigrid_pressure_solve(float stable_dt, float density, int V_cycles_count);
            void solve_pressure_system(float absolute_tolerance_of_error, float relative_tolerance_of_error, int stagnated_cycles_count_tolerance, float density, float stable_dt);
        };
        
        
        void project_pressure(float stable_dt, float density, GLuint vx_grid, GLuint vy_grid, GLuint vz_grid); 
        void divergence_cal(GLuint vx_grid, GLuint vy_grid, GLuint vz_grid);

};
