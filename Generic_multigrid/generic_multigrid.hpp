#pragma once

#include <glad/glad.h>
#include <vector>

#include "../shader_manager/shader_manager.hpp"
#include "../Linear_systems/LinearSystem.hpp"

enum class linear_solver_status
{
    CONVERGED_ABSOLUTE_TOLERANCE,
    CONVERGED_RELATIVE_TOLERANCE,
    MAX_CYCLES_REACHED,
    STAGNATED,
    DIVERGING_FROM_ANSWER
};


class Generic_multigrid
{
    private:
    std::vector<GLuint> levels_RHS_SSBOs_arr;
    std::vector<GLuint> levels_correction_SSBOs_arr;
    std::vector<GLuint> levels_residual_SSBOs_arr;
    std::vector<GLuint> L2_reductions_SSBOs_arr;

    ShaderManager& shadermanager;

    LinearSystem &linearsystem;
    common_stencil &coefficients_stencil;

    //Helper functions
    
    void Generic_multigrid_solve(int V_cycles_count);
    
    public:
    Generic_multigrid(
        ShaderManager& shadermanager_in,
        LinearSystem &linearsystem, 
        common_stencil &Common_stencil
    );
    
    float calculate_l2_norm();      // PUT THIS BACK INTO THE PRIVATE FUNCS AFTER TESTS

    //Final caller
    int multigrid_solve_system(float absolute_tolerance_of_error, float relative_tolerance_of_error, int stagnated_cycles_count_tolerance);
};