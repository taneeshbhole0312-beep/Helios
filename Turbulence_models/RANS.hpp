#pragma once

#include "../Linear_systems/LinearSystem.hpp"
#include "../Orchestrator/orchestrator.hpp"
#include "../Generic_multigrid/generic_multigrid.hpp"
#include "k_TurbulenceModels.hpp"
#include <iostream>

// class common_stencil is declared in Linear_systems/LinearSystem.hpp (Generic_multigrid needs it, and RANS.hpp includes Generic_multigrid); its members are defined in RANS.cpp

class RANS
{
    private:

        // DELETE LATER
        float stream_func_dy(
        float x,
        float y,
        float u_Not,
        float shear_distance,
        float epsilon,
        float num_of_waves,
        float HEIGHT,
        float WIDTH);
        float stream_func_dx(
        float x,
        float y,
        float u_Not,
        float shear_distance,
        float epsilon,
        float num_of_waves,
        float HEIGHT,
        float WIDTH);

        friend class k_TurbulenceModels;        // the k-epsilon initialization (epsilon branch) calls initialize_velocity_fields()

        GLuint create_SSBO(const std::vector<float> &data);

        //---------------------------------------- NONLINEAR CONVERGENCE (iterates of the SAME physical time step)
        // previous nonlinear iterate of each unknown: filled by store_nonlinear_iterate()
        GLuint previous_iterate_u;
        GLuint previous_iterate_v;
        GLuint previous_iterate_w;
        GLuint previous_iterate_k;
        GLuint previous_iterate_eps;
        
        // one reduction chain sized for (cells_x + 1) x (cells_y + 1) x (cells_z + 1): every level is at least as large as
        // the same level of any topology (cell, x/y/z face), so all five fields share it
        std::vector <GLuint> nonlinear_reduction_chain;
        
        // 10 final scalars (5 numerators, 5 denominators) gathered on the GPU, read back with ONE transfer
        GLuint nonlinear_reduction_results;
        
        // sum over the field of (new - old)^2 (old_field != 0) or new^2 (old_field == 0), final scalar -> results[slot]
        void reduce_sum_of_squares(GLuint new_field, GLuint old_field, int field_x, int field_y, int field_z, int result_slot);

        // HELPERS
        void generic_restriction_staggered(LinearSystem &linear_system, int level, std::vector <GLuint> vel_fields);
        
        // TRUE FUNCTIONS
        void add_viscosity_contribution_momentum(LinearSystem &linear_system_momentum);
        
        // OVERLOADED FUNCTIONS FOR MAKING THE COEFFICIENTS AT COARSER LEVELS
        void add_viscosity_contribution_momentum( int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, fieldTopology topology, GLuint &mu_eff_at_coarse_level, Levels_info &level_info);

        // MOMENTUM WALL FUNCTION (Spalding, implicit: aP += rho u_tau^2 A / |U_t|), fine and coarse
        void add_wall_function_contribution_momentum(LinearSystem &linear_system_momentum, float spalding_wall_smoothness);
        void add_wall_function_contribution_momentum(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, fieldTopology topology, GLuint &current_level_u_field, GLuint &current_level_v_field, GLuint &current_level_w_field, float spalding_wall_smoothness, Levels_info &level_info);
        
        configuration &config;
        
        public:
        const MultigridHierarchy &hierarchy;      // single source of truth for the multigrid level geometry (owned by the caller, outlives RANS)
        ShaderManager &shadermanager;
        int cells_x;
        int cells_y;
        int cells_z;
        float domain_size_x;
        float domain_size_y;
        float domain_size_z;
        float density;
        float dynamic_viscosity;
        std::vector <GLuint> u_vel_hierarcy;
        std::vector <GLuint> v_vel_hierarcy;
        std::vector <GLuint> w_vel_hierarcy;
        std::vector <GLuint> mu_eff_hierarchy;
        
        LinearSystem u_system;        
        LinearSystem v_system;        
        LinearSystem w_system;   
        GLuint mu_eff;

        common_stencil coefficients_stencil;
        
        
        // ONE-TO-ONE COPY OF THE EXPLICIT SOLVER'S VELOCITY INITIALIZATION (explicit_simulation::MAC_vx_grid / MAC_vy_grid / MAC_vz_grid),
        // built on CPU vectors and uploaded into the RANS-owned u / v / w SSBOs
        void initialize_velocity_fields();

        RANS(const MultigridHierarchy &hierarchy_in, ShaderManager &shadermanager_in, configuration &config);      // cells and domain size come from the hierarchy (level 0)
        
        //COMMON HELPER FUNCTIONS FOR MULTIPLE EQUATIONS AT FINEST LEVEL          
        void restrict_vels(bool use_previous_iterate_velocities);       // RESTRICTS ALL THE WAY TO THE LAST LEVELS
        void restrict_mu_eff();       // RESTRICTS ALL THE WAY TO THE LAST LEVELS
        
        //COMMON HELPER FUNCTIONS FOR MULTIPLE EQUATIONS AT COARSER LEVELS
        void clear_coefficients(StencilOperator &stencil);

        // x = max(x, floor_value) over the first element_count floats of a scalar field SSBO (NaN is left untouched, see Kernels/clamp_to_floor.comp)
        void clamp_field_to_floor(GLuint field, int element_count, float floor_value);
        
        
        // TRUE FINE LEVEL FUNCTIONS COMMON FOR MULTIPLE EQUATIONS
        void add_time_contribution(float dt, LinearSystem &linear_system);
        void add_advection_contribution(LinearSystem &linear_system, bool use_previous_iterate_velocities);
        
        
        // OVERLOADED FUNCTIONS FOR MAKING THE COEFFICIENTS AT COARSER LEVELS (COMMON FOR MULTIPLE EQUATIONS)
        void add_time_contribution(float dt, int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, fieldTopology topology, StencilOperator &coefficients_at_coarse_level, Levels_info &level_info);
        void add_advection_contribution(int current_cells_x, int current_cells_y, int current_cells_z, GLuint &current_level_u_field, GLuint &current_level_v_field, GLuint &current_level_w_field,fieldTopology topology, StencilOperator &coefficients_at_coarse_level, Levels_info &level_info);
        

        //FINAL ONE LINER CALLERS
        void construct_u_momentum_equation(float stable_dt, float spalding_wall_smoothness);
        void construct_v_momentum_equation(float stable_dt, float spalding_wall_smoothness);
        void construct_w_momentum_equation(float stable_dt, float spalding_wall_smoothness);

        void build_u_coefficients_hierarchy(float dt, float spalding_wall_smoothness);
        void build_v_coefficients_hierarchy(float dt, float spalding_wall_smoothness);
        void build_w_coefficients_hierarchy(float dt, float spalding_wall_smoothness);

        //NONLINEAR CONVERGENCE
        // copies the current u, v, w, k, eps into the previous-iterate buffers (GPU -> GPU)
        void store_nonlinear_iterate(k_TurbulenceModels &k_model, k_TurbulenceModels::destruction_variable &eps_model);

        // max over u, v, w, k, eps of  sqrt( sum((new - old)^2) / sum(new^2) ),  old = last store_nonlinear_iterate()
        float calculate_nonlinear_convergence(k_TurbulenceModels &k_model, k_TurbulenceModels::destruction_variable &eps_model);
    };