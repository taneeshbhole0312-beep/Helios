#pragma once


#include <memory>
#include "../Linear_systems/LinearSystem.hpp"

class RANS;             // forward declared: RANS.hpp includes this header so RANS can orchestrate K / epsilon

class k_TurbulenceModels
{
    private:
        friend class RANS; 
        
        RANS &rans;             // We use some common data and functions form class RANS in class Turbulence models
        bool is_omega_based;
        bool is_SST;        // Always False when is_omega_based is false, and can be true or false when it is true 

        class SST_additions
        {
            private:
                k_TurbulenceModels &k_model;

                //COARSE LEVEL FUNCTIONS REQUIRED
                void compute_blended_coefficient_field(int current_cells_x, int current_cells_y, int current_cells_z, GLuint wall_dists_ssbo_for_coarse_level, GLuint blended_coefficient_at_coarse_level, GLuint k_field_at_coarse_level, GLuint omega_field_at_coarse_level); 

                void compute_structured_mesh_cell_min_dists(int current_cells_x, int current_cells_y, int current_cells_z, GLuint wall_dists_ssbo_for_coarse_level);

            public:
                GLuint min_wall_dists;
                GLuint blended_coefficient;     // This is the field for the value of F1 across the domain
                std::vector <GLuint> blended_coefficient_hierarchy;
                std::vector <GLuint> min_wall_dists_hierarchy;

                // FLOOR OF THE omega THAT DIVIDES THE CROSS DIFFUSION SOURCE (and the F1 arguments).  omega_min = factor * U_ref / L_ref  [1/s]
                static constexpr float omega_floor_factor = 1.0f;           // c in omega_min = c U_ref / L_ref
                float omega_min = 0.0f;

                SST_additions(k_TurbulenceModels &k_model_in);

                // omega_min = c * U_ref / min(L_turbulence, longest domain side). Prints a warning (and leaves omega_min = 0) when U_ref or L is not a positive number.
                void set_omega_floor(float reference_velocity, float turbulence_length_scale);

                void compute_structured_mesh_cell_min_dists();
                void compute_blended_coefficient_field(GLuint omega_field);    
                void assemble_cross_diffusion_contribution();
                void assemble_coarse_level_hierarchy_blended_coefficient(std::vector <GLuint> &destruction_variable_hierarchy);

                void SST_preprocessing(GLuint &omega_field, std::vector <GLuint> &destruction_variable_hierarchy);
        };

        //Private helper    
        GLuint create_SSBO(const std::vector<float> &data);

        //FINEST LEVEL CONTRIBUTIONS FOR K EQUATION 
        //transient and advective contributions are made via shared functions from RANS.cpp
        void assemble_diffusion_contribution_k();

        void assemble_destruction_contribution_k(GLuint &destruction_variable);
        void assemble_production_contribution_k(GLuint &destruction_variable);
        
        //COARSER LEVEL OVERLOADED FUNCTION FOR THE SPECIALIZED (HARMONIC INTERPOLATION BASED) DIFFUSION CONTRIBUTION FOR K EQUATION
        void assemble_diffusion_contribution_k(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, GLuint mu_eff_at_coarse_level, GLuint blended_coefficient_at_coarse_level);     // blended_coefficient_at_coarse_level: F1 restricted to this level, only read when is_SST

        //COARSER LEVEL OVERLOADED FUNCTION FOR THE DESTRUCTION CONTRIBUTION OF THE K EQUATION (no production on coarse levels)
        void assemble_destruction_contribution_k(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, GLuint k_at_coarse_level, GLuint destruction_variable_at_coarse_level);

        //WALL TREATMENT (SPALDING) OF k, ONE FUNCTION PER MODEL: WRITES k IN EVERY WALL CELL AND PINS THE k FINE ROW IN THE SHARED STENCIL SET
        void apply_k_wall_treatment_for_k_eps_model(float spalding_wall_smoothness);
        void apply_k_wall_treatment_for_k_omega_model(float spalding_wall_smoothness);

        //COARSER LEVEL OVERLOADS: HOMOGENEOUS CORRECTION CONDITION (e = 0) ON THE WALL ROWS OF THE COARSE K OPERATOR
        void apply_k_wall_treatment_for_k_eps_model(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level);
        void apply_k_wall_treatment_for_k_omega_model(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level);

        public:
            std::vector <GLuint> k_hierarchy;
            LinearSystem KSystem;
            std::unique_ptr <SST_additions> sst;          //null ptr when non-SST model is selected

            k_TurbulenceModels(RANS &rans, bool is_omega_based_system, bool is_SST_in);       // is_SST_in is ignored (forced false) unless is_omega_based_system is true

            GLuint &k_field();

            //Final one liner callers
            void construct_K_equation(float stable_dt , GLuint &destruction_variable, float spalding_wall_smoothness);
            void construct_K_equation_hierarchy(float stable_dt , std::vector <GLuint> &destruction_variable_hierarchy);

            class destruction_variable
            {
                private:
                    friend class RANS;          // the RANS-level orchestration (not written yet) will be the caller of the initialization below

                    k_TurbulenceModels &k_model;

                    //transient and advective contributions are made via shared functions from RANS.cpp
                    void assemble_diffusion_contribution_destruction_variable();

                    void assemble_destruction_contribution_destruction_variable();
                    void assemble_production_contribution_destruction_variable();

                    //COARSER LEVEL OVERLOADED FUNCTION FOR THE SPECIALIZED (HARMONIC INTERPOLATION BASED) DIFFUSION CONTRIBUTION FOR K EQUATION
                    void assemble_diffusion_contribution_destruction_variable(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, GLuint mu_eff_at_coarse_level, GLuint blended_coefficient_at_coarse_level);     // blended_coefficient_at_coarse_level: F1 restricted to this level, only read when is_SST

                    //COARSER LEVEL OVERLOADED FUNCTION FOR THE DESTRUCTION CONTRIBUTION OF THE DESTRUCTION VARIABLE EQUATION (no production on coarse levels)
                    void assemble_destruction_contribution_destruction_variable(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level, GLuint k_at_coarse_level, GLuint destruction_variable_at_coarse_level);

                    //WALL TREATMENT (SPALDING), ONE FUNCTION PER TRANSPORTED VARIABLE (the k wall functions live in k_TurbulenceModels):
                    //WRITES THE VARIABLE IN EVERY WALL CELL AND PINS THAT VARIABLE'S FINE ROW IN THE SHARED STENCIL SET
                    void apply_eps_wall_treatment_for_k_eps_model(float spalding_wall_smoothness);      //epsilon of the k-epsilon model
                    void apply_omega_wall_treatment_for_k_omega_model(float spalding_wall_smoothness);  //omega of the k-omega model (also prescribes mu_eff in the wall cells)

                    //COARSER LEVEL OVERLOADS: HOMOGENEOUS CORRECTION CONDITION (e = 0) ON THE WALL ROWS OF THAT VARIABLE'S COARSE OPERATOR
                    void apply_eps_wall_treatment_for_k_eps_model(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level);
                    void apply_omega_wall_treatment_for_k_omega_model(int CurrentCells_x, int CurrentCells_y, int CurrentCells_z, StencilOperator &coefficients_at_coarse_level);

                    
                public:
                    LinearSystem destruction_variable_system;

                    destruction_variable(k_TurbulenceModels &k_model);
                    
                    // Restricted old k and old epsilon (cell centered), 0-based: [0] = N/2, [1] = N/4, ...
                    std::vector <GLuint> destruction_variable_hierarchy;
                    void restrict_k_and_destruction_variable();       // RESTRICTS ALL THE WAY TO THE LAST LEVELS

                    // returns false (and changes nothing) if the turbulence length scale is not a positive number
                    bool initialize_k_epsilon_RANS_fields(float reference_velocity, float turbulence_intensity, float turbulence_length_scale, float spalding_wall_smoothness);

                    // same as above for k-omega (K_omega_initialize.comp); ends with the omega wall treatment and mu_eff = mu + rho k / omega
                    bool initialize_k_omega_RANS_fields(float reference_velocity, float turbulence_intensity, float turbulence_length_scale, float spalding_wall_smoothness, bool is_SST);

                    GLuint &destruction_variable_field();

                    // x = max(x, tiny positive floor) on k AND on the destruction variable (omega / epsilon). Call right after a linear solve of k or of the
                    // destruction variable, BEFORE anything reads them: it removes the rounding-level negative values the multigrid leaves in near-zero regions.
                    // NaN is not repaired (the monitors must still see it).
                    void enforce_positivity();

                    //Final one liner callers
                    void construct_destruction_variable_equation(float stable_dt, float spalding_wall_smoothness);
                    void construct_destruction_variable_equation_hierarchy(float stable_dt);

                    //mu_eff = mu + rho C_mu k^2 / eps   (from the current k and eps fields)
                    void update_mu_eff_k_eps();

                    //mu_eff = mu + rho * k / omega
                    void update_mu_eff_k_omega();

                    //mu_eff is limited by formulated limiters
                    void update_mu_eff_k_omega_SST();
                };
};
