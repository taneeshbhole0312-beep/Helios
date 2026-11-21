#pragma once

#include <vector>
#include "glad/glad.h"
#include <cmath>
#include <algorithm>
#include "../shader_manager/shader_manager.hpp"
#include "multigrid_hierarchy.hpp"
#include "../configuration.hpp"

enum class fieldTopology
{
    CELL_CENTERED,
    XFACE_CENTERED,
    YFACE_CENTERED,
    ZFACE_CENTERED
};

struct Field
{
    GLuint data;
    fieldTopology topology;
};

struct StencilOperator
{
    GLuint aP;
    GLuint aW;
    GLuint aE;
    GLuint aS;
    GLuint aN;
    GLuint aB;
    GLuint aT;
};

// Element (DOF) counts of one coarse level for ONE field topology. Derived from the shared MultigridHierarchy cell counts,
// never generated independently.
struct Levels_info
{
    int level_x_elements;
    int level_y_elements;
    int level_z_elements;
};

class common_stencil         // INSANE memory saving      (members are defined in Turbulence_models/RANS.cpp)
{
    private:
        GLuint create_SSBO(const std::vector<float> &data);     // helper

    public:
        StencilOperator coefficients;            //finest level (standard level)
        std::vector <StencilOperator> multigrid_levels_coeffiecients_arr;                //coarser levels

        GLuint RHS_main;

        common_stencil(const MultigridHierarchy &hierarchy);
};


class LinearSystem
{
    private:
        GLuint create_SSBO(const std::vector<float> &data);
        ShaderManager &shadermanager;
        configuration &config;
        
    public:
        const MultigridHierarchy &hierarchy;        // shared level geometry (cell counts, spacing) of every system

        GLuint previous_time_step_field;
        GLuint pre_previous_time_step_field;        //For BDF2, Initialized to null if other mode selected
        Field field;
        std::vector <Levels_info> levels_info;      // COARSE levels only: element [k] is level k + 1 (element counts of this topology)

        int cells_x;
        int cells_y;
        int cells_z;

        int x_elements;
        int y_elements;
        int z_elements;

        LinearSystem(const MultigridHierarchy &hierarchy_in, ShaderManager &shadermanager_in, fieldTopology topology, configuration &config);

        std::vector <int> GetLevelDimension(int level) const;
        void save_previous_time_step_field();           // Saves all the previosu time steps needed by the selected TemporalScheme$2
        void seed_time_history();                       // sets previous (and, for BDF2, pre-previous) time step = the current field: call once on the initial condition

};