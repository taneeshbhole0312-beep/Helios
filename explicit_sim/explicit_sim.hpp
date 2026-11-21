#pragma once

#include <glad/glad.h>
#include <vector>
#include <cmath>

#include "../shader_manager/shader_manager.hpp"

class explicit_simulation
{
private:
    int& cells_x;
    int& cells_y;
    int& cells_z;

    float& cell_side_x;
    float& cell_side_y;
    float& cell_side_z;

    float& WIDTH;
    float& HEIGHT;

    int frame_buffer_w;
    int frame_buffer_h;

    ShaderManager& shadermanager;

    // chain of ever-smaller buffers for the max-|v| reduction: N -> ceil(N/256) -> ... -> 1
    std::vector<GLuint> reduction_for_CFL_SSBOs_arr;

    void flat_groups_to_3d(
        int totalGroups,
        int& gx,
        int& gy,
        int& gz
    );

    struct MAC_vx
    {
        std::vector<float> vx;
        std::vector<float> vx_orignal;
    };

    struct MAC_vy
    {
        std::vector<float> vy;
        std::vector<float> vy_orignal;
    };

    struct MAC_vz
    {
        std::vector<float> vz;
        std::vector<float> vz_orignal;
    };

    GLuint create_SSBO(const std::vector<float>& data);

    MAC_vx MAC_vx_grid(
        int cells_x,
        int cells_y,
        int cells_z,
        float cell_side_y,
        float HEIGHT
    );

    MAC_vy MAC_vy_grid(
        int cells_x,
        int cells_y,
        int cells_z,
        float cell_side_x,
        float WIDTH
    );

    MAC_vz MAC_vz_grid(
        int cells_x,
        int cells_y,
        int cells_z
    );

public:

    GLuint vx_grid_SSBO;
    GLuint vy_grid_SSBO;
    GLuint vz_grid_SSBO;

    GLuint vx_grid_orignal_SSBO;
    GLuint vy_grid_orignal_SSBO;
    GLuint vz_grid_orignal_SSBO;

    GLuint Buffer_vx_grid_SSBO;
    GLuint Buffer_vy_grid_SSBO;
    GLuint Buffer_vz_grid_SSBO;

    explicit_simulation(
        int& cells_x_in,
        int& cells_y_in,
        int& cells_z_in,
        float& cell_side_x_in,
        float& cell_side_y_in,
        float& cell_side_z_in,
        float& WIDTH_in,
        float& HEIGHT_in,
        int frame_buffer_w_in,
        int frame_buffer_h_in,
        ShaderManager& shadermanager_in
    );

    void initialize();

    //------------------------------------------------------------ explicit solver operators
    void viscosity_cal(
        float &stable_dt,
        float &viscosity,
        float density
    );

    void WENOZ_advect(
        float &stable_dt
    );

    void SSPRK2_blend_step();

    float velocity_component_reduction(
        GLuint velocity,
        int total_elements
    );
};