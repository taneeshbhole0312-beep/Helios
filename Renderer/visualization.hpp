#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>

#include "../shader_manager/shader_manager.hpp"
#include "../explicit_sim/explicit_sim.hpp"
#include "../Debuggers/debuggers.hpp"
#include "../Linear_systems/LinearSystem.hpp"       // fieldTopology

class camera
{
private:
    float camera_speed = 400.0f;
    float mouse_sensitivity = 0.12f;
    float world_up[3] = { 0, 1, 0 };

    void calc_forward_camera(camera& camera);
    void calc_right_camera(camera& camera);
    void calc_up_camera(camera& camera);
    void camera_update(camera& camera);

public:
    float halfWidth;
    float halfHeight;

    float x;
    float y;
    float z;

    float yaw;
    float pitch;

    float forward[3];
    float right[3];
    float up[3];

    camera(
        float FOV_degree,
        float camera_x,
        float camera_y,
        float camera_z,
        float yaw_in,
        float pitch_in,
        int fb_w,
        int fb_h
    );

    void camera_controls(
        GLFWwindow* window,
        camera& camera,
        float frame_dt
    );

    void set_camera_position(
        float x_in,
        float y_in,
        float z_in
    );
};

class renderer_functions;      // Forward declaration due to shit design

class light
{
    private:
        renderer_functions &render;

    public:
        
    
    float x;
        float y;
        float z;
        
        float WIDTH;
        float HEIGHT;
        float DEPTH;
        
        float light_ds;
        float intensity;
        
        int cells_x;
        int cells_y;
        int cells_z;

        float extinction_total;

        light(
            float light_x,
            float light_y,
            float light_z,
            int cells_x_in,
            int cells_y_in,
            int cells_z_in,
            float light_intensity,
            float extinction_total_in,
            float WIDTH_in,
            float HEIGHT_in,
            float DEPTH_in,
            float light_ds_in,
            renderer_functions &renderer_in
        );
        
        void dispatchLightTransmittance(ShaderManager& shadermanager);
    };
    
    
    class renderer_functions
    {
    private:
        GLuint image_texture;
    
        camera &Camera;
        
        int cells_x;
        int cells_y;
        int cells_z;
        
        float cell_side_x;
        float cell_side_y;
        float cell_side_z;
        
        float WIDTH;
        float HEIGHT;
        float DEPTH;
        
        float extinction_total;
        float extinction_via_scattering;
        float mean_cosine;
        
        GLuint quadVAO = 0;
        GLuint quadVBO = 0;
        
        void flat_groups_to_3d(
            int totalGroups,
            int& gx,
            int& gy,
            int& gz
        );
        
        std::vector <float> create_smoke(int cells_x, int cells_y, int cells_z); // helper

        // scratch buffers of the max |field| reduction chain used by render_normalized_field_slice (built on first use)
        std::vector <GLuint> field_max_reduction_SSBOs;

    public:
        GLuint smokeMatter_SSBO;
        GLuint lightTransmittance_SSBO;
        GLuint smokeMatter_orignal_SSBO;
        GLuint smokeMatter_SSBO_buffer;

        renderer_functions(
            int cells_x_in,
            int cells_y_in,
            int cells_z_in,
            float cell_side_x_in,
            float cell_side_y_in,
            float cell_side_z_in,
            float WIDTH_in,
            float HEIGHT_in,
            float DEPTH_in,
            float extinction_total_in,
            float extinction_via_scattering_in,
            float mean_cosine_in,
            int fb_w,
            int fb_h,
            camera &Camera
        );

        void smoke_advection_dispatch(ShaderManager &shadermanager, GLuint vx_grid, GLuint vy_grid, GLuint vz_grid, float stable_dt);

        void ray_march_dispatch(
            ShaderManager& shadermanager,
            int imageW,
            int imageH,
            float raymarch_ds,
            light &Light
        );

        void create_fullscreen_quad();

        void draw_fullscreen_texture(
            ShaderManager &shadermanager,
            GLuint &texture,
            int fb_w,
            int fb_h
        );

        void final_smoke_blend(
            ShaderManager &shadermanager
        );

        // Shows ANY field SSBO of the given topology (CELL / XFACE / YFACE / ZFACE, sized for this renderer's cells_x/y/z)
        // on the z-slice `slice_id`: finds max |field| with the existing CFL max-reduction kernel, then the
        // field_normalized_slice kernel divides by it and colours the result. Nothing normalised is stored.
        // The picture ends up on the default framebuffer, so glReadPixels works exactly as after dual_mode_render.
        void render_normalized_field_slice(
            ShaderManager &shadermanager,
            GLuint field_ssbo,
            fieldTopology topology,
            int slice_id,
            int fb_w,
            int fb_h
        );

        void dual_mode_render(
            int render_mode, 
            ShaderManager &shadermanager, 
            debuggers debug,  
            int slice_id,
            int fb_w, 
            int fb_h, 
            float raymarch_ds,
            light &Light
        );
};