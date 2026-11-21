#include "visualization.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

//helper
GLuint create_SSBO(const std::vector<float> &data)
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




void camera::calc_forward_camera(camera &camera)
{
    float yaw = camera.yaw * 3.1415926536 / 180.0f;
    float pitch = camera.pitch * 3.1415926536 / 180.0f;              // conversion to radians

    camera.forward[0] = cos(pitch) * cos(yaw);
    camera.forward[1] = sin(pitch);
    camera.forward[2] = cos(pitch) * sin(yaw);
}

void camera::calc_right_camera(camera &camera)
{
    float up[3] = { world_up[0], world_up[1], world_up[2] };

    float right_x = camera.forward[1] * up[2] - camera.forward[2] * up[1];

    float right_y = camera.forward[2] * up[0] - camera.forward[0] * up[2];

    float right_z = camera.forward[0] * up[1] - camera.forward[1] * up[0];

    float mag = sqrt( right_x * right_x + right_y * right_y + right_z * right_z );

    // Forward is too close to world_up.
    if (mag < 1e-6f)
    {
        up[0] = 0.0f;
        up[1] = 0.0f; //determines that mag is too low for the right, so the up is too much aligned with the world up, so to avoid division by 0, do this
        up[2] = 1.0f;

        right_x = camera.forward[1] * up[2] - camera.forward[2] * up[1];

        right_y = camera.forward[2] * up[0] - camera.forward[0] * up[2];

        right_z = camera.forward[0] * up[1] - camera.forward[1] * up[0];

        mag = sqrt( right_x * right_x + right_y * right_y + right_z * right_z);
    }

    camera.right[0] = right_x / mag;
    camera.right[1] = right_y / mag;
    camera.right[2] = right_z / mag;
}

void camera::calc_up_camera(camera &camera)
{
    float up_x = camera.right[1] * camera.forward[2] - camera.right[2] * camera.forward[1];
    float up_y = camera.right[2] * camera.forward[0] - camera.right[0] * camera.forward[2];
    float up_z = camera.right[0] * camera.forward[1] - camera.right[1] * camera.forward[0];


    camera.up[0] = up_x;
    camera.up[1] = up_y;
    camera.up[2] = up_z;
}

void camera::camera_update(camera &camera)
{
    calc_forward_camera(camera);
    calc_right_camera(camera);
    calc_up_camera(camera);
}

camera::camera(float FOV_degree, float camera_x, float camera_y, float camera_z, float yaw_in, float pitch_in, int fb_w, int fb_h)
{
    x = camera_x;
    y = camera_y;
    z = camera_z;
    halfHeight = tan((FOV_degree * 3.1415926535 / 180.0f) * 0.5);
    float aspect = float(fb_w) / float(fb_h);
    halfWidth = aspect * halfHeight;
    pitch = pitch_in;
    yaw = yaw_in;
}

void camera::camera_controls(GLFWwindow *window, camera &camera, float frame_dt)
{
    //--------------------------------------------------------------- rotation
    static bool  looking = false;
    static double last_x = 0.0;
    static double last_y = 0.0;

    double mouse_x, mouse_y;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        if (!looking)
        {
            // first frame of the drag: anchor here so the view doesn't snap
            looking = true;
            last_x = mouse_x;
            last_y = mouse_y;
        }

        float delta_x = float(mouse_x - last_x);
        float delta_y = float(mouse_y - last_y);

        camera.yaw += delta_x * mouse_sensitivity;
        camera.pitch -= delta_y * mouse_sensitivity;

        // Stop just short of the poles. Exactly +-90 makes forward parallel to world up,
        // which collapses the right vector to zero and NaNs the whole camera basis.
        if (camera.pitch > 89.0f)  camera.pitch = 89.0f;
        if (camera.pitch < -89.0f) camera.pitch = -89.0f;
    }
    else
    {
        looking = false;
    }

    last_x = mouse_x;
    last_y = mouse_y;

    // rebuild forward/right/up from the angles we just changed
    camera_update(camera);

    //--------------------------------------------------------------- translation
    float speed = camera_speed * frame_dt;

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        speed *= 4.0f;
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.x += camera.forward[0] * speed;
        camera.y += camera.forward[1] * speed;
        camera.z += camera.forward[2] * speed;
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.x -= camera.forward[0] * speed;
        camera.y -= camera.forward[1] * speed;
        camera.z -= camera.forward[2] * speed;
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.x += camera.right[0] * speed;
        camera.y += camera.right[1] * speed;
        camera.z += camera.right[2] * speed;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.x -= camera.right[0] * speed;
        camera.y -= camera.right[1] * speed;
        camera.z -= camera.right[2] * speed;
    }

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    {
        camera.y += speed;
    }

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        camera.y -= speed;
    }

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

void camera::set_camera_position(float x_in, float y_in, float z_in)
{
    x = x_in;
    y = y_in;
    z = z_in;
}


light::light(float light_x, float light_y, float light_z, int cells_x_in, int cells_y_in, int cells_z_in, float light_intensity, float extinction_total_in, float WIDTH_in, float HEIGHT_in, float DEPTH_in, float light_ds_in, renderer_functions &renderer_in)
: render(renderer_in)
{
    x = light_x;
    y = light_y;
    z = light_z;
    cells_x = cells_x_in;
    cells_y = cells_y_in;
    cells_z = cells_z_in;
    intensity = light_intensity;
    extinction_total = extinction_total_in;
    WIDTH = WIDTH_in;
    HEIGHT = HEIGHT_in;
    DEPTH = DEPTH_in;
    light_ds = light_ds_in;
}

void light::dispatchLightTransmittance(ShaderManager &shadermanager)
{
    glUseProgram(shadermanager.transmittance_program);

    glUniform1f(shadermanager.light_transmittance_volume_width_loc, WIDTH);
    glUniform1f(shadermanager.light_transmittance_volume_height_loc, HEIGHT);
    glUniform1f(shadermanager.light_transmittance_volume_depth_loc, DEPTH);
    glUniform1f(shadermanager.light_transmittance_for_light_ds_loc, light_ds);
    glUniform1i(shadermanager.light_transmittance_field_stride_loc, cells_x);
    glUniform1i(shadermanager.light_transmittance_field_layer_stride_loc, cells_x * cells_y);
    glUniform1i(shadermanager.light_transmittance_field_x_elements_loc, cells_x);
    glUniform1i(shadermanager.light_transmittance_field_y_elements_loc, cells_y);
    glUniform1i(shadermanager.light_transmittance_field_z_elements_loc, cells_z);
    glUniform1f(shadermanager.light_transmittance_extinction_total_loc, extinction_total);

    // Density field
    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        render.smokeMatter_SSBO
    );

    // Output T_L field
    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        render.lightTransmittance_SSBO
    );

    // 8x8x8 local workgroup
    const int GROUP_SIZE = 8;

    GLuint groups_x = (cells_x + GROUP_SIZE - 1) / GROUP_SIZE;

    GLuint groups_y = (cells_y + GROUP_SIZE - 1) / GROUP_SIZE;

    GLuint groups_z = (cells_z + GROUP_SIZE - 1) / GROUP_SIZE;

    glDispatchCompute(
        groups_x,
        groups_y,
        groups_z
    );

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}


void renderer_functions::flat_groups_to_3d(int totalGroups, int &gx, int &gy, int &gz)     //WARNING: this func blocks the max groups per axis to 256 square
{
    const int maxPerAxis = 65535;

    gx = std::min(totalGroups, maxPerAxis);
    int remaining = (totalGroups + gx - 1) / gx;

    gy = std::min(remaining, maxPerAxis);
    gz = (remaining + gy - 1) / gy;
}

renderer_functions::renderer_functions(int cells_x_in, int cells_y_in, int cells_z_in, float cell_side_x_in, float cell_side_y_in,float cell_side_z_in,float WIDTH_in, float HEIGHT_in, float  DEPTH_in, float extinction_total_in, float extinction_via_scattering_in, float mean_cosine_in, int fb_w, int fb_h, camera &Camera)
: Camera(Camera)
{
    cells_x = cells_x_in;
    cells_y = cells_y_in;
    cells_z = cells_z_in;
    cell_side_x = cell_side_x_in;
    cell_side_y = cell_side_y_in;
    cell_side_z = cell_side_z_in;
    extinction_total = extinction_total_in;
    extinction_via_scattering = extinction_via_scattering_in;
    WIDTH = WIDTH_in;
    HEIGHT = HEIGHT_in;
    DEPTH = DEPTH_in;
    mean_cosine = mean_cosine_in;

    std::vector <float> smokeMatter = create_smoke(cells_x, cells_y, cells_z);
    smokeMatter_SSBO = create_SSBO(smokeMatter);
    smokeMatter_SSBO_buffer = create_SSBO(smokeMatter);
    smokeMatter_orignal_SSBO = create_SSBO(smokeMatter);

    
    lightTransmittance_SSBO = smokeMatter_orignal_SSBO; // MEMORY CONSERVATION TRICK. smokeMatter_orignal_SSBO is refreshed everytime before use

    //texture explicit_simulation for raymarching and rendering
    glGenTextures(1, &image_texture);

    glBindTexture(GL_TEXTURE_2D, image_texture);

    glTexStorage2D(
        GL_TEXTURE_2D,
        1,
        GL_RGBA32F,
        fb_w,
        fb_h);

    // Required. The default minification filter is GL_NEAREST_MIPMAP_LINEAR, but this
    // texture only ever has one level, so leaving the default risks the sampler being
    // treated as incomplete and returning black on some drivers.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

}

std::vector<float> renderer_functions::create_smoke(int cells_x, int cells_y, int cells_z)
{
    std::vector<float> smoke(
        cells_x * cells_y * cells_z,
        0.0f
    );

    int mid_y = cells_y / 2;
    int mid_z = cells_z / 2;
    int mid_x = cells_x / 2;

    for (int z = 0; z < cells_z; z++)
    {
        for (int y = 0; y < cells_y; y++)
        {
            for (int x = 0; x < cells_x; x++)
            {
                int idx = z * cells_y * cells_x + y * cells_x + x;

                if (x >= mid_y - 8 && x <= mid_x + 8 &&
                    z >= mid_z - 8 && z <= mid_z + 8)
                {
                    smoke[idx] = 2.5f;
                }

                // if (y < 10 && y > 0)
                // {
                //     smoke[idx] = 2.5f;
                // }
            }
        }
    }

    return smoke;
}


void renderer_functions::smoke_advection_dispatch(ShaderManager &shadermanager, GLuint vx_grid, GLuint vy_grid, GLuint vz_grid, float stable_dt)
{
    glUseProgram(shadermanager.smoke_advection_program);

    glUniform1i(shadermanager.smoke_advection_cells_x_loc, cells_x);
    glUniform1i(shadermanager.smoke_advection_cells_y_loc, cells_y);
    glUniform1i(shadermanager.smoke_advection_cells_z_loc, cells_z);
    glUniform1f(shadermanager.smoke_advection_cell_side_x_loc, cell_side_x);
    glUniform1f(shadermanager.smoke_advection_cell_side_y_loc, cell_side_y);
    glUniform1f(shadermanager.smoke_advection_cell_side_z_loc, cell_side_z);
    glUniform1f(glGetUniformLocation(shadermanager.smoke_advection_program, "dt"), stable_dt);

    // MAC velocity fields
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, vx_grid);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vy_grid);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vz_grid);

    // Smoke input/output
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, smokeMatter_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, smokeMatter_SSBO_buffer);


    // One thread per smoke cell
    int gx = (int(cells_x) + 7u) / 8u;
    int gy = (int(cells_y) + 7u) / 8u;
    int gz = (int(cells_z) + 7u) / 8u;

    glDispatchCompute(gx, gy, gz);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::swap(smokeMatter_SSBO, smokeMatter_SSBO_buffer);
}

void renderer_functions::ray_march_dispatch(ShaderManager &shadermanager, int imageW, int imageH, float raymarch_ds, light &Light)
{
    glUseProgram(shadermanager.ray_march_program);

    glUniform1f(shadermanager.ray_march_halfHeight_loc, Camera.halfHeight);
    glUniform1f(shadermanager.ray_march_halfWidth_loc, Camera.halfWidth);
    glUniform1i(shadermanager.ray_march_image_width_loc, imageW);
    glUniform1i(shadermanager.ray_march_image_height_loc, imageH);
    glUniform1f(shadermanager.ray_march_volume_depth_loc, DEPTH);
    glUniform1f(shadermanager.ray_march_volume_height_loc, HEIGHT);
    glUniform1f(shadermanager.ray_march_volume_width_loc, WIDTH);
    glUniform1f(shadermanager.ray_march_ds_loc, raymarch_ds);
    glUniform1i(shadermanager.ray_march_field_stride_loc, cells_x);
    glUniform1i(shadermanager.ray_march_field_layer_stride_loc, cells_x * cells_y);
    glUniform1i(shadermanager.ray_march_field_x_elements_loc, cells_x);
    glUniform1i(shadermanager.ray_march_field_y_elements_loc, cells_y);
    glUniform1i(shadermanager.ray_march_field_z_elements_loc, cells_z);
    glUniform1f(shadermanager.ray_march_light_x_loc, Light.x);
    glUniform1f(shadermanager.ray_march_light_y_loc, Light.y);
    glUniform1f(shadermanager.ray_march_light_z_loc, Light.z);
    glUniform1f(shadermanager.ray_march_light_intensity_loc, Light.intensity);
    glUniform1f(shadermanager.ray_march_extinction_total_loc, extinction_total);
    glUniform1f(shadermanager.ray_march_extinction_via_scattering_loc, extinction_via_scattering);
    glUniform1f(shadermanager.ray_march_for_light_ds_loc, Light.light_ds);
    glUniform1f(shadermanager.ray_march_mean_cosine_loc, mean_cosine);
    glUniform1f(shadermanager.ray_march_smoke_color_max, 2.5);
    glUniform1f(shadermanager.ray_march_smoke_color_min, 0.0);

    // output image
    glBindImageTexture(
        0,
        image_texture,
        0,
        GL_FALSE,
        0,
        GL_WRITE_ONLY,
        GL_RGBA32F);

    // sampling field
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, smokeMatter_SSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, lightTransmittance_SSBO);

    glUniform3f(shadermanager.ray_march_cameraPosition_loc, Camera.x, Camera.y, Camera.z);
    glUniform3f(shadermanager.ray_march_cameraForward_loc,  Camera.forward[0], Camera.forward[1], Camera.forward[2]);
    glUniform3f(shadermanager.ray_march_cameraRight_loc,    Camera.right[0],   Camera.right[1],   Camera.right[2]);
    glUniform3f(shadermanager.ray_march_cameraUp_loc,       Camera.up[0],      Camera.up[1],      Camera.up[2]);


    // launch one thread per pixel
    const GLuint gx = (GLuint(imageW) + 15u) / 16u;
    const GLuint gy = (GLuint(imageH) + 15u) / 16u;
    glDispatchCompute(gx, gy, 1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void renderer_functions::create_fullscreen_quad()
{
    float quad[] = {
        // pos      // uv
        -1.0f, -1.0f,  0.0f, 0.0f,
        1.0f, -1.0f,  1.0f, 0.0f,
        1.0f,  1.0f,  1.0f, 1.0f,

        -1.0f, -1.0f,  0.0f, 0.0f,
        1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
}

void renderer_functions::draw_fullscreen_texture(ShaderManager &shadermanager, GLuint &texture, int fb_w, int fb_h)
{
    // Nothing else in the program ever sets the viewport, so do it here. Without this
    // the blit keeps using whatever size the viewport defaulted to at context creation
    // and the image is stretched or cropped after any window resize.
    glViewport(0, 0, fb_w, fb_h);

    // The default framebuffer was never being cleared anywhere in the program, so any
    // frame that failed to draw left stale/uninitialized contents on screen instead of a
    // known background. Clearing here makes a dropped frame obvious rather than silent.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shadermanager.displayProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    GLint loc = glGetUniformLocation(shadermanager.displayProgram, "renderTex");
    glUniform1i(loc, 0);

    glUniform3f(shadermanager.fragment_backgroundColor_loc, 0.02f, 0.02f, 0.05f);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void renderer_functions::final_smoke_blend(ShaderManager &shadermanager)
{
    glUseProgram(shadermanager.final_blend_SSPRK2_program);

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        0,
        smokeMatter_orignal_SSBO
    );

    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        1,
        smokeMatter_SSBO
    );

    glUniform1ui(shadermanager.final_blend_SSPRK2_count_loc, cells_x * cells_y * cells_z);

    {
        int groups = (cells_y * cells_x * cells_z + 255) / 256;
        int gx, gy, gz;
        flat_groups_to_3d(groups, gx, gy, gz);
        glDispatchCompute(gx, gy, gz);
    }

}

void renderer_functions::render_normalized_field_slice(ShaderManager &shadermanager, GLuint field_ssbo, fieldTopology topology, int slice_id, int fb_w, int fb_h)
{
    // number of DOFs of each topology for this renderer's cell counts
    int total_elements = cells_x * cells_y * cells_z;

    if (topology == fieldTopology::XFACE_CENTERED)
        total_elements = (cells_x + 1) * cells_y * cells_z;
    else if (topology == fieldTopology::YFACE_CENTERED)
        total_elements = cells_x * (cells_y + 1) * cells_z;
    else if (topology == fieldTopology::ZFACE_CENTERED)
        total_elements = cells_x * cells_y * (cells_z + 1);

    // -------- scratch chain for the reduction, built once. It is sized for the largest topology, so any topology fits
    //          (the chain for n elements never needs more or bigger levels than the chain for max_elements >= n)
    if (field_max_reduction_SSBOs.empty())
    {
        int max_elements = std::max(std::max((cells_x + 1) * cells_y * cells_z, cells_x * (cells_y + 1) * cells_z), cells_x * cells_y * (cells_z + 1));

        int n = max_elements;

        while (n > 1)
        {
            n = (n + 255) / 256;

            GLuint level_ssbo;
            glGenBuffers(1, &level_ssbo);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, level_ssbo);
            glBufferData(GL_SHADER_STORAGE_BUFFER, n * sizeof(float), nullptr, GL_DYNAMIC_COPY);

            field_max_reduction_SSBOs.push_back(level_ssbo);
        }
    }

    // -------- max |field| with the EXISTING CFL_reduction kernel (it takes abs() itself). Same loop as the CFL
    //          reduction in the orchestrator, but the result is left on the GPU: the next kernel reads it directly
    glUseProgram(shadermanager.CFl_comp_program);

    GLuint reduction_input = field_ssbo;
    int n = total_elements;
    int level = 0;

    while (n > 1)
    {
        int groups = (n + 255) / 256;

        int gx, gy, gz;
        flat_groups_to_3d(groups, gx, gy, gz);

        glUniform1ui(shadermanager.cfl_elements_count_loc, (GLuint)n);
        glUniform1ui(shadermanager.cfl_total_groups_loc, (GLuint)groups);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, reduction_input);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, field_max_reduction_SSBOs[level]);

        glDispatchCompute(gx, gy, gz);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        reduction_input = field_max_reduction_SSBOs[level];
        n = groups;
        level++;
    }

    // (a field with a single element would skip the loop above; every real field has many elements)

    // -------- normalise inside the kernel and colour one slice into the ray-march image texture
    glUseProgram(shadermanager.field_normalized_slice_program);

    glUniform1i(shadermanager.field_slice_image_width_loc, fb_w);
    glUniform1i(shadermanager.field_slice_image_height_loc, fb_h);
    glUniform1i(shadermanager.field_slice_cells_x_loc, cells_x);
    glUniform1i(shadermanager.field_slice_cells_y_loc, cells_y);
    glUniform1i(shadermanager.field_slice_cells_z_loc, cells_z);
    glUniform1i(shadermanager.field_slice_topology_loc, int(topology));
    glUniform1i(shadermanager.field_slice_slice_id_loc, slice_id);

    glBindImageTexture(0, image_texture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, field_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, reduction_input);     // element 0 = max |field|

    glDispatchCompute((GLuint(fb_w) + 15u) / 16u, (GLuint(fb_h) + 15u) / 16u, 1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    draw_fullscreen_texture(shadermanager, image_texture, fb_w, fb_h);
}

void renderer_functions::dual_mode_render(int render_mode, ShaderManager &shadermanager, debuggers debug, int slice_id, int fb_w, int fb_h, float raymarch_ds, light &Light)
{
    if(render_mode == 0)
    {
        Light.dispatchLightTransmittance(shadermanager);

        ray_march_dispatch(shadermanager, fb_w, fb_h, raymarch_ds, Light);

        draw_fullscreen_texture(shadermanager, image_texture, fb_w, fb_h);
    }

}
