// Runs the UNMODIFIED Orchestrator::run_rans_k_eps for a bounded simulated time, with main.cpp's configuration.
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "orchestrator/orchestrator.hpp"

int main(int argc, char **argv)
{
    float U = argc > 1 ? (float)atof(argv[1]) : 10.0f, I = argc > 2 ? (float)atof(argv[2]) : 0.05f, L = argc > 3 ? (float)atof(argv[3]) : 1.0f;
    float steps = argc > 4 ? (float)atof(argv[4]) : 1.0f;
    float dt = argc > 5 ? (float)atof(argv[5]) : 1.0f / 60.0f;
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow *win = glfwCreateWindow(64, 64, "orch", nullptr, nullptr);
    glfwMakeContextCurrent(win); gladLoadGL();
    ShaderManager sm;
    configuration c{};
    c.cells_x = c.cells_y = c.cells_z = 32;
    c.WIDTH = c.HEIGHT = c.DEPTH = 200.0f;
    c.cell_side_x = c.cell_side_y = c.cell_side_z = 200.0f / 32.0f;
    c.frame_buffer_w = c.frame_buffer_h = 64;
    c.density = 1.225f; c.dynamic_viscosity = 1.8e-5f;
    c.V_cycles_count = 2; c.pressure_coarsest_grid = 8;
    c.dt_max = dt;
    c.simulation_needed_time = steps * dt * 0.999f;     // stop after `steps` physical steps
    std::cout << "ORCH CONFIG U=" << U << " I=" << I << " L=" << L << " dt=" << dt << " steps=" << steps << std::endl;
    Orchestrator o(c, sm);
    o.run_rans_k_eps(9.8f, U, I, L);
    std::cout << "ORCH DONE glGetError=0x" << std::hex << glGetError() << std::endl;
    glfwTerminate();
}
