#pragma once

//------------------------------------------Numerical methods enums-------------------------------------
enum class TemporalScheme
{
    BACKWARD_EULER,
    BDF2
};

enum class ConvectionScheme
{
    UPWIND,         // first order scheme
    SECOND_ORDER_DEFERRED                   // easier architecture for 2nd order
};

enum class FluxLimiter
{
    NONE,
    MINMOD,
    VAN_LEER,
    SUPERBEE
};



struct configuration
{
    // grid / domain
    int cells_x;
    int cells_y;
    int cells_z;
    float cell_side_x;
    float cell_side_y;
    float cell_side_z;
    float WIDTH;
    float HEIGHT;
    float DEPTH;

    int frame_buffer_w;
    int frame_buffer_h;

    // fluid
    float density;
    float dynamic_viscosity;

    // pressure multigrid
    int pressure_coarsest_grid;             // coarsest resolution of the pressure V-cycle (cells per axis)

    // time stepping
    float dt_max;                           // upper bound on the time step when the fluid is slow / at rest
    float simulation_needed_time;           // the run ends once this much simulated time has passed
    float frames_per_sec_offline_sim;

    TemporalScheme temporal_scheme;         // Time advancement schemes
    ConvectionScheme advection_scheme;      //Spatial discreatization scheme
    FluxLimiter flux_limiter;               // Gradients limiter inside the advection schemes
};