#include "multigrid_hierarchy.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

MultigridHierarchy::MultigridHierarchy(int cells_x, int cells_y, int cells_z, float domain_size_x, float domain_size_y, float domain_size_z, int min_coarse_cells, int coarsening_factor)
:   domain_x(domain_size_x),
    domain_y(domain_size_y),
    domain_z(domain_size_z)
{
    // ---------------------------------------- validation
    if (cells_x <= 0 || cells_y <= 0 || cells_z <= 0)
        throw std::invalid_argument("MultigridHierarchy: cell counts must be positive");

    if (!(std::isfinite(domain_size_x) && domain_size_x > 0.0f) ||
        !(std::isfinite(domain_size_y) && domain_size_y > 0.0f) ||
        !(std::isfinite(domain_size_z) && domain_size_z > 0.0f))
        throw std::invalid_argument("MultigridHierarchy: domain sizes must be positive finite numbers");

    if (coarsening_factor < 2)
        throw std::invalid_argument("MultigridHierarchy: the coarsening factor must be an integer >= 2");

    if (min_coarse_cells < 1)
        throw std::invalid_argument("MultigridHierarchy: min_coarse_cells must be >= 1");

    // ---------------------------------------- level generation (iterative, no logarithms)
    GridLevelInfo finest;
    finest.cells_x = cells_x;
    finest.cells_y = cells_y;
    finest.cells_z = cells_z;
    finest.dx = domain_size_x / cells_x;
    finest.dy = domain_size_y / cells_y;
    finest.dz = domain_size_z / cells_z;

    levels.push_back(finest);

    while (true)
    {
        const GridLevelInfo &fine = levels.back();


        const int fine_cells[3]   = { fine.cells_x, fine.cells_y, fine.cells_z };
        const float fine_spacing[3] = { fine.dx, fine.dy, fine.dz };

        // smallest spacing over ALL axes, also over those that can no longer coarsen: once the strongly coupled axis is exhausted, the
        // weakly coupled ones are not coarsened further (that would only create a more and more anisotropic coarse grid)
        float smallest_spacing = std::min(fine_spacing[0], std::min(fine_spacing[1], fine_spacing[2]));
        const float spacing_tolerance = std::sqrt((float)coarsening_factor);        // Not a strict mathematical threshold, a heuristic one

        int next[3];
        bool any_axis = false;

        for (int axis = 0; axis < 3; axis++)
        {
            const bool can_coarsen = (fine_cells[axis] / coarsening_factor) >= min_coarse_cells;
            const bool coarsen_axis = can_coarsen && fine_spacing[axis] <= spacing_tolerance * smallest_spacing * 1.0001f;

            next[axis] = coarsen_axis ? fine_cells[axis] / coarsening_factor : fine_cells[axis];
            any_axis = any_axis || coarsen_axis;
        }

        if (!any_axis)
            break;                                              // nothing left that can and should be coarsened

        float new_smallest = 1e30f, new_largest = 0.0f;

        for (int axis = 0; axis < 3; axis++)
        {
            const float new_spacing = (axis == 0 ? domain_size_x : axis == 1 ? domain_size_y : domain_size_z) / next[axis];
            new_smallest = std::min(new_smallest, new_spacing);
            new_largest  = std::max(new_largest,  new_spacing);
        }

        const float fine_ratio = std::max(fine_spacing[0], std::max(fine_spacing[1], fine_spacing[2])) / smallest_spacing;

        if (new_largest / new_smallest > std::max(fine_ratio, spacing_tolerance) * 1.0001f)
            break;

        GridLevelInfo coarse;
        coarse.cells_x = next[0];
        coarse.cells_y = next[1];
        coarse.cells_z = next[2];
        coarse.dx = domain_size_x / next[0];
        coarse.dy = domain_size_y / next[1];
        coarse.dz = domain_size_z / next[2];

        levels.push_back(coarse);
    }

    // Both multigrid solvers start every V-cycle with a restriction to level 1, so a grid that cannot be coarsened once is an
    // invalid configuration, not a silent single-level solve.
    if (levels.size() < 2)
        throw std::invalid_argument("MultigridHierarchy: " + std::to_string(cells_x) + " x " + std::to_string(cells_y) + " x " + std::to_string(cells_z) +
                                    " cannot be coarsened even once with factor " + std::to_string(coarsening_factor) +
                                    " and min_coarse_cells " + std::to_string(min_coarse_cells));
}

const GridLevelInfo &MultigridHierarchy::level(int level_index) const
{
    return levels.at(level_index);
}

int MultigridHierarchy::level_count() const
{
    return static_cast<int>(levels.size());
}

int MultigridHierarchy::coarse_level_count() const
{
    return static_cast<int>(levels.size()) - 1;
}

float MultigridHierarchy::domain_size_x() const { return domain_x; }
float MultigridHierarchy::domain_size_y() const { return domain_y; }
float MultigridHierarchy::domain_size_z() const { return domain_z; }
