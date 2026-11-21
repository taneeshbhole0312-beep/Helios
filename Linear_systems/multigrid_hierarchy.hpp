#pragma once

#include <vector>

// Geometry of one multigrid level (level 0 = the finest, simulation grid).
// Cell counts are the CELL counts of the level; a staggered field derives its element counts from them
// (XFACE: cells_x + 1, ...), see LinearSystem.
struct GridLevelInfo
{
    int cells_x;
    int cells_y;
    int cells_z;

    // uniform spacing of this level: domain_size / cells (never "previous spacing * coarsening factor",
    // which is wrong as soon as the cell count does not divide exactly, e.g. 125 -> 62)
    float dx;
    float dy;
    float dz;
};

//=================================================================================================================================
// SINGLE SOURCE OF TRUTH FOR THE MULTIGRID LEVEL GEOMETRY
//
// Every multigrid-related component (LinearSystem, common_stencil, RANS hierarchies, the turbulence hierarchies, the generic and
// the pressure multigrid) takes its level count, cell counts and spacings from one instance of this class.
//
// Level l + 1 is built from level l by SEMI-COARSENING: an axis is coarsened with N(l+1) = floor(N(l) / coarsening_factor) while it can
// stay at or above min_coarse_cells, and only the axes whose spacing is within sqrt(factor) of the smallest spacing of ALL axes are coarsened at a given step;
// an axis that cannot (or should not yet) coarsen keeps its cell count (transfer ratio 1 on that axis). A step that would make the coarse spacings more anisotropic than the current level (beyond sqrt(factor)) is not taken. Generation stops when no axis can (and should) be coarsened. For an isotropic grid every axis coarsens together, which reproduces the previous all-axes behaviour.
//
// Immutable after construction.
//=================================================================================================================================
class MultigridHierarchy
{
    public:
        MultigridHierarchy(int cells_x, int cells_y, int cells_z, float domain_size_x, float domain_size_y, float domain_size_z, int min_coarse_cells, int coarsening_factor = 2);

        // level 0 = finest
        const GridLevelInfo &level(int level_index) const;

        int level_count() const;            // finest + coarse levels
        int coarse_level_count() const;     // number of restriction steps (the old "passes")

        float domain_size_x() const;
        float domain_size_y() const;
        float domain_size_z() const;

    private:
        std::vector <GridLevelInfo> levels;

        float domain_x;
        float domain_y;
        float domain_z;
};
