"""Post-build solvation stage: add water and small ions to a built hydrogel.

Submodules:
    add_water: estimates the number of CG water beads needed to reach the
        configured gel weight fraction (mass-balance model).
    add_small_ion: inserts ion species and neutralizes the system through
        staged ``gmx genion`` runs.

The bundled ``water.gro`` / ``water.itp`` files are the reference Martini
water templates used by the solvation step.
"""
