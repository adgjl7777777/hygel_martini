import os
from config_params.read_json import Config
from core_utils import packer, topology_updater

def calculate_water_molecules(mode):
    """
    Calculates the number of water molecules to add based on the configuration.
    """
    # --- Constants from the original script ---
    MASS_MONOMER = 232
    MASS_BIS = (297+34)*2
    MASS_WATER = 72

    sim_params = Config.get_param('simulation_parameters')
    nmer = sim_params['segment_length']
    gel_wt = sim_params['gel_weight_fraction']
    num_cell = sim_params['number_of_cells']

    print(f"Calculation mode: {mode}")
    print(f"Nmer: {nmer}, Gel weight fraction: {gel_wt}, NumCell: {num_cell}")

    unit_mer = nmer * 4 * 4
    tot_mer = unit_mer * (num_cell / 2)**3
    pol_mass = tot_mer * MASS_MONOMER

    if mode == 'full':
        tot_bis_mass = 8 * (num_cell / 2)**3 * MASS_BIS
        total_gel_mass = pol_mass + tot_bis_mass
    else: # monomer_only
        total_gel_mass = pol_mass

    if not (0 < gel_wt < 1):
        raise ValueError("Error: gel_weight_fraction must be between 0 and 1.")
    
    water_mass = (total_gel_mass / gel_wt) - total_gel_mass
    n_water = int(water_mass / MASS_WATER)
    
    print(f"Total gel mass: {total_gel_mass:.2f}")
    print(f"Required water mass: {water_mass:.2f}")
    print(f"Number of water molecules to add: {n_water}")

    return n_water