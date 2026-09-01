"""Ion insertion helpers used by the final hydrogel packing stage.

The main routine in this module wraps repeated ``gmx grompp`` + ``gmx genion``
calls because the project frequently needs to insert several ion species with a
specific order:

1. Add staged anions one-by-one except the last species.
2. Add staged cations one-by-one except the last species.
3. Add the final cation/anion pair together with ``-neutral``.

That ordering keeps the topology file synchronized with the gradually updated
coordinate file while preserving user-configured ion identities.

Ownership and call sites
    Called by the build pipeline after water has been added (the counts that
    the water stage reserves for ion replacement come from this module's
    :func:`resolve_effective_ion_plan`, which
    :mod:`hygel_martini.hydrogel_builder.add_series.add_water` imports).

Inputs and outputs
    Inputs are the solvated ``.gro``, the ``.top`` topology, and the
    ``add_series_parameters.add_small_ion`` config block. Ions are inserted
    by replacing solvent molecules in place via ``gmx genion``, so the
    topology's ``[molecules]`` section is updated by GROMACS itself; a final
    reorder pass keeps the coordinate file consistent with that section.

Invariants
    * Ion entries carry ``ion_name``, integer ``charge`` (e), ``number``
      (molecule count), and an optional ``additional_add`` flag marking the
      species usable for charge compensation.
    * The compensation math is seeded from ``random_seed`` so the plan
      predicted up front matches the one executed here.
"""

import itertools
import os
import random
import shutil
import subprocess
import sys
import copy

import numpy as np

from hygel_martini.hydrogel_builder.config_params.config import Config
from hygel_martini.hydrogel_builder.core_utils.runtime.geo_opt import _run_with_logs


def _run_checked(cmd, label, cwd=None, env=None, input_text=None):
    """Run a subprocess through the shared logging wrapper.

    Args:
        cmd: Command argv list.
        label: Short tag used by the logging wrapper to name the step.
        cwd: Working directory for the subprocess.
        env: Environment mapping (e.g. with ``GMX_INCLUDE`` set).
        input_text: Text piped to stdin (used to answer genion's solvent
            group prompt).

    Returns:
        subprocess.CompletedProcess: The finished process.

    Raises:
        subprocess.CalledProcessError: If the command exits non-zero.
    """
    proc = _run_with_logs(cmd, label, log_path=None, cwd=cwd, env=env, input_text=input_text)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
    return proc


def _partition_ion_definitions(ion_list):
    """Split configured ions into primary and compensating pools.

    Ions flagged ``additional_add`` are set aside as the compensation pool
    whose counts may be increased later to neutralize the system; the rest
    are inserted exactly as configured. Zero-charge entries only contribute
    to the totals and land in no pool.

    Args:
        ion_list: List of ion config dicts (may be None).

    Returns:
        tuple: ``(cations, anions, extra_cations, extra_anions,
        total_charge)`` where the lists reference the original dicts and
        ``total_charge`` is the summed ``charge * number`` in e.
    """
    cations = []
    anions = []
    extra_cations = []
    extra_anions = []
    total_charge = 0.0

    for ion in ion_list or []:
        charge = float(ion.get("charge", 0.0) or 0.0)
        count = float(ion.get("number", 0) or 0)
        total_charge += charge * count

        if charge > 0:
            (extra_cations if ion.get("additional_add", False) else cations).append(ion)
        elif charge < 0:
            (extra_anions if ion.get("additional_add", False) else anions).append(ion)

    return cations, anions, extra_cations, extra_anions, total_charge


def _ensure_compensation_pool(primary_pool, compensation_pool, label):
    """Guarantee that a compensation pool contains at least one ion species.

    The final ``genion -neutral`` step needs one cation and one anion
    species, so if the user marked none as ``additional_add`` the first
    primary species is moved (popped, not copied) into the pool.

    Args:
        primary_pool: List of non-compensating ion dicts; mutated on demand.
        compensation_pool: List of compensating ion dicts; mutated on demand.
        label: ``"cation"`` or ``"anion"``, used in the error message.

    Raises:
        ValueError: If both pools are empty for this polarity.
    """
    if compensation_pool:
        return
    if not primary_pool:
        raise ValueError(f"No {label} ions are available for neutralization.")
    compensation_pool.append(primary_pool.pop(0))


def _apply_residual_charge(total_charge, compensation_pool, seed):
    """Increase compensation-ion counts so the staged system can be neutralized.

    Distributes ``|total_charge|`` across the pool in proportion to the
    configured counts (floored to integers). Any leftover charge that the
    floor step drops is covered by brute-force enumeration of small integer
    additions (each species count 0..max|charge| extra), with one valid
    combination chosen by a seeded RNG for reproducibility. Pool entries'
    ``number`` fields are mutated in place.

    Args:
        total_charge: Net system charge (e) that this pool must cancel; its
            sign is opposite to the pool's ion charges.
        compensation_pool: Ion dicts whose ``number`` may be increased.
        seed: Seed for the tie-breaking RNG.

    Raises:
        ValueError: If the pool's effective charge is zero, or the residual
            charge cannot be represented by the enumerated combinations.
    """
    if abs(total_charge) <= 1e-6 or not compensation_pool:
        return

    ratios = np.array(
        [float(ion.get("number", 0) or 0) for ion in compensation_pool],
        dtype=float,
    )
    charges = np.array(
        [float(ion.get("charge", 0.0) or 0.0) for ion in compensation_pool],
        dtype=float,
    )

    basic_charge = float(np.dot(ratios, charges))
    if abs(basic_charge) <= 1e-12:
        raise ValueError("Compensation-ion pool has zero effective charge.")

    total_multiplier = abs(total_charge / basic_charge)
    base_solution = np.floor(ratios * total_multiplier).astype(int)
    satisfied_charge = abs(float(np.dot(base_solution, charges)))
    remained_charge = abs(total_charge) - satisfied_charge

    if remained_charge > 1e-6:
        max_charge = max(int(abs(charge)) for charge in charges if abs(charge) > 0)
        valid_solutions = []
        for candidate in itertools.product(range(max_charge + 1), repeat=len(charges)):
            if abs(abs(float(np.dot(candidate, charges))) - remained_charge) < 1e-6:
                valid_solutions.append(candidate)

        if not valid_solutions:
            raise ValueError(
                "Could not represent the residual charge with the configured "
                "compensation ions."
            )

        rng = random.Random(seed)
        base_solution += np.array(rng.choice(valid_solutions), dtype=int)

    for idx, extra_count in enumerate(base_solution):
        compensation_pool[idx]["number"] += int(extra_count)


def resolve_effective_ion_plan(ion_params, seed=None):
    """Predict the effective ion counts after compensation adjustments.

    Runs the same partition/compensation pipeline that
    :func:`run_genion_for_neutralization` executes, but on a deep copy of
    the config, so callers (notably the water stage) can budget for the
    final ion counts without side effects.

    Args:
        ion_params: The ``add_small_ion`` config block (may be None/empty).
        seed: RNG seed for residual-charge tie-breaking; defaults to
            ``simulation_parameters.random_seed``.

    Returns:
        list[dict]: Copied ion dicts with adjusted ``number`` values, anions
        first then cations, each polarity ordered primaries-then-reversed-
        compensation to mirror the genion insertion order.
    """
    ion_list = copy.deepcopy((ion_params or {}).get("ions") or [])
    if not ion_list:
        return []

    (
        cation_list,
        anion_list,
        additional_cation_list,
        additional_anion_list,
        total_charge,
    ) = _partition_ion_definitions(ion_list)

    _ensure_compensation_pool(cation_list, additional_cation_list, "cation")
    _ensure_compensation_pool(anion_list, additional_anion_list, "anion")

    if seed is None:
        seed = Config.get_param("simulation_parameters").get("random_seed", 0)

    if total_charge > 1e-6:
        _apply_residual_charge(total_charge, additional_anion_list, seed)
    elif total_charge < -1e-6:
        _apply_residual_charge(total_charge, additional_cation_list, seed)

    additional_anion_list.reverse()
    additional_cation_list.reverse()
    anion_list.extend(additional_anion_list)
    cation_list.extend(additional_cation_list)
    return anion_list + cation_list


def run_genion_for_neutralization(input_gro, output_gro, topology_file, sim_params, ion_params, solvent_name):
    """Run GROMACS genion to add ions and neutralize the system.

    Executes the staged plan described in the module docstring: for every
    anion species but the last, then every cation species but the last, a
    minimal ``grompp`` + ``genion`` pair replaces solvent molecules in the
    current coordinate file; the last anion/cation pair is inserted together
    with ``-neutral`` so GROMACS absorbs any remaining net charge. The
    topology's ``[molecules]`` section is updated by genion itself. Finally
    the coordinate file is post-processed and its water/ion blocks reordered
    to match the topology.

    Args:
        input_gro (str): Path to the solvated input .gro file.
        output_gro (str): Path for the final output .gro file.
        topology_file (str): Path to the topology file (.top); it must
            already reside in ``sim_params['output_dir']``, where all
            commands run.
        sim_params (dict): Simulation parameters from config (output dir,
            GROMACS executable and include path, random seed).
        ion_params (dict): The ``add_small_ion`` config block. Its ion
            dicts are mutated in place by the compensation step.
        solvent_name (str): Solvent residue name, piped to genion's group
            prompt to choose which molecules get replaced.

    Returns:
        dict: ``{"output_gro": <path>, "ion_counts": {name: count}}`` where
        the counts reflect the compensation-adjusted plan.

    Raises:
        subprocess.CalledProcessError: If any grompp/genion call fails.
        ValueError: If neutralization is impossible with the configured
            ion species.
    """
    print("="*50)
    print("Running GROMACS genion to add ions and neutralize the system...")
    print("="*50)

    output_dir = sim_params['output_dir']
    gmx_exec_path = sim_params.get('gromacs_executable_path', 'gmx_mpi')
    gmx_include_path = sim_params.get('gromacs_include_path')

    # Make local copies of input files in the output directory to avoid path issues
    local_topo_file = os.path.basename(topology_file)
    # shutil.copy(topology_file, os.path.join(output_dir, local_topo_file)) # Removed this line

    current_gro_for_genion = "system_before_ions.gro"
    shutil.copy(input_gro, os.path.join(output_dir, current_gro_for_genion))


    # 2. Run gmx grompp to create a .tpr file
    env = os.environ.copy()
    if gmx_include_path:
        env['GMX_INCLUDE'] = gmx_include_path

    ion_list = ion_params.get("ions") or []
    (
        cation_list,
        anion_list,
        additional_cation_list,
        additional_anion_list,
        total_charge,
    ) = _partition_ion_definitions(ion_list)

    _ensure_compensation_pool(cation_list, additional_cation_list, "cation")
    _ensure_compensation_pool(anion_list, additional_anion_list, "anion")

    seed = Config.get_param("simulation_parameters").get("random_seed", 0)
    if total_charge > 1e-6:
        _apply_residual_charge(total_charge, additional_anion_list, seed)
    elif total_charge < -1e-6:
        _apply_residual_charge(total_charge, additional_cation_list, seed)
    additional_anion_list.reverse()
    additional_cation_list.reverse()
    anion_list.extend(additional_anion_list)
    cation_list.extend(additional_cation_list)
    genion_count = 0
    # Stage 1: insert every anion species except the last, one grompp+genion
    # pair per species, threading the growing .gro through each iteration.
    for i in range(len(anion_list)-1):
        temp_mdp_file = os.path.join(output_dir, "temp_for_genion.mdp")
        temp_tpr_file = os.path.join(output_dir, "temp_for_genion.tpr")

        # 1. Create a minimal .mdp file for grompp
        with open(temp_mdp_file, 'w') as f:
            f.write("title       = Minimal MDP for grompp\nintegrator  = steep\nnsteps      = 0\n")

        # 2. Run gmx grompp to create a .tpr file

        grompp_command = [
            gmx_exec_path, 'grompp',
            '-f', os.path.basename(temp_mdp_file),
            '-c', current_gro_for_genion,
            '-p', local_topo_file,
            '-o', os.path.basename(temp_tpr_file),
            '-maxwarn', '2' # Allow warnings for box size and charge
        ]
        _run_checked(grompp_command, "grompp", cwd=output_dir, env=env)
        
        # 3. Run gmx genion
        next_gro_file = f"{os.path.splitext(output_gro)[0]}_{genion_count}.gro"
        genion_command = [
            gmx_exec_path, 'genion',
            '-s', os.path.basename(temp_tpr_file),
            '-o', next_gro_file,
            '-p', local_topo_file,
            '-nname', anion_list[i]["ion_name"],
            '-nn', str(int(anion_list[i]["number"])),
            '-nq', str(int(anion_list[i]["charge"])),
        ]

        _run_checked(genion_command, "genion", cwd=output_dir, env=env, input_text=solvent_name)
        current_gro_for_genion = next_gro_file
        # 4. Clean up temporary files
        print("\nCleaning up temporary files...")
        for temp_file in [temp_mdp_file, temp_tpr_file, os.path.join(output_dir, "mdout.mdp")]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        genion_count += 1
    # Stage 2: same one-by-one insertion for every cation species but the
    # last.
    for i in range(len(cation_list)-1):
        temp_mdp_file = os.path.join(output_dir, "temp_for_genion.mdp")
        temp_tpr_file = os.path.join(output_dir, "temp_for_genion.tpr")

        # 1. Create a minimal .mdp file for grompp
        with open(temp_mdp_file, 'w') as f:
            f.write("title       = Minimal MDP for grompp\nintegrator  = steep\nnsteps      = 0\n")

        # 2. Run gmx grompp to create a .tpr file

        grompp_command = [
            gmx_exec_path, 'grompp',
            '-f', os.path.basename(temp_mdp_file),
            '-c', current_gro_for_genion,
            '-p', local_topo_file,
            '-o', os.path.basename(temp_tpr_file),
            '-maxwarn', '2' # Allow warnings for box size and charge
        ]
        _run_checked(grompp_command, "grompp", cwd=output_dir, env=env)
            
        # 3. Run gmx genion
        next_gro_file = f"{os.path.splitext(output_gro)[0]}_{genion_count}.gro"
        genion_command = [
            gmx_exec_path, 'genion',
            '-s', os.path.basename(temp_tpr_file),
            '-o', next_gro_file,
            '-p', local_topo_file,
            '-pname', cation_list[i]["ion_name"],
            '-np', str(int(cation_list[i]["number"])),
            '-pq', str(int(cation_list[i]["charge"])),
        ]

        _run_checked(genion_command, "genion", cwd=output_dir, env=env, input_text=solvent_name)
        current_gro_for_genion = next_gro_file
        # 4. Clean up temporary files
        print("\nCleaning up temporary files...")
        for temp_file in [temp_mdp_file, temp_tpr_file, os.path.join(output_dir, "mdout.mdp")]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        genion_count += 1
    # Stage 3: insert the final anion/cation pair together with -neutral so
    # genion cancels any charge left over from the staged insertions.
    temp_mdp_file = os.path.join(output_dir, "temp_for_genion.mdp")
    temp_tpr_file = os.path.join(output_dir, "temp_for_genion.tpr")

    # 1. Create a minimal .mdp file for grompp
    with open(temp_mdp_file, 'w') as f:
        f.write("title       = Minimal MDP for grompp\nintegrator  = steep\nnsteps      = 0\n")

    grompp_command = [
        gmx_exec_path, 'grompp',
        '-f', os.path.basename(temp_mdp_file),
        '-c', current_gro_for_genion,
        '-p', local_topo_file,
        '-o', os.path.basename(temp_tpr_file),
        '-maxwarn', '2' # Allow warnings for box size and charge
    ]
    _run_checked(grompp_command, "grompp", cwd=output_dir, env=env)
    final_gro = f"{os.path.splitext(output_gro)[0]}_end.gro"
    # 3. Run gmx genion
    genion_command = [
        gmx_exec_path, 'genion',
        '-s', os.path.basename(temp_tpr_file),
        '-o', final_gro,
        '-p', local_topo_file,
        '-nname', anion_list[-1]["ion_name"],
        '-nn', str(int(anion_list[-1]["number"])),
        '-nq', str(int(anion_list[-1]["charge"])),
        '-pname', cation_list[-1]["ion_name"],
        '-np', str(int(cation_list[-1]["number"])),
        '-pq', str(int(cation_list[-1]["charge"])),
        '-neutral'
    ]

    _run_checked(genion_command, "genion", cwd=output_dir, env=env, input_text=solvent_name)

    # Post-process the gro file: keep original ion names and reorder
    final_gro_path = os.path.join(output_dir, final_gro)

    try:
        with open(final_gro_path, 'r') as f:
            lines = f.readlines()

        if len(lines) >= 3:
            new_lines = lines[:2]  # Title and atom count

            for line in lines[2:-1]:  # Atom lines
                new_lines.append(line)

            new_lines.append(lines[-1])  # Box vectors

            with open(final_gro_path, 'w') as f:
                f.writelines(new_lines)
            
            # Copy the final processed file to the originally requested output_gro path
            shutil.copy(final_gro_path, output_gro)

            print(f"Final structure with ions saved to '{output_gro}'.")
            Config.debug_log(f"Ion post-processing complete: {output_gro}")

            # Reorder water and ions to match [molecules] ordering (water first, then ions)
            ion_name_order = [ion.get('ion_name') for ion in ion_params.get('ions', []) if ion.get('ion_name')]
            _reorder_water_and_ions(output_gro, solvent_name, ion_name_order)


    except FileNotFoundError:
        print(f"Warning: Output file '{final_gro_path}' not found for post-processing.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during .gro file post-processing: {e}", file=sys.stderr)

    # 4. Clean up temporary files
    print("\nCleaning up temporary files...")
    for temp_file in [temp_mdp_file, temp_tpr_file, os.path.join(output_dir, "mdout.mdp")]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("\n" + "="*50)
    print(f"Ion addition complete. Final file: '{output_gro}'")
    print("="*50)
    ion_counts_summary = {}
    for ion in ion_params.get('ions', []):
        ion_name = ion.get('ion_name')
        number = int(ion.get('number', 0) or 0)
        if ion_name and number > 0:
            ion_counts_summary[ion_name] = ion_counts_summary.get(ion_name, 0) + number

    return {"output_gro": output_gro, "ion_counts": ion_counts_summary}


def _reorder_water_and_ions(gro_path, water_resname, ion_names):
    """Reorder a GRO file so water and ions follow topology ordering.

    genion leaves inserted ions at the coordinate positions of the waters
    they replaced, which scatters them through the water block. This pass
    rewrites the file as ``[everything else] + [waters] + [ions in config
    order]`` so the atom order matches the topology's ``[molecules]``
    section. Ions that genion emitted with the generic ``ION`` residue name
    are renamed back to their specific ion name (matched via the atom-name
    column). Failures are logged but never raised, leaving the file as-is.

    Args:
        gro_path: Path of the GRO file to rewrite in place.
        water_resname: Residue name of the solvent block.
        ion_names: Ion residue names in the desired (config) order.
    """
    try:
        with open(gro_path, 'r') as f:
            lines = f.readlines()
        if len(lines) < 3:
            return
        title = lines[0]
        natoms = int(lines[1].strip())
        atom_lines = lines[2:2 + natoms]
        box_line = lines[2 + natoms] if len(lines) >= 3 + natoms else lines[-1]

        prefix = []
        waters = []
        ions_by_name = {name: [] for name in ion_names}
        for line in atom_lines:
            resname = line[5:10].strip()
            atomname = line[10:15].strip()
            # Normalize ions that come back with generic 'ION' resname by using atom name
            if resname == 'ION' and atomname in ion_names:
                resname_use = atomname
                # rewrite resname field to the specific ion name for consistency with topology
                line = line[:5] + f"{resname_use:<5}" + line[10:]
            else:
                resname_use = resname
            if resname_use == water_resname:
                waters.append(line)
            elif resname_use in ion_names:
                ions_by_name[resname_use].append(line)
            else:
                prefix.append(line)

        reordered = prefix + waters
        for name in ion_names:
            reordered.extend(ions_by_name.get(name, []))

        with open(gro_path, 'w') as f:
            f.write(title)
            f.write(f"{natoms:5d}\n")
            for line in reordered:
                f.write(line)
            f.write(box_line)
        print(f"[INFO] Reordered water/ion blocks in {gro_path}")
        Config.debug_log(f"[INFO] Reordered waters/ions in {gro_path}")
    except Exception as exc:
        print(f"[WARN] Failed to reorder {gro_path}: {exc}", file=sys.stderr)
        Config.debug_log(f"[WARN] Failed to reorder {gro_path}: {exc}")
