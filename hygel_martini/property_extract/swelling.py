"""Swelling/composition analysis from GROMACS topology and energy files.

Owns :class:`SwellingAnalyzer`, which combines bead counts parsed from
``.top``/``.itp`` files with the box-volume time series of an
``energy.xvg`` to report two observables through
:class:`.result.PropertyResult`:

* ``loading_qm`` — count-based initial composition ratio, always
  computable without any MD output
  (``validation_role="composition_check_only"``, direct experimental
  comparison revoked because a fixed-water NPT box is not equilibrium
  swelling);
* ``polymer_volume_fraction`` — bead-volume estimate against the
  time-averaged box volume (``validation_role="direct"``).

Called by the extractor adapters (:mod:`.extractors.composition`,
:mod:`.extractors.swelling`) and usable directly.  Missing Volume
columns or empty averaging windows raise ``ValueError`` here; the
extractor layer converts those into refusal statuses.
"""
import os
import re
import numpy as np
from .gmx_utils import parse_xvg
from .result import PropertyResult


class SwellingAnalyzer:
    """Composition and volume-fraction calculator for one built system.

    Holds the as-built bead counts and per-bead constants; methods
    combine them with an optional energy time series.  Construct
    directly from counts, or via :meth:`from_files` to parse counts
    out of top/itp files.
    """

    def __init__(
        self,
        n_polymer_beads,
        n_solvent_beads,
        polymer_bead_mass=45.0,
        solvent_bead_mass=72.0,
        polymer_bead_vol_nm3=0.065,
    ):
        """Store bead counts and per-bead mass/volume constants.

        Args:
            n_polymer_beads: Polymer bead count (e.g. EO beads).
            n_solvent_beads: Solvent bead count (e.g. Martini W).
            polymer_bead_mass: Mass per polymer bead in amu.
                Default 45.0 (Martini 3 PEO SN3r).
            solvent_bead_mass: Mass per solvent bead in amu.
                Default 72.0 (Martini W = 4 x 18).
            polymer_bead_vol_nm3: Approximate volume per polymer bead
                in nm^3.  Default 0.065 (force-field dependent).
        """
        self.n_polymer_beads = n_polymer_beads
        self.n_solvent_beads = n_solvent_beads
        self.polymer_bead_mass = polymer_bead_mass
        self.solvent_bead_mass = solvent_bead_mass
        self.polymer_bead_vol_nm3 = polymer_bead_vol_nm3

    def composition_summary(self) -> PropertyResult:
        """Summarize the as-built composition without any MD output.

        Always computable — needs no energy XVG.  ``loading_qm`` is a
        count-based initial composition ratio; in a fixed-water NPT
        simulation it differs from the experimental equilibrium Qm, so
        the result revokes direct experimental comparison and carries
        ``validation_role="composition_check_only"``.

        Returns:
            Computed PropertyResult ``loading_qm`` with the bead counts
            and interpretation note in metadata.
        """
        loading_qm = self.calculate_loading_qm()
        return PropertyResult(
            property="loading_qm",
            value=loading_qm,
            status="computed",
            direct_experiment_comparison_allowed=False,
            validation_role="composition_check_only",
            metadata={
                "note": (
                    "count-based initial composition ratio; "
                    "not equilibrium swelling (fixed-water NPT)"
                ),
                "n_polymer_beads": self.n_polymer_beads,
                "n_solvent_beads": self.n_solvent_beads,
            },
        )

    def calculate_loading_qm(self) -> float:
        """loading_qm = (m_polymer + m_solvent) / m_polymer"""
        mass_dry = self.n_polymer_beads * self.polymer_bead_mass
        mass_wet = mass_dry + self.n_solvent_beads * self.solvent_bead_mass
        return mass_wet / mass_dry

    def calculate_phi(self, box_volume_nm3: float) -> float:
        """polymer volume fraction phi = V_polymer / V_box"""
        v_poly = self.n_polymer_beads * self.polymer_bead_vol_nm3
        return v_poly / box_volume_nm3

    def analyze_trajectory(self, energy_xvg, start_time_ps=0) -> PropertyResult:
        """Compute polymer_volume_fraction from an energy XVG's Volume.

        Averages the Volume column (nm^3) over frames at or after
        ``start_time_ps`` and converts it to a volume fraction via
        :meth:`calculate_phi`.  The reported ``phi_std`` propagates the
        volume standard deviation through phi = V_poly / V_box.

        Args:
            energy_xvg: Path to a ``gmx energy`` XVG containing a
                Volume column (matched by substring).
            start_time_ps: Discard frames before this time (ps).

        Returns:
            Computed PropertyResult ``polymer_volume_fraction``
            (``validation_role="direct"``, direct comparison allowed).

        Raises:
            ValueError: No Volume column in the file, or no frames at
                or after ``start_time_ps``.
        """
        data = parse_xvg(energy_xvg)
        times = data['time']

        vol_key = next((k for k in data if 'Volume' in k), None)
        if vol_key is None:
            raise ValueError(f"Volume 데이터를 찾을 수 없습니다: {energy_xvg}")

        volumes = data[vol_key]
        mask = times >= start_time_ps
        if mask.sum() == 0:
            raise ValueError(
                f"start_time_ps={start_time_ps} ps 이후 데이터가 없습니다. "
                f"trajectory 마지막 시간: {times[-1]} ps"
            )

        avg_vol = np.mean(volumes[mask])
        std_vol = np.std(volumes[mask])
        phi = self.calculate_phi(avg_vol)
        phi_std = self.polymer_bead_vol_nm3 * self.n_polymer_beads * (std_vol / avg_vol ** 2)

        return PropertyResult(
            property="polymer_volume_fraction",
            value=phi,
            status="computed",
            direct_experiment_comparison_allowed=True,
            validation_role="direct",
            metadata={
                "phi_std": phi_std,
                "method": f"bead_volume (vol_per_bead={self.polymer_bead_vol_nm3} nm3)",
                "vol_avg_nm3": avg_vol,
                "vol_std_nm3": std_vol,
                "start_time_ps": start_time_ps,
                "n_frames_used": int(mask.sum()),
            },
        )

    @classmethod
    def from_files(
        cls,
        top_file,
        itp_file,
        polymer_bead_mass=45.0,
        solvent_bead_mass=72.0,
        polymer_bead_vol_nm3=0.065,
        polymer_residue_name=None,
        polymer_atom_name=None,
        solvent_molecule_names='W',
    ):
        """Build an analyzer by counting beads in top/itp files.

        Solvent beads are counted from the ``[ molecules ]`` section of
        ``top_file`` (summing the counts of matching molecule names);
        polymer beads from the ``[ atoms ]`` section of ``itp_file``
        (a missing itp yields a zero polymer count rather than an
        error).

        Args:
            top_file: GROMACS ``.top`` with a ``[ molecules ]`` section.
            itp_file: Polymer ``.itp`` whose ``[ atoms ]`` lines are
                filtered by residue/atom name.
            polymer_bead_mass: Mass per polymer bead (amu).
            solvent_bead_mass: Mass per solvent bead (amu).
            polymer_bead_vol_nm3: Volume per polymer bead (nm^3).
            polymer_residue_name: Residue name to select in
                ``[ atoms ]``; None counts every atom line.
            polymer_atom_name: Atom name to select in ``[ atoms ]``;
                None counts every atom line.
            solvent_molecule_names: Solvent molecule name(s) in
                ``[ molecules ]``; str or list of str.

        Returns:
            A SwellingAnalyzer initialized with the parsed counts.
        """
        if isinstance(solvent_molecule_names, str):
            solvent_molecule_names = [solvent_molecule_names]
        solvent_name_set = set(solvent_molecule_names)

        n_solvent = 0
        in_molecules = False
        with open(top_file, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(';') or stripped == '':
                    continue
                if stripped.startswith('['):
                    m = re.search(r'\[\s*(\w+)\s*\]', stripped)
                    in_molecules = bool(m and m.group(1) == 'molecules')
                    continue
                if in_molecules:
                    parts = stripped.split()
                    if len(parts) >= 2 and parts[0] in solvent_name_set:
                        n_solvent += int(parts[1])

        # ITP atoms line: nr(0) type(1) resnr(2) residu(3) atom(4) cgnr(5) charge(6) [mass(7)]
        n_polymer = 0
        if os.path.exists(itp_file):
            with open(itp_file, 'r') as f:
                section = None
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith(';') or stripped == '':
                        continue
                    if stripped.startswith('['):
                        m = re.search(r'\[\s*(\w+)\s*\]', stripped)
                        section = m.group(1) if m else None
                        continue
                    if section == 'atoms':
                        parts = stripped.split()
                        if len(parts) < 5:
                            continue
                        residu = parts[3]
                        atom = parts[4]
                        residue_ok = (polymer_residue_name is None) or (residu == polymer_residue_name)
                        atom_ok = (polymer_atom_name is None) or (atom == polymer_atom_name)
                        if residue_ok and atom_ok:
                            n_polymer += 1

        return cls(
            n_polymer_beads=n_polymer,
            n_solvent_beads=n_solvent,
            polymer_bead_mass=polymer_bead_mass,
            solvent_bead_mass=solvent_bead_mass,
            polymer_bead_vol_nm3=polymer_bead_vol_nm3,
        )
