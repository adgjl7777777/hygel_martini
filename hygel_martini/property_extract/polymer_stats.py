"""Chain-level conformation statistics via MDAnalysis.

Owns :class:`PolymerStats`, which loads a structure (and optional
trajectory) into an MDAnalysis Universe and reports radius of
gyration, per-chain end-to-end distance, and a nearest-neighbor
bond-correlation persistence-length estimate.  Distances are in the
Universe's native units (Angstrom for .gro input).

Unlike the gated extractors, this module returns raw numpy values, not
PropertyResult objects, and degrades with warnings instead of refusal
statuses: without bond information (bare .gro input) chains fall back
to residue-based splitting and the persistence length degrades to 0.0
with a UserWarning.
"""
import warnings
import numpy as np
import MDAnalysis as mda
try:
    # Older/newer MDAnalysis layouts may not expose NoDataError; fall
    # back to Exception so the except clauses below stay valid.
    from MDAnalysis.exceptions import NoDataError
except ImportError:
    NoDataError = Exception

class PolymerStats:
    """Conformation statistics for the selected polymer atoms.

    Wraps one MDAnalysis Universe; every method iterates the loaded
    trajectory (or the single frame when only a structure file was
    given).
    """

    def __init__(self, gro_file, traj_file=None,
                 selection="resname PEO or resname HYDROGEL"):
        """Load the system and select the polymer atoms.

        Args:
            gro_file: Structure/topology file for the Universe.
            traj_file: Optional trajectory; omitted means single-frame
                analysis on ``gro_file``.
            selection: MDAnalysis selection string for polymer atoms.
                Default ``'resname PEO or resname HYDROGEL'`` — set it
                explicitly for your system (e.g.
                ``"resname PEO and name EO"``).
        """
        self.u = mda.Universe(gro_file, traj_file) if traj_file else mda.Universe(gro_file)
        self.polymer = self.u.select_atoms(selection)

    def calculate_rg(self):
        """Calculate the radius of gyration of the whole selection.

        Returns:
            1-D array with one Rg value per trajectory frame (a single
            value for structure-only input), in the Universe's distance
            unit.
        """
        rg_list = []
        if hasattr(self.u.trajectory, 'ts'):
            for ts in self.u.trajectory:
                rg_list.append(self.polymer.atoms.radius_of_gyration())
        else:
            rg_list.append(self.polymer.atoms.radius_of_gyration())
        return np.array(rg_list)

    def calculate_end_to_end(self):
        """Calculate the mean per-chain end-to-end distance per frame.

        Chains are the bonded fragments of the selection.  When bond
        information is absent or the selection is one giant fragment
        (typical for bare .gro input or a fully crosslinked network),
        it warns and splits by residue instead — inaccurate for
        crosslinked networks.  End-to-end uses the first and last atom
        of each chain in file order; chains shorter than 2 atoms are
        skipped.

        Returns:
            1-D array of frame-wise mean end-to-end distances (frames
            with no valid chain are omitted), in the Universe's
            distance unit.
        """
        try:
            chains = self.polymer.fragments
            if len(chains) == 0 or len(chains[0]) == len(self.polymer):
                raise NoDataError("single fragment — fallback to residue split")
        except NoDataError:
            warnings.warn(
                ".gro 단독 입력: bond 정보 없어 residue 단위로 chain을 분리합니다. "
                "crosslinked network에서는 부정확할 수 있습니다.",
                UserWarning,
            )
            chains = self.polymer.split('residue')

        results = []
        for ts in self.u.trajectory:
            chain_dists = []
            for fragment in chains:
                if len(fragment) < 2:
                    continue
                start = fragment.atoms.positions[0]
                end = fragment.atoms.positions[-1]
                dist = np.linalg.norm(end - start)
                chain_dists.append(dist)
            if chain_dists:
                results.append(np.mean(chain_dists))
        return np.array(results)

    def estimate_persistence_length(self):
        """Estimate the persistence length from bond-bond correlation.

        Uses the worm-like-chain relation ``<cos(theta)> = exp(-s/Lp)``
        evaluated only at nearest-neighbor bond pairs of the current
        frame:  ``Lp = -<l_bond> / ln(<cos theta>)`` per fragment,
        averaged over fragments.  Fragments with ``<cos theta> <= 0``
        contribute nothing (the log would be undefined).

        Returns:
            Mean Lp over fragments in the Universe's distance unit, or
            0.0 (with a UserWarning) when bond information is missing
            or no fragment yields a positive correlation.
        """
        # Simplified version: Lp = <R^2> / (2 * L_contour) for worm-like chain in limit
        # Or direct correlation:
        try:
            frags = self.polymer.fragments
        except NoDataError:
            warnings.warn("bond 정보 없어 persistence length를 계산할 수 없습니다.", UserWarning)
            return 0.0
        lps = []
        for fragment in frags:
            positions = fragment.atoms.positions
            bonds = positions[1:] - positions[:-1]
            bond_lengths = np.linalg.norm(bonds, axis=1)
            avg_bond_len = np.mean(bond_lengths)
            
            # Normalize bonds to unit vectors
            u_bonds = bonds / bond_lengths[:, np.newaxis]
            
            # Correlation for nearest neighbors
            dot_products = np.sum(u_bonds[:-1] * u_bonds[1:], axis=1)
            avg_cos = np.mean(dot_products)
            
            if avg_cos > 0:
                lp = -avg_bond_len / np.log(avg_cos)
                lps.append(lp)
        
        return np.mean(lps) if lps else 0.0
