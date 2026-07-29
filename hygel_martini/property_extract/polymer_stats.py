import warnings
import numpy as np
import MDAnalysis as mda
try:
    from MDAnalysis.exceptions import NoDataError
except ImportError:
    NoDataError = Exception

class PolymerStats:
    def __init__(self, gro_file, traj_file=None,
                 selection="resname PEO or resname HYDROGEL"):
        """
        selection: MDAnalysis selection string for polymer atoms.
                   기본값은 'resname PEO or resname HYDROGEL'.
                   시스템에 맞게 명시적으로 지정할 것 (예: "resname PEO and name EO").
        """
        self.u = mda.Universe(gro_file, traj_file) if traj_file else mda.Universe(gro_file)
        self.polymer = self.u.select_atoms(selection)

    def calculate_rg(self):
        """Calculate Radius of Gyration over the trajectory."""
        rg_list = []
        if hasattr(self.u.trajectory, 'ts'):
            for ts in self.u.trajectory:
                rg_list.append(self.polymer.atoms.radius_of_gyration())
        else:
            rg_list.append(self.polymer.atoms.radius_of_gyration())
        return np.array(rg_list)

    def calculate_end_to_end(self):
        """Calculate end-to-end distance for individual chains."""
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
        """
        Estimate Persistence Length (Lp) from bond-bond correlation.
        <cos(theta)> = exp(-s/Lp) where s is distance along contour.
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
