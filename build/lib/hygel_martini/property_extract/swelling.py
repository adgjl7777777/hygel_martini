import os
import re
import numpy as np
from .gmx_utils import parse_xvg
from .result import PropertyResult


class SwellingAnalyzer:
    def __init__(
        self,
        n_polymer_beads,
        n_solvent_beads,
        polymer_bead_mass=45.0,
        solvent_bead_mass=72.0,
        polymer_bead_vol_nm3=0.065,
    ):
        """
        n_polymer_beads      : polymer bead count (e.g. EO beads)
        n_solvent_beads      : solvent bead count (e.g. Martini W)
        polymer_bead_mass    : mass per polymer bead [amu].  Default 45.0 (M3 PEO SN3r)
        solvent_bead_mass    : mass per solvent bead [amu].  Default 72.0 (Martini W = 4 x 18)
        polymer_bead_vol_nm3 : approximate bead volume [nm^3]. Default 0.065 (FF dependent)
        """
        self.n_polymer_beads = n_polymer_beads
        self.n_solvent_beads = n_solvent_beads
        self.polymer_bead_mass = polymer_bead_mass
        self.solvent_bead_mass = solvent_bead_mass
        self.polymer_bead_vol_nm3 = polymer_bead_vol_nm3

    def composition_summary(self) -> PropertyResult:
        """
        energy xvg 없이도 항상 계산 가능한 초기 조성 요약.
        loading_qm 은 count-based initial composition ratio.
        water 수가 고정된 NPT 시뮬레이션에서는 실험 equilibrium Qm 과 다름.
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
        """
        energy xvg 파일에서 Volume 을 읽어 polymer_volume_fraction 계산.
        start_time_ps: 이 시간 이후 데이터만 평균에 사용 [ps]
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
        """
        top/itp 파일로부터 n_polymer_beads, n_solvent_beads 를 읽어 인스턴스 생성.

        polymer_residue_name  : ITP [ atoms ]에서 선택할 residue 이름. None 이면 전체 count.
        polymer_atom_name     : ITP [ atoms ]에서 선택할 atom 이름. None 이면 전체 count.
        solvent_molecule_names: top [ molecules ]에서 solvent 이름. str 또는 list[str].
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
