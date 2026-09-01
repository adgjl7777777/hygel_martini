"""GROMACS MDP generation for the rheology/dynamics MD protocols.

Owns :class:`MDProtocolGenerator`, which renders MDP text (Martini
standard nonbonded settings baked in) for the NEMD shear runs consumed
by :mod:`.rheology` / :mod:`.extractors.rheology_nemd` and for
high-frequency energy runs intended for Green-Kubo stress ACFs.  It
only writes MDP content — wiring top/gro and running ``grompp`` are
the caller's job.

Draft status is explicit: the shear MDP's physical soundness
(ensemble/barostat choice for anisotropic coupling combined with
``deform``) still needs verification against the GROMACS manual, and
the generated files say so in their comments.
"""
import os


class MDProtocolGenerator:
    """GROMACS MDP file generator (draft status).

    Draft: the shear MDP's physical soundness (ensemble/barostat
    choice) still needs verification against the GROMACS manual.  The
    class generates MDP text/files only — it neither wires top/gro
    files nor runs ``grompp``.
    """

    # Martini 표준 비결합 파라미터
    # (Martini standard nonbonded block shared by every generated MDP:
    #  Verlet lists, 1.1 nm cutoffs, reaction-field electrostatics.)
    _MARTINI_NONBONDED = """
; Nonbonded (Martini standard)
cutoff-scheme       = Verlet
nstlist             = 20
vdwtype             = Cut-off
rvdw                = 1.1
coulombtype         = Reaction-Field
rcoulomb            = 1.1
epsilon_rf          = 15
"""

    def __init__(self, base_mdp_path=None):
        """Store an optional base MDP path (currently unused by getters)."""
        self.base_mdp = base_mdp_path

    def get_shear_mdp(
        self,
        shear_rate,
        temperature=310.15,
        nsteps=10000000,
        dt=0.02,
        tau_t=1.0,
        tau_p=12.0,
        ref_p=1.013,
        compressibility=4.5e-5,
        nstenergy=1000,
    ):
        """Render the NEMD shear MDP text.

        Warning: the anisotropic pressure coupling + ``deform``
        combination must still be verified against the GROMACS manual
        before quantitative use.

        Args:
            shear_rate: ``deform`` XY component in nm/ps.
            temperature: Reference temperature in K.
            nsteps: Number of MD steps.
            dt: Timestep in ps.
            tau_t: Thermostat time constant in ps.
            tau_p: Barostat time constant in ps.
            ref_p: Reference pressure in bar.
            compressibility: Compressibility in bar^-1 (diagonal
                components only; off-diagonals fixed at 0).
            nstenergy: Energy output interval in steps.

        Returns:
            Complete MDP file contents as a string (Martini nonbonded
            block included).
        """
        compr = f"{compressibility:.2e}"
        return (
            f"integrator          = md\n"
            f"nsteps              = {nsteps}\n"
            f"dt                  = {dt}\n"
            f"comm-mode           = Linear\n"
            f"nstxout             = 5000\n"
            f"nstvout             = 5000\n"
            f"nstfout             = 0\n"
            f"nstenergy           = {nstenergy}\n"
            f"nstlog              = 5000\n"
            + self._MARTINI_NONBONDED +
            f"\n; T-coupling\n"
            f"tcoupl              = v-rescale\n"
            f"tc-grps             = System\n"
            f"tau_t               = {tau_t}\n"
            f"ref_t               = {temperature}\n"
            f"\n; P-coupling with shear (초안 — 물리적 적합성 검증 필요)\n"
            f"pcoupl              = Parrinello-Rahman\n"
            f"pcoupltype          = anisotropic\n"
            f"tau_p               = {tau_p}\n"
            f"compressibility     = {compr} {compr} {compr} 0 0 0\n"
            f"ref_p               = {ref_p} {ref_p} {ref_p} 0 0 0\n"
            f"deform              = 0 0 0 {shear_rate} 0 0  ; Shear in XY plane\n"
        )

    def get_dynamics_mdp(
        self,
        temperature=310.15,
        nsteps=5000000,
        dt=0.02,
        tau_t=1.0,
        tau_p=12.0,
        ref_p=1.013,
        compressibility=4.5e-5,
        nstenergy=10,
    ):
        """Render an MDP for high-frequency energy output (Green-Kubo).

        Same Martini nonbonded block with isotropic Parrinello-Rahman
        coupling; ``nstenergy`` defaults to 10 steps so the
        stress-stress autocorrelation can be resolved for Green-Kubo
        viscosity.

        Args:
            temperature: Reference temperature in K.
            nsteps: Number of MD steps.
            dt: Timestep in ps.
            tau_t: Thermostat time constant in ps.
            tau_p: Barostat time constant in ps.
            ref_p: Reference pressure in bar.
            compressibility: Compressibility in bar^-1.
            nstenergy: Energy output interval in steps (small on
                purpose).

        Returns:
            Complete MDP file contents as a string.
        """
        compr = f"{compressibility:.2e}"
        return (
            f"integrator          = md\n"
            f"nsteps              = {nsteps}\n"
            f"dt                  = {dt}\n"
            f"nstenergy           = {nstenergy}  ; ACF용 고빈도\n"
            f"nstxout             = 5000\n"
            + self._MARTINI_NONBONDED +
            f"\ntcoupl              = v-rescale\n"
            f"tc-grps             = System\n"
            f"tau_t               = {tau_t}\n"
            f"ref_t               = {temperature}\n"
            f"\npcoupl              = Parrinello-Rahman\n"
            f"pcoupltype          = isotropic\n"
            f"tau_p               = {tau_p}\n"
            f"ref_p               = {ref_p}\n"
            f"compressibility     = {compr}\n"
        )

    def create_shear_series(self, output_dir, rates=None, temperature=310.15, **mdp_kwargs):
        """Create per-shear-rate directories with their shear MDPs.

        Makes ``<output_dir>/shear_<rate>/shear.mdp`` for every rate
        (directories created as needed).  Only MDP files are written —
        top/gro wiring and ``grompp`` runs are separate steps.

        Args:
            output_dir: Parent directory for the series.
            rates: Shear rates in nm/ps; default
                ``[0.0001, 0.001, 0.01]``.
            temperature: Reference temperature in K.
            **mdp_kwargs: Extra arguments forwarded to
                :meth:`get_shear_mdp`.

        Returns:
            List of the created MDP file paths.
        """
        if rates is None:
            rates = [0.0001, 0.001, 0.01]
        created = []
        for rate in rates:
            path = os.path.join(output_dir, f"shear_{rate}")
            os.makedirs(path, exist_ok=True)
            mdp_path = os.path.join(path, "shear.mdp")
            with open(mdp_path, "w") as f:
                f.write(self.get_shear_mdp(rate, temperature=temperature, **mdp_kwargs))
            created.append(mdp_path)
        return created
