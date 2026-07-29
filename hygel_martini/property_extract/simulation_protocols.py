import os


class MDProtocolGenerator:
    """
    GROMACS MDP 파일 생성기.
    초안 상태: shear MDP의 물리적 적합성(ensemble, barostat 선택)은 GROMACS manual 기준으로
    별도 검증 필요. 이 클래스는 MDP 파일만 생성하며, top/gro 연결이나 grompp 실행은 수행하지 않음.
    """

    # Martini 표준 비결합 파라미터
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
        """
        NEMD shear MDP 생성.
        shear_rate     : deform XY 성분 [nm/ps]
        temperature    : 온도 [K]
        nsteps         : MD 스텝 수
        dt             : 타임스텝 [ps]
        tau_t          : thermostat 시간상수 [ps]
        tau_p          : barostat 시간상수 [ps]
        ref_p          : 기준 압력 [bar]
        compressibility: 압축률 [bar^-1]
        nstenergy      : energy 출력 간격 [steps]

        경고: anisotropic pressure coupling + deform 조합은 GROMACS manual 기준 검증 필요.
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
        """
        Green-Kubo ACF용 고빈도 에너지 수집 MDP.
        nstenergy를 작게 설정해 stress-stress ACF를 계산할 수 있게 함.
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
        """
        전단 속도 계열 디렉터리와 MDP 파일 생성.
        rates: 전단 속도 목록 [nm/ps]. 기본값 [0.0001, 0.001, 0.01].
        경고: MDP 파일만 생성. top/gro 연결 및 grompp 실행은 별도 수행 필요.
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
