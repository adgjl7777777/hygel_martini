#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${1:-${script_dir}/work/synthetic_bond_demo}"
python_bin="${PYTHON_BIN:-python}"

if [[ -e "${project_root}" ]]; then
    echo "Refusing to overwrite existing path: ${project_root}" >&2
    exit 2
fi

"${python_bin}" "${script_dir}/configure_demo.py" "${project_root}"
protocol_module="hygel_martini.param_opt.qm_to_martini.protocol"
"${python_bin}" -m "${protocol_module}" validate "${project_root}"
"${python_bin}" -m "${protocol_module}" seal "${project_root}"

for gate in E0 E1 E2 E3 E4 E5 E6; do
    "${python_bin}" -m "${protocol_module}" evaluate \
        "${project_root}" "${project_root}/evidence/${gate}.yaml" --commit
done

"${python_bin}" -m "${protocol_module}" validate "${project_root}"
"${python_bin}" -m "${protocol_module}" status "${project_root}"

