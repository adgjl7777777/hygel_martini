#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# launcher_utils.sh 탐색: 설치된 패키지 경로 우선
LAUNCHER_UTILS_PATH=$(python3 -c \
  "from pathlib import Path; import hygel_martini; \
   print(Path(hygel_martini.__file__).parent / 'bash_settings' / 'common' / 'launcher_utils.sh')" \
  2>/dev/null || true)
if [ -z "$LAUNCHER_UTILS_PATH" ] || [ ! -f "$LAUNCHER_UTILS_PATH" ]; then
  echo "[ERROR] launcher_utils.sh를 찾을 수 없습니다. hygel_martini가 설치되어 있나요? (pip install -e .)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$LAUNCHER_UTILS_PATH"

# 환경 설정 (environment.sh 자동 로드)
setup_hygel_env "$SCRIPT_DIR"

# GROMACS 설정 — 하드코딩 없음
GMX_CMD="${GMX_CMD:-gmx_mpi}"
GMXRC_PATH="${GMXRC_PATH:-}"
if [ -n "$GMXRC_PATH" ] && [ -f "$GMXRC_PATH" ]; then
  source_optional_script "GMXRC_PATH" "$GMXRC_PATH"
fi

usage() {
  cat <<EOF
Usage:
  bash run_property_extract.sh property_extract.yaml
  bash run_property_extract.sh --extract-only property_extract.yaml
  bash run_property_extract.sh --check-gmx
  bash run_property_extract.sh --help

Options:
  --extract-only   gmx energy xvg 추출만 수행하고 종료
  --check-gmx      GROMACS 실행 가능 여부 확인
  --help           이 도움말 출력

Shell environment:
  environment.sh 가 같은 디렉터리에 있으면 자동 로드됩니다.
  ENVIRONMENT_FILE, PYTHON_BIN, GMX_CMD, GMXRC_PATH 환경 변수로 재정의 가능.
  GROMACS 실행 파일은 property_extract.yaml의 gromacs.executable 또는
  GMX_BIN 환경 변수 또는 PATH 자동 탐색 순으로 결정됩니다.
EOF
}

EXTRACT_ONLY_FLAG=""
CHECK_GMX=0

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)      usage; exit 0 ;;
    --check-gmx)    CHECK_GMX=1; shift ;;
    --extract-only) EXTRACT_ONLY_FLAG="--extract-only"; shift ;;
    *)              break ;;
  esac
done

if [ "$CHECK_GMX" -eq 1 ]; then
  "$GMX_CMD" --version
  exit 0
fi

if [ $# -eq 0 ]; then
  usage; exit 0
fi

CONFIG_ARG="$1"
if [[ "$CONFIG_ARG" = /* ]]; then
  CONFIG_PATH="$CONFIG_ARG"
else
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
require_python_module "hygel_martini.property_extract"
# shellcheck disable=SC2086
"$PYTHON_BIN" -m hygel_martini.property_extract --config "$CONFIG_PATH" $EXTRACT_ONLY_FLAG
