# param_opt launchers

Shared shell entry points for qm-to-Martini parameterization and analysis.

## Scripts

- `run_qm_to_martini.sh`: generic package launcher for `python -m hygel_martini.param_opt.qm_to_martini`.
- `run_postprocess_sweep.sh`: shared sweep runner for analysis directories containing `configs/*.yaml`.
- `run_trim_summary.sh`: summarize `*_trim_info.json` files under a compare root.
- `run_trim_sensitivity.sh`: run C/D/S energy trim sensitivity diagnostics.
- `run_trim_threshold_samet_analysis.sh`: analyze an existing `trim_threshold_samet` set.

All scripts honor `PYTHON_BIN`. The project-oriented scripts use `PROJECT_DIR=$PWD` by default, so run them from a project directory or set `PROJECT_DIR` explicitly.
