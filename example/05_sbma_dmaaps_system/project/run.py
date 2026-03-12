import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
from hydrogel_builder.config_params.generator import run_hydrogel_example

def main():
    """
    Main entry point to run the hydrogel generation.
    """
    config_path = None
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        # Prefer YAML, fall back to JSON for backward compatibility
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in ("maker.yaml", "maker.yml", "maker.json"):
            guess = os.path.join(base_dir, candidate)
            if os.path.exists(guess):
                config_path = guess
                break

    if not config_path:
        print("Usage: python run.py <path_to_config.(yaml|yml|json)>", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"Error: config file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)

    # Resolve to an absolute path
    absolute_json_path = os.path.abspath(config_path)

    print("="*50)
    print(f"Starting hydrogel generation with: {os.path.basename(absolute_json_path)}")
    print("="*50)

    run_hydrogel_example(absolute_json_path)

if __name__ == "__main__":
    main()
