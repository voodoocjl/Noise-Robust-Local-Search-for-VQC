#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

python -m nrls.scripts.test_simulation --csv_file 'simulation_results/Hidden_48/No margins/QML_Hidden_48d.csv' --weights_dir 'simulation_results/Hidden_48/No margins/weights'
python -m nrls.scripts.test_simulation --csv_file 'simulation_results/Hidden_48/Margins/QML_Hidden_48d.csv' --weights_dir 'simulation_results/Hidden_48/Margins/weights'
