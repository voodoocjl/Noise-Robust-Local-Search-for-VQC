#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

task="{
'task': 'QML_Linear_48d',
'n_qubits': 12,
'n_layers': 4,
'fold': 1,
'option': 'mix_reg',
'regular': False,
'num_processes': 2,
'enable_group_avg': True,
'qubit_group_sizes': [6, 6],
'target_single_diff': 10,
'target_enta_diff': 10
}"

python -m nrls.search.NRLS -task "$task"