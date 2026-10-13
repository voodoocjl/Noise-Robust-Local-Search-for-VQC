#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

task="{
'task': 'MNIST_10',
'option': 'mix_reg',
'regular': True,
'n_qubits': 10,
'n_layers': 4,
'fold': 1,
'num_processes': 2,
'target_single_diff': 10,
'target_enta_diff': 10
}"

python -m nrls.search.NRLS -task "$task"
