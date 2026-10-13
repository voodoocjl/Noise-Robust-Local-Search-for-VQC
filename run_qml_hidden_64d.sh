#!/bin/bash

task="{
'task': 'QML_Hidden_64d',
'n_qubits': 16,
'n_layers': 4,
'fold': 1,
'option': 'mix_reg',
'regular': False,
'num_processes': 2,
'enable_group_avg': True,
'qubit_group_sizes': [8, 8],
'target_single_diff': 10,
'target_enta_diff': 10
}"

python hello.py -task "$task"
