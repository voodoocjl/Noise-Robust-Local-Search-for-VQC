#!/bin/bash

task="{
'task': 'MNIST_4',
'option': 'mix_no_reg',
'regular': False,
'n_qubits': 4,
'n_layers': 4,
'fold': 1,
'num_processes': 2,
'target_single_diff': 10,
'target_enta_diff': 10
}"

python hello.py -task "$task"
