#!/bin/bash

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

python hello.py -task "$task"
