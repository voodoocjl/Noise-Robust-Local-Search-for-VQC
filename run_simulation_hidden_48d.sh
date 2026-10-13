#!/bin/bash

python test_simulation.py --csv_file 'simulation_results/Hidden_48/No margins/QML_Hidden_48d.csv' --weights_dir 'simulation_results/Hidden_48/No margins/weights'
python test_simulation.py --csv_file 'simulation_results/Hidden_48/Margins/QML_Hidden_48d.csv' --weights_dir 'simulation_results/Hidden_48/Margins/weights'
