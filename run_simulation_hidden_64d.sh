#!/bin/bash

python test_simulation.py --csv_file 'simulation_results/Hidden_64/No margins/QML_Hidden_64d.csv' --weights_dir 'simulation_results/Hidden_64/No margins/weights'
python test_simulation.py --csv_file 'simulation_results/Hidden_64/Margins/QML_Hidden_64d.csv' --weights_dir 'simulation_results/Hidden_64/Margins/weights'
