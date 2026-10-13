# Run Experiments With Shell Scripts

All commands below assume you are in the project root:

```bash
cd HELLO_AAAI_27

python cleaner.py
```

Run cleaner.py to empty results, states, and weights for each new task.

## 1) MNIST-4 Search Experiment

```bash
bash run_mnist_4.sh
```

What it does:

- Runs hello.py with the MNIST_4 task configuration from run_mnist_4.sh.
- Executes the search and training loop.
- Writes output files to:
  - results/MNIST_4_mix_no_reg.csv
  - results/MNIST_4_mix_no_reg_fine.csv
  - weights/weight_ITERATION_TIMESTAMP
  - states/ (search state snapshots)

## 2) MNIST-10 Search Experiment

```bash
bash run_mnist_10.sh
```

What it does:

- Runs hello.py with the MNIST_10 task configuration from run_mnist_10.sh.
- Executes the search and training loop.
- Writes output files to:
  - results/MNIST_10_mix_reg.csv
  - results/MNIST_10_mix_reg_fine.csv
  - weights/weight_ITERATION_TIMESTAMP
  - states/ (search state snapshots)

## 3) QML Linear-48d Search Experiment

```bash
bash run_qml_linear_48d.sh
```

What it does:

- Runs hello.py with the QML_Linear_48d task configuration from run_qml_linear_48d.sh.
- Executes the search and training loop.
- Writes output files to:
  - results/QML_Linear_48d_mix_reg.csv
  - results/QML_Linear_48d_mix_reg_fine.csv
  - weights/weight_ITERATION_TIMESTAMP
  - states/ (search state snapshots)

## 4) QML Linear-64d Search Experiment

```bash
bash run_qml_linear_64d.sh
```

What it does:

- Runs hello.py with the QML_Linear_64d task configuration from run_qml_linear_64d.sh.
- Executes the search and training loop.
- Writes output files to:
  - results/QML_Linear_64d_mix_reg.csv
  - results/QML_Linear_64d_mix_reg_fine.csv
  - weights/weight_ITERATION_TIMESTAMP
  - states/ (search state snapshots)

## 5) QML Hidden-64d Search Experiment

```bash
bash run_qml_hidden_64d.sh
```

What it does:

- Runs hello.py with the QML_Hidden_64d task configuration from run_qml_hidden_64d.sh.
- Executes the search and training loop.
- Writes output files to:
  - results/QML_Hidden_64d_mix_reg.csv
  - results/QML_Hidden_64d_mix_reg_fine.csv
  - weights/weight_ITERATION_TIMESTAMP
  - states/ (search state snapshots)

## 6) Hidden-48d Simulation Evaluation (No Margins + Margins)

```bash
bash run_simulation_hidden_48d.sh
python cleaner.py
```

What it does:

- To reproduce Figure. 5
- Evaluates architectures listed in each CSV.
- Loads matching model weights from each weights folder.

## 8) Draw Circuits From a CSV

The `arch_code` column stores the single-qubit and entangling-gate encoding for
each circuit. To write one PNG per row:

```bash
python draw_circuits.py QML_Hidden_48d_mix_reg_fine.csv
```

Images are written to `QML_Hidden_48d_mix_reg_fine_circuits/`. A different
destination can be selected with `--output-dir`.

- Writes result CSV beside each input CSV as:
  - simulation_results/Hidden_48/No margins/fake_backend_QML_Hidden_48d.csv
  - simulation_results/Hidden_48/Margins/fake_backend_QML_Hidden_48d.csv

## 7) Hidden-64d Simulation Evaluation (No Margins + Margins)

```bash
bash run_simulation_hidden_64d.sh
python cleaner.py
```

What it does:

- To reproduce Figure. 5
- Evaluates architectures listed in each CSV.
- Loads matching model weights from each weights folder.
- Writes result CSV beside each input CSV as:
  - simulation_results/Hidden_64/No margins/fake_backend_QML_Hidden_64d.csv
  - simulation_results/Hidden_64/Margins/fake_backend_QML_Hidden_64d.csv
