# **Hardware‑Level Transpiled Results**

## Project layout

The repository is now organized into a small package structure so it is no longer a single flat folder:

```text
nrls/
  core/        # shared utility and model configuration code
  data/        # dataset and preprocessing helpers
  models/      # quantum model, VAE, and classifier implementations
  search/      # NRLS search, sampling, and scheme logic
  scripts/     # runnable utilities and evaluation entry points
```

Compatibility shims remain at the project root so existing scripts such as `python NRLS.py` and `from Arguments import Arguments` keep working while the codebase is gradually migrated to the package tree.

All circuits are transpiled using **Qiskit’s Heron backend (FakeTorino)**. After transpilation, we extract native gate statistics (CZ, RZ, SX) and mapped depth using `count_ops()`. All methods use identical transpiler settings.

### **Transpiled Metrics and Path counts of the Best‑Found Circuits**

#### **LS-48d**

| Method | CZ | RZ | SX | Depth | Path |
| --- | --- | --- | --- | --- | --- |
| DQAS | 141 | 780 | 557 | 350 | 390 |
| QuantumNAS | 147 | 684 | 512 | 264 | 540 |
| Q‑MCTS | 177 | 702 | 577 | 327 | 1005 |
| TF‑QAS | 96 | 1035 | 582 | 295 | 1218 |
| NSGA‑II | 96 | 789 | 426 | 292 | 831 |
| **NRLS** | **132** | **669** | **489** | **226** | **273** |

#### **LS-64d**

| Method | CZ | RZ | SX | Depth | Path |
| --- | --- | --- | --- | --- | --- |
| DQAS | 303 | 994 | 907 | 578 | 1352 |
| QuantumNAS | 348 | 1080 | 1031 | 573 | 2362 |
| Q‑MCTS | 268 | 1002 | 854 | 640 | 1650 |
| TF‑QAS | 333 | 1069 | 1013 | 482 | 1676 |
| NSGA‑II | 330 | 1075 | 994 | 606 | 2006 |
| **NRLS** | **277** | **1031** | **896** | **688** | **1321** |

#### **HM-48d**

| Method | CZ | RZ | SX | Depth | Path |
| --- | --- | --- | --- | --- | --- |
| DQAS | 195 | 657 | 573 | 330 | 294 |
| QuantumNAS | 189 | 715 | 611 | 375 | 528 |
| Q‑MCTS | 192 | 846 | 665 | 428 | 426 |
| TF‑QAS | 192 | 951 | 725 | 444 | 825 |
| NSGA‑II | 150 | 1068 | 726 | 483 | 540 |
| **NRLS** | **195** | **1023** | **762** | **373** | **390** |

#### **HM-64d**

| Method | CZ | RZ | SX | Depth | Path |
| --- | --- | --- | --- | --- | --- |
| DQAS | 340 | 1097 | 1037 | 1660 | 1792 |
| QuantumNAS | 314 | 1052 | 962 | 552 | 1571 |
| Q‑MCTS | 322 | 1008 | 947 | 627 | 1481 |
| TF‑QAS | 344 | 1000 | 995 | 559 | 1564 |
| NSGA‑II | 289 | 1244 | 1017 | 706 | 2090 |
| **NRLS** | **295** | **1258** | **1038** | **588** | **1732** |

#### **MNIST-10**

| Method | CZ | RZ | SX | Depth | Path |
| --- | --- | --- | --- | --- | --- |
| DQAS | 125 | 816 | 578 | 429 | 540 |
| QuantumNAS | 184 | 848 | 694 | 488 | 1026 |
| Q‑MCTS | 156 | 808 | 624 | 482 | 798 |
| TF‑QAS | 176 | 778 | 620 | 611 | 1788 |
| NSGA‑II | 188 | 888 | 716 | 688 | 1410 |
| **NRLS** | **176** | **726** | **590** | **480** | **1026** |

# Run Experiments With Shell Scripts

All commands below assume you are in the project root:

```bash
python cleaner.py
```

Run cleaner.py to empty results, states, and weights for each new task.

## 1) MNIST-4 Search Experiment

```bash
bash run_mnist_4.sh
```

What it does:

- Runs NRLS.py with the MNIST_4 task configuration from run_mnist_4.sh.
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

- Runs NRLS.py with the MNIST_10 task configuration from run_mnist_10.sh.
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

- Runs NRLS.py with the QML_Linear_48d task configuration from run_qml_linear_48d.sh.
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

- Runs NRLS.py with the QML_Linear_64d task configuration from run_qml_linear_64d.sh.
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

- Runs NRLS.py with the QML_Hidden_64d task configuration from run_qml_hidden_64d.sh.
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

- Evaluates architectures listed in each CSV.
- Loads matching model weights from each weights folder.
- Writes result CSV beside each input CSV as:
  - simulation_results/Hidden_64/No margins/fake_backend_QML_Hidden_64d.csv
  - simulation_results/Hidden_64/Margins/fake_backend_QML_Hidden_64d.csv
