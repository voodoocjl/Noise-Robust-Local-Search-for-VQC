import pickle
import random
from itertools import product

num_layers = 4
num_qubits = 6
num_single = 2 * num_layers
num_enta = num_layers


def generate_search_space(number, category, n_qubits, n_layers):
    if category not in ["single", "enta"]:
        raise ValueError("category must be 'single' or 'enta'")
    if number <= 0:
        return []

    if category == "single":
        max_unique = n_qubits * (2 ** (2 * n_layers))
    else:
        max_unique = n_qubits ** (n_layers + 1)

    target_size = min(number, max_unique)
    unique_rows = set()

    if category == "single":
        # row format: [start_qubit] + 2*n_layers binary values
        while len(unique_rows) < target_size:
            start_value = random.randint(1, n_qubits)
            row = tuple([start_value] + [random.randint(0, 1) for _ in range(2 * n_layers)])
            unique_rows.add(row)
    else:
        # row format: [start_qubit] + n_layers target qubits in [1, n_qubits]
        while len(unique_rows) < target_size:
            start_value = random.randint(1, n_qubits)
            row = tuple([start_value] + [random.randint(1, n_qubits) for _ in range(n_layers)])
            unique_rows.add(row)

    return [list(row) for row in unique_rows]
