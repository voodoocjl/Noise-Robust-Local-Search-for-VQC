import argparse
import ast
import csv
from pathlib import Path
from typing import List, Sequence, Tuple
from nrls.models.FusionModel import single_enta_to_design

import numpy as np


def parse_arch_code(arch_value: str) -> Tuple[List[List[int]], List[List[int]], Tuple[int, int]]:
    """Parse one CSV arch_code cell into single rows, enta rows, and arch code."""

    parsed = ast.literal_eval(arch_value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError("arch_code must be a list with [single_rows, enta_rows]")

    single_rows, enta_rows = parsed
    if not single_rows:
        raise ValueError("single_rows is empty")

    single_rows = sorted(single_rows, key=lambda row: row[0])
    enta_rows = sorted(enta_rows, key=lambda row: row[0])

    tail_width = len(single_rows[0]) - 1
    if tail_width <= 0 or tail_width % 2 != 0:
        raise ValueError("single_rows encoding width is invalid")

    n_layers = tail_width // 2
    n_qubits = max(int(row[0]) for row in single_rows)
    if n_qubits <= 0:
        raise ValueError("inferred n_qubits is invalid")

    expected_enta_width = 1 + n_layers
    for row in enta_rows:
        if len(row) != expected_enta_width:
            raise ValueError("enta row width does not match inferred n_layers")

    arch_code = (n_qubits, n_layers)

    return single_rows, enta_rows, arch_code


def _op_wires(op: Tuple[str, List[int], int]) -> List[int]:
    gate_name, wires, _ = op
    if gate_name == "C(U3)":
        return [int(wires[0]), int(wires[1])]
    if gate_name in {"U3", "data"}:
        return [int(wires[0])]
    return []


def design_to_adj_matrix(design: Sequence[Tuple[str, List[int], int]]) -> np.ndarray:
    """Build an operation DAG where each edge connects to the next op on the same wire."""

    op_count = len(design)
    adj = np.zeros((op_count, op_count), dtype=float)

    for i, op in enumerate(design):
        wires = _op_wires(op)
        if not wires:
            continue

        for wire in wires:
            for j in range(i + 1, op_count):
                if wire in _op_wires(design[j]):
                    adj[i, j] = 1.0
                    break

    return adj


def add_virtual_start_node(adj_matrix: np.ndarray) -> np.ndarray:
    """Add a synthetic source connected to all entry nodes."""

    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adjacency matrix must be square")

    node_count = adj_matrix.shape[0]
    augmented = np.zeros((node_count + 1, node_count + 1), dtype=float)
    augmented[1:, 1:] = adj_matrix

    in_degrees = np.count_nonzero(adj_matrix, axis=0)
    entry_nodes = [idx for idx, degree in enumerate(in_degrees) if degree == 0]
    if entry_nodes:
        augmented[0, [idx + 1 for idx in entry_nodes]] = 1.0

    return augmented


def path_based_proxy(adj_matrix: np.ndarray) -> int:
    """Count all directed paths from synthetic source to synthetic sink."""

    augmented = add_virtual_start_node(adj_matrix)
    augmented = np.pad(augmented, ((0, 1), (0, 1)), mode="constant")

    out_degrees = np.count_nonzero(adj_matrix, axis=1)
    exit_nodes = [idx for idx, degree in enumerate(out_degrees) if degree == 0]
    if not exit_nodes:
        exit_nodes = [adj_matrix.shape[0] - 1]

    for node_idx in exit_nodes:
        augmented[node_idx + 1, -1] = 1.0

    edge_mask = augmented > 0
    path_counts = np.zeros(augmented.shape[0], dtype=np.int64)
    path_counts[0] = 1

    for node in range(augmented.shape[0]):
        successors = np.nonzero(edge_mask[node])[0]
        if successors.size:
            path_counts[successors] += path_counts[node]

    return int(path_counts[-1])


def compute_paths(
    input_csv: Path,
    output_csv: Path,
    arch_column: str,
    path_column: str,
    error_column: str,
) -> Tuple[int, int]:
    """Read CSV, compute path count for each arch_code row, and write a new CSV."""

    total = 0
    failed = 0

    with input_csv.open("r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        if arch_column not in reader.fieldnames:
            raise ValueError(f"Column '{arch_column}' not found in CSV")

        out_fields = list(reader.fieldnames)
        if path_column not in out_fields:
            out_fields.append(path_column)
        if error_column not in out_fields:
            out_fields.append(error_column)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                total += 1
                row[path_column] = ""
                row[error_column] = ""
                try:
                    single_rows, enta_rows, arch_code = parse_arch_code(row[arch_column])
                    design = single_enta_to_design(single_rows, enta_rows, arch_code, fold)
                    adj_matrix = design_to_adj_matrix(design)
                    row[path_column] = path_based_proxy(adj_matrix)
                except Exception as exc:
                    failed += 1
                    row[error_column] = str(exc)
                writer.writerow(row)

    return total, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute path proxy numbers for all arch_code rows in a CSV")
    parser.add_argument("input_csv", type=Path, nargs="?", help="Input CSV path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_stem>_with_paths.csv next to input)",
    )
    parser.add_argument("--arch-column", default="arch_code", help="Column name containing the [single, enta] encoding")
    parser.add_argument("--path-column", default="path_number", help="Name of output path count column")
    parser.add_argument("--error-column", default="path_error", help="Name of output error column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv
    input_csv = Path('QML_Hidden_64d.csv') if input_csv is None else input_csv
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv = args.output
    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_with_paths.csv")

    total, failed = compute_paths(
        input_csv=input_csv,
        output_csv=output_csv,
        arch_column=args.arch_column,
        path_column=args.path_column,
        error_column=args.error_column,
    )

    print(f"Processed {total} rows")
    print(f"Failures: {failed}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    fold = 2  # Default fold value for design conversion
    main()
