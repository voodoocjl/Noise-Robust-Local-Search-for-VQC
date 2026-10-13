"""Draw the circuits encoded in an arch_code CSV column."""

import argparse
import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import UGate


def parse_arch_code(value: str):
    """Return single-qubit rows, entangling rows, qubit count, and layer count."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError("arch_code must contain [single_rows, enta_rows]")

    single_rows, enta_rows = parsed
    if not single_rows or not enta_rows:
        raise ValueError("arch_code contains an empty section")

    single_rows = sorted(single_rows, key=lambda row: int(row[0]))
    enta_rows = sorted(enta_rows, key=lambda row: int(row[0]))
    qubit_count = len(single_rows)
    if [int(row[0]) for row in single_rows] != list(range(1, qubit_count + 1)):
        raise ValueError("single rows must contain consecutive 1-based qubit indices")
    if [int(row[0]) for row in enta_rows] != list(range(1, qubit_count + 1)):
        raise ValueError("entangling rows must contain consecutive 1-based qubit indices")

    single_width = len(single_rows[0]) - 1
    if single_width <= 0 or single_width % 2:
        raise ValueError("single rows must contain pairs of values per layer")
    layer_count = single_width // 2
    if any(len(row) != single_width + 1 for row in single_rows):
        raise ValueError("single rows have inconsistent widths")
    if any(len(row) != layer_count + 1 for row in enta_rows):
        raise ValueError("entangling rows do not match the layer count")

    return single_rows, enta_rows, qubit_count, layer_count


def draw_arch_code(arch_code: str, output_path: Path, title: str) -> None:
    """Create one PNG for one encoded circuit."""
    single_rows, enta_rows, qubit_count, layer_count = parse_arch_code(arch_code)
    circuit = QuantumCircuit(qubit_count)

    data_gate = Gate("DATA", 1, [], label="DATA")
    for layer in range(layer_count):
        for row in single_rows:
            qubit = int(row[0]) - 1
            data_bit = int(row[1 + 2 * layer])
            rotation_bit = int(row[2 + 2 * layer])
            gate_code = f"{data_bit}{rotation_bit}"

            if gate_code in {"10", "11"}:
                circuit.append(data_gate, [qubit])
            if gate_code in {"01", "11"}:
                circuit.u(
                    Parameter(f"t_{layer}_{qubit}"),
                    Parameter(f"p_{layer}_{qubit}"),
                    Parameter(f"l_{layer}_{qubit}"),
                    qubit,
                )

        for row in enta_rows:
            control = int(row[0]) - 1
            target = int(row[1 + layer]) - 1
            if control == target:
                continue
            controlled_u = UGate(
                Parameter(f"ct_{layer}_{control}"),
                Parameter(f"cp_{layer}_{control}"),
                Parameter(f"cl_{layer}_{control}"),
            ).control(1)
            circuit.append(controlled_u, [control, target])

        if layer < layer_count - 1:
            circuit.barrier()

    figure = circuit.draw(output="mpl", fold=-1, idle_wires=True)
    figure.suptitle(title, fontsize=12)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw one PNG for every circuit in an arch_code CSV column"
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input_stem>_circuits)",
    )
    parser.add_argument("--arch-column", default="arch_code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_csv.with_name(f"{args.input_csv.stem}_circuits")
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or args.arch_column not in reader.fieldnames:
            raise ValueError(f"CSV does not contain an '{args.arch_column}' column")

        for row_number, row in enumerate(reader, start=1):
            iteration = row.get("iteration") or str(row_number)
            output_path = output_dir / f"iteration_{iteration.zfill(3)}.png"
            draw_arch_code(
                row[args.arch_column],
                output_path,
                f"Iteration {iteration} | 12 qubits | encoded circuit",
            )
            print(output_path)


if __name__ == "__main__":
    main()