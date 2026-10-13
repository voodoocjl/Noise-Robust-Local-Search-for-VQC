#!/usr/bin/env python
"""
Test script to evaluate Linear_48d structures with different backends.
Reads Linear_48d.csv and evaluates each structure (from iteration 1 onwards)
using: TorchQuantum (tq), Qiskit noise-free, and Qiskit with heron_r1 noise.

Usage:
    conda run -n tq_density python test_linear48d.py

Output:
    test_results_linear48d_online.csv - Incremental results written per structure
"""

import os
import csv
import ast
import argparse
import multiprocessing as mp
from typing import List, Sequence, Tuple

import torch
import numpy as np
from Arguments import Arguments
from datasets import MNISTDataLoaders, qml_Dataloaders
from schemes_noise import Scheme_eval
from FusionModel import QNet, single_enta_to_design


def parse_arch_code_from_csv(arch_code_str):
    """Parse arch_code string from CSV into single and enta lists."""
    try:
        arch_code = ast.literal_eval(arch_code_str)
        single = arch_code[0]
        enta = arch_code[1]
        return single, enta
    except Exception as e:
        print(f"Error parsing arch_code: {e}")
        return None, None


def find_weight_file(iteration_num, weights_dir='weights'):
    """Find weight file for given iteration."""
    pattern = f"weight_{iteration_num}_"
    for fname in os.listdir(weights_dir):
        if fname.startswith(pattern):
            return os.path.join(weights_dir, fname)
    return None


def _op_wires(op: Tuple[str, List[int], int]) -> List[int]:
    gate_name, wires, _ = op
    if gate_name == "C(U3)":
        return [int(wires[0]), int(wires[1])]
    if gate_name in {"U3", "data"}:
        return [int(wires[0])]
    return []


def design_to_adj_matrix(design: Sequence[Tuple[str, List[int], int]]) -> np.ndarray:
    """Build operation DAG adjacency by linking next op on each wire."""
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


def path_based_proxy(adj_matrix: np.ndarray) -> int:
    """Count directed paths from synthetic source to synthetic sink in DAG."""
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adjacency matrix must be square")

    n = adj_matrix.shape[0]
    augmented = np.zeros((n + 2, n + 2), dtype=float)
    augmented[1:n + 1, 1:n + 1] = adj_matrix

    in_degrees = np.count_nonzero(adj_matrix, axis=0)
    entry_nodes = [idx for idx, deg in enumerate(in_degrees) if deg == 0]
    for idx in entry_nodes:
        augmented[0, idx + 1] = 1.0

    out_degrees = np.count_nonzero(adj_matrix, axis=1)
    exit_nodes = [idx for idx, deg in enumerate(out_degrees) if deg == 0]
    if not exit_nodes and n > 0:
        exit_nodes = [n - 1]
    for idx in exit_nodes:
        augmented[idx + 1, -1] = 1.0

    edge_mask = augmented > 0
    path_counts = np.zeros(augmented.shape[0], dtype=np.int64)
    path_counts[0] = 1
    for node in range(augmented.shape[0]):
        successors = np.nonzero(edge_mask[node])[0]
        if successors.size:
            path_counts[successors] += path_counts[node]

    return int(path_counts[-1])


def build_transpiled_metrics(design, base_task, backend_name='heron_r1'):
    """Build transpiled template and return key transpilation metrics."""
    args = Arguments(**base_task)
    args.backend = 'qi'
    args.noise = False
    args.name = backend_name

    model = QNet(args, design)
    transpiled_circuit = model.QuantumLayer.transpiled_qc_template
    ops = dict(transpiled_circuit.count_ops())
    depth = transpiled_circuit.depth()
    if depth is None:
        depth = transpiled_circuit.decompose(reps=10).depth()

    return {
        'depth': int(depth) if depth is not None else 'N/A',
        'size': int(transpiled_circuit.size()),
        'CZ': int(ops.get('cz', 0)),
        'RZ': int(ops.get('rz', 0)),
        'SX': int(ops.get('sx', 0)),
    }


def has_required_header(output_file, required_fieldnames):
    """Return True if existing CSV header exactly matches required fields."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return False
    try:
        with open(output_file, 'r', newline='') as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            return header == required_fieldnames
    except Exception:
        return False


def evaluate_fake_backend_worker(design, base_task, weight_path, result_queue, correct_counter, total_counter):
    """Run fake backend evaluation and keep partial accuracy progress in shared counters."""
    try:
        args = Arguments(**base_task)
        args.backend = 'qi'
        args.noise = True
        args.qiskit_seed = 170
        args.shots = 10000
        args.name = base_task.get('_backend_name', 'heron_r1')

        if base_task['task'].startswith('QML'):
            dataloader = qml_Dataloaders(args)
        else:
            dataloader = MNISTDataLoaders(args, base_task['task'])

        _, _, test_loader = dataloader

        # Collect full test set upfront so targets can be indexed per chunk
        all_images_list, all_targets_list = [], []
        for feed_dict in test_loader:
            all_images_list.append(feed_dict['image'])
            all_targets_list.append(feed_dict['digit'])
        all_images = torch.cat(all_images_list, dim=0)
        all_targets = torch.cat(all_targets_list, dim=0)

        model = QNet(args, design).to(args.device)
        model.load_state_dict(torch.load(weight_path, map_location=args.device), strict=False)
        model.eval()

        # After each estimator chunk (chunk_size=1 -> one sample), update shared counters
        # using the same post-processing path as QNet.forward (group average + log_softmax).
        def per_chunk_callback(chunk_start, chunk_end, chunk_results):
            chunk_exp_val = torch.tensor(chunk_results, dtype=torch.float32, device=args.device)
            chunk_exp_val = model._apply_group_average(chunk_exp_val, args.n_qubits)
            chunk_output = torch.nn.functional.log_softmax(chunk_exp_val, dim=1)
            chunk_targets = all_targets[chunk_start:chunk_end].to(args.device)
            _, indices = chunk_output.topk(1, dim=1)
            corr = indices.eq(chunk_targets.view(-1, 1).expand_as(indices)).sum().item()
            correct_counter.value += corr
            total_counter.value += (chunk_end - chunk_start)

        args.per_chunk_callback = per_chunk_callback

        with torch.no_grad():
            output = model(all_images.to(args.device), args.n_qubits, args.task)

        total = all_targets.shape[0]
        _, indices = output.topk(1, dim=1)
        correct = indices.eq(all_targets.to(args.device).view(-1, 1).expand_as(indices)).sum().item()
        result_queue.put(float(correct / total) if total else None)
    except Exception as exc:
        result_queue.put(f"ERROR: {exc}")


def evaluate_fake_backend_with_timeout(design, base_task, weight_path, timeout_seconds):
    """Evaluate fake backend in a subprocess and return partial accuracy if timeout occurs."""
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    correct_counter = ctx.Value('i', 0)
    total_counter = ctx.Value('i', 0)
    process = ctx.Process(
        target=evaluate_fake_backend_worker,
        args=(design, base_task, weight_path, result_queue, correct_counter, total_counter)
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        partial_total = total_counter.value
        partial_correct = correct_counter.value
        process.terminate()
        process.join()
        partial_acc = (partial_correct / partial_total) if partial_total else None
        return partial_acc, True, partial_total

    if not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, str) and result.startswith('ERROR:'):
            print(result)
            return None, False, total_counter.value
        return result, False, total_counter.value

    return None, False, total_counter.value


def main(csv_file, weights_dir):
    csv_dir = os.path.dirname(csv_file)
    csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
    output_name = f'fake_backend_{csv_stem}.csv'
    output_file = os.path.join(csv_dir, output_name) if csv_dir else output_name
    transpile_backend_name = 'heron_r1'
    fake_timeout_seconds = 2000  # 2000 seconds per structure for fake backend
    start_iteration =  20 # Set to 1 to evaluate from iteration 1; e.g., 6 starts from iteration 6

    # ── Select which backends to evaluate ──────────────────────────────────
    # Comment out any entry to skip that backend.
    backends_to_run = [
        'tq',            # TorchQuantum (fast, no noise)
        # 'qi_noise_free', # Qiskit Estimator, no noise model
        'heron_r1',      # FakeTorino  (133-qubit Heron r1 snapshot)
        # 'eagle_r3',      # FakeKyiv    (127-qubit Eagle r3 snapshot)
    ]
    # ───────────────────────────────────────────────────────────────────────

    # Task configuration (same as in schemes_noise.py)
    if csv_stem == 'QML_Hidden_64d':
        base_task = {
            'task': 'QML_Hidden_64d',
            'n_qubits': 16,
            'n_layers': 4,
            'fold': 1,
            'option': 'mix_reg',
            'regular': True,
            'num_processes': 1,
            'enable_group_avg': True,
            'qubit_group_sizes': [8, 8],
            'estimator_job_chunk_size': 1,
            }
    else:
        base_task = {
                'task': 'QML_Hidden_48d',
                'n_qubits': 12,
                'n_layers': 4,
                'fold': 1,
                'option': 'mix_reg',
                'regular': True,
                'num_processes': 1,
                'enable_group_avg': True,
                'qubit_group_sizes': [6, 6],
                'estimator_job_chunk_size': 1,
                }
        

    # Read CSV and evaluate each structure, writing one row immediately.
    # Append mode keeps previous results instead of overwriting.
    fieldnames = [
        'iteration',
        'depth',
        'size',
        'CZ',
        'RZ',
        'SX',
        'path_number',
        'weight_file',
        'csv_acc',
        'tq_acc',
        'qi_noise_free_acc',
        'qi_heron_r1_acc',
        'qi_eagle_r3_acc',
        'status'
    ]

    output_has_required_header = has_required_header(output_file, fieldnames)
    file_mode = 'a' if output_has_required_header else 'w'
    if file_mode == 'w' and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Existing {output_file} header does not match required columns; overwriting with new header.")

    with open(csv_file, 'r', newline='') as in_f, open(output_file, file_mode, newline='') as out_f:
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if file_mode == 'w':
            writer.writeheader()
            out_f.flush()

        for row in reader:
            iteration = int(row['iteration'])

            if iteration < start_iteration:
                print(f"Skipping iteration {iteration} (< start_iteration={start_iteration})")
                continue

            row_out = {
                'iteration': iteration,
                'depth': 'N/A',
                'size': 'N/A',
                'CZ': 'N/A',
                'RZ': 'N/A',
                'SX': 'N/A',
                'path_number': 'N/A',
                'weight_file': 'N/A',
                'csv_acc': row['ACC'] if row.get('ACC') else 'N/A',
                'tq_acc': 'N/A',
                'qi_noise_free_acc': 'N/A',
                'qi_heron_r1_acc': 'N/A',
                'qi_eagle_r3_acc': 'N/A',
                'status': 'N/A'
            }

            print(f"\n{'='*60}")
            print(f"Evaluating Iteration {iteration}")
            print(f"{'='*60}")

            single, enta = parse_arch_code_from_csv(row['arch_code'])
            if single is None or enta is None:
                row_out['status'] = 'PARSE_ERROR'
                writer.writerow(row_out)
                out_f.flush()
                continue

            arch_code = [base_task['n_qubits'], base_task['n_layers']]
            try:
                design = single_enta_to_design(single, enta, arch_code, base_task['fold'])

                adj_matrix = design_to_adj_matrix(design)
                row_out['path_number'] = path_based_proxy(adj_matrix)

                transpiled_metrics = build_transpiled_metrics(design, base_task, transpile_backend_name)
                row_out.update(transpiled_metrics)
            except Exception as e:
                print(f"Error generating design/metrics for iteration {iteration}: {e}")
                row_out['status'] = 'DESIGN_ERROR'
                writer.writerow(row_out)
                out_f.flush()
                continue

            weight_path = find_weight_file(iteration, weights_dir)
            if weight_path is None:
                print(f"Weight file not found for iteration {iteration}")
                row_out['status'] = 'WEIGHT_NOT_FOUND'
                writer.writerow(row_out)
                out_f.flush()
                continue

            row_out['weight_file'] = weight_path
            print(f"Using weight: {weight_path}")

            if 'tq' in backends_to_run:
                try:
                    print("  Running tq...", flush=True)
                    result_tq = Scheme_eval(design, base_task, weight_path, backend='tq')
                    row_out['tq_acc'] = float(result_tq['acc'])
                    print(f"  tq acc: {row_out['tq_acc']:.6f}")
                except Exception as e:
                    row_out['tq_acc'] = 'N/A'
                    print(f"  tq acc: N/A (error: {e})")

            if 'qi_noise_free' in backends_to_run:
                try:
                    print("  Running qi noise-free...", flush=True)
                    result_qi_nf = Scheme_eval(
                        design, base_task, weight_path,
                        backend='qi', noise=False, seed=170, shots=10000, name='yorktown'
                    )
                    row_out['qi_noise_free_acc'] = float(result_qi_nf['acc'])
                    print(f"  qi noise-free acc: {row_out['qi_noise_free_acc']:.6f}")
                except Exception as e:
                    row_out['qi_noise_free_acc'] = 'N/A'
                    print(f"  qi noise-free acc: N/A (error: {e})")

            if 'heron_r1' in backends_to_run:
                print("  Running qi heron_r1 (fake, timeout-enabled)...", flush=True)
                fake_acc, timed_out, processed_samples = evaluate_fake_backend_with_timeout(
                    design, base_task, weight_path, fake_timeout_seconds
                )
                if timed_out:
                    print(
                        f"Iteration {iteration} fake backend timed out (>{fake_timeout_seconds}s); "
                        f"using {processed_samples} processed samples for partial accuracy."
                    )
                    row_out['qi_heron_r1_acc'] = fake_acc if fake_acc is not None else 'N/A'
                    row_out['status'] = 'FAKE_TIMEOUT_PARTIAL' if fake_acc is not None else 'FAKE_TIMEOUT'
                    if fake_acc is None:
                        print("  qi heron_r1 acc: N/A (timeout before any sample finished)")
                    else:
                        print(f"  qi heron_r1 acc: {row_out['qi_heron_r1_acc']} (partial, {processed_samples} samples)")
                else:
                    row_out['qi_heron_r1_acc'] = fake_acc if fake_acc is not None else 'N/A'
                    row_out['status'] = 'OK'
                    print(f"  qi heron_r1 acc: {row_out['qi_heron_r1_acc']}")

            if 'eagle_r3' in backends_to_run:
                print("  Running qi eagle_r3 / FakeKyiv (fake, timeout-enabled)...", flush=True)
                eagle_task = {**base_task, '_backend_name': 'eagle_r3'}
                eagle_acc, eagle_timed_out, eagle_processed = evaluate_fake_backend_with_timeout(
                    design, eagle_task, weight_path, fake_timeout_seconds
                )
                if eagle_timed_out:
                    print(
                        f"Iteration {iteration} eagle_r3 timed out (>{fake_timeout_seconds}s); "
                        f"using {eagle_processed} processed samples for partial accuracy."
                    )
                    row_out['qi_eagle_r3_acc'] = eagle_acc if eagle_acc is not None else 'N/A'
                    if row_out['status'] == 'OK':
                        row_out['status'] = 'EAGLE_TIMEOUT_PARTIAL' if eagle_acc is not None else 'EAGLE_TIMEOUT'
                    if eagle_acc is None:
                        print("  qi eagle_r3 acc: N/A (timeout before any sample finished)")
                    else:
                        print(f"  qi eagle_r3 acc: {row_out['qi_eagle_r3_acc']} (partial, {eagle_processed} samples)")
                else:
                    row_out['qi_eagle_r3_acc'] = eagle_acc if eagle_acc is not None else 'N/A'
                    if row_out['status'] not in ('FAKE_TIMEOUT_PARTIAL', 'FAKE_TIMEOUT'):
                        row_out['status'] = 'OK'
                    print(f"  qi eagle_r3 acc: {row_out['qi_eagle_r3_acc']}")

            writer.writerow(row_out)
            out_f.flush()

    print(f"\nResults saved incrementally to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate structures from a CSV file using weights from a directory.'
    )
    parser.add_argument('--csv_file', required=False, help='Path to input CSV file containing architectures.')
    parser.add_argument('--weights_dir', required=False, help='Directory containing iteration weight files.')
    args = parser.parse_args()

    # args.csv_file = 'simulation_results/Hidden_48/No margins/QML_Hidden_48d.csv'
    # args.weights_dir = 'simulation_results/Hidden_48/No margins/weights'

    main(args.csv_file, args.weights_dir)
