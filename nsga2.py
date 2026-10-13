import numpy as np
import torch
import random
import json
import csv
import os
import torch.multiprocessing as mp
from Arguments import Arguments
from FusionModel import single_enta_to_design, cir_to_matrix
from schemes import Scheme, Scheme_eval

# ====================== NSGA2 ======================
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

import pickle
import datetime
import copy

# ====================== 从你的MCTS直接搬过来的函数 ======================
def get_list_dimensions(lst):
    if isinstance(lst, (list, tuple)) and not isinstance(lst[0], (int, float)):
        return 1 + get_list_dimensions(lst[0])
    else:
        return 1

def insert_job(change_code, job_input):
    job = copy.deepcopy(job_input)
    if not isinstance(job[0], list):
        job = [job]
    qubit = [sub[0] for sub in job]
    if change_code is not None:
        for change in change_code:
            if change[0] not in qubit:
                job.append(change)
    return job

def count_gates(arch, arch_code=None, fold=1, coeff=None):
    if arch_code is None:
        args = Arguments()
        qubits = int(args.n_qubits / args.fold)
        layers = args.n_layers
    else:
        qubits = int(arch_code[0] / fold)
        layers = arch_code[1]
    x = [[0, 0, i] * 4 for i in range(1, qubits + 1)]
    x = np.transpose(x, (1, 0))
    x = np.sign(abs(x - arch))

    stat = {
        'uploading': x[[3 * i for i in range(layers)]].sum(),
        'single': x[[3 * i + 1 for i in range(layers)]].sum(),
        'enta': x[[3 * i + 2 for i in range(layers)]].sum()
    }
    total = stat['uploading'] + stat['single'] + stat['enta']
    return total, stat


def get_resume_state_path(task):
    os.makedirs('states', exist_ok=True)
    task_name = str(task.get('task', 'task')).replace('/', '_').replace(' ', '_')
    option_name = str(task.get('option', 'run')).replace('/', '_').replace(' ', '_')
    return os.path.join('states', f'nsga2_{task_name}_{option_name}.pkl')


def save_resume_state(agent, task, completed_round, last_res=None):
    state = {
        'completed_round': int(completed_round),
        'task': copy.deepcopy(task),
        'arch_code': copy.deepcopy(agent.ARCH_CODE),
        'fold': agent.fold,
        'single': copy.deepcopy(agent.single),
        'enta': copy.deepcopy(agent.enta),
        'base_single': copy.deepcopy(agent.base_single),
        'base_enta': copy.deepcopy(agent.base_enta),
        'weight': copy.deepcopy(agent.weight),
        'weight_file': getattr(agent, 'weight_file', None),
        'task_name': getattr(agent, 'task_name', None),
        'search_history': copy.deepcopy(getattr(agent, 'search_history', [])),
        'last_X': None if last_res is None else np.asarray(last_res.X).copy(),
        'last_F': None if last_res is None else np.asarray(last_res.F).copy(),
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'timestamp': datetime.datetime.now().isoformat()
    }
    if torch.cuda.is_available():
        state['torch_cuda_random_state_all'] = torch.cuda.get_rng_state_all()

    with open(get_resume_state_path(task), 'wb') as f:
        pickle.dump(state, f)


def load_resume_state(task):
    state_path = get_resume_state_path(task)
    if not os.path.exists(state_path):
        return None

    with open(state_path, 'rb') as f:
        return pickle.load(f)


def restore_agent_from_state(agent, state):
    agent.single = copy.deepcopy(state['single'])
    agent.enta = copy.deepcopy(state['enta'])
    agent.base_single = copy.deepcopy(state.get('base_single', state['single']))
    agent.base_enta = copy.deepcopy(state.get('base_enta', state['enta']))
    agent.weight = copy.deepcopy(state['weight'])
    agent.weight_file = state.get('weight_file')
    agent.search_history = copy.deepcopy(state.get('search_history', []))
    agent.completed_round = int(state.get('completed_round', len(agent.search_history)))

    if 'python_random_state' in state:
        random.setstate(state['python_random_state'])
    if 'numpy_random_state' in state:
        np.random.set_state(state['numpy_random_state'])
    if 'torch_random_state' in state:
        torch.set_rng_state(state['torch_random_state'])
    if torch.cuda.is_available() and 'torch_cuda_random_state_all' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda_random_state_all'])

    return agent

# ====================== 完全和MCTS一样的初始化 ======================
def create_agent(task, arch_code,epoches=1):
    
    class SimpleAgent:
        def __init__(self, fold, arch_code):
            
            self.fold = fold
            self.ARCH_CODE = arch_code
            self.weight = None
            self.explorations = {'regular': [0.001, 0.002, 0.002]}
            self.task_name = task['task'] + '_' + task['option']

            nq = int(arch_code[0] / fold)
            nl = arch_code[1]
            self.single = [[i] + [1] * 2 * nl for i in range(1, nq + 1)]
            self.enta = [[i] + [i + 1] * nl for i in range(1, nq)] + [[nq] + [1] * nl]
            self.base_single = copy.deepcopy(self.single)
            self.base_enta = copy.deepcopy(self.enta)
            self.weight_file = None
            self.search_history = []
            self.completed_round = 0

    agent = SimpleAgent(task['fold'], arch_code)

    if task.get('resume', True):
        state = load_resume_state(task)
        if state is not None:
            agent = restore_agent_from_state(agent, state)
            print(f"🔁 检测到断点，已恢复到第 {agent.completed_round} 轮结束后的状态")
            return agent

    design = single_enta_to_design(agent.single, agent.enta, arch_code, task['fold'])

    best_model, report = Scheme(design, task, 'init', epoches, None, 'save')
    os.makedirs('weights', exist_ok=True)
    current_time = datetime.datetime.now().strftime('%m-%d-%H')
    weight_name = f'weight_init_{current_time}.pth'
    weight_path = os.path.join('weights', weight_name)
    torch.save(best_model.state_dict(), weight_path)
    agent.weight = copy.deepcopy(best_model.state_dict())
    agent.weight_file = weight_name
    agent.search_history = []
    agent.completed_round = 0

    with open(f'results/{agent.task_name}_fine.csv', 'a+', newline='') as f:
        csv.writer(f).writerow([0, [agent.single, agent.enta], report['mae']])

    save_resume_state(agent, task, completed_round=0)
    print(f"✅ 初始化完成 ")
    return agent

# ====================== 针对完整电路编码的 NSGA-II ======================
def encode_architecture(single, enta):
    genes = []
    for row in single:
        genes.extend(int(v) for v in row[1:])
    for row in enta:
        genes.extend(int(v) for v in row[1:])
    return np.array(genes, dtype=int)


def decode_architecture(vector, arch_code, fold=1):
    qubits = int(arch_code[0] / fold)
    layers = arch_code[1]
    vector = np.asarray(np.rint(vector), dtype=int).tolist()

    single_gene_count = qubits * 2 * layers
    enta_gene_count = qubits * layers
    expected_len = single_gene_count + enta_gene_count
    if len(vector) != expected_len:
        raise ValueError(f"编码长度不匹配: got={len(vector)}, expected={expected_len}")

    single_flat = vector[:single_gene_count]
    enta_flat = vector[single_gene_count:]

    single = []
    offset = 0
    for q in range(1, qubits + 1):
        row = [q] + single_flat[offset: offset + 2 * layers]
        single.append(row)
        offset += 2 * layers

    enta = []
    offset = 0
    for q in range(1, qubits + 1):
        row = [q]
        for _ in range(layers):
            target = int(enta_flat[offset])
            target = max(1, min(qubits, target))
            row.append(target)
            offset += 1
        enta.append(row)

    return single, enta


def get_single_qubit_gene_indices(problem, qubit_idx):
    single_start = qubit_idx * problem.single_genes_per_qubit
    single_end = single_start + problem.single_genes_per_qubit
    return list(range(single_start, single_end))


def get_enta_qubit_gene_indices(problem, qubit_idx):
    enta_start = problem.single_gene_count + qubit_idx * problem.enta_genes_per_qubit
    enta_end = enta_start + problem.enta_genes_per_qubit
    return list(range(enta_start, enta_end))


def apply_mutation_to_indices(mutated, change_indices, problem, enta_step_threshold=1):
    for idx in change_indices:
        if idx < problem.single_gene_count:
            mutated[idx] = 1 - int(mutated[idx])
        else:
            current = int(mutated[idx])
            lower = int(problem.xl[idx])
            upper = int(problem.xu[idx])
            if enta_step_threshold is None:
                candidates = [v for v in range(lower, upper + 1) if v != current]
            else:
                step = max(1, int(enta_step_threshold))
                left = max(lower, current - step)
                right = min(upper, current + step)
                candidates = [v for v in range(left, right + 1) if v != current]
                if not candidates:
                    candidates = [v for v in range(lower, upper + 1) if v != current]
            if candidates:
                mutated[idx] = random.choice(candidates)
    return mutated


def mutate_vector(vector, problem, mutation_threshold, enta_step_threshold=1, qubit_mutation_prob=0.3):
    mutated = np.asarray(vector, dtype=int).copy()
    max_single_changes = int(max(1, min(mutation_threshold, problem.single_genes_per_qubit)))
    max_enta_changes = int(max(1, min(mutation_threshold, problem.enta_genes_per_qubit)))

    selected_single_qubits = [
        q for q in range(problem.n_effective_qubits)
        if np.random.rand() < qubit_mutation_prob
    ]
    selected_enta_qubits = [
        q for q in range(problem.n_effective_qubits)
        if np.random.rand() < qubit_mutation_prob
    ]

    if not selected_single_qubits and not selected_enta_qubits:
        fallback_qubit = np.random.randint(0, problem.n_effective_qubits)
        if np.random.rand() < 0.5:
            selected_single_qubits = [fallback_qubit]
        else:
            selected_enta_qubits = [fallback_qubit]

    for qubit_idx in selected_single_qubits:
        single_indices = get_single_qubit_gene_indices(problem, qubit_idx)
        n_changes = random.randint(1, min(max_single_changes, len(single_indices)))
        change_indices = np.random.choice(single_indices, size=n_changes, replace=False)
        mutated = apply_mutation_to_indices(mutated, change_indices, problem, enta_step_threshold)

    for qubit_idx in selected_enta_qubits:
        enta_indices = get_enta_qubit_gene_indices(problem, qubit_idx)
        n_changes = random.randint(1, min(max_enta_changes, len(enta_indices)))
        change_indices = np.random.choice(enta_indices, size=n_changes, replace=False)
        mutated = apply_mutation_to_indices(mutated, change_indices, problem, enta_step_threshold)

    return mutated


class ArchitectureSampling(Sampling):
    def __init__(self, base_vector, mutation_threshold=3, enta_step_threshold=1, qubit_mutation_prob=0.3):
        super().__init__()
        self.base_vector = np.asarray(base_vector, dtype=int)
        self.mutation_threshold = mutation_threshold
        self.enta_step_threshold = enta_step_threshold
        self.qubit_mutation_prob = qubit_mutation_prob

    def _do(self, problem, n_samples, **kwargs):
        candidates = []
        seen = set()
        base_tuple = tuple(self.base_vector.tolist())

        max_attempts = max(50, n_samples * 30)
        attempts = 0
        while len(candidates) < n_samples and attempts < max_attempts:
            attempts += 1
            candidate = mutate_vector(
                self.base_vector,
                problem,
                self.mutation_threshold,
                self.enta_step_threshold,
                self.qubit_mutation_prob,
            ).astype(int)
            candidate_tuple = tuple(candidate.tolist())

            if candidate_tuple == base_tuple:
                continue
            if candidate_tuple in seen:
                continue

            seen.add(candidate_tuple)
            candidates.append(candidate)

        if len(candidates) < n_samples:
            raise RuntimeError(
                f"无法生成足够的非 base 候选结构: need={n_samples}, got={len(candidates)}"
            )

        return np.array(candidates, dtype=int)


class ArchitectureCrossover(Crossover):
    def __init__(self, prob=0.9):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=int)

        for k in range(n_matings):
            parent_a = X[0, k].copy()
            parent_b = X[1, k].copy()
            child_a = parent_a.copy()
            child_b = parent_b.copy()

            if np.random.rand() < self.prob:
                mask = np.random.rand(n_var) < 0.5
                child_a[mask] = parent_b[mask]
                child_b[mask] = parent_a[mask]

            Y[0, k, :] = child_a
            Y[1, k, :] = child_b

        return Y


class ThresholdMutation(Mutation):
    def __init__(self, prob=1.0, mutation_threshold=3, enta_step_threshold=1, qubit_mutation_prob=0.3):
        super().__init__()
        self.prob = prob
        self.mutation_threshold = mutation_threshold
        self.enta_step_threshold = enta_step_threshold
        self.qubit_mutation_prob = qubit_mutation_prob

    def _do(self, problem, X, **kwargs):
        Y = np.asarray(X, dtype=int).copy()
        for i in range(len(Y)):
            if np.random.rand() < self.prob:
                Y[i] = mutate_vector(
                    Y[i],
                    problem,
                    self.mutation_threshold,
                    self.enta_step_threshold,
                    self.qubit_mutation_prob,
                )
        return Y


def get_mutated_qubits(base_rows, new_rows):
    changed = []
    for base_row, new_row in zip(base_rows, new_rows):
        if list(base_row[1:]) != list(new_row[1:]):
            changed.append(int(base_row[0]))
    return changed


def evaluate_architecture(single, enta, agent, task):
    try:
        design = single_enta_to_design(single, enta, agent.ARCH_CODE, agent.fold)
        short_epochs = int(task.get('candidate_epochs', 2))
        best_model, report = Scheme(design, task, agent.weight, short_epochs, verbs=1)
        acc = report['mae']
        mat = cir_to_matrix(single, enta, agent.ARCH_CODE, agent.fold)
        gates, gate_stat = count_gates(mat, agent.ARCH_CODE, agent.fold)
        mutated_single_qubits = get_mutated_qubits(agent.base_single, single)
        mutated_enta_qubits = get_mutated_qubits(agent.base_enta, enta)
        print(f"变异qubit | single={mutated_single_qubits} | enta={mutated_enta_qubits}")
        meta = {
            'single': copy.deepcopy(single),
            'enta': copy.deepcopy(enta),
            'mutated_single_qubits': mutated_single_qubits,
            'mutated_enta_qubits': mutated_enta_qubits,
            'acc': acc,
            'gates': gates,
            'gate_stat': gate_stat,
            'weight': copy.deepcopy(best_model.state_dict()),
            'objective': [-acc, gates]
        }
        return [-acc, gates], meta
    except Exception as e:
        print("评估错误:", e)
        meta = {
            'single': copy.deepcopy(single),
            'enta': copy.deepcopy(enta),
            'mutated_single_qubits': [],
            'mutated_enta_qubits': [],
            'acc': 0.001,
            'gates': 999,
            'gate_stat': None,
            'weight': None,
            'objective': [-0.001, 999]
        }
        return [-0.001, 999], meta


def candidate_score(meta, task):
    acc = float(meta['acc'])
    gates = float(meta['gates'])
    if task.get('regular', False):
        penalty = float(task.get('gate_penalty_lambda', 0.001)) * gates
        return acc - penalty
    return acc


def select_best_candidate(res, problem, task):
    best_idx = None
    best_score = -float('inf')
    best_meta = None

    for i, x in enumerate(np.asarray(res.X, dtype=int)):
        key = tuple(np.asarray(x, dtype=int).tolist())
        meta = problem.evaluated_meta[key]
        score = candidate_score(meta, task)
        if score > best_score:
            best_score = score
            best_idx = i
            best_meta = meta

    return best_idx, best_meta, best_score


class QuantumProblem(Problem):
    def __init__(self, agent, task, mutation_threshold=3, enta_step_threshold=1):
        self.agent = agent
        self.task = task
        self.evaluated = {}
        self.evaluated_meta = {}

        qubits = int(agent.ARCH_CODE[0] / agent.fold)
        layers = agent.ARCH_CODE[1]
        self.n_effective_qubits = qubits
        self.single_genes_per_qubit = 2 * layers
        self.enta_genes_per_qubit = layers
        self.genes_per_qubit = self.single_genes_per_qubit + self.enta_genes_per_qubit
        self.single_gene_count = qubits * self.single_genes_per_qubit
        self.enta_gene_count = qubits * self.enta_genes_per_qubit
        self.base_vector = encode_architecture(agent.base_single, agent.base_enta)
        self.mutation_threshold = int(max(1, mutation_threshold))
        self.enta_step_threshold = enta_step_threshold

        xl = np.array([0] * self.single_gene_count + [1] * self.enta_gene_count, dtype=int)
        xu = np.array([1] * self.single_gene_count + [qubits] * self.enta_gene_count, dtype=int)
        super().__init__(n_var=len(self.base_vector), n_obj=2, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in np.asarray(X, dtype=int):
            key = tuple(np.asarray(x, dtype=int).tolist())
            if key in self.evaluated:
                F.append(self.evaluated[key])
            else:
                single, enta = decode_architecture(x, self.agent.ARCH_CODE, self.agent.fold)
                val, meta = evaluate_architecture(single, enta, self.agent, self.task)
                self.evaluated[key] = val
                self.evaluated_meta[key] = meta
                F.append(val)
        out["F"] = np.array(F)

# ====================== 运行 ======================
def run_nsga2(agent, task):
    mutation_threshold = task.get('mutation_threshold', 3)
    enta_step_threshold = task.get('enta_step_threshold', 1)
    mutation_prob = task.get('mutation_prob', 0.35)
    pop_size = task.get('pop_size', 10)
    n_offsprings = task.get('n_offsprings', 10)
    n_gen = task.get('n_gen', 1)
    search_rounds = task.get('search_rounds', 10)
    finetune_epochs = int(task.get('finetune_epochs', 20))
    search_seed = task.get('search_seed', None)

    last_res = None
    start_round = int(getattr(agent, 'completed_round', len(getattr(agent, 'search_history', []))))

    if start_round >= search_rounds:
        print(f"✅ 检测到搜索已经完成，共 {search_rounds} 轮；如需重跑请关闭 resume 或删除断点文件。")
        return None

    try:
        for round_idx in range(start_round, search_rounds):
            print(f"\n===== NSGA-II 搜索轮次 {round_idx + 1}/{search_rounds} =====")
            print(f"候选结构短训练轮数: {task.get('candidate_epochs', 2)} | 最优结构强化训练轮数: {finetune_epochs}")

            problem = QuantumProblem(agent, task, mutation_threshold, enta_step_threshold)
            algorithm = NSGA2(
                pop_size=pop_size,
                n_offsprings=n_offsprings,
                sampling=ArchitectureSampling(
                    problem.base_vector,
                    mutation_threshold,
                    enta_step_threshold,
                    qubit_mutation_prob=mutation_prob,
                ),
                crossover=ArchitectureCrossover(prob=task.get('crossover_prob', 0.9)),
                mutation=ThresholdMutation(
                    prob=1.0,
                    mutation_threshold=mutation_threshold,
                    enta_step_threshold=enta_step_threshold,
                    qubit_mutation_prob=mutation_prob,
                ),
                eliminate_duplicates=True
            )
            round_seed = None if search_seed is None else search_seed + round_idx
            res = minimize(problem, algorithm, ('n_gen', n_gen), seed=round_seed, verbose=True)
            last_res = res

            best_idx, best_meta, best_score = select_best_candidate(res, problem, task)
            if best_meta is None:
                raise RuntimeError('NSGA-II 未找到可用候选结构')

            best_single = copy.deepcopy(best_meta['single'])
            best_enta = copy.deepcopy(best_meta['enta'])
            best_design = single_enta_to_design(best_single, best_enta, agent.ARCH_CODE, agent.fold)

            print(f"当轮最优候选: idx={best_idx}, ACC={best_meta['acc']:.4f}, 门数={best_meta['gates']:.0f}, 评分={best_score:.4f}")
            print(f"best训练前变异qubit | single={best_meta.get('mutated_single_qubits', [])} | enta={best_meta.get('mutated_enta_qubits', [])}")
            best_model, best_report = Scheme(best_design, task, agent.weight, finetune_epochs, verbs=0)

            agent.single = copy.deepcopy(best_single)
            agent.enta = copy.deepcopy(best_enta)
            agent.base_single = copy.deepcopy(best_single)
            agent.base_enta = copy.deepcopy(best_enta)
            agent.weight = copy.deepcopy(best_model.state_dict())

            current_time = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
            weight_name = f'weight_{round_idx + 1}_{current_time}.pth'
            torch.save(best_model.state_dict(), os.path.join('weights', weight_name))
            agent.weight_file = weight_name
            agent.search_history.append({
                'round': round_idx + 1,
                'candidate_acc': float(best_meta['acc']),
                'candidate_gates': float(best_meta['gates']),
                'finetuned_acc': float(best_report['mae']),
                'single': copy.deepcopy(best_single),
                'enta': copy.deepcopy(best_enta)
            })
            agent.completed_round = round_idx + 1

            with open(f'results/{agent.task_name}_fine.csv', 'a+', newline='') as f:
                csv.writer(f).writerow([
                    round_idx + 1,
                    [best_single, best_enta],
                    best_report['mae'],
                    best_meta['acc'],
                    best_meta['gates']
                ])

            save_resume_state(agent, task, completed_round=agent.completed_round, last_res=res)

    except KeyboardInterrupt:
        print("\n⚠️ 运行被中断，已自动保存断点。下次直接重新运行即可继续。")
        save_resume_state(agent, task, completed_round=getattr(agent, 'completed_round', start_round), last_res=last_res)
        raise

    return last_res

# ====================== 主函数 ======================
if __name__ == "__main__":
    mp.set_start_method('spawn')

    task = {
        'task': 'QML_Linear_64d',
        'n_qubits': 16,
        'n_layers': 4,
        'fold': 1,
        'option': 'nsga2',
        'regular': False,
        'num_processes': 1,
        'mutation_threshold': 4,
        'enta_step_threshold': 4,
        'mutation_prob': 0.5,
        'crossover_prob': 0.9,
        'pop_size': 10,
        'n_offsprings': 10,
        'n_gen': 1,
        'candidate_epochs': 2,
        'finetune_epochs': 20,
        'search_rounds': 30,
        'gate_penalty_lambda': 0.001,
        'search_seed': None,
        'global_seed': None,
        'enable_group_avg': True,
        'qubit_group_sizes': [8, 8],
        'resume': True,
    }

    global_seed = task.get('global_seed', None)
    if global_seed is not None:
        random.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)

    arch_code = [task['n_qubits'], task['n_layers']]
    args = Arguments(**task)
    os.makedirs('results', exist_ok=True)

    agent = create_agent(task, arch_code, epoches=1)
    res = run_nsga2(agent, task)

    # ====================== 输出 + 保存结果 ======================
    if res is not None:
        acc_list = [-f[0] for f in res.F]
        gate_list = [f[1] for f in res.F]
        structure_list = [decode_architecture(x, agent.ARCH_CODE, agent.fold) for x in res.X]
    else:
        mat = cir_to_matrix(agent.single, agent.enta, agent.ARCH_CODE, agent.fold)
        gates, _ = count_gates(mat, agent.ARCH_CODE, agent.fold)
        last_acc = agent.search_history[-1]['finetuned_acc'] if agent.search_history else 0.0
        acc_list = [last_acc]
        gate_list = [gates]
        structure_list = [(agent.single, agent.enta)]

    print("\n===== Pareto 前沿 =====")
    for i, (acc, gates, arch) in enumerate(zip(acc_list, gate_list, structure_list)):
        single, enta = arch
        print(f"【{i+1}】 ACC={acc:.4f} | 门数={gates:.0f} | single={single} | enta={enta}")

    # 保存结果到 CSV 文件
    with open("nsga2_pareto_result.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "准确率(ACC)", "总门数", "single编码", "enta编码"])
        for i, (acc, gates, arch) in enumerate(zip(acc_list, gate_list, structure_list)):
            single, enta = arch
            writer.writerow([i+1, round(acc, 4), int(gates), json.dumps(single, ensure_ascii=False), json.dumps(enta, ensure_ascii=False)])

    print(f"\n✅ 结果已保存至：nsga2_pareto_result.csv | 断点文件：{get_resume_state_path(task)}")
