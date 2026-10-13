import copy

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError
)

TWO_QUBIT_GATES = {'cx', 'cz', 'rzz', 'ecr'}

# ==============================================
# 【芯片噪声配置中心】—— 新增芯片只在这里加！
# ==============================================
CHIP_CONFIGS = {
    'yorktown':{
        "num_qubits": 4,  # 测试用4量子比特
        "single_err": 1e-3,  # SX门误差 (median)
        "two_err": 4e-2,  # CZ门误差 (median)
        "t1_us": 40,  # T1 (μs)
        "t2_us": 40,  # T2 (μs)
        "readout_err": 8e-2,  # 测量误差 (median)
        "single_gate_ns": 50,  # 单比特门时间
        "two_gate_ns": 300,  # 双比特门时间
        "basis_gates": ['cx', 'id', 'rz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    "heron_r1": {
        "num_qubits": 133,  # 测试用4量子比特，实际为133
        "single_err": 2.999e-4,  # SX门误差 (median)
        "two_err": 2.589e-3,  # CZ门误差 (median)
        "t1_us": 175.85,  # T1 (μs)
        "t2_us": 134.17,  # T2 (μs)
        "readout_err": 2.95e-2,  # 测量误差 (median)
        "single_gate_ns": 50,  # 单比特门时间
        "two_gate_ns": 300,  # 双比特门时间
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x'],
        "coupling_map": None
    },
    "heron_r2": {
        "num_qubits": 4,  # 测试用4量子比特，实际为156
        "single_err": 2.263e-4,  # SX门误差
        "two_err": 2.194e-3,  # CZ门误差
        "t1_us": 238.22,  # T1 (μs)
        "t2_us": 123.16,  # T2 (μs)
        "readout_err": 0.0117,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    "heron_r3": {
        "num_qubits": 4,  # 测试用4量子比特，实际为156
        "single_err": 1.484e-4,  # SX门误差
        "two_err": 1.199e-3,  # CZ门误差
        "t1_us": 282.14,  # T1 (μs)
        "t2_us": 327.59,  # T2 (μs)
        "readout_err": 5.25e-3,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'rzz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    "nighthawk_r1": {
        "num_qubits": 120,
        "single_err": 2.38e-4,  # SX门误差
        "two_err": 2.594e-3,  # CZ门误差
        "t1_us": 357.64,  # T1 (μs)
        "t2_us": 273.63,  # T2 (μs)
        "readout_err": 2.14e-2,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "basis_gates": ['cz', 'id', 'rx', 'rz', 'sx', 'x'],
        "coupling_map": None
    },
    "eagle_r3": {
        "num_qubits": 4,  # 测试用4量子比特，实际为127
        "single_err": 2.594e-4,  # SX门误差
        "two_err": 7.782e-3,  # ECR门误差
        "t1_us": 241.79,  # T1 (μs)
        "t2_us": 138.99,  # T2 (μs)
        "readout_err": 3.00e-2,  # 测量误差
        "single_gate_ns": 50,
        "two_gate_ns": 300,
        "basis_gates": ['ecr', 'id', 'rz', 'sx', 'x', 'measure'],
        "coupling_map": None
    },
    'toronto':{
        "num_qubits": 27,
        "single_err": 5.0318e-4,  # SX门误差
        "two_err": 8.94e-3,  # ECR门误差
        "t1_us": 112.79,  # T1 (μs)
        "t2_us": 153.99,  # T2 (μs)
        "readout_err": 5.00e-2,  # 测量误差
        "single_gate_ns": 568,
        "two_gate_ns": 6000,
        "basis_gates": [ 'id', 'rz', 'sx', 'x', 'cx'],
        "coupling_map": None
    },
}

# print(f"\n\n可选择芯片列表：{list(CHIP_CONFIGS.keys())}\n\n")
def get_chip_config(chip_name: str):
    if chip_name not in CHIP_CONFIGS:
        raise ValueError(f"芯片 {chip_name} 不存在！可选项：{list(CHIP_CONFIGS.keys())}")

    return copy.deepcopy(CHIP_CONFIGS[chip_name])


def create_noise_model(chip_name: str):
    """
    根据芯片名自动创建带对应噪声的 NoiseModel
    """
    cfg = get_chip_config(chip_name)

    # 单位转换
    t1_ns = cfg["t1_us"] * 1000
    t2_ns = cfg["t2_us"] * 1000

    # 噪声模型
    noise_model = NoiseModel()

    # ============== 单比特门噪声 ==============
    # 基础单比特门噪声：对有物理脉冲的单比特门叠加退极化和热弛豫。
    err_single = depolarizing_error(cfg["single_err"], 1)
    thermal_single = thermal_relaxation_error(t1_ns, t2_ns, cfg["single_gate_ns"])
    active_single_error = err_single.compose(thermal_single)

    active_single_qubit_gates = [
        gate for gate in cfg["basis_gates"]
        if gate not in TWO_QUBIT_GATES and gate not in {'measure', 'rz', 'id'}
    ]
    if active_single_qubit_gates:
        noise_model.add_all_qubit_quantum_error(active_single_error, active_single_qubit_gates)

    # IBM 风格后端里 rz 通常是 virtual gate，不额外施加门误差。
    if 'id' in cfg["basis_gates"]:
        noise_model.add_all_qubit_quantum_error(thermal_single, ['id'])

    # ============== 双比特门噪声 ==============
    # 基础双比特门噪声：两个 qubit 都施加热弛豫，再叠加双比特退极化。
    err_two = depolarizing_error(cfg["two_err"], 2)
    thermal_two_single = thermal_relaxation_error(t1_ns, t2_ns, cfg["two_gate_ns"])
    thermal_two = thermal_two_single.tensor(thermal_two_single)
    two_qubit_error = err_two.compose(thermal_two)

    two_qubit_gates = [
        gate for gate in cfg["basis_gates"]
        if gate in TWO_QUBIT_GATES
    ]
    if two_qubit_gates:
        noise_model.add_all_qubit_quantum_error(two_qubit_error, two_qubit_gates)

    # ============== 测量噪声 ==============
    read_err = ReadoutError([
        [1 - cfg["readout_err"], cfg["readout_err"]],
        [cfg["readout_err"], 1 - cfg["readout_err"]]
    ])
    noise_model.add_all_qubit_readout_error(read_err)

    return noise_model