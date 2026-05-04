import numpy as np
from pymatching import Matching
from collections import defaultdict
import functools

from config import DEBUG_UF_GEOMETRY, DECODER_CONFIG
from tools.pre_processing import process_syndrome
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices

try:
    import stimcircuits
except ImportError:
    stimcircuits = None

def build_syndrome_from_detector_shot(detector_coords, syndrome_shot, code_type, L, num_x_stabs, num_z_stabs, repetitions):
    """Build syndrome dict/array structures from one detector shot."""
    syndrome_dict_x = defaultdict(int)
    syndrome_dict_z = defaultdict(int)
    syndrome_array_x = np.zeros((num_x_stabs, repetitions + 1))
    syndrome_array_z = np.zeros((num_z_stabs, repetitions + 1))

    for j in range(len(detector_coords)):
        if syndrome_shot[j] == 1:
            if code_type == 'rotated':
                col = detector_coords[j][0]
                row = detector_coords[j][1]
                time_step = detector_coords[j][2]
                if (col + row) % 2 == 1:
                    syndrome_dict_x[(row, col - 1, time_step)] = 1
                else:
                    syndrome_dict_z[(row - 1, col, time_step)] = 1
            elif code_type == 'toric':
                col = detector_coords[j][0] // 2
                row = detector_coords[j][1] // 2
                time_step = detector_coords[j][2]
                if detector_coords[j][1] % 2 == 0:
                    syndrome_dict_x[(row, col, time_step)] = 1
                    syndrome_array_x[int(col * L + row), int(time_step)] = 1
                else:
                    syndrome_dict_z[(row, col, time_step)] = 1
                    syndrome_array_z[int(col * L + row), int(time_step)] = 1

    return syndrome_dict_x, syndrome_dict_z, syndrome_array_x, syndrome_array_z

def batchify_noise_generation(func):
    """Batch decorator specifically for noise generation functions (generator version, low memory)

    Usage:
    @batchify_noise_generation
    def generate_random_error_and_syndrome(..., rng=None):
        ...

    # Call (generate one sample each time)
    for result in generate_random_error_and_syndrome(..., batch_size=1000):
        ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batch_size = kwargs.pop("batch_size", 1)
        if batch_size == 1:
            yield func(*args, **kwargs)
            return

        seed_seq = np.random.SeedSequence()
        child_seeds = seed_seq.spawn(batch_size)

        for seed in child_seeds:
            rng = np.random.default_rng(seed)
            yield func(*args, rng=rng, **kwargs)

    return wrapper


@batchify_noise_generation

def generate_random_error_and_syndrome(
    L, code_structure,
    before_round_data_error_rate=0.0,
    before_measure_error_rate=0.0,
    error_type="depolarize",
    rng=None
):
    """Generate random errors and corresponding syndrome
    Args:
        L: Grid size
        Hx: X stabilizer check matrix
        Hz: Z stabilizer check matrix
        repetitions: Number of measurement rounds
        before_round_data_error_rate: Data error probability
        before_measure_error_rate: Measurement error probability
        error_type: Error type ("bit_flip", "polarize")
        rng: numpy.random.Generator instance (passed by decorator)
    Returns:
        tuple: (noise_total_x, noise_total_z, noisy_syndrome_x, noisy_syndrome_z)
    """
    rng = rng or np.random.default_rng()
    repetitions = code_structure.repetitions
    Hx = code_structure.H_x
    Hz = code_structure.H_z
    # error_channel = code_structure.error_channel

    num_stabs_x, num_qubits = Hx.shape
    num_stabs_z, _ = Hz.shape

    noise_type = np.zeros((num_qubits, repetitions), dtype=np.int8)
    pure_syndrome_x = np.zeros((num_stabs_x, repetitions+2), dtype=np.int8)
    pure_syndrome_z = np.zeros((num_stabs_z, repetitions+2), dtype=np.int8)

    if error_type == "depolarize":
        rand_arr = rng.random((num_qubits, repetitions))
        error_mask = rand_arr < before_round_data_error_rate
        num_errors_occ = np.sum(error_mask)
        if num_errors_occ > 0:
            noise_type[error_mask] = (np.floor(rng.random(num_errors_occ) * 3) + 1).astype(np.int8)
        effective_error_x = ((noise_type == 1) | (noise_type == 2)).astype(np.int8)
        effective_error_z = ((noise_type == 2) | (noise_type == 3)).astype(np.int8)

    elif error_type == "phenomenological":
        # Generate phenomenological data noise. By default, X/Y/Z are uniform;
        # when enabled in config, biased Pauli probabilities are used.
        rand_arr = rng.random((num_qubits, repetitions+2))
        error_mask = rand_arr < before_round_data_error_rate
        
        # Initialize noise type array
        noise_type = np.zeros((num_qubits, repetitions+2), dtype=np.int8)
        use_bias = bool(DECODER_CONFIG.get('enable_biased_noise', False))
        if use_bias:
            pauli_probs_cfg = DECODER_CONFIG.get('biased_noise_pauli_probs', {})
            px = float(pauli_probs_cfg.get('x', 1.0 / 3.0))
            py = float(pauli_probs_cfg.get('y', 1.0 / 3.0))
            pz = float(pauli_probs_cfg.get('z', 1.0 / 3.0))
            pauli_probs = np.asarray([px, py, pz], dtype=float)
            pauli_probs = np.clip(pauli_probs, 0.0, None)
            prob_sum = pauli_probs.sum()
            if prob_sum <= 0:
                pauli_probs = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
            else:
                pauli_probs = pauli_probs / prob_sum
        
        # For each error position, randomly assign X(1), Y(2), Z(3) noise
        for i in range(repetitions+2):
            error_positions = np.where(error_mask[:, i])[0]
            if len(error_positions) > 0:
                if use_bias:
                    noise_type[error_positions, i] = rng.choice(
                        np.asarray([1, 2, 3], dtype=np.int8),
                        size=len(error_positions),
                        p=pauli_probs,
                    ).astype(np.int8)
                else:
                    # Randomly assign 1(X), 2(Y), 3(Z) with uniform probability.
                    noise_type[error_positions, i] = rng.integers(1, 4, size=len(error_positions)).astype(np.int8)
        
        # Calculate effective X and Z errors
        # X errors: type 1 (X) or 2 (Y)
        # Z errors: type 2 (Y) or 3 (Z)
        effective_error_x = ((noise_type == 1) | (noise_type == 2)).astype(np.int8)
        effective_error_z = ((noise_type == 2) | (noise_type == 3)).astype(np.int8)

        # No errors in first round
        error_mask[:, 0] = 0
        effective_error_x[:, 0] = 0
        effective_error_z[:, 0] = 0

        noise_total_x = effective_error_x[:, -1]
        noise_total_z = effective_error_z[:, -1]

        for i in range(repetitions + 2):
            syndrome_x = (Hx @ effective_error_x[:, i]) % 2
            syndrome_z = (Hz @ effective_error_z[:, i]) % 2
            pure_syndrome_x[:, i] = syndrome_x
            pure_syndrome_z[:, i] = syndrome_z

        pure_syndrome_x[:, 0] = 0
        pure_syndrome_z[:, 0] = 0

        meas_error_x = (rng.random((num_stabs_x, repetitions+2)) < before_measure_error_rate).astype(np.int8)
        meas_error_x[:, [0, -1]] = 0
        noisy_syndrome_x = (pure_syndrome_x + meas_error_x) % 2
        noisy_syndrome_x = np.abs(np.diff(noisy_syndrome_x, axis=1))

        meas_error_z = (rng.random((num_stabs_z, repetitions+2)) < before_measure_error_rate).astype(np.int8)
        meas_error_z[:, [0, -1]] = 0
        noisy_syndrome_z = (pure_syndrome_z + meas_error_z) % 2
        noisy_syndrome_z = np.abs(np.diff(noisy_syndrome_z, axis=1))

        actual_logX = (noise_total_x @ code_structure.logicals_x.T) % 2
        actual_logZ = (noise_total_z @ code_structure.logicals_z.T) % 2

        # if np.any(noise_total_x != 0) or np.any(noise_total_z != 0):
        #     print(f"Debugging, stop here, noise_syndrome: {noise_total_x}")

        syndrome_dict_x = process_syndrome(noisy_syndrome_x, code_structure, error_channel='x')
        syndrome_dict_z = process_syndrome(noisy_syndrome_z, code_structure, error_channel='z')

    else:
        raise ValueError(f"Unknown error_type: {error_type}")

    
    return actual_logX, actual_logZ, syndrome_dict_x, syndrome_dict_z, noisy_syndrome_x, noisy_syndrome_z

def get_noise_generator(config_type="custom", **kwargs):
    """获取噪声生成器
    
    Args:
        config_type: 噪声类型 ("custom", "stim")
        **kwargs: 传递给具体生成器的参数
    
    Returns:
        生成器函数，每次调用返回 (noise_x, noise_z, syndrome_x, syndrome_z)
    """
    if config_type == "stim":
        return _stim_noise_generator(**kwargs)
    else:  # "custom"
        return _custom_noise_generator(**kwargs)

def _stim_noise_generator(L, before_round_data_error_rate, before_measure_error_rate, num_shots, **kwargs):
    """使用stim生成的噪声"""
    code_structure = kwargs.get('code_structure')
    if code_structure is None:
        raise ValueError("code_structure is None")
    else:
        Hx = code_structure.H_x
        Hz = code_structure.H_z
        logX = code_structure.logicals_x
        logZ = code_structure.logicals_z
        weight_value = np.log((1 - before_round_data_error_rate) / before_round_data_error_rate) if before_round_data_error_rate > 0 else 1
        timelike_weight = np.log((1 - before_measure_error_rate) / before_measure_error_rate) if before_measure_error_rate > 0 else 1
    
        code_type = code_structure.code_type
        num_stabs = code_structure.num_stabs_x
        repetitions = int(code_structure.repetitions)  # 确保repetitions是整数类型
        L = int(code_structure.L)  # 确保L是整数类型
    
    channel = kwargs.get('channel', 'x')

    if code_structure.code_type == 'toric':
        if channel == 'x':
            code_name = "toric_code:unrotated_memory_x"
        elif channel == 'z':
            code_name = "toric_code:unrotated_memory_z"
    elif code_type == 'planar':
        if channel == 'x':
            code_name = "surface_code:unrotated_memory_x"
        elif channel == 'z':
            code_name = "surface_code:unrotated_memory_z"
    elif code_type == 'rotated':
        if channel == 'x':
            code_name = "surface_code:rotated_memory_x"
        elif channel == 'z':
            code_name = "surface_code:rotated_memory_z"
    else:
        raise ValueError(f"未知 code_type: {code_type}")


    if DEBUG_UF_GEOMETRY:
        print(f"code_name: {code_name}")
        print(f"L: {L}")
        print(f"repetitions: {repetitions}")
        print(f"before_round_data_error_rate: {before_round_data_error_rate}")
        print(f"before_measure_error_rate: {before_measure_error_rate}")
        print(f"num_shots: {num_shots}")

    circuit = stimcircuits.generate_circuit(code_name,
                                    distance=L,
                                    rounds=repetitions,
                                    after_clifford_depolarization=before_round_data_error_rate,
                                    before_round_data_depolarization=before_round_data_error_rate,
                                    after_reset_flip_probability=0,
                                    before_measure_flip_probability=before_round_data_error_rate)

    detector_coords = circuit.get_detector_coordinates()
    if DEBUG_UF_GEOMETRY:
        print(f"detector_coords: {detector_coords}")
    dem = circuit.detector_error_model(decompose_errors=True)
    H = detector_error_model_to_check_matrices(dem).check_matrix
    sampler = circuit.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
    

    mwpm_backend = DECODER_CONFIG.get('mwpm_backend', 'dem_unweighted')
    use_unweighted_hx = (mwpm_backend == 'hx_manual_unweighted') or DECODER_CONFIG.get('mwpm_report_dual', True)
    if channel == 'x':
        matching = Matching(
            Hx,
            weights=1.0 if use_unweighted_hx else weight_value,
            repetitions=repetitions+1,
            timelike_weights=1.0 if use_unweighted_hx else timelike_weight,
            faults_matrix=logX
        )
    else:
        matching = Matching(
            Hz,
            weights=1.0 if use_unweighted_hx else weight_value,
            repetitions=repetitions+1,
            timelike_weights=1.0 if use_unweighted_hx else timelike_weight,
            faults_matrix=logZ
        )

    for i in range(num_shots):
        # 从stim结果中提取噪声和syndrome
        # 这里需要根据实际的stim输出格式进行调整
        actual_x = actual_observables[i]  # 需要根据实际格式调整
        actual_z = actual_observables[i]  # 需要根据实际格式调整
        # syndrome_x = syndrome[i]  # 需要根据实际格式调整
        # syndrome_z = syndrome[i]  # 需要根据实际格式调整

        # if code_type == 'rotated_surface_code':
        syndrome_dict_x = defaultdict(int)
        syndrome_dict_z = defaultdict(int)

        if DEBUG_UF_GEOMETRY:
            print(f"NEW Sample: syndrome[i,:]: {syndrome[i,:]}")

        syndrome_reorder_x = np.zeros((Hx.shape[0],repetitions+1))
        syndrome_reorder_z = np.zeros((Hz.shape[0],repetitions+1))
        for j in range(len(detector_coords)):
            if syndrome[i,j] == 1:
                if DEBUG_UF_GEOMETRY:
                    print(f"detector_coords[j]: {detector_coords[j]}")
                if code_type == 'rotated':
                    col = detector_coords[j][0]
                    row = detector_coords[j][1]
                    time = detector_coords[j][2]
                    if (col + row) % 2 == 1:
                        syndrome_dict_x[(row, col - 1, time)] = 1
                    else:##z error
                        syndrome_dict_z[(row - 1, col, time)] = 1                    
                elif code_type == 'toric':
                    col = detector_coords[j][0]//2
                    row = detector_coords[j][1]//2
                    time = detector_coords[j][2]
                    if detector_coords[j][1] % 2 == 0:
                        syndrome_dict_x[(row, col, time)] = 1
                        syndrome_reorder_x[int(col * L + row), int(time)] = 1
                    else:##z error
                        syndrome_dict_z[(row, col, time)] = 1
                        syndrome_reorder_z[int(col * L + row), int(time)] = 1


        yield actual_x, actual_z, syndrome_dict_x, syndrome_dict_z, syndrome_reorder_x, syndrome_reorder_z, matching

def _custom_noise_generator(L, after_clifford_depolarization, before_round_data_error_rate, before_measure_error_rate, num_shots, **kwargs):
    """使用自定义函数生成的噪声"""
    # 直接使用装饰器修饰的函数，它会自动处理batch_size
    channel = kwargs.get('channel', 'x')
    code_structure = kwargs.get('code_structure')
    if code_structure is None:
        raise ValueError("code_structure is None")
    else:
        code_type = code_structure.code_type
        num_stabs = code_structure.num_stabs_x
        repetitions = code_structure.repetitions
        Hx = code_structure.H_x
        Hz = code_structure.H_z
        logX = code_structure.logicals_x
        logZ = code_structure.logicals_z
        weight_value = np.log((1 - before_round_data_error_rate) / (before_round_data_error_rate)) if before_round_data_error_rate > 0 else 1
        timelike_weight = np.log((1 - before_measure_error_rate) / before_measure_error_rate) if before_measure_error_rate > 0 else 1
        
    if channel == 'x':
        matching = Matching(
            Hx,
            weights=weight_value,
            repetitions=repetitions+1,
            timelike_weights=timelike_weight,
            faults_matrix=logX
        )
    else:
        matching = Matching(
            Hz,
            weights=weight_value,
            repetitions=repetitions+1,
            timelike_weights=timelike_weight,
            faults_matrix=logZ
        )
    for result in generate_random_error_and_syndrome(
        L=L, code_structure=code_structure,
        before_round_data_error_rate=before_round_data_error_rate,
        before_measure_error_rate=before_measure_error_rate,
        error_type="phenomenological",#phenomenological
        batch_size=num_shots,
    ):
        yield result[0], result[1], result[2], result[3], result[4], result[5], matching


def simple_batchify(func):
    """简化的批量装饰器，专门用于噪声生成函数
    
    使用方法：
    @simple_batchify
    def generate_noise(...):
        # 返回 (noise_x, noise_z, syndrome_x, syndrome_z)
        pass
    
    # 调用
    result = generate_noise(..., batch_size=10)  # 生成10个样本
    """
    def wrapper(*args, batch_size=1, **kwargs):
        if batch_size == 1:
            return func(*args, **kwargs)
        
        # 保存原始随机状态
        original_state = np.random.get_state()
        
        # 获取单个样本的结果来确定形状
        single_result = func(*args, **kwargs)
        
        if isinstance(single_result, tuple):
            # 为每个返回值创建批量数组
            batch_results = []
            for item in single_result:
                if hasattr(item, 'shape'):
                    batch_shape = (batch_size,) + item.shape
                    batch_array = np.zeros(batch_shape, dtype=item.dtype)
                    batch_results.append(batch_array)
                else:
                    batch_results.append([None] * batch_size)
            
            # 生成批量数据
            for batch_idx in range(batch_size):
                np.random.seed(original_state[1] + batch_idx)
                single_batch_result = func(*args, **kwargs)
                
                for i, item in enumerate(single_batch_result):
                    if hasattr(item, 'shape'):
                        batch_results[i][batch_idx] = item
                    else:
                        batch_results[i][batch_idx] = item
            
            # 恢复原始随机状态
            np.random.set_state(original_state)
            return tuple(batch_results)
        else:
            # 处理单个返回值的情况
            if hasattr(single_result, 'shape'):
                batch_shape = (batch_size,) + single_result.shape
                batch_array = np.zeros(batch_shape, dtype=single_result.dtype)
                
                for batch_idx in range(batch_size):
                    np.random.seed(original_state[1] + batch_idx)
                    batch_array[batch_idx] = func(*args, **kwargs)
                
                np.random.set_state(original_state)
                return batch_array
            else:
                results = []
                for batch_idx in range(batch_size):
                    np.random.seed(original_state[1] + batch_idx)
                    results.append(func(*args, **kwargs))
                
                np.random.set_state(original_state)
                return results
    
    # 保持原函数的文档字符串
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    
    return wrapper

# 使用简化的装饰器
@simple_batchify
def generate_random_error_and_syndrome_simple(L, Hx, Hz, repetitions=1, before_round_data_error_rate=0.0, before_measure_error_rate=0.0, error_type="bit_flip"):
    """生成随机错误和对应的syndrome（简化版本）
    
    Args:
        L: 格子大小
        Hx: X稳定子的校验矩阵
        Hz: Z稳定子的校验矩阵
        repetitions: 测量轮数
        before_round_data_error_rate: 数据错误概率
        before_measure_error_rate: 测量错误概率
        error_type: 错误类型 ("bit_flip", "polarize")
        batch_size: 批量大小（通过装饰器处理）
    
    Returns:
        tuple: (noise_total_x, noise_total_z, noisy_syndrome_x, noisy_syndrome_z)
    """
    # 获取校验矩阵尺寸：
    num_stabs_x, num_qubits = Hx.shape
    num_stabs_z, _ = Hz.shape

    # 生成随机数
    rand_arr = np.random.rand(num_qubits, repetitions)
    noise_type = np.zeros((num_qubits, repetitions), dtype=np.int8)
    pure_syndrome_x = np.zeros((num_stabs_x, repetitions+2), dtype=np.int8)
    pure_syndrome_z = np.zeros((num_stabs_z, repetitions+2), dtype=np.int8)
    
    if error_type == "polarize":
        error_mask = rand_arr < before_round_data_error_rate
        num_errors_occ = np.sum(error_mask)
        if num_errors_occ > 0:
            noise_type[error_mask] = (np.floor(np.random.rand(num_errors_occ) * 3) + 1).astype(np.int8)
        effective_error_x = ((noise_type == 1) | (noise_type == 2)).astype(np.int8)
        effective_error_z = ((noise_type == 2) | (noise_type == 3)).astype(np.int8)
    elif error_type == "bit_flip":
        # 为X和Z错误分别生成随机数
        rand_arr_x = np.random.rand(num_qubits, repetitions+1)
        rand_arr_z = np.random.rand(num_qubits, repetitions+1)
        
        # 分别生成X和Z错误
        error_mask_x = rand_arr_x < before_round_data_error_rate
        error_mask_z = rand_arr_z < before_round_data_error_rate
        error_mask_x[:, 0] = 0
        error_mask_z[:, 0] = 0
        
        # 直接设置X和Z错误，不使用noise_type
        effective_error_x = error_mask_x.astype(np.int8)
        effective_error_z = error_mask_z.astype(np.int8)
    
    # 计算累积错误
    noise_total_x = effective_error_x[:, -1]
    noise_total_z = effective_error_z[:, -1]
    
    # --- Syndrome 计算 ---
    for i in range(repetitions + 1):
        syndrome_x = (Hx @ effective_error_x[:, i]) % 2
        syndrome_z = (Hz @ effective_error_z[:, i]) % 2
        pure_syndrome_x[:, i] = syndrome_x
        pure_syndrome_z[:, i] = syndrome_z
    
    pure_syndrome_x[:, 0] = 0
    pure_syndrome_z[:, 0] = 0
    
    # --- 测量噪声模拟 ---
    # 对于 X 通道：X 错误会翻转 syndrome，所以加入测量噪声
    meas_error_x = (np.random.rand(num_stabs_x, repetitions+1) < before_measure_error_rate).astype(np.int8)
    meas_error_x[:, -1] = 0  # 最后一轮保持完美
    meas_error_x[:, 0] = 0
    noisy_syndrome_x = (pure_syndrome_x + meas_error_x) % 2
    noisy_syndrome_x = np.abs(np.diff(noisy_syndrome_x, axis=1))
    
    # 对于 Z 通道：Z 错误会翻转 syndrome，所以加入测量噪声
    meas_error_z = (np.random.rand(num_stabs_z, repetitions+1) < before_measure_error_rate).astype(np.int8)
    meas_error_z[:, -1] = 0  # 最后一轮保持完美
    meas_error_z[:, 0] = 0
    noisy_syndrome_z = (pure_syndrome_z + meas_error_z) % 2
    noisy_syndrome_z = np.abs(np.diff(noisy_syndrome_z, axis=1))
    
    return noise_total_x, noise_total_z, noisy_syndrome_x, noisy_syndrome_z
