# 数据收集配置
import os as _os
import json as _json
from re import T


DATA_COLLECTION_CONFIG = {
    'enable_data_collection': False,   # 启用数据收集
    'save_dir': 'cluster_data',       # 数据保存目录
    'save_syndrome': True,           # 是否保存syndrome数据
    'save_code_structure': True,     # 是否保存code结构
    'save_cluster_candidates': True, # 是否保存cluster候选解
    'filter_zero_syndrome': True,    # 是否过滤掉syndrome全为0的数据
}

# 解码器配置
DECODER_CONFIG = {
    'use_uf': True,                  # 使用UF解码器
    'use_mwpm': True,                # 使用MWPM解码器
    'use_bposd': False,               # 使用BP-OSD解码器
    # BPOSD backend controls:
    # - dem_graph: use DEM check matrix with detector-shot input (circuit-level reference).
    # - shared_graph: use Hx/Hz-derived shared spacetime graph (fair algorithmic comparison with UF/manual MWPM).
    'bposd_backend': 'shared_graph',
    'bposd_report_dual': False,       # 同时记录 dem_graph 与 shared_graph 两套 BPOSD 结果
    'use_peel_listdecoding': True,              # 使用UF列表解码
    # UF list-decoding randomization policy controls.
    # uf_random_mode:
    # - fixed: same seed for all shots/candidates (deterministic baseline)
    # - per_shot: different seed per shot
    # - per_candidate: different seed per shot and per candidate (max diversity)
    # - correlated: intentionally grouped/correlated streams for stress testing
    'uf_random_mode': 'per_candidate',
    'uf_random_base_seed': 42,
    'uf_random_correlation_group': 8,
    'use_peel_efficient': True,      # 使用connected subgraph进行peeling解码 (disabled for L=15 hw bug)
    'use_ablation_baseline': False,  # 使用ablation baseline进行解码
    'use_ablation_mbuffer_only': False,   # 使用ablation mbuffer进行解码
    'use_ablation_dsuopt_only': False,       # 使用ablation dsu进行解码
    'use_ablation_graphcompression': False,       # Based on mbuffer and dsuopt
    'use_ablation_growskipping': False,       # Based on mbuffer and dsuopt

    
    # 噪声生成配置
    'noise_source': 'stim',        # 噪声来源: 'custom' 或 'stim'
    'use_dem_priors_for_bposd': False,   # BPOSD在stim下是否使用DEM逐变量先验
    'stim_compare_full_observables': False,  # stim下是否按完整observables向量判错
    # Custom phenomenological noise bias controls.
    # When enabled, Pauli errors are sampled with the configured X/Y/Z probabilities.
    'enable_biased_noise': False,
    'biased_noise_pauli_probs': {
        'x': 1/3.0,
        'y': 1/3.0,
        'z': 1/3.0,
    },
    # MWPM fairness controls for stim noise.
    # - dem_unweighted: decode detector shots with DEM-derived graph and uniform edge weights.
    # - dem_weighted: decode detector shots with DEM-derived graph and native DEM weights.
    # - hx_manual_unweighted: decode mapped syndrome_array with Hx-based space-time graph and uniform edge weights.
    'mwpm_backend': 'hx_manual_unweighted',
    'mwpm_report_dual': False,            # 同时记录 DEM 与 Hx-manual 两套 MWPM 结果及分歧率
    'mwpm_report_dem_weighted': False,    # 同时记录 DEM-weighted 结果并输出与其他实现的对照
    'plot_dual_mwpm': False,              # 画图时是否同时展示 DEM/Hx 两条 MWPM 曲线（若结果中存在）
    'plot_mwpm_disagree': False,         # 画图时是否在终端打印 MWPM 两实现分歧率
    'plot_dual_bposd': False,             # 画图时是否同时展示 BPOSD dem_graph/shared_graph 两条曲线
    # Recommended protocol for reproducible MWPM fairness checks.
    'mwpm_fairness_protocol': {
        'Ls': [3, 5],
        'ps': [5e-3, 1e-2],
        'num_shots': 8000,
        'key_points_num_shots': 12000,
    },
    # 'use_stim_noise': False,         # 是否使用stim生成的噪声（备选配置）
    
    # 时间测量配置
    'enable_timing': False,           # 启用时间测量功能
    'timing_verbose': False,           # 是否打印详细的时间统计信息
    
    # 操作统计配置
    'enable_operation_counting': False,      # 是否启用操作统计
    'enable_hardware_cycle_estimation': False,      # 是否启用硬件cycle估计
    'operation_counting_verbose': False,    # 是否详细打印操作统计
    'save_operation_stats': False,          # 是否保存操作统计到文件
    'operation_stats_file': 'operation_stats.json',  # 操作统计保存文件名

    'verbose_top': False,                  # 是否详细打印top level信息
    'batch_progress_every': 1000,           # 控制批次打印频率
    'progress_bar': True,                  # 是否显示进度条
    'progress_step_percent': 4,            # 进度条步长
}

# ABLATION_CONFIG = {
#     'if_graph_compression': True,
#     'if_grow_skipping': True,
#     'if_no_dsu_opt': True,
#     'if_no_mb_bufffer': True,
#     }

# Allow runtime overrides from environment (used by multiprocessing workers).
_env_overrides = _os.environ.get('UF_DECODER_CONFIG_OVERRIDES')
if _env_overrides:
    try:
        _overrides = _json.loads(_env_overrides)
        if isinstance(_overrides, dict):
            DECODER_CONFIG.update(_overrides)
    except (ValueError, TypeError):
        pass

# Add a new debug flag
# Set to True to print detailed debug information for UF geometry, cluster growth, and peeling
DEBUG_UF_GEOMETRY = False
DEBUG_3D_CODE = False
