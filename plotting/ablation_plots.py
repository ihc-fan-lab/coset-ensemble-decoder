import os
import json
import pickle
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def numpy_to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_serializable(v) for v in obj]
    return obj


def legend_if_any() -> None:
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if any(label and not str(label).startswith('_') for label in labels):
        plt.legend()


def plot_fig5_ops_stages(results: Dict[str, Any], L: int, L_index: int, ps: List[float], 
                         output_dir: str, file_ext: str = 'pdf', 
                         variant_key: str = 'raw_latency_all_L_ablation_graphcompression',
                         variant_name: str = 'graphcompression') -> None:
    """绘制图5：ablation变体的 Cluster/Peeling 延迟对比（随 p 的柱状图）
    
    Args:
        results: 实验结果字典
        L: 代码距离
        L_index: L的索引
        ps: 物理错误率列表
        output_dir: 输出目录
        file_ext: 文件扩展名
        variant_key: 要使用的数据源键名，默认为 graphcompression
        variant_name: 变体名称，用于文件名和颜色配置
    """
    # ISCA 常用配色（Tableau 10 风格）
    palette = {
        'baseline': '#1f77b4',          # 蓝
        'graphcompression': '#d62728',  # 红
        'mbuffer': '#ff7f0e',           # 橙（MemoryAccess Opt）
    }
    
    if variant_key not in results or len(results[variant_key]) <= L_index:
        return
    
    baseline_latency_for_L = results[variant_key][L_index]
    cluster_latency_means = []
    peeling_latency_means = []
    
    for p_idx, _ in enumerate(ps):
        shot_dicts = baseline_latency_for_L[p_idx] if p_idx < len(baseline_latency_for_L) else []
        # 读取 cluster 和 peeling 的延迟（cycles）
        cluster_latencies = [sd.get('cluster_operations', 0) for sd in shot_dicts if isinstance(sd, dict)]
        peeling_latencies = [sd.get('peeling_operations', 0) for sd in shot_dicts if isinstance(sd, dict)]
        cluster_latency_means.append(float(np.mean(cluster_latencies)) if cluster_latencies else 0.0)
        peeling_latency_means.append(float(np.mean(peeling_latencies)) if peeling_latencies else 0.0)
    
    x = np.arange(len(ps))
    width = 0.35
    plt.figure(figsize=(10, 6))
    
    bar1 = plt.bar(
        x - width/2, cluster_latency_means, width=width,
        label='Clustering latency', color=palette['baseline'], edgecolor='black', linewidth=1.0
    )
    bar2 = plt.bar(
        x + width/2, peeling_latency_means, width=width,
        label='Spanning Tree&Peeling latency', color=palette.get(variant_name, palette['graphcompression']), 
        edgecolor='black', linewidth=1.0
    )
    
    plt.xticks(x, [f'{p:.5f}' for p in ps], rotation=0)
    plt.ylabel('Latency (cycles)')
    plt.xlabel('Physical Error Rate')
    plt.grid(True, axis='y')
    legend_if_any()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig5_ops_{variant_name}_stages_L{L}.{file_ext}'), dpi=200)
    plt.close()
    
    # 保存绘图数据到 JSON 文件
    plot_data = {
        'L': L,
        'variant_name': variant_name,
        'variant_key': variant_key,
        'ps': numpy_to_serializable(ps),
        'cluster_latency_means': numpy_to_serializable(cluster_latency_means),
        'peeling_latency_means': numpy_to_serializable(peeling_latency_means),
    }
    json_filename = os.path.join(output_dir, f'fig5_ops_{variant_name}_stages_L{L}_data.json')
    with open(json_filename, 'w') as f:
        json.dump(plot_data, f, indent=2)


def _extract_mean_latency_per_p(raw_latency_for_L: List[List[Dict[str, Any]]], ps: List[float], key: str = 'total_cycles') -> List[float]:
    means = []
    for p_idx, _ in enumerate(ps):
        shot_dicts = raw_latency_for_L[p_idx] if p_idx < len(raw_latency_for_L) else []
        if not shot_dicts:
            means.append(0.0)
            continue
        values = [sd.get(key, 0) for sd in shot_dicts if isinstance(sd, dict)]
        means.append(float(np.mean(values)) if values else 0.0)
    return means


_PERCENTILE_LEVELS = [0, 10, 25, 50, 75, 90, 95, 99, 100]


def _extract_percentiles_per_p(
    raw_latency_for_L: List[List[Dict[str, Any]]],
    ps: List[float],
    key: str = 'total_cycles',
) -> Dict[str, List[float]]:
    """Compute compact percentile statistics per p value.

    Returns a dict mapping "pXX" -> [value_for_p0, value_for_p1, ...].
    Uses numpy for efficiency even with 1M+ shots.
    """
    result: Dict[str, List[float]] = {f'p{q}': [] for q in _PERCENTILE_LEVELS}
    result['mean'] = []
    result['std'] = []

    for p_idx, _ in enumerate(ps):
        shot_dicts = raw_latency_for_L[p_idx] if p_idx < len(raw_latency_for_L) else []
        if not shot_dicts:
            for q in _PERCENTILE_LEVELS:
                result[f'p{q}'].append(0.0)
            result['mean'].append(0.0)
            result['std'].append(0.0)
            continue

        values = np.array(
            [sd.get(key, 0) for sd in shot_dicts if isinstance(sd, dict)],
            dtype=np.float64,
        )
        if len(values) == 0:
            for q in _PERCENTILE_LEVELS:
                result[f'p{q}'].append(0.0)
            result['mean'].append(0.0)
            result['std'].append(0.0)
            continue

        pcts = np.percentile(values, _PERCENTILE_LEVELS)
        for q, v in zip(_PERCENTILE_LEVELS, pcts):
            result[f'p{q}'].append(float(v))
        result['mean'].append(float(np.mean(values)))
        result['std'].append(float(np.std(values)))

    return result


def _binomial_sem(vals: List[float], n: Optional[int]) -> List[float]:
    if not n:
        return [0.0 for _ in vals]
    ys = np.array(vals, dtype=float)
    eps = 1e-12
    sem = np.sqrt(np.clip(ys, 0.0, 1.0) * np.clip(1.0 - ys, 0.0, 1.0) / float(max(n, 1)))
    sem = np.minimum(sem, np.maximum(ys - eps, 0.0))
    return sem.tolist()


def get_helios_cycles(L: int, p: float) -> float:
    # 来自 uf_test_utils.UFTester._get_helios_cycles 的查表数据
    helios_data = {
        3: {0.0005: 11.1527, 0.00075: 11.2768, 0.001: 11.3173, 0.00125: 11.4417, 0.0015: 11.4872},
        5: {0.0005: 11.9882, 0.00075: 12.4421, 0.001: 12.9897, 0.00125: 13.5605, 0.0015: 13.9666},
        7: {0.0005: 14.398, 0.00075: 16.2252, 0.001: 17.6502, 0.00125: 19.2076, 0.0015: 20.7541},
        9: {0.0005: 19.4252, 0.00075: 23.245, 0.001: 26.3648, 0.00125: 30.2762, 0.0015: 33.038},
        11: {0.0005: 27.0971, 0.00075: 33.5048, 0.001: 39.7536, 0.00125: 44.6698, 0.0015: 48.9028}
    }
    if L in helios_data and p in helios_data[L]:
        return helios_data[L][p]
    return 20.0


def get_micro_blossom_cycles(L: int, p: float) -> float:
    # 来自 uf_test_utils.UFTester._get_micro_blossom_cycles 的查表数据
    micro_blossom_data = {
        3: {0.0005: 5.6302, 0.00075: 6.3692, 0.001: 7.3894, 0.00125: 8.0564, 0.0015: 9.0906},
        5: {0.0005: 13.1732, 0.00075: 17.6634, 0.001: 23.1238, 0.00125: 27.4426, 0.0015: 33.1738},
        7: {0.0005: 31.4794, 0.00075: 45.1838, 0.001: 59.2154, 0.00125: 73.0846, 0.0015: 86.0702},
        9: {0.0005: 62.9546, 0.00075: 92.9466, 0.001: 121.5564, 0.00125: 151.5874, 0.0015: 181.8154},
        11: {0.0005: 112.723, 0.00075: 166.1346, 0.001: 219.101, 0.00125: 274.604, 0.0015: 331.2032}
    }
    if L in micro_blossom_data and p in micro_blossom_data[L]:
        return micro_blossom_data[L][p]
    return 50.0


def compute_fidelity(ler: float, cycles: float, d: int = 1) -> float:
    if ler <= 0:
        return 1.0
    system_fidelity = (1 - 2 * ler) ** (cycles / d)
    return max(0.0, system_fidelity)


def aggregate_plots_data(results: Dict[str, Any], Ls: List[int], ps: List[float], L_index: int, include_error_bars: bool = False, num_shots: Optional[int] = None) -> Dict[str, Any]:
    L = Ls[L_index]
    out: Dict[str, Any] = {'L': L, 'ps': ps}

    # LER 系列
    def get_series(key: str) -> List[float]:
        if key in results and len(results[key]) > L_index:
            return numpy_to_serializable(results[key][L_index])
        return []

    out['ler'] = {
        'mwpm': get_series('log_errors_all_L_mwpm'),
        'mwpm_dem_unweighted': get_series('log_errors_all_L_mwpm_dem_unweighted'),
        'mwpm_dem_weighted': get_series('log_errors_all_L_mwpm_dem_weighted'),
        'mwpm_hx_manual_unweighted': get_series('log_errors_all_L_mwpm_hx_manual_unweighted'),
        'mwpm_disagree_hx_vs_dem_unweighted': get_series('log_errors_all_L_mwpm_disagree_rate'),
        'mwpm_disagree_hx_vs_dem_weighted': get_series('log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'),
        'mwpm_disagree_dem_unweighted_vs_dem_weighted': get_series('log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'),
        'bposd': get_series('log_errors_all_L_bposd'),
        'bposd_dem_graph': get_series('log_errors_all_L_bposd_dem_graph'),
        'bposd_shared_graph': get_series('log_errors_all_L_bposd_shared_graph'),
        'bposd_disagree': get_series('log_errors_all_L_bposd_disagree_rate'),
        'uf': get_series('log_errors_all_L_uf'),
        'uf_peel_votemax': get_series('log_errors_all_L_uf_peel_votemax'),
        'uf_efficient_votemax': get_series('log_errors_all_L_uf_peel_efficient_votemax'),
        'ablation_baseline': get_series('log_errors_all_L_uf_ablation_baseline_votemax'),
        'ablation_mbuffer_only': get_series('log_errors_all_L_uf_ablation_mbuffer_only_votemax'),
        'ablation_dsuopt_only': get_series('log_errors_all_L_uf_ablation_dsuopt_only_votemax'),
        'ablation_graphcompression': get_series('log_errors_all_L_uf_ablation_graphcompression_votemax'),
        'ablation_growskipping': get_series('log_errors_all_L_uf_ablation_growskipping_votemax'),
    }
    if include_error_bars and num_shots:
        ler_sem: Dict[str, List[float]] = {}
        for k, v in out['ler'].items():
            if isinstance(v, list) and len(v) == len(ps):
                ler_sem[k] = _binomial_sem(v, num_shots)
            else:
                ler_sem[k] = []
        out['ler_sem'] = ler_sem

    # 各变体的平均总延迟（用于图4）
    variants_latency_keys = [
        ('raw_latency_all_L_peel_efficient', 'efficient_total'),
        ('raw_latency_all_L_ablation_baseline', 'baseline_total'),
        ('raw_latency_all_L_ablation_mbuffer_only', 'mbuffer_only_total'),
        ('raw_latency_all_L_ablation_dsuopt_only', 'dsuopt_only_total'),
        ('raw_latency_all_L_ablation_graphcompression', 'graphcompression_total'),
        ('raw_latency_all_L_ablation_growskipping', 'growskipping_total'),
    ]
    latency_means: Dict[str, List[float]] = {}
    for key, label in variants_latency_keys:
        if key in results and len(results[key]) > L_index:
            latency_means[label] = _extract_mean_latency_per_p(results[key][L_index], ps, key='total_cycles')
        else:
            latency_means[label] = []
    out['latency_means'] = latency_means

    latency_percentiles: Dict[str, Dict[str, List[float]]] = {}
    for key, label in variants_latency_keys:
        if key in results and len(results[key]) > L_index:
            latency_percentiles[label] = _extract_percentiles_per_p(
                results[key][L_index], ps, key='total_cycles',
            )
        else:
            latency_percentiles[label] = {}
    out['latency_percentiles'] = latency_percentiles

    # baseline 的 Ops 平均数（用于图5）
    baseline_key = 'raw_latency_all_L_ablation_baseline'
    peeling_ops_means = []
    baseline_ops_means = []
    if baseline_key in results and len(results[baseline_key]) > L_index:
        baseline_latency_for_L = results[baseline_key][L_index]
        for p_idx, _ in enumerate(ps):
            shot_dicts = baseline_latency_for_L[p_idx] if p_idx < len(baseline_latency_for_L) else []
            peeling_ops = [sd.get('Peeling_OPs', 0) for sd in shot_dicts if isinstance(sd, dict)]
            baseline_ops = [sd.get('Baseline_OPs', 0) for sd in shot_dicts if isinstance(sd, dict)]
            peeling_ops_means.append(float(np.mean(peeling_ops)) if peeling_ops else 0.0)
            baseline_ops_means.append(float(np.mean(baseline_ops)) if baseline_ops else 0.0)
    out['ops_means_baseline'] = {
        'Peeling_OPs': peeling_ops_means,
        'Baseline_OPs': baseline_ops_means,
    }

    # Fidelity 曲线（用于图6）
    fidelity = {}
    # MWPM
    if 'log_errors_all_L_mwpm' in results and len(results['log_errors_all_L_mwpm']) > L_index:
        ler = results['log_errors_all_L_mwpm'][L_index]
        infid = []
        for p, v in zip(ps, ler):
            cycles = get_micro_blossom_cycles(L, p) * 0.0232
            f = compute_fidelity(float(v), float(cycles), d=1)
            infid.append(1 - f)
        fidelity['mwpm'] = infid
    # UF
    if 'log_errors_all_L_uf' in results and len(results['log_errors_all_L_uf']) > L_index:
        ler = results['log_errors_all_L_uf'][L_index]
        infid = []
        for p, v in zip(ps, ler):
            cycles = get_helios_cycles(L, p) * 0.0133
            f = compute_fidelity(float(v), float(cycles), d=1)
            infid.append(1 - f)
        fidelity['uf'] = infid
    # UF efficient (用追踪到的 total_cycles 平均×比例)
    key_eff = 'raw_latency_all_L_peel_efficient'
    if 'log_errors_all_L_uf_peel_efficient_votemax' in results and key_eff in results and len(results['log_errors_all_L_uf_peel_efficient_votemax']) > L_index and len(results[key_eff]) > L_index:
        ler = results['log_errors_all_L_uf_peel_efficient_votemax'][L_index]
        latency_L = results[key_eff][L_index]
        infid = []
        n = min(len(ps), len(ler), len(latency_L))
        for p_idx in range(n):
            shots = latency_L[p_idx] if p_idx < len(latency_L) else []
            total_means = np.mean([sd.get('total_cycles', 0) for sd in shots]) if shots else 0.0
            cycles = float(total_means) * 0.005
            f = compute_fidelity(float(ler[p_idx]), float(cycles), d=1)
            infid.append(1 - f)
        if infid:
            fidelity['uf_efficient'] = infid
    out['fidelity'] = fidelity

    if include_error_bars and num_shots:
        out['meta'] = {'num_shots': int(num_shots)}
    return out


def save_plots_data(results: Dict[str, Any], Ls: List[int], ps: List[float], L_index: int, output_dir: str, include_error_bars: bool = False, num_shots: Optional[int] = None) -> None:
    ensure_dir(output_dir)
    agg = aggregate_plots_data(results, Ls, ps, L_index, include_error_bars=include_error_bars, num_shots=num_shots)
    with open(os.path.join(output_dir, 'plots_data.json'), 'w') as f:
        json.dump(numpy_to_serializable(agg), f, indent=2)


def plot_ablation_figures(results: Dict[str, Any], Ls: List[int], ps: List[float], L_index: int = 0, output_dir: str = 'plots', file_ext: str = 'png', include_error_bars: bool = False, num_shots: Optional[int] = None) -> None:
    ensure_dir(output_dir)
    if not Ls:
        raise ValueError('Ls 为空')
    if L_index < 0 or L_index >= len(Ls):
        raise ValueError('L_index 越界')
    L = Ls[L_index]

    # ISCA 风格全局设定：较粗线条、大字号、去掉上右边框，适度网格
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'font.size': 16,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
    })

    # ISCA 常用配色（Tableau 10 风格）
    palette = {
        'mwpm': '#1f77b4',              # 蓝
        'uf': '#ff7f0e',                # 橙
        'hardware': '#2ca02c',          # 绿（UF efficient）
        'peel': '#d62728',              # 红（peel list）
        # ablations（与图4顺序一致）
        'baseline': '#1f77b4',          # 蓝
        'dsuopt': '#9467bd',            # 紫
        'mbuffer': '#ff7f0e',           # 橙（MemoryAccess Opt）
        'growskipping': '#8c564b',      # 棕
        'graphcompression': '#d62728',  # 红
    }

    def _binomial_sem(vals: List[float], n: Optional[int]) -> List[float]:
        if not include_error_bars or not n:
            return [0.0 for _ in vals]
        ys = np.array(vals, dtype=float)
        eps = 1e-12
        sem = np.sqrt(np.clip(ys, 0.0, 1.0) * np.clip(1.0 - ys, 0.0, 1.0) / float(max(n, 1)))
        sem = np.minimum(sem, np.maximum(ys - eps, 0.0))
        return sem.tolist()

    # 1) MWPM, UF, UF_peel_efficient LER 对比
    plt.figure(figsize=(8, 6))
    if 'log_errors_all_L_mwpm' in results and len(results['log_errors_all_L_mwpm']) > L_index:
        y = results['log_errors_all_L_mwpm'][L_index]
        n = min(len(ps), len(y))
        if n > 0:
            xp, yp = ps[:n], y[:n]
            if include_error_bars and num_shots:
                ye = _binomial_sem(y, num_shots)[:n]
                plt.errorbar(xp, yp, yerr=ye, marker='o', linestyle='-', color=palette['mwpm'], label='Micro-blossom (MWPM)', capsize=3)
            else:
                plt.plot(xp, yp, marker='o', linestyle='-', color=palette['mwpm'], label='Micro-blossom (MWPM)')
    if 'log_errors_all_L_uf' in results and len(results['log_errors_all_L_uf']) > L_index:
        y = results['log_errors_all_L_uf'][L_index]
        n = min(len(ps), len(y))
        if n > 0:
            xp, yp = ps[:n], y[:n]
            if include_error_bars and num_shots:
                ye = _binomial_sem(y, num_shots)[:n]
                plt.errorbar(xp, yp, yerr=ye, marker='s', linestyle='-', color=palette['uf'], label='Helios (UF)', capsize=3)
            else:
                plt.plot(xp, yp, marker='s', linestyle='-', color=palette['uf'], label='Helios (UF)')
    if 'log_errors_all_L_uf_peel_efficient_votemax' in results and len(results['log_errors_all_L_uf_peel_efficient_votemax']) > L_index:
        y = results['log_errors_all_L_uf_peel_efficient_votemax'][L_index]
        n = min(len(ps), len(y))
        if n > 0:
            xp, yp = ps[:n], y[:n]
            if include_error_bars and num_shots:
                ye = _binomial_sem(y, num_shots)[:n]
                plt.errorbar(xp, yp, yerr=ye, marker='^', linestyle='-', color=palette['hardware'], label='Ours', capsize=3)
            else:
                plt.plot(xp, yp, marker='^', linestyle='-', color=palette['hardware'], label='Ours')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Logical Error Rate')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    legend_if_any()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig1_ler_mwpm_uf_ufe_L{L}.{file_ext}'), dpi=200)
    plt.close()

    # 2) MWPM, UF, UF_peel（listdecoding votemax）LER 对比
    plt.figure(figsize=(8, 6))
    if 'log_errors_all_L_mwpm' in results and len(results['log_errors_all_L_mwpm']) > L_index:
        y = results['log_errors_all_L_mwpm'][L_index]
        n = min(len(ps), len(y))
        if n > 0:
            xp, yp = ps[:n], y[:n]
            if include_error_bars and num_shots:
                ye = _binomial_sem(y, num_shots)[:n]
                plt.errorbar(xp, yp, yerr=ye, marker='o', linestyle='-', color=palette['mwpm'], label='Micro-blossom (MWPM)', capsize=3)
            else:
                plt.plot(xp, yp, marker='o', linestyle='-', color=palette['mwpm'], label='Micro-blossom (MWPM)')
    if 'log_errors_all_L_uf' in results and len(results['log_errors_all_L_uf']) > L_index:
        y = results['log_errors_all_L_uf'][L_index]
        n = min(len(ps), len(y))
        if n > 0:
            xp, yp = ps[:n], y[:n]
            if include_error_bars and num_shots:
                ye = _binomial_sem(y, num_shots)[:n]
                plt.errorbar(xp, yp, yerr=ye, marker='s', linestyle='-', color=palette['uf'], label='Helios (UF)', capsize=3)
            else:
                plt.plot(xp, yp, marker='s', linestyle='-', color=palette['uf'], label='Helios (UF)')
    if 'log_errors_all_L_uf_peel_votemax' in results and len(results['log_errors_all_L_uf_peel_votemax']) > L_index:
        y = results['log_errors_all_L_uf_peel_votemax'][L_index]
        n = min(len(ps), len(y))
        if n > 0:
            xp, yp = ps[:n], y[:n]
            if include_error_bars and num_shots:
                ye = _binomial_sem(y, num_shots)[:n]
                plt.errorbar(xp, yp, yerr=ye, marker='D', linestyle='-', color=palette['peel'], label='Ours (software)', capsize=3)
            else:
                plt.plot(xp, yp, marker='D', linestyle='-', color=palette['peel'], label='Ours (software)')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Logical Error Rate')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    legend_if_any()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig2_ler_mwpm_uf_peel_L{L}.{file_ext}'), dpi=200)
    plt.close()

    # 3) UF_efficient 与五个 ablation LER 对比
    plt.figure(figsize=(9, 6))
    if 'log_errors_all_L_uf_peel_efficient_votemax' in results and len(results['log_errors_all_L_uf_peel_efficient_votemax']) > L_index:
        y = results['log_errors_all_L_uf_peel_efficient_votemax'][L_index]
        n = min(len(ps), len(y))
        if n > 0:
            xp, yp = ps[:n], y[:n]
            if include_error_bars and num_shots:
                ye = _binomial_sem(y, num_shots)[:n]
                plt.errorbar(xp, yp, yerr=ye, marker='^', linestyle='-', color=palette['hardware'], label='Fully Optimized', capsize=3)
            else:
                plt.plot(xp, yp, marker='^', linestyle='-', color=palette['hardware'], label='Fully Optimized')
    ablation_series = [
        ('log_errors_all_L_uf_ablation_baseline_votemax', 'Baseline', palette['baseline']),
        ('log_errors_all_L_uf_ablation_dsuopt_only_votemax', 'DSU Opt Only', palette['dsuopt']),
        ('log_errors_all_L_uf_ablation_mbuffer_only_votemax', 'MemoryAccess Opt Only', palette['mbuffer']),
        ('log_errors_all_L_uf_ablation_growskipping_votemax', 'Cluster Skipping Only', palette['growskipping']),
        ('log_errors_all_L_uf_ablation_graphcompression_votemax', 'Graph Compression Only', palette['graphcompression']),
    ]
    for key, label, color in ablation_series:
        if key in results and len(results[key]) > L_index:
            y = results[key][L_index]
            n = min(len(ps), len(y))
            if n > 0:
                xp, yp = ps[:n], y[:n]
                ls = '-' if label == 'Baseline' else '--'
                if include_error_bars and num_shots:
                    ye = _binomial_sem(y, num_shots)[:n]
                    plt.errorbar(xp, yp, yerr=ye, marker='o', linestyle=ls, label=label, color=color, capsize=3)
                else:
                    plt.plot(xp, yp, marker='o', linestyle=ls, label=label, color=color)
    plt.yscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Logical Error Rate')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    legend_if_any()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig3_ler_efficient_ablations_L{L}.{file_ext}'), dpi=200)
    plt.close()

    # 4) UF_efficient 与五个 ablation 的总 latency 对比柱状图（按 p 分组）
    # 顺序要求：baseline, dsuopt_only, mbuffer_only, growskipping, graphcompression, efficient
    variants_latency_keys = [
        ('raw_latency_all_L_ablation_baseline', 'Baseline', palette['baseline']),
        ('raw_latency_all_L_ablation_dsuopt_only', 'DSU Opt Only', palette['dsuopt']),
        ('raw_latency_all_L_ablation_mbuffer_only', 'MemoryAccess Opt Only', palette['mbuffer']),
        ('raw_latency_all_L_ablation_growskipping', 'Cluster Skipping Only', palette['growskipping']),
        ('raw_latency_all_L_ablation_graphcompression', 'Graph Compression Only', palette['graphcompression']),
        ('raw_latency_all_L_peel_efficient', 'Fully Optimized', palette['hardware']),
    ]
    means_per_variant = {}
    for key, label, _color in variants_latency_keys:
        if key in results and len(results[key]) > L_index:
            means = _extract_mean_latency_per_p(results[key][L_index], ps, key='total_cycles')
            # 仅当该变体在当前 L 上存在非零均值时纳入绘图，避免空柱
            if any(v > 0 for v in means):
                means_per_variant[label] = means
    # 绘制分组柱状图
    if means_per_variant:
        x = np.arange(len(ps))
        width = 0.12
        plt.figure(figsize=(12, 6))
        # 纯色填充 + 黑边框（无 hatch）
        items = [(label, color, means_per_variant[label]) for _, label, color in variants_latency_keys if label in means_per_variant]
        for i, (label, color, means) in enumerate(items):
            offsets = (i - (len(items)-1)/2) * width
            plt.bar(
                x + offsets, means, width=width, label=label,
                color=color, edgecolor='black', linewidth=1.0
            )
        plt.xticks(x, [f'{p:.5f}' for p in ps], rotation=0)
        plt.ylabel('Total Cycles')
        plt.xlabel('Physical Error Rate')
        plt.grid(True, axis='y')
        legend_if_any()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'fig4_latency_efficient_ablations_L{L}.{file_ext}'), dpi=200)
        plt.close()

    # 5) ablation graphcompression 的 Cluster/Peeling 操作数对比（随 p 的柱状图）
    plot_fig5_ops_stages(
        results=results,
        L=L,
        L_index=L_index,
        ps=ps,
        output_dir=output_dir,
        file_ext=file_ext,
        variant_key='raw_latency_all_L_ablation_graphcompression',
        variant_name='graphcompression'
    )

    # 6) System Fidelity 对比：MWPM、UF、UF_efficient
    plt.figure(figsize=(9, 6))
    # 准备三条曲线
    curves = []
    if 'log_errors_all_L_mwpm' in results and len(results['log_errors_all_L_mwpm']) > L_index:
        ler = results['log_errors_all_L_mwpm'][L_index]
        n = min(len(ps), len(ler))
        infid = []
        for p, v in zip(ps[:n], ler[:n]):
            cycles = get_micro_blossom_cycles(L, p) * 0.0232
            f = compute_fidelity(v, cycles, d=1)
            infid.append(1 - f)
        if infid:
            curves.append(('Micro-blossom (MWPM)', ps[:n], infid, {'marker': 'o', 'linestyle': '-', 'color': palette['mwpm']}))
    if 'log_errors_all_L_uf' in results and len(results['log_errors_all_L_uf']) > L_index:
        ler = results['log_errors_all_L_uf'][L_index]
        n = min(len(ps), len(ler))
        infid = []
        for p, v in zip(ps[:n], ler[:n]):
            cycles = get_helios_cycles(L, p) * 0.0133
            f = compute_fidelity(v, cycles, d=1)
            infid.append(1 - f)
        if infid:
            curves.append(('Helios (UF)', ps[:n], infid, {'marker': 's', 'linestyle': '-', 'color': palette['uf']}))
    # UF efficient：用跟踪到的 total_cycles 平均，并按 0.005 缩放
    key_eff = 'raw_latency_all_L_peel_efficient'
    if 'log_errors_all_L_uf_peel_efficient_votemax' in results and key_eff in results and len(results['log_errors_all_L_uf_peel_efficient_votemax']) > L_index and len(results[key_eff]) > L_index:
        ler = results['log_errors_all_L_uf_peel_efficient_votemax'][L_index]
        latency_L = results[key_eff][L_index]
        n = min(len(ps), len(ler), len(latency_L))
        infid = []
        for p_idx in range(n):
            shots = latency_L[p_idx] if p_idx < len(latency_L) else []
            total_means = np.mean([sd.get('total_cycles', 0) for sd in shots]) if shots else 0.0
            cycles = total_means * 0.005
            f = compute_fidelity(ler[p_idx], cycles, d=1)
            infid.append(1 - f)
        if infid:
            curves.append(('Ours', ps[:n], infid, {'marker': '^', 'linestyle': '-', 'color': palette['hardware']}))
    # 画图
    for label, xs, ys, style in curves:
        plt.semilogy(xs, ys, label=label, **style)
    plt.xlabel('Physical Error Rate')
    plt.ylabel('1 - System Fidelity')
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    legend_if_any()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig6_fidelity_mwpm_uf_efficient_L{L}.{file_ext}'), dpi=200)
    plt.close()


def plot_and_save_all(results: Dict[str, Any], Ls: List[int], ps: List[float], L_index: int = 0, output_root: str = 'analysis_outputs', file_ext: str = 'pdf', include_error_bars: bool = False, num_shots: Optional[int] = None) -> None:
    ensure_dir(output_root)
    # 仅保存用于作图的聚合数据（避免保存每个shot的明细）
    save_plots_data(results, Ls, ps, L_index, output_root, include_error_bars=include_error_bars, num_shots=num_shots)
    # 再绘图
    plot_ablation_figures(results, Ls, ps, L_index=L_index, output_dir=output_root, file_ext=file_ext, include_error_bars=include_error_bars, num_shots=num_shots)


