import os
import numpy as np
import matplotlib.pyplot as plt

from config import DECODER_CONFIG

def _has_dual_mwpm_data(results):
    return (
        DECODER_CONFIG.get('plot_dual_mwpm', True)
        and 'log_errors_all_L_mwpm_dem_unweighted' in results
        and 'log_errors_all_L_mwpm_hx_manual_unweighted' in results
    )

def _has_dem_weighted_data(results):
    return (
        DECODER_CONFIG.get('plot_dual_mwpm', True)
        and 'log_errors_all_L_mwpm_dem_weighted' in results
    )

def _has_dual_bposd_data(results):
    return (
        DECODER_CONFIG.get('plot_dual_bposd', True)
        and 'log_errors_all_L_bposd_dem_graph' in results
        and 'log_errors_all_L_bposd_shared_graph' in results
    )

def plot_results(self, Ls, ps, results, num_candidates):
    """绘制实验结果
    Args:
        Ls: 格子大小列表
        ps: 错误概率列表
        results: 实验结果字典
        num_candidates: 候选解数量
    """
    plt.figure(figsize=(12,9))
    
    # 打印结果数据
    print("\n" + "="*80)
    print("Experimental Results Summary")
    print("="*80)
    print(f"Physical Error Rate: {ps}")
    print(f"Number of Candidates: {num_candidates}")
    print(f"Number of Experiments: {results['num_shots']}")
    print("-"*80)
    
    # 遍历不同的 L 值，对每个 L 分别绘制不同解码器的逻辑错误率曲线
    for L in Ls:
        print(f"Plotting for L={L}...")
        
        # 打印当前L的数据
        print(f"\nL={L} Results Data:")
        print("-"*60)
        
        # 获取当前L对应的所有错误率数据
        current_errors = {}
        current_errors['mwpm'] = None
        current_errors['mwpm_dem'] = None
        current_errors['mwpm_dem_weighted'] = None
        current_errors['mwpm_hx'] = None
        current_errors['bposd'] = None
        current_errors['bposd_dem_graph'] = None
        current_errors['bposd_shared_graph'] = None
        current_errors['mwpm_disagree'] = None
        current_errors['bposd_dem_graph'] = None
        current_errors['bposd_shared_graph'] = None
        
        # 根据配置添加相应的错误率数据
        if DECODER_CONFIG['use_mwpm'] and L in Ls:
            idx = Ls.index(L)
            current_errors['mwpm'] = results['log_errors_all_L_mwpm'][idx]
            if _has_dual_mwpm_data(results):
                current_errors['mwpm_dem'] = results['log_errors_all_L_mwpm_dem_unweighted'][idx]
                current_errors['mwpm_hx'] = results['log_errors_all_L_mwpm_hx_manual_unweighted'][idx]
                if 'log_errors_all_L_mwpm_disagree_rate' in results:
                    current_errors['mwpm_disagree'] = results['log_errors_all_L_mwpm_disagree_rate'][idx]
            if _has_dem_weighted_data(results):
                current_errors['mwpm_dem_weighted'] = results['log_errors_all_L_mwpm_dem_weighted'][idx]
        
        if DECODER_CONFIG['use_uf'] and L in Ls:
            current_errors['uf'] = results['log_errors_all_L_uf'][Ls.index(L)]
        
        if DECODER_CONFIG['use_bposd'] and L in Ls:
            idx = Ls.index(L)
            current_errors['bposd'] = results['log_errors_all_L_bposd'][idx]
            if _has_dual_bposd_data(results):
                current_errors['bposd_dem_graph'] = results['log_errors_all_L_bposd_dem_graph'][idx]
                current_errors['bposd_shared_graph'] = results['log_errors_all_L_bposd_shared_graph'][idx]

        if DECODER_CONFIG['use_peel_listdecoding'] and L in Ls:
            current_errors['uf_peel_list'] = results['log_errors_all_L_uf_peel_list'][Ls.index(L)]
            current_errors['uf_peel_minweight'] = results['log_errors_all_L_uf_peel_minweight'][Ls.index(L)]
            current_errors['uf_peel_votemax'] = results['log_errors_all_L_uf_peel_votemax'][Ls.index(L)]
            current_errors['uf_peel_syndrome'] = results['log_errors_all_L_uf_peel_syndrome'][Ls.index(L)]

        if DECODER_CONFIG['use_peel_efficient'] and L in Ls:
            current_errors['uf_peel_efficient_list'] = results['log_errors_all_L_uf_peel_efficient_list'][Ls.index(L)]
            current_errors['uf_peel_efficient_minweight'] = results['log_errors_all_L_uf_peel_efficient_minweight'][Ls.index(L)]
            current_errors['uf_peel_efficient_votemax'] = results['log_errors_all_L_uf_peel_efficient_votemax'][Ls.index(L)]
            current_errors['uf_peel_efficient_syndrome'] = results['log_errors_all_L_uf_peel_efficient_syndrome'][Ls.index(L)]
        
        
        # 分别绘制每种解码器的结果
        if DECODER_CONFIG['use_mwpm'] and current_errors['mwpm'] is not None:
            if current_errors['mwpm_dem'] is not None and current_errors['mwpm_hx'] is not None:
                plt.errorbar(
                    ps, current_errors['mwpm_dem'],
                    yerr=(current_errors['mwpm_dem']*(1-current_errors['mwpm_dem'])/results['num_shots'])**0.5,
                    fmt='o-', color='blue', capsize=4, markersize=11, linewidth=2.5, label=f"MWPM-DEM(unweighted), L={L}"
                )
                plt.errorbar(
                    ps, current_errors['mwpm_hx'],
                    yerr=(current_errors['mwpm_hx']*(1-current_errors['mwpm_hx'])/results['num_shots'])**0.5,
                    fmt='s--', color='#1f77b4', capsize=4, markersize=10, linewidth=2.0, label=f"MWPM-Hx-manual(unweighted), L={L}"
                )
                if current_errors['mwpm_dem_weighted'] is not None:
                    plt.errorbar(
                        ps, current_errors['mwpm_dem_weighted'],
                        yerr=(current_errors['mwpm_dem_weighted']*(1-current_errors['mwpm_dem_weighted'])/results['num_shots'])**0.5,
                        fmt='^-', color='navy', capsize=4, markersize=9, linewidth=2.0, label=f"MWPM-DEM(weighted), L={L}"
                    )
                print(f"MWPM Decoder (Dual, L={L}):")
                for idx_p, (p, dem_v, hx_v) in enumerate(zip(ps, current_errors['mwpm_dem'], current_errors['mwpm_hx'])):
                    dem_eb = (dem_v * (1 - dem_v) / results['num_shots']) ** 0.5
                    hx_eb = (hx_v * (1 - hx_v) / results['num_shots']) ** 0.5
                    msg = f"  p={p:.4f}: DEM-unw={self._format_error_label(dem_v)} ± {self._format_error_label(dem_eb)}, Hx={self._format_error_label(hx_v)} ± {self._format_error_label(hx_eb)}"
                    if current_errors['mwpm_dem_weighted'] is not None and idx_p < len(current_errors['mwpm_dem_weighted']):
                        demw_v = current_errors['mwpm_dem_weighted'][idx_p]
                        demw_eb = (demw_v * (1 - demw_v) / results['num_shots']) ** 0.5
                        msg += f", DEM-w={self._format_error_label(demw_v)} ± {self._format_error_label(demw_eb)}"
                    print(msg)
                if DECODER_CONFIG.get('plot_mwpm_disagree', False) and current_errors['mwpm_disagree'] is not None:
                    print(f"  disagree_rate: {[float(v) for v in current_errors['mwpm_disagree']]}")
                print()
            else:
                plt.errorbar(ps, current_errors['mwpm'], 
                            yerr=(current_errors['mwpm']*(1-current_errors['mwpm'])/results['num_shots'])**0.5,
                            fmt='o-', color='blue', capsize=4, markersize=14, linewidth=3, label=f"MWPM, L={L}")
                # 添加数值标签
                for x, y in zip(ps, current_errors['mwpm']):
                    plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                
                # 打印MWPM解码器数据
                print(f"MWPM Decoder (L={L}):")
                for p, error_rate in zip(ps, current_errors['mwpm']):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"  p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                print()
        
        if DECODER_CONFIG['use_uf'] and current_errors['uf'] is not None:
            line = plt.errorbar(ps, current_errors['uf'], 
                        yerr=(current_errors['uf']*(1-current_errors['uf'])/results['num_shots'])**0.5,
                        fmt='*-', color='orange', capsize=4, markersize=14, linewidth=3, label=f"UF, L={L}")
            # 添加数值标签
            for x, y in zip(ps, current_errors['uf']):
                plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            # 打印UF解码器数据
            print(f"UF Decoder (L={L}):")
            for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf'])):
                # 计算误差棒数值
                error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                print(f"  p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            print()
        
        if DECODER_CONFIG['use_bposd'] and current_errors['bposd'] is not None:
            if current_errors['bposd_dem_graph'] is not None and current_errors['bposd_shared_graph'] is not None:
                plt.errorbar(ps, current_errors['bposd_dem_graph'], 
                            yerr=(current_errors['bposd_dem_graph']*(1-current_errors['bposd_dem_graph'])/results['num_shots'])**0.5,
                            fmt='o-', color='black', capsize=4, markersize=10, linewidth=2.5, label=f"BP-OSD(DEM-graph), L={L}")
                plt.errorbar(ps, current_errors['bposd_shared_graph'], 
                            yerr=(current_errors['bposd_shared_graph']*(1-current_errors['bposd_shared_graph'])/results['num_shots'])**0.5,
                            fmt='d--', color='gray', capsize=4, markersize=9, linewidth=2.0, label=f"BP-OSD(shared-graph), L={L}")
                print(f"BP-OSD Decoder (Dual, L={L}):")
                for p, dg, sg in zip(ps, current_errors['bposd_dem_graph'], current_errors['bposd_shared_graph']):
                    dg_e = (dg * (1 - dg) / results['num_shots']) ** 0.5
                    sg_e = (sg * (1 - sg) / results['num_shots']) ** 0.5
                    print(f"  p={p:.4f}: DEM-graph={self._format_error_label(dg)} ± {self._format_error_label(dg_e)}, shared-graph={self._format_error_label(sg)} ± {self._format_error_label(sg_e)}")
                print()
            else:
                line = plt.errorbar(ps, current_errors['bposd'], 
                            yerr=(current_errors['bposd']*(1-current_errors['bposd'])/results['num_shots'])**0.5,
                            fmt='o-', color='black', capsize=4, markersize=14, linewidth=3, label=f"BP-OSD, L={L}")
                # 添加数值标签
                for x, y in zip(ps, current_errors['bposd']):
                    plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                
                # 打印BP-OSD解码器数据
                print(f"BP-OSD Decoder (L={L}):")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['bposd'])):
                    # 计算误差棒数值
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"  p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                print()


        if DECODER_CONFIG['use_peel_listdecoding'] and current_errors['uf_peel_list'] is not None:

            line = plt.errorbar(ps, current_errors['uf_peel_votemax'], 
                        yerr=(current_errors['uf_peel_votemax']*(1-current_errors['uf_peel_votemax'])/results['num_shots'])**0.5,
                        fmt='s--', color='purple', linewidth=2, markersize=8, capsize=6,
                        label=f"Algorithm Model-Majority Voting, L={L}")
            # 添加数值标签
            for x, y in zip(ps, current_errors['uf_peel_votemax']):
                plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            # 打印Peeling List Decoding数据
            print(f"Peeling List Decoding (L={L}):")
            print(f"  Majority Voting:")
            for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_votemax'])):
                error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            
            # 如果有其他peeling方法的数据，也打印出来
            if current_errors.get('uf_peel_list') is not None:
                print(f"  Theoretical Optimal:")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_list'])):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            
            if current_errors.get('uf_peel_minweight') is not None:
                print(f"  MaxLikelihood Selection:")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_minweight'])):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            
            if current_errors.get('uf_peel_syndrome') is not None:
                print(f"  Syndrome Selection:")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_syndrome'])):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            print()



        if DECODER_CONFIG['use_peel_efficient'] and current_errors['uf_peel_efficient_list'] is not None:
            
            line = plt.errorbar(ps, current_errors['uf_peel_efficient_votemax'], 
                        yerr=(current_errors['uf_peel_efficient_votemax']*(1-current_errors['uf_peel_efficient_votemax'])/results['num_shots'])**0.5,
                        fmt='*-', color='green', linewidth=3, markersize=16, capsize=6,
                        label=f"Hardware Golden Model-Majority Voting, L={L}")
            # 添加数值标签
            for x, y in zip(ps, current_errors['uf_peel_efficient_votemax']):
                plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            # 打印Efficient Decoding数据
            print(f"Efficient Decoding (L={L}):")
            print(f"  Majority Voting:")
            for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_efficient_votemax'])):
                error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            

            # 如果有其他efficient方法的数据，也打印出来
            if current_errors.get('uf_peel_efficient_list') is not None:
                print(f"  Theoretical Optimal:")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_efficient_list'])):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            
            if current_errors.get('uf_peel_efficient_minweight') is not None:
                print(f"  Minimal Weight Selection:")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_efficient_minweight'])):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            
            if current_errors.get('uf_peel_efficient_syndrome') is not None:
                print(f"  Syndrome Selection:")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_efficient_syndrome'])):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
            print()

        

    # 设置图表属性
    plt.xlabel("Physical error rate", fontsize=20)
    plt.ylabel("Logical error rate", fontsize=20)
    plt.legend(loc="best", fontsize=12)
    plt.title("Logical Error Rate Comparison", fontsize=24)
    plt.grid(True)
    plt.tight_layout()
    # plt.loglog()  # 设置x轴和y轴都为对数刻度
    # plt.xscale('log')  # 只设置x轴为对数刻度
    plt.yscale('log')  # 只设置y轴为对数刻度

    # 设置刻度标签大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # 修改所有数值标签的大小
    for text in plt.gca().texts:
        text.set_fontsize(10)
    
    # 保存图像
    # 生成包含L、list_size、ps范围、shot数目的文件名
    L_str = '-'.join(map(str, Ls))
    ps_range = f"p{ps[0]:.3f}-{ps[-1]:.3f}"
    save_filename = f"plot_L{L_str}_list{num_candidates}_ps{ps_range}_shots{results['num_shots']}.png"
    save_path = os.path.join(self.save_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {save_path}")
        
    plt.show()

def plot_latency(self, Ls, ps, results, code_type = 'toric_code'):
    """绘制平均解码延迟与物理错误率的关系图，并使用小提琴图显示数据分布
    Args:
        Ls: 格子大小列表
        ps: 错误概率列表
        results: 实验结果字典，包含 'raw_latency_all_L'
        code_type: 代码类型
    """
    
    # 新增：绘制性能追踪器数据
    self._plot_performance_tracker_latency(Ls, ps, results, code_type)

def _plot_performance_tracker_latency(self, Ls, ps, results, code_type):
    """使用性能追踪器数据绘制延迟图表，按L值和p值分别分析"""
    if 'raw_latency_all_L' not in results or not results['raw_latency_all_L']:
        print("Warning: No performance tracker data available.")
        return
    
    print("\n" + "="*80)
    print("Performance Tracker Latency Distribution Data Summary")
    print("="*80)
    print(f"Code Type: {code_type}")
    print(f"Physical Error Rate: {ps}")
    print(f"Number of Experiments: {results['num_shots']}")
    print("-"*80)
    
    # 为每个L值创建单独的图表
    for L_idx, L in enumerate(Ls):
        if L_idx >= len(results['raw_latency_all_L']):
            continue
            
        latency_data_for_L = results['raw_latency_all_L'][L_idx]
        if not latency_data_for_L:
            print(f"Warning: No latency data for L={L}")
            continue
        
        print(f"\nAnalyzing L={L}...")
        
        # 按p值收集数据
        p_data = {}  # {p_value: {'cluster': [...], 'peeling': [...], 'total': [...], 'baseline': [...]}}
        
        for p_idx, p in enumerate(ps):
            if p_idx >= len(latency_data_for_L) or not latency_data_for_L[p_idx]:
                continue
                
            cluster_latencies = []
            peeling_latencies = []
            total_latencies = []
            baseline_latencies = []
            
            for shot_data in latency_data_for_L[p_idx]:
                if isinstance(shot_data, dict):
                    cluster_latencies.append(shot_data.get('cluster_operations', 0))
                    peeling_latencies.append(shot_data.get('peeling_operations', 0))
                    total_latencies.append(shot_data.get('total_cycles', 0))
                    baseline_latencies.append(shot_data.get('estimated_baseline_latency', 0))
            
            if cluster_latencies:  # 只有当有数据时才添加
                p_data[p] = {
                    'cluster': cluster_latencies,
                    'peeling': peeling_latencies,
                    'total': total_latencies,
                    'baseline': baseline_latencies
                }
        
        if not p_data:
            print(f"Warning: No valid data found for L={L}")
            continue
        
        # 创建图表 (2x2布局)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Hardware Performance Tracker - L={L} Latency Analysis', fontsize=18)
        
        # 1. Cluster和Peeling模块延迟占比（100%堆叠柱状图）
        p_values = sorted(p_data.keys())
        cluster_avgs = []
        peeling_avgs = []
        cluster_ratios = []
        peeling_ratios = []
        
        for p in p_values:
            cluster_avg = np.mean(p_data[p]['cluster'])
            peeling_avg = np.mean(p_data[p]['peeling'])
            total_cycles = cluster_avg + peeling_avg
            
            cluster_avgs.append(cluster_avg)
            peeling_avgs.append(peeling_avg)
            
            if total_cycles > 0:
                cluster_ratios.append(cluster_avg / total_cycles * 100)
                peeling_ratios.append(peeling_avg / total_cycles * 100)
            else:
                cluster_ratios.append(0)
                peeling_ratios.append(0)
        
        x_pos = np.arange(len(p_values))
        bars1 = axes[0, 0].bar(x_pos, cluster_ratios, label='Cluster Latency', 
                               color='#1f77b4', alpha=0.8)
        bars2 = axes[0, 0].bar(x_pos, peeling_ratios, bottom=cluster_ratios, 
                               label='Peeling Latency', color='#2ca02c', alpha=0.8)
        
        # 添加数值标注
        for i, (cluster_ratio, peeling_ratio) in enumerate(zip(cluster_ratios, peeling_ratios)):
            # 标注cluster百分比
            axes[0, 0].text(i, cluster_ratio/2, f'{cluster_ratio:.1f}%', 
                           ha='center', va='center', fontweight='bold', fontsize=14)
            # 标注peeling百分比
            axes[0, 0].text(i, cluster_ratio + peeling_ratio/2, f'{peeling_ratio:.1f}%', 
                           ha='center', va='center', fontweight='bold', fontsize=14)
        
        axes[0, 0].set_title('Module Latency Distribution (100% Stacked)', fontsize=18)
        axes[0, 0].set_xlabel('Physical Error Rate', fontsize=16)
        axes[0, 0].set_ylabel('Percentage (%)', fontsize=16)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([f'{p:.3f}' for p in p_values], fontsize=14)
        axes[0, 0].legend(fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        
        # 2. 解码器Fidelity对比图
        # 提取不同解码器的数据（参考plot_results函数的逻辑）
        decoder_data = {}
        
        # 根据DECODER_CONFIG获取各解码器的LER数据（参考plot_results逻辑）
        if DECODER_CONFIG['use_mwpm'] and 'log_errors_all_L_mwpm' in results and L_idx < len(results['log_errors_all_L_mwpm']):
            mwpm_ler = results['log_errors_all_L_mwpm'][L_idx]
            decoder_data['MWPM'] = {'ler': mwpm_ler, 'color': '#1f77b4', 'marker': 'o'}
        
        if DECODER_CONFIG['use_uf'] and 'log_errors_all_L_uf' in results and L_idx < len(results['log_errors_all_L_uf']):
            uf_ler = results['log_errors_all_L_uf'][L_idx]
            decoder_data['UF'] = {'ler': uf_ler, 'color': '#ff7f0e', 'marker': 's'}
        
        # 使用peel_listdecoding的votemax结果
        if DECODER_CONFIG['use_peel_listdecoding'] and 'log_errors_all_L_uf_peel_votemax' in results and L_idx < len(results['log_errors_all_L_uf_peel_votemax']):
            peel_list_ler = results['log_errors_all_L_uf_peel_votemax'][L_idx]
            decoder_data['Peel List'] = {'ler': peel_list_ler, 'color': '#e377c2', 'marker': 'D'}
        
        # 使用peel_efficient的votemax结果
        if DECODER_CONFIG['use_peel_efficient'] and 'log_errors_all_L_uf_peel_efficient_votemax' in results and L_idx < len(results['log_errors_all_L_uf_peel_efficient_votemax']):
            peel_eff_ler = results['log_errors_all_L_uf_peel_efficient_votemax'][L_idx]
            decoder_data['Peel Efficient'] = {'ler': peel_eff_ler, 'color': '#2ca02c', 'marker': '^'}
        
        # 绘制Fidelity对比图（使用1-fidelity和对数坐标）
        ax2 = axes[0, 1]
        
        if decoder_data:
            # 为每个解码器计算并绘制1-fidelity
            for decoder_name, decoder_info in decoder_data.items():
                if len(decoder_info['ler']) == len(p_values):
                    # 计算fidelity和1-fidelity
                    fidelities = []
                    infidelities = []
                    for p_idx, p in enumerate(p_values):
                        ler = decoder_info['ler'][p_idx]
                        
                        # 根据解码器类型计算周期数
                        if decoder_name == 'Peel Efficient':
                            # Peel Efficient: 使用total average周期数，乘以0.002
                            cycles = np.mean(p_data[p]['total']) * 0.005
                        elif decoder_name == 'Peel List':
                            # Peel List: 使用baseline周期数，乘以0.002
                            cycles = np.mean(p_data[p]['baseline']) * 0.005
                        elif decoder_name == 'UF':
                            # UF (helios): 根据L和p查表，乘以0.0133
                            cycles = self._get_helios_cycles(L, p) * 0.0133
                        elif decoder_name == 'MWPM':
                            # MWPM (micro-blossom): 根据L和p查表，乘以0.0232
                            cycles = self._get_micro_blossom_cycles(L, p) * 0.0232
                        else:
                            # 默认使用total周期数
                            cycles = np.mean(p_data[p]['total'])
                        
                        fidelity = self.get_fidelity(ler, cycles, d=1)
                        fidelities.append(fidelity)
                        infidelities.append(1 - fidelity)
                    
                    # 绘制1-fidelity曲线（对数坐标）
                    ax2.semilogy(p_values, infidelities, 
                               marker=decoder_info['marker'], linestyle='-',
                               color=decoder_info['color'], linewidth=2, markersize=8,
                               label=f'{decoder_name}')
                    
                    # 添加数值标注（显示原始fidelity值）
                    for p, fidelity, infidelity in zip(p_values, fidelities, infidelities):
                        ax2.annotate(f'{fidelity:.6f}', (p, infidelity), 
                                   textcoords="offset points", xytext=(0,10), 
                                   ha='center', fontsize=10)
            
            ax2.set_xlabel('Physical Error Rate', fontsize=16)
            ax2.set_ylabel('1 - System Fidelity (log scale)', fontsize=16)
            ax2.set_title('Decoder Infidelity Comparison', fontsize=18)
            ax2.legend(fontsize=14)
            ax2.grid(True, alpha=0.3)
            # 设置y轴范围，显示1e-6到1的范围
            ax2.set_ylim(1e-6, 1)
            # 设置坐标轴刻度字体大小
            ax2.tick_params(axis='both', which='major', labelsize=14)
        else:
            # 如果没有解码器数据，显示延迟对比
            width = 0.35
            bars1 = axes[0, 1].bar(x_pos - width/2, cluster_avgs, width, 
                                   label='Cluster Latency', color='#1f77b4', alpha=0.7)
            bars2 = axes[0, 1].bar(x_pos + width/2, peeling_avgs, width, 
                                   label='Peeling Latency', color='#2ca02c', alpha=0.7)
            
            axes[0, 1].set_title('Average Module Latency by Error Rate', fontsize=18)
            axes[0, 1].set_xlabel('Physical Error Rate', fontsize=16)
            axes[0, 1].set_ylabel('Average Cycles', fontsize=16)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels([f'{p:.3f}' for p in p_values], fontsize=14)
            axes[0, 1].legend(fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
        
        # 3. Total Latency和Baseline Latency对比（小提琴图）
        total_data = []
        baseline_data = []
        violin_positions = []
        labels = []
        
        for i, p in enumerate(p_values):
            total_data.append(p_data[p]['total'])
            baseline_data.append(p_data[p]['baseline'])
            violin_positions.extend([i*2, i*2+0.8])  # 为每个p值创建两个位置
            labels.extend([f'Total\np={p:.3f}', f'Baseline\np={p:.3f}'])
        
        # 绘制小提琴图
        all_violin_data = []
        for i, p in enumerate(p_values):
            all_violin_data.append(p_data[p]['total'])
            all_violin_data.append(p_data[p]['baseline'])
        
        parts = axes[1, 0].violinplot(all_violin_data, positions=violin_positions, 
                                     showmeans=True, showmedians=True, showextrema=True, widths=0.6)
        
        # 设置颜色
        for i, pc in enumerate(parts['bodies']):
            if i % 2 == 0:  # Total latency
                pc.set_facecolor('#ff7f0e')
                pc.set_alpha(0.7)
            else:  # Baseline latency
                pc.set_facecolor('#d62728')
                pc.set_alpha(0.7)
        
        # 添加平均值和最大值标注
        for i, p in enumerate(p_values):
            total_avg = np.mean(p_data[p]['total'])
            total_max = np.max(p_data[p]['total'])
            baseline_avg = np.mean(p_data[p]['baseline'])
            baseline_max = np.max(p_data[p]['baseline'])
            
            # Total latency标注
            axes[1, 0].text(i*2, total_avg, f'Avg: {total_avg:.1f}', 
                           ha='center', va='bottom', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            axes[1, 0].text(i*2, total_max, f'Max: {total_max:.1f}', 
                           ha='center', va='bottom', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7))
            
            # Baseline latency标注
            axes[1, 0].text(i*2+0.8, baseline_avg, f'Avg: {baseline_avg:.1f}', 
                           ha='center', va='bottom', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            axes[1, 0].text(i*2+0.8, baseline_max, f'Max: {baseline_max:.1f}', 
                           ha='center', va='bottom', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7))
        
        axes[1, 0].set_title('Total vs Baseline Latency Distribution by Error Rate', fontsize=18)
        axes[1, 0].set_ylabel('Latency (Cycles)', fontsize=16)
        axes[1, 0].set_xticks(violin_positions)
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#ff7f0e', alpha=0.7, label='Total Latency'),
                         Patch(facecolor='#d62728', alpha=0.7, label='Baseline Latency')]
        axes[1, 0].legend(handles=legend_elements, loc='upper left', fontsize=14)
        
        # 4. 详细延迟统计表格
        axes[1, 1].axis('off')
        table_data = []
        headers = ['Error Rate', 'Cluster Avg', 'Peeling Avg', 'Total Avg', 'Baseline Avg', 
                  'Total Max', 'Baseline Max']
        table_data.append(headers)
        
        for p in p_values:
            row = [
                f'{p:.5f}',
                f'{np.mean(p_data[p]["cluster"]):.1f}',
                f'{np.mean(p_data[p]["peeling"]):.1f}',
                f'{np.mean(p_data[p]["total"]):.1f}',
                f'{np.mean(p_data[p]["baseline"]):.1f}',
                f'{np.max(p_data[p]["total"]):.1f}',
                f'{np.max(p_data[p]["baseline"]):.1f}'
            ]
            table_data.append(row)
        
        # 创建表格
        table = axes[1, 1].table(cellText=table_data[1:], colLabels=table_data[0],
                               cellLoc='center', loc='center', fontsize=12)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E6E6FA')
            table[(0, i)].set_text_props(weight='bold')
        
        axes[1, 1].set_title(f'L={L} Latency Statistics Summary', fontsize=18)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, f"performance_tracker_latency_L{L}_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance tracker latency analysis for L={L} saved to: {save_path}")
        plt.show()
        
        # 打印详细统计信息
        print(f"\n=== L={L} Performance Tracker Statistics Summary ===")
        for p in p_values:
            cluster_avg = np.mean(p_data[p]['cluster'])
            peeling_avg = np.mean(p_data[p]['peeling'])
            total_avg = np.mean(p_data[p]['total'])
            baseline_avg = np.mean(p_data[p]['baseline'])
            total_max = np.max(p_data[p]['total'])
            baseline_max = np.max(p_data[p]['baseline'])
            
            print(f"p={p:.3f}: Cluster={cluster_avg:.1f}, Peeling={peeling_avg:.1f}, "
                  f"Total={total_avg:.1f} (max={total_max:.1f}), "
                  f"Baseline={baseline_avg:.1f} (max={baseline_max:.1f})")
            
            if total_avg > 0:
                print(f"         Cluster/Total: {cluster_avg/total_avg*100:.1f}%, "
                      f"Peeling/Total: {peeling_avg/total_avg*100:.1f}%")
        print("=" * 60)

def plot_results_with_varying_styles(self, Ls, ps, results, num_candidates, code_type = 'toric_code'):
    """绘制实验结果，根据不同的L值使用不同的线型和颜色
    Args:
        Ls: 格子大小列表
        ps: 错误概率列表
        results: 实验结果字典
        num_candidates: 候选解数量
    """
    plt.figure(figsize=(12,9))
    
    # 打印结果数据
    print("\n" + "="*80)
    print("Experimental Results Summary (Varying Styles)")
    print("="*80)
    print(f"Code Type: {code_type}")
    print(f"Physical Error Rate: {ps}")
    print(f"Number of Candidates: {num_candidates}")
    print(f"Number of Experiments: {results['num_shots']}")
    print("-"*80)
    
    # 定义不同代码类型对应的样式组
    line_styles = ['-', '--', '-.', ':']
    
    # 定义每种类型的固定颜色
    type_colors = {
        'mwpm': '#1f77b4',      # 蓝色
        'mwpm_dem': '#1f77b4',
        'mwpm_hx': '#0d4f8b',
        'mwpm_dem_weighted': 'navy',
        'uf': '#ff7f0e',        # 橙色
        'bposd': 'black',     # 绿色
        'uf_peel_list': '#d62728',      # 红色
        'uf_peel_minweight': '#9467bd',  # 紫色
        # 'uf_peel_votemin': '#8c564b',    # 棕色     
        'uf_peel_votemax': '#e377c2',    # 粉色
        'uf_peel_syndrome': '#7f7f7f',   # 灰色
        # 'uf_peel_topological': '#bcbd22'  # 黄绿色
        'uf_pratical_list': '#8c564b',    # 棕色
        'uf_pratical_minweight': '#17becf',  # 青色
        'uf_pratical_votemax': '#ff9896',    # 浅红色
        'uf_pratical_syndrome': '#98df8a',   # 浅绿色
    }
    
    # 遍历不同的 L 值，对每个 L 分别绘制不同解码器的逻辑错误率曲线
    for L_idx, L in enumerate(Ls):
        print(f"Plotting for L={L}...")
        
        # 打印当前L的数据
        print(f"\nL={L} Results Data (Varying Styles):")
        print("-"*60)
        
        # 获取当前L对应的所有错误率数据
        current_errors = {}
        current_errors['mwpm'] = None
        current_errors['mwpm_dem'] = None
        current_errors['mwpm_dem_weighted'] = None
        current_errors['mwpm_hx'] = None
        
        # 根据配置添加相应的错误率数据
        if DECODER_CONFIG['use_mwpm'] and L in Ls:
            idx = Ls.index(L)
            current_errors['mwpm'] = results['log_errors_all_L_mwpm'][idx]
            if _has_dual_mwpm_data(results):
                current_errors['mwpm_dem'] = results['log_errors_all_L_mwpm_dem_unweighted'][idx]
                current_errors['mwpm_hx'] = results['log_errors_all_L_mwpm_hx_manual_unweighted'][idx]
            if _has_dem_weighted_data(results):
                current_errors['mwpm_dem_weighted'] = results['log_errors_all_L_mwpm_dem_weighted'][idx]
        
        if DECODER_CONFIG['use_uf'] and L in Ls:
            current_errors['uf'] = results['log_errors_all_L_uf'][Ls.index(L)]

        if DECODER_CONFIG['use_bposd'] and L in Ls:
            idx = Ls.index(L)
            current_errors['bposd'] = results['log_errors_all_L_bposd'][idx]
            if _has_dual_bposd_data(results):
                current_errors['bposd_dem_graph'] = results['log_errors_all_L_bposd_dem_graph'][idx]
                current_errors['bposd_shared_graph'] = results['log_errors_all_L_bposd_shared_graph'][idx]
        
        if DECODER_CONFIG['use_peel_listdecoding'] and L in Ls:
            current_errors['uf_peel_list'] = results['log_errors_all_L_uf_peel_list'][Ls.index(L)]
            current_errors['uf_peel_minweight'] = results['log_errors_all_L_uf_peel_minweight'][Ls.index(L)]
            # current_errors['uf_peel_votemin'] = results['log_errors_all_L_uf_peel_votemin'][Ls.index(L)]
            current_errors['uf_peel_votemax'] = results['log_errors_all_L_uf_peel_votemax'][Ls.index(L)]
            current_errors['uf_peel_syndrome'] = results['log_errors_all_L_uf_peel_syndrome'][Ls.index(L)]
            # current_errors['uf_peel_topological'] = results['log_errors_all_L_uf_peel_topological'][Ls.index(L)]
            # 添加practical错误率数据
            current_errors['uf_pratical_list'] = results['log_errors_all_L_uf_pratical_list'][Ls.index(L)]
            current_errors['uf_pratical_minweight'] = results['log_errors_all_L_uf_pratical_minweight'][Ls.index(L)]
            current_errors['uf_pratical_votemax'] = results['log_errors_all_L_uf_pratical_votemax'][Ls.index(L)]
            current_errors['uf_pratical_syndrome'] = results['log_errors_all_L_uf_pratical_syndrome'][Ls.index(L)]
        
        # 打印调试信息
        print(f"L={L}, ps length={len(ps)}")
        for key, value in current_errors.items():
            if value is not None:
                print(f"{key} length={len(value)}")
        
        # 分别绘制每种解码器的结果
        if DECODER_CONFIG['use_mwpm'] and current_errors['mwpm'] is not None:
            if current_errors['mwpm_dem'] is not None and current_errors['mwpm_hx'] is not None:
                has_dmw = current_errors['mwpm_dem_weighted'] is not None
                dmw_len_ok = (not has_dmw) or (len(ps) == len(current_errors['mwpm_dem_weighted']))
                if len(ps) == len(current_errors['mwpm_dem']) and len(ps) == len(current_errors['mwpm_hx']) and dmw_len_ok:
                    plt.errorbar(
                        ps, current_errors['mwpm_dem'],
                        yerr=(current_errors['mwpm_dem']*(1-current_errors['mwpm_dem'])/results['num_shots'])**0.5,
                        fmt=f'o{line_styles[L_idx % len(line_styles)]}',
                        color=type_colors['mwpm_dem'],
                        capsize=4,
                        markersize=10,
                        linewidth=2.5,
                        label="MWPM-DEM(unweighted)" if L_idx == 0 else None
                    )
                    plt.errorbar(
                        ps, current_errors['mwpm_hx'],
                        yerr=(current_errors['mwpm_hx']*(1-current_errors['mwpm_hx'])/results['num_shots'])**0.5,
                        fmt=f's{line_styles[L_idx % len(line_styles)]}',
                        color=type_colors['mwpm_hx'],
                        capsize=4,
                        markersize=8,
                        linewidth=2.0,
                        label="MWPM-Hx-manual(unweighted)" if L_idx == 0 else None
                    )
                    if has_dmw:
                        plt.errorbar(
                            ps, current_errors['mwpm_dem_weighted'],
                            yerr=(current_errors['mwpm_dem_weighted']*(1-current_errors['mwpm_dem_weighted'])/results['num_shots'])**0.5,
                            fmt=f'^{line_styles[L_idx % len(line_styles)]}',
                            color=type_colors['mwpm_dem_weighted'],
                            capsize=4,
                            markersize=8,
                            linewidth=2.0,
                            label="MWPM-DEM(weighted)" if L_idx == 0 else None
                        )
                else:
                    print(f"Warning: Length mismatch for dual MWPM - ps: {len(ps)}, dem: {len(current_errors['mwpm_dem'])}, hx: {len(current_errors['mwpm_hx'])}")
            else:
                if len(ps) == len(current_errors['mwpm']):
                    line = plt.errorbar(ps, current_errors['mwpm'], 
                                yerr=(current_errors['mwpm']*(1-current_errors['mwpm'])/results['num_shots'])**0.5,
                                fmt=f'{line_styles[L_idx % len(line_styles)]}', 
                                color=type_colors['mwpm'], 
                                capsize=4, 
                                markersize=14, 
                                linewidth=3, 
                                label="MWPM" if L_idx == 0 else None)
                    # 添加数值标签
                    for x, y in zip(ps, current_errors['mwpm']):
                        plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                else:
                    print(f"Warning: Length mismatch for MWPM - ps: {len(ps)}, mwpm: {len(current_errors['mwpm'])}")
        
        if DECODER_CONFIG['use_uf'] and current_errors['uf'] is not None:
            if len(ps) == len(current_errors['uf']):
                line = plt.errorbar(ps, current_errors['uf'], 
                            yerr=(current_errors['uf']*(1-current_errors['uf'])/results['num_shots'])**0.5,
                            fmt=f'{line_styles[L_idx % len(line_styles)]}', 
                            color=type_colors['uf'], 
                            capsize=4, 
                            markersize=18, 
                            linewidth=3, 
                            label="UF" if L_idx == 0 else None)
                # 添加数值标签
                for x, y in zip(ps, current_errors['uf']):
                    plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                
                # 打印UF解码器数据
                print(f"UF Decoder (L={L}):")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf'])):
                    # 计算误差棒数值
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"  p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                print()
            else:
                print(f"Warning: Length mismatch for UF - ps: {len(ps)}, uf: {len(current_errors['uf'])}")

        if DECODER_CONFIG['use_bposd'] and current_errors['bposd'] is not None:
            if current_errors['bposd_dem_graph'] is not None and current_errors['bposd_shared_graph'] is not None:
                if len(ps) == len(current_errors['bposd_dem_graph']) and len(ps) == len(current_errors['bposd_shared_graph']):
                    plt.errorbar(ps, current_errors['bposd_dem_graph'], 
                                yerr=(current_errors['bposd_dem_graph']*(1-current_errors['bposd_dem_graph'])/results['num_shots'])**0.5,
                                fmt=f'o{line_styles[L_idx % len(line_styles)]}',
                                color=type_colors['bposd'],
                                linewidth=2.3,
                                markersize=8,
                                capsize=6,
                                label="BP-OSD(DEM-graph)" if L_idx == 0 else None)
                    plt.errorbar(ps, current_errors['bposd_shared_graph'], 
                                yerr=(current_errors['bposd_shared_graph']*(1-current_errors['bposd_shared_graph'])/results['num_shots'])**0.5,
                                fmt=f'd{line_styles[L_idx % len(line_styles)]}',
                                color='gray',
                                linewidth=2.0,
                                markersize=7,
                                capsize=6,
                                label="BP-OSD(shared-graph)" if L_idx == 0 else None)
                else:
                    print(f"Warning: Length mismatch for dual BPOSD - ps: {len(ps)}, dem_graph: {len(current_errors['bposd_dem_graph'])}, shared_graph: {len(current_errors['bposd_shared_graph'])}")
            elif len(ps) == len(current_errors['bposd']):
                line = plt.errorbar(ps, current_errors['bposd'], 
                            yerr=(current_errors['bposd']*(1-current_errors['bposd'])/results['num_shots'])**0.5,
                            fmt=f'{line_styles[L_idx % len(line_styles)]}', 
                            color=type_colors['bposd'],     
                            linewidth=2, 
                            markersize=8, 
                                capsize=6,
                            label="BP-OSD" if L_idx == 0 else None)
                # 添加数值标签
                for x, y in zip(ps, current_errors['bposd']):   
                    plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                
                # 打印BP-OSD解码器数据（包含误差棒）
                print(f"BP-OSD Decoder (L={L}):")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['bposd'])):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"  p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                print()
            else:
                print(f"Warning: Length mismatch for BP-OSD - ps: {len(ps)}, bposd: {len(current_errors['bposd'])}")

        if DECODER_CONFIG['use_peel_listdecoding'] and current_errors['uf_peel_list'] is not None:
            if len(ps) == len(current_errors['uf_peel_list']):
                line = plt.errorbar(ps, current_errors['uf_peel_list'], 
                            yerr=(current_errors['uf_peel_list']*(1-current_errors['uf_peel_list'])/results['num_shots'])**0.5,
                            fmt=f'{line_styles[L_idx % len(line_styles)]}', 
                            color=type_colors['uf_peel_list'], 
                            linewidth=2, 
                            markersize=8, 
                            capsize=6,
                            label="UF Peeling List Decoding-Theoretical optimal" if L_idx == 0 else None)
                # 添加数值标签
                for x, y in zip(ps, current_errors['uf_peel_list']):
                    plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                
                # 打印Peeling List Decoding数据
                print(f"Peeling List Decoding (L={L}):")
                print(f"  Theoretical Optimal:")
                for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_list'])):
                    error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                    print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                
                # 如果有其他peeling方法的数据，也打印出来
                if current_errors.get('uf_peel_minweight') is not None:
                    print(f"  MaxLikelihood Selection:")
                    for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_minweight'])):
                        error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                        print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                
                if current_errors.get('uf_peel_votemax') is not None:
                    print(f"  Majority Voting:")
                    for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_votemax'])):
                        error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                        print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                
                if current_errors.get('uf_peel_syndrome') is not None:
                    print(f"  Syndrome Selection:")
                    for i, (p, error_rate) in enumerate(zip(ps, current_errors['uf_peel_syndrome'])):
                        error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                        print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                print()
            else:
                print(f"Warning: Length mismatch for UF Peeling List - ps: {len(ps)}, uf_peel_list: {len(current_errors['uf_peel_list'])}")

            # 其他peel list decoding方法使用相同的线型但不同的颜色
            for i, (key, label) in enumerate([
                # ('uf_peel_minweight', 'MaxLikelihood Selection'),
                # ('uf_peel_votemin', 'Minority Voting'),
                # ('uf_peel_votemax', 'Majority Voting'),
                ('uf_peel_syndrome', 'Syndrome Similarity'),
                # ('uf_peel_topological', 'Topological Similarity')
            ]):
                if current_errors[key] is not None:
                    if len(ps) == len(current_errors[key]):
                        line = plt.errorbar(ps, current_errors[key], 
                                    yerr=(current_errors[key]*(1-current_errors[key])/results['num_shots'])**0.5,
                                    fmt=f'{line_styles[L_idx % len(line_styles)]}', 
                                    color=type_colors[key], 
                                    linewidth=2, 
                                    markersize=8, 
                                    capsize=6,
                                    label=f"UF Peeling List Decoding-{label}" if L_idx == 0 else None)
                        # 添加数值标签
                        for x, y in zip(ps, current_errors[key]):
                            plt.annotate(self._format_error_label(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
                        
                        # 打印数据（包含误差棒）
                        print(f"  {label}:")
                        for i, (p, error_rate) in enumerate(zip(ps, current_errors[key])):
                            error_bar = (error_rate * (1 - error_rate) / results['num_shots']) ** 0.5
                            print(f"    p={p:.4f}: {self._format_error_label(error_rate)} ± {self._format_error_label(error_bar)}")
                    else:
                        print(f"Warning: Length mismatch for {key} - ps: {len(ps)}, {key}: {len(current_errors[key])}")
    
    # 设置图表属性
    plt.xlabel("Physical error rate", fontsize=18)
    plt.ylabel("Logical error rate", fontsize=18)
    plt.legend(loc="best", fontsize=12)
    plt.title("Comparison of List Decoding and MWPM/UF Decoders", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    
    # 设置y轴为对数刻度
    plt.gca().set_yscale('log')
    
    # 设置刻度标签大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 修改所有数值标签的大小
    for text in plt.gca().texts:
        text.set_fontsize(10)
    
    # 保存图像
    # 生成包含L、list_size、ps范围、shot数目的文件名
    L_str = '-'.join(map(str, Ls))
    ps_range = f"p{ps[0]:.3f}-{ps[-1]:.3f}"
    save_filename = f"plot_L{L_str}_list{num_candidates}_ps{ps_range}_shots{results['num_shots']}.png"
    save_path = os.path.join(self.save_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {save_path}")
        
    plt.show()
