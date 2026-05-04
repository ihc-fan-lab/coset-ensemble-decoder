from functools import wraps
from typing import Callable, Any

class CycleContext:
    def __init__(self):
        self.current_cycle = 0  # 全局周期计数器
        self.module_dependencies = {
            'peeling': 'cluster'   # precompute依赖cluster
        }
        self.module_end_cycles = {}  # 存储每个模块的结束周期
        self.module_latencies = {}  # 存储每个模块的延迟统计
        self.module_OPs = {}  # 存储每个模块的操作数统计

    def get_current_cycle(self) -> int:
        """获取当前周期数"""
        return self.current_cycle
    
    def get_module_start_cycle(self, module_name: str) -> int:
        """获取模块的开始周期"""
        if module_name in self.module_dependencies:
            dep_module = self.module_dependencies[module_name]
            return self.module_end_cycles.get(dep_module, 0)
        return self.current_cycle
    
    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        peeling_ops = self.module_OPs.get('Peeling_OPs', 0)
        baseline_ops = self.module_OPs.get('Baseline_OPs', 0)
        peeling_latency = self.module_latencies.get('peeling', 0)
        cluster_latency = self.module_latencies.get('cluster', 0)
        
        # 避免除零错误，安全计算estimated_baseline_latency
        if peeling_ops > 0:
            estimated_baseline_latency = cluster_latency + peeling_latency * baseline_ops / peeling_ops
        else:
            # 如果没有peeling操作，baseline latency就是cluster latency
            estimated_baseline_latency = cluster_latency
        min_peeling_latency = self.module_latencies.get('min_peeling', 0)
        cluster_stalls = self.module_latencies.get('cluster_stalls', 0)
        return {
            'total_cycles': self.current_cycle,
            'module_latencies': self.module_latencies.copy(),
            'cluster_latency': cluster_latency,
            'cluster_stalls': cluster_stalls,
            'cluster_busy': cluster_latency - cluster_stalls,
            'peeling_latency': self.module_latencies.get('peeling', 0),
            'spanning_tree_latency': self.module_latencies.get('spanning_tree', 0),
            'peeling_only_latency': self.module_latencies.get('peeling_only', 0),
            'min_peeling_latency': min_peeling_latency,
            'min_total_cycles': cluster_latency + min_peeling_latency,
            'Peeling_OPs': self.module_OPs.get('Peeling_OPs', 0),
            'Baseline_OPs': self.module_OPs.get('Baseline_OPs', 0),
            'Estimated_Baseline_latency': estimated_baseline_latency
        }
    
    def reset_to_zero(self):
        """将所有性能统计重置为0"""
        self.current_cycle = 0
        self.module_latencies = {}
        self.module_end_cycles = {}
    
    @staticmethod
    def create_zero_stats() -> dict:
        """创建全零的性能统计信息"""
        return {
            'total_cycles': 0,
            'module_latencies': {},
            'cluster_latency': 0,
            'cluster_stalls': 0,
            'cluster_busy': 0,
            'peeling_latency': 0,
            'spanning_tree_latency': 0,
            'peeling_only_latency': 0,
            'min_peeling_latency': 0,
            'min_total_cycles': 0,
            'Peeling_OPs': 0,
            'Baseline_OPs': 0,
            'Estimated_Baseline_latency': 0
        }


# cycle_tracker.py
def track_cycles_with_callback(module_name: str, default_latency: int = 1):
    """
    支持内部状态回调的装饰器
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cycle_ctx = kwargs.get('cycle_ctx')
            if cycle_ctx is None:
                cycle_ctx = CycleContext()
                kwargs['cycle_ctx'] = cycle_ctx
            
            start_cycle = cycle_ctx.get_current_cycle()
            
            if module_name == 'cluster':
                _cluster_state_collector = {
                    'pipeline_level': 7, # 7 is the pipeline level for the cluster module
                    'num_ins': 0,
                    'num_stalls': 0
                }
                kwargs['_state_collector'] = _cluster_state_collector
            elif module_name == 'peeling':
                _peeling_state_collector = {
                    'spanning_tree_cycle': 0,
                    'peeling_cycle': 0,
                    'Peeling_OPs': 0
                }
                kwargs['_state_collector'] = _peeling_state_collector

            # 执行原函数
            result = func(*args, **kwargs)
            
            # 推进周期 - 根据不同模块计算latency
            state_collector = kwargs.get('_state_collector', {})
            if module_name == 'cluster':
                # cluster模块：基于pipeline_level和num_ins
                pipeline_level = state_collector.get('pipeline_level', 6)
                num_ins = state_collector.get('num_ins', 0)
                num_stalls = state_collector.get('num_stalls', 0)
                cluster_latency = pipeline_level + num_ins
                cycle_ctx.current_cycle += cluster_latency
                cycle_ctx.module_latencies['cluster'] = cluster_latency
                cycle_ctx.module_latencies['cluster_stalls'] = num_stalls
            elif module_name == 'peeling':
                # peeling模块：基于peeling_count
                st_peeling_latency = state_collector.get('spanning_tree_cycle', 0)
                peeling_latency = state_collector.get('peeling_cycle', 0)
                total_peeling_latency = st_peeling_latency + peeling_latency
                cycle_ctx.current_cycle += total_peeling_latency + 3 # 3 is the pipeline latency for the peeling module
                cycle_ctx.module_latencies['peeling'] = total_peeling_latency + 3
                cycle_ctx.module_latencies['spanning_tree'] = st_peeling_latency
                cycle_ctx.module_latencies['peeling_only'] = peeling_latency
                min_st = state_collector.get('min_spanning_tree_cycle', 0)
                min_peel = state_collector.get('min_peeling_cycle', 0)
                cycle_ctx.module_latencies['min_peeling'] = min_st + min_peel + 3
                cycle_ctx.module_OPs['Peeling_OPs'] = state_collector.get('Peeling_OPs', 0)
                cycle_ctx.module_OPs['Baseline_OPs'] = state_collector.get('Baseline_OPs', 0)
            else:
                cycle_ctx.current_cycle += default_latency
                cycle_ctx.module_latencies[module_name] = default_latency
            
            
            return result
        return wrapper
    return decorator


def calculate_dynamic_latency(state, default_latency):
    """根据状态计算延迟（保留原有函数以兼容性）"""
    grow_order_size = state.get('grow_order_size', 0)
    merge_count = state.get('merge_count', 0)
    
    # 基于grow_order_size和merge_count计算延迟
    base_latency = max(default_latency, grow_order_size // 2)
    merge_penalty = merge_count * 2
    
    return base_latency + merge_penalty