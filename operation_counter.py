from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import random
from config import DECODER_CONFIG

class OperationCounter:
    """操作计数器类，支持多个实例独立统计"""
    
    def __init__(self, name="default"):
        """初始化操作计数器
        Args:
            name: 计数器实例的名称
        """
        self.name = name
        self.counts = {
            # UF阶段操作
            'cluster_grow_operations': 0,      # grow函数中的操作次数
            'cluster_fusion_operations': 0,    # 簇合并操作次数
            'cluster_boundary_checks': 0,      # 边界检查次数
            'cluster_find_operations': 0,      # find函数调用次数
            'cluster_union_operations': 0,     # union函数调用次数
            'uf_virtual_stab_adds': 0,         # 虚拟stabilizer添加次数

            'tree_operations': 0,      # 生成树操作次数     

            # Peeling阶段操作
            'peeling_leaf_operations': 0,      # 叶子节点处理次数
            'peeling_edge_flips': 0,           # 边翻转次数
            'peeling_post_precessing': 0,     # syndrome更新次数


            # 额外操作
            'extra_ops': 0,                    # 额外操作次数
            
            # 时间统计（可选）
            'uf_time_ms': 0,                   # UF阶段耗时（毫秒）
            'peeling_time_ms': 0,              # Peeling阶段耗时（毫秒）
            'total_time_ms': 0,                # 总耗时（毫秒）
        }
        
        # 重置计数器
        self.reset()
    
    def reset(self):
        """重置所有计数器"""
        for key in self.counts:
            self.counts[key] = 0
    
    def increment(self, operation_name, count=1):
        """增加指定操作的计数
        Args:
            operation_name: 操作名称
            count: 增加的次数，默认为1
        """
        # 只有在启用操作统计时才执行
        # if not DECODER_CONFIG.get('enable_operation_counting', True) and:
        #     return
            
        if operation_name in self.counts:
            self.counts[operation_name] += count
        else:
            # 如果操作名称不存在，自动添加
            self.counts[operation_name] = count
    
    def get_count(self, operation_name):
        """获取指定操作的计数
        Args:
            operation_name: 操作名称
        Returns:
            int: 操作计数
        """
        return self.counts.get(operation_name, 0)
    
    def get_all_counts(self):
        """获取所有操作的计数
        Returns:
            dict: 所有操作的计数字典
        """
        return self.counts.copy()
    
    def get_stage_counts(self):
        """获取各个阶段的操作计数
        Returns:
            dict: 各阶段操作计数字典
        """
        stage_counts = {
            'cluster_operations': (
                self.counts['cluster_grow_operations'] 
                # self.counts['cluster_fusion_operations'] + 
                # self.counts['cluster_boundary_checks'] + 
                # self.counts['cluster_find_operations'] + 
                # self.counts['cluster_union_operations'] + 
                # self.counts['uf_virtual_stab_adds']
            ),
            'tree_operations': self.counts['tree_operations'],
            'peeling_operations': (
                self.counts['peeling_leaf_operations']
                # self.counts['peeling_edge_flips']
                # self.counts['peeling_post_precessing'] + 
                # self.counts['peeling_tree_operations']
            )
            # 'extra_ops': self.counts['extra_ops']
        }
        stage_counts['total_operations'] = stage_counts['cluster_operations'] + stage_counts['tree_operations'] + stage_counts['peeling_operations'] # + stage_counts['extra_ops']
        return stage_counts
    
    def print_summary(self):
        """打印操作统计摘要"""
        # 只有在启用详细统计时才打印
        if not DECODER_CONFIG.get('operation_counting_verbose', False):
            return
        
        # 计算各阶段操作数
        stage_counts = self.get_stage_counts()
        
        print(f"\n=== {self.name} 解码操作统计摘要 ===")
        
        # Cluster阶段详细操作
        print("Cluster阶段操作:")
        print(f"  生长操作: {self.counts['cluster_grow_operations']}")
        print(f"  合并操作: {self.counts['cluster_fusion_operations']}")
        print(f"  边界检查: {self.counts['cluster_boundary_checks']}")
        print(f"  Find操作: {self.counts['cluster_find_operations']}")
        print(f"  Union操作: {self.counts['cluster_union_operations']}")
        print(f"  虚拟stabilizer添加: {self.counts['uf_virtual_stab_adds']}")
        print(f"  Cluster阶段总操作数: {stage_counts['cluster_operations']}")
        
        print("\nTree阶段操作:")
        print(f"  生成树操作: {self.counts['tree_operations']}")
        
        
        print("\nPeeling阶段操作:")
        print(f"  叶子节点处理: {self.counts['peeling_leaf_operations']}")
        print(f"  边翻转: {self.counts['peeling_edge_flips']}") 
        # print(f"  生成树操作: {self.counts['peeling_tree_operations']}")
        print(f"  Post-precessing: {self.counts['peeling_post_precessing']}")        
        print(f"  Peeling阶段总操作数: {stage_counts['peeling_operations']}")

        print(f"  额外操作: {self.counts['extra_ops']}")
        
        print(f"\n总操作数: {stage_counts['total_operations']}")
        
        # 可选的时间统计
        if self.counts['total_time_ms'] > 0:
            print(f"\n时间统计:")
            print(f"  UF阶段: {self.counts['uf_time_ms']:.2f}ms")
            # print(f"  Tree阶段: {self.counts['tree_time_ms']:.2f}ms")
            print(f"  Peeling阶段: {self.counts['peeling_time_ms']:.2f}ms")
            print(f"  总时间: {self.counts['total_time_ms']:.2f}ms")
        
        print("=" * 40)
    
    def export_to_dict(self):
        """导出统计数据为字典格式，便于保存或分析"""
        export_data = self.counts.copy()
        export_data.update(self.get_stage_counts())
        export_data['counter_name'] = self.name
        return export_data
    
    def merge(self, other_counter):
        """合并另一个计数器的数据
        Args:
            other_counter: 另一个OperationCounter实例
        """
        for key, value in other_counter.counts.items():
            if key in self.counts:
                self.counts[key] += value
            else:
                self.counts[key] = value

    def merge_from_dict(self, counter_dict):
        """从字典数据合并计数器
        Args:
            counter_dict: 包含计数器数据的字典
        """
        if 'counter_name' in counter_dict:
            # 这是从export_to_dict()导出的数据
            for key, value in counter_dict.items():
                if key != 'counter_name' and key in self.counts:
                    self.counts[key] += value
        else:
            # 这是直接的计数字典
            for key, value in counter_dict.items():
                if key in self.counts:
                    self.counts[key] += value

    def merge_from_pickle(self, pickle_data):
        """从pickle数据合并计数器
        Args:
            pickle_data: pickle序列化的计数器数据
        """
        import pickle
        counter_dict = pickle.loads(pickle_data)
        self.merge_from_dict(counter_dict)

    def export_to_pickle(self):
        """导出计数器数据为pickle格式，便于进程间传输
        Returns:
            bytes: pickle序列化的数据
        """
        import pickle
        return pickle.dumps(self.export_to_dict())

# 全局实例管理器
class OperationCounterManager:
    """操作计数器管理器，管理多个计数器实例"""
    
    def __init__(self):
        self.counters = {}
        self.default_counter = None
        self._create_default_counter()
    
    def _create_default_counter(self):
        """创建默认计数器"""
        self.default_counter = OperationCounter("default")
        self.counters["default"] = self.default_counter
    
    def create_counter(self, name):
        """创建新的计数器实例
        Args:
            name: 计数器名称
        Returns:
            OperationCounter: 新创建的计数器实例
        """
        if name in self.counters:
            # print(f"警告: 计数器 '{name}' 已存在，将返回现有实例")
            return self.counters[name]
        
        counter = OperationCounter(name)
        self.counters[name] = counter
        return counter
    
    def get_counter(self, name="default"):
        """获取指定的计数器实例
        Args:
            name: 计数器名称，默认为"default"
        Returns:
            OperationCounter: 计数器实例
        """
        if name not in self.counters:
            print(f"警告: 计数器 '{name}' 不存在，创建新实例")
            return self.create_counter(name)
        return self.counters[name]
    
    def set_default_counter(self, name):
        """设置默认计数器
        Args:
            name: 计数器名称
        """
        if name not in self.counters:
            print(f"错误: 计数器 '{name}' 不存在")
            return
        self.default_counter = self.counters[name]
    
    def reset_counter(self, name="default"):
        """重置指定的计数器
        Args:
            name: 计数器名称，默认为"default"
        """
        counter = self.get_counter(name)
        counter.reset()
    
    def reset_all_counters(self):
        """重置所有计数器"""
        for counter in self.counters.values():
            counter.reset()
    
    def print_all_summaries(self):
        """打印所有计数器的统计摘要"""
        for name, counter in self.counters.items():
            counter.print_summary()
    
    def get_all_counters(self):
        """获取所有计数器实例
        Returns:
            dict: 所有计数器的字典 {name: counter}
        """
        return self.counters.copy()
    
    def remove_counter(self, name):
        """删除指定的计数器
        Args:
            name: 计数器名称
        """
        if name == "default":
            print("错误: 不能删除默认计数器")
            return
        
        if name in self.counters:
            del self.counters[name]
            print(f"已删除计数器 '{name}'")
        else:
            print(f"计数器 '{name}' 不存在")

    def merge_from_other_process(self, counter_name, pickle_data):
        """从其他进程合并计数器数据
        Args:
            counter_name: 计数器名称
            pickle_data: pickle序列化的计数器数据
        """
        counter = self.get_counter(counter_name)
        counter.merge_from_pickle(pickle_data)

    def get_all_pickle_data(self):
        """获取所有计数器的pickle数据，用于进程间传输
        Returns:
            dict: {counter_name: pickle_data}
        """
        result = {}
        for name, counter in self.counters.items():
            result[name] = counter.export_to_pickle()
        return result

# 全局管理器实例
global_manager = OperationCounterManager()

# 兼容性函数 - 使用默认计数器
def get_global_counter():
    """获取全局操作计数器实例（默认计数器）"""
    return global_manager.default_counter

def reset_global_counter():
    """重置全局操作计数器（默认计数器）"""
    global_manager.default_counter.reset()

def increment_operation(operation_name, if_ops_count=True, count=1, peeling_list_decoding = False, efficient_decoding = False):
    """全局函数：增加操作计数
    Args:
        operation_name: 操作名称
        count: 增加的次数，默认为1
        counter_name: 计数器名称，默认为"default"
    """
    # if uf_decoding == True:
    #     counter = global_manager.get_counter("uf_decoding")
    #     counter.increment(operation_name, count)
    if if_ops_count == True:
        if peeling_list_decoding == True:
            counter = global_manager.get_counter("peeling_list_decoding_ops")
            counter.increment(operation_name, count)
        elif efficient_decoding == True:
            counter = global_manager.get_counter("efficient_decoding_ops")
            counter.increment(operation_name, count)
        else:
            counter = global_manager.get_counter("uf_decoding_ops")
            counter.increment(operation_name, count)
    else:
        if peeling_list_decoding == True:
            counter = global_manager.get_counter("peeling_list_decoding_cycles")
            counter.increment(operation_name, count)
        elif efficient_decoding == True:
            counter = global_manager.get_counter("efficient_decoding_cycles")
            counter.increment(operation_name, count)
        else:
            counter = global_manager.get_counter("uf_decoding_cycles")
            counter.increment(operation_name, count)

    # counter = global_manager.get_counter(counter_name)
    # counter.increment(operation_name, count)


def increment_operation_efficient(operation_name, if_ops_count=True, count=1):
    """全局函数：增加操作计数
    Args:
        operation_name: 操作名称
        count: 增加的次数，默认为1
        counter_name: 计数器名称，默认为"default"
    """
    # if uf_decoding == True:
    #     counter = global_manager.get_counter("uf_decoding")
    #     counter.increment(operation_name, count)
    if if_ops_count == True:
        counter = global_manager.get_counter("efficient_decoding_ops")
        counter.increment(operation_name, count)
    else:
        counter = global_manager.get_counter("efficient_decoding_cycles")
        counter.increment(operation_name, count)

# 新的便捷函数
def create_counter(name):
    """创建新的计数器实例
    Args:
        name: 计数器名称
    Returns:
        OperationCounter: 新创建的计数器实例
    """
    return global_manager.create_counter(name)

def get_counter(name="default"):
    """获取指定的计数器实例
    Args:
        name: 计数器名称，默认为"default"
    Returns:
        OperationCounter: 计数器实例
    """
    return global_manager.get_counter(name)

def reset_counter(name="default"):
    """重置指定的计数器
    Args:
        name: 计数器名称，默认为"default"
    """
    global_manager.reset_counter(name)

def print_all_summaries():
    """打印所有计数器的统计摘要"""
    global_manager.print_all_summaries() 