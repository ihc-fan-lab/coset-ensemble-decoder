class DecoderPipeline:
    def __init__(self):
        self.steps = []
        
    def add_step(self, step_func, **step_kwargs):
        """添加一个解码步骤到管道中
        Args:
            step_func: 解码步骤函数
            step_kwargs: 传递给函数的参数
        """
        self.steps.append((step_func, step_kwargs))
        return self
        
    def execute(self, syndrome, code_structure, error_type='x',
                actual_logicals=None, data_collector=None, noise_level=None,
                global_time_info=None):
        """执行解码管道
        Args:
            syndrome: 综合征数据
            code_structure: 代码结构对象
            channel: 错误通道类型 ('x' 或 'z')
            actual_logicals: 实际逻辑错误（用于评估）
            data_collector: 数据收集器
            noise_level: 噪声水平
        Returns:
            result: 解码结果
        """
        result = None
        cluster_info = None  # 存储cluster信息
        cluster_peeling_info = None  # 存储cluster peeling信息
        root_vertex = None  # 存储根节点信息
        virtual_vertex = None  # 存储虚拟节点信息
        
        for i, (step_func, step_kwargs) in enumerate(self.steps):
            # print(f"\n执行第 {i+1} 步: {step_func.__name__}")
            # print(f"步骤特定参数: {step_kwargs}")
            
            if i== 0:
                # 第一步，传入syndrome
                step_result = step_func(syndrome=syndrome, 
                                        global_time_info=global_time_info,
                                        code_structure=code_structure, 
                                        error_type=error_type, 
                                        **step_kwargs)
                
                    
                # print(f"Grow Stage Result: {result}")
            elif i == 1:
                # 后续步骤，处理多个候选解
                # 如果上一步返回了多个候选解，对每个候选解分别处理
                all_corrections = []
                all_weights = []
                
                # 检查 result 是否为 None 或空
                if step_result is None:
                    print("Warning: First step returned None result")
                    return None
                
                for candidate in step_result:
                    # 将cluster_info、cluster_peeling_info、root_vertex、virtual_vertex也传递给peeling算法
                    erasure,cluster_erasure_info,root_vertex,virtual_vertex, cluster_superedge_info = candidate
                    correction, weight = step_func(
                        erasure=erasure, 
                        syndrome=syndrome, 
                        num_faults=code_structure.num_qubits, 
                        code_structure=code_structure, 
                        error_type=error_type, 
                        cluster_info=cluster_erasure_info,  # 传递cluster信息
                        cluster_superedge_info=cluster_superedge_info,  # 传递cluster peeling信息
                        root_vertex=root_vertex,  # 传递根节点信息
                        virtual_vertex=virtual_vertex,  # 传递虚拟节点信息
                        global_time_info=global_time_info,
                        **step_kwargs
                    )
                    all_corrections.append(correction)
                    all_weights.append(weight)
                
                # 将所有候选解的校正和权重组合成一个列表
                result_corrections = []
                result_weights = []
                
                for state in all_corrections:
                    for correction in state:
                        result_corrections.append(correction)
                for weight in all_weights:
                    for w in weight:
                        result_weights.append(w)

                result = result_corrections, result_weights
            
            # print(f"输出结果类型: {type(result)}")
        # print(f"Peeling Stage Result: {result}")
        
        # 如果有cluster信息，将其添加到结果中
        # if cluster_info is not None:
        #     if isinstance(result, tuple):
        #         return result + (cluster_info,)
        #     else:
        #         return result, cluster_info
        
        return result

# 预定义的管道配置
def create_standard_pipeline():
    """创建标准解码管道"""
    from cluster import union_find_decoder
    from peeling import peeling_decoder
    
    pipeline = DecoderPipeline()
    pipeline.add_step(union_find_decoder)
    pipeline.add_step(peeling_decoder)
    return pipeline

# def create_hardware_goldenmodel_pipeline(num_candidates_cluster=1, num_candidates_peeling=1):
#     from cluster import union_find_decoder
#     from peeling import peeling_decoder
    
#     pipeline = DecoderPipeline()
#     pipeline.add_step(union_find_decoder, list_decoding_method=3)
#     pipeline.add_step(peeling_decoder,
#                      num_candidates=num_candidates_peeling)
#     return pipeline


# def create_cluster_list_pipeline(num_candidates_cluster=1, num_candidates_peeling=1):
#     """创建使用cluster列表解码的管道"""
#     from cluster import union_find_decoder
#     from peeling import peeling_decoder
    
#     pipeline = DecoderPipeline()
#     # 第一步：使用list decoding生成多个候选解
#     pipeline.add_step(union_find_decoder, 
#                      list_decoding_method=4,
#                      num_candidates=num_candidates_cluster)
#     # 第二步：对每个候选解分别进行peeling
#     pipeline.add_step(peeling_decoder)
#     return pipeline

def create_peeling_list_pipeline(num_candidates_cluster=1, num_candidates_peeling=1):
    """创建使用peeling列表解码的管道"""
    from cluster import union_find_decoder
    from peeling import peeling_decoder
    
    pipeline = DecoderPipeline()
    pipeline.add_step(union_find_decoder, list_decoding_method=1)
    pipeline.add_step(peeling_decoder,
                     num_candidates=num_candidates_peeling)
    return pipeline


def create_efficient_peeling_list_pipeline(num_candidates_cluster=1, num_candidates_peeling=1):
    """创建使用peeling列表解码的管道,使用connected subgraph进行高效解码"""
    from cluster import union_find_decoder
    from peeling import peeling_decoder
    
    pipeline = DecoderPipeline()
    pipeline.add_step(union_find_decoder, list_decoding_method=2)
    pipeline.add_step(peeling_decoder, 
                      num_candidates=num_candidates_peeling)
                    #   use_connected_subgraph=True)
    return pipeline


def create_hardware_peeling_list_pipeline(num_candidates_cluster=1, num_candidates_peeling=1):
    """创建使用peeling列表解码的管道,使用connected subgraph进行高效解码"""
    from cluster import union_find_decoder
    from peeling import peeling_decoder
    
    pipeline = DecoderPipeline()
    pipeline.add_step(union_find_decoder, list_decoding_method=3)
    pipeline.add_step(peeling_decoder, 
                      num_candidates=num_candidates_peeling)
                    #   use_connected_subgraph=True)
    return pipeline