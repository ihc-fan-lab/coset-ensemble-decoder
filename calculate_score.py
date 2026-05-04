from collections import defaultdict
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import random
from config import DECODER_CONFIG


def calculate_syndrome_match(correction, syndrome, H):
    """计算syndrome匹配度
    Args:
        correction: 预测的校正
        syndrome: 原始syndrome
        H: 校验矩阵
    Returns:
        match_score: 匹配度分数 (0-1)
    """
    # 确保correction是1D数组
    correction = correction.reshape(H.shape[1], 1)
    
    # 直接计算预测的syndrome
    predicted_syndrome = (H @ correction) % 2
    predicted_syndrome = predicted_syndrome.reshape(-1)  # 将predicted_syndrome展平为一维数组   
    # 确保syndrome和predicted_syndrome维度一致
    syndrome_final = syndrome.sum(axis=1) % 2
    syndrome_final = syndrome_final.reshape(-1)  # 将syndrome展平为一维数组
    
    # 计算匹配度
    total_points = len(syndrome_final)
    if total_points == 0:
        return 1.0
    
    matches = np.sum(predicted_syndrome == syndrome_final)
    return matches / total_points


def calculate_topological_score(correction, code_structure):
    """计算拓扑评分
    Args:
        correction: 预测的校正
        code_structure: 代码结构对象
    Returns:
        topological_score: 拓扑评分 (0-1)
    """
    # 将correction转换为边的列表
    edges = []
    for i, bit in enumerate(correction):
        if bit == 1:
            qubit_coords = (0,0) #code_structure.index_to_coord(i)#((0,0), (1,1))
            # 添加相关的边
            for dx, dy in [(0,0), (0,1), (1,0), (1,1)]:
                coord1 = ((qubit_coords[0] + dx) % code_structure.L, 
                         (qubit_coords[1] + dy) % code_structure.L)
                coord2 = ((qubit_coords[0] + dx + 1) % code_structure.L, 
                         (qubit_coords[1] + dy) % code_structure.L)
                edges.append((coord1, coord2))
    
    # 检查是否形成闭合路径
    if not edges:
        return 0.0
    
    # 构建图
    graph = defaultdict(list)
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    
    # 检查连通性
    visited = set()
    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
    
    # 从任意顶点开始DFS
    start_vertex = edges[0][0]
    dfs(start_vertex)
    
    # 如果所有顶点都被访问，说明是连通的
    is_connected = len(visited) == len(graph)
    
    # 计算平均路径长度
    path_lengths = []
    for start in graph:
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        path_lengths.extend(distances.values())
    
    avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0
    
    # 综合评分
    connectivity_score = 1.0 if is_connected else 0.5
    path_length_score = 1.0 / (1.0 + avg_path_length)
    
    return 0.6 * connectivity_score + 0.4 * path_length_score


def calculate_physical_error_rate(syndrome_dict, code_structure):
    """估算物理错误率
    Args:
        syndrome_dict: syndrome字典
        code_structure: 代码结构对象
    Returns:
        p: 估算的物理错误率
    """
    num_syndromes = len(syndrome_dict)
    total_qubits = code_structure.num_qubits
    return num_syndromes / (4 * total_qubits)  # 每个错误平均影响4个syndrome点

def multi_solution(correction, weights, code_structure, syndrome_dict, syndrome_array, actual_logicals):
    if not isinstance(correction, list):
        correction = [correction]
    correction = np.array(correction)
    
    # 计算每个candidate的logical error预测
    predicted_logicals = []
    for c in correction:
        # 确保c是1D数组
        c = np.array(c).flatten()
        # 计算logical error
        # 确保logicals_x.T的维度与c匹配
        logicals_t = code_structure.logicals_x.T.toarray() #if isinstance(code_structure.logicals_x, csr_matrix) else code_structure.logicals_z.T.toarray()
        pred = (c @ logicals_t) % 2
        predicted_logicals.append(pred)

    min_errors = float('inf')
    min_error_logicals = None
    for i, pred in enumerate(predicted_logicals):
        physical_errors = weights[i]
        if physical_errors < min_errors:
            min_errors = physical_errors
            min_error_logicals = pred

    # if DECODER_CONFIG['use_uf_weight']:
    weight_candidate = min_error_logicals
    # else:
    #     weight_candidate = predicted_logicals[0]


    
    # # 如果只有一个预测结果，所有返回值都相同
    # if len(predicted_logicals) == 1:
    #     single_pred = predicted_logicals[0]
    #     return predicted_logicals, single_pred, single_pred, single_pred
    
    # 初始化变量
    # min_errors = float('inf')
    # random_logicals = None
    min_vote_logicals = None
    max_vote_logicals = None
    
    # # 第一次遍历：找到最小错误数，并记录对应的预测
    # for i, pred in enumerate(predicted_logicals):
    #     physical_errors = np.sum(all_corrections[i])
    #     if physical_errors < min_errors:
    #         min_errors = physical_errors
    #         random_logicals = pred
    
    # 第二次遍历：收集所有具有最小错误数的预测
    min_error_predictions = []
    for i, pred in enumerate(predicted_logicals):
        if weights[i] == min_errors:
            min_error_predictions.append(pred)
    
    # # 如果只有一个最小错误预测，所有返回值都相同
    # if len(min_error_predictions) == 1:
    #     single_min_pred = min_error_predictions[0]
    #     return predicted_logicals, single_min_pred, single_min_pred, single_min_pred
    
    # 如果有多个最小错误预测，统计它们的出现次数
    pred_counts = {}
    for pred in min_error_predictions:
        pred_key = tuple(pred)
        pred_counts[pred_key] = pred_counts.get(pred_key, 0) + 1
    
    # 找到出现次数最多和最少的预测
    max_count = max(pred_counts.values())
    min_count = min(pred_counts.values())
    most_common_predictions = [pred for pred, count in pred_counts.items() if count == max_count]
    least_common_predictions = [pred for pred, count in pred_counts.items() if count == min_count]
    
    # 设置返回值
    min_vote_logicals = np.array(list(least_common_predictions[0]))
    max_vote_logicals = np.array(list(most_common_predictions[0]))
########################################################################################################
    predicted_logicals = np.array(predicted_logicals)

    # 计算每个candidate的多个评分
    syndrome_scores = []
    topological_scores = []
    weight_scores = []
    
    for i in range(len(predicted_logicals)):
        # 1. 计算syndrome匹配度
        # if DECODER_CONFIG['use_uf_syndrome']:
        syndrome_score = calculate_syndrome_match(correction[i], syndrome_array, code_structure.H_x)
        syndrome_scores.append(syndrome_score)
        
        # 2. 计算拓扑评分
        # if DECODER_CONFIG['use_uf_topological']:
        topological_score = calculate_topological_score(correction[i], code_structure)
        topological_scores.append(topological_score)
        

    
    # 选择各个评分最高的candidate
    # if DECODER_CONFIG['use_uf_syndrome']:
    # 找到所有syndrome_score最大的候选
    max_syndrome_score = np.max(syndrome_scores)
    max_score_indices = np.where(syndrome_scores == max_syndrome_score)[0]
    
    # 在syndrome_score最高的候选中，选择correction最少的
    min_correction = float('inf')
    best_candidate = None
    for idx in max_score_indices:
        # 计算当前候选的correction数目
        correction_count = np.sum(predicted_logicals[idx])
        if correction_count < min_correction:
            min_correction = correction_count
            best_candidate = predicted_logicals[idx]
    
    syndrome_candidate = best_candidate
    # 找到syndrome_score最大的候选
    # syndrome_candidate = predicted_logicals[np.argmax(syndrome_scores)]
    # if DECODER_CONFIG['use_uf_topological']:
    topological_candidate = predicted_logicals[np.argmax(topological_scores)]
    # else:
    #     topological_candidate = predicted_logicals[0]





    return predicted_logicals, weight_candidate, min_vote_logicals, max_vote_logicals, syndrome_candidate, topological_candidate