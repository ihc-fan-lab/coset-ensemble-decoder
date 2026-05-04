from collections import defaultdict, deque
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import random

from sympy.abc import D
from config import DECODER_CONFIG, DEBUG_UF_GEOMETRY
from operation_counter import increment_operation, increment_operation_efficient
from find_subgraph import build_superedge_graph
from tools.post_precessing import edge_to_qubit_index
from typing import Dict, List, Set, Tuple, Union, Any, cast
from software.peeling_compression import peeling_decoder as peeling_decoder_compression



def find(el, clusters, boundaries, code_structure):
    """查找元素的根节点，使用路径压缩优化
    Args:
        el: 要查找的元素
        clusters: 簇的字典，每个元素存储[父节点, 大小, 奇偶性, 是否到达边界, 初始syndrome点集合, syndrome点连接映射]
    Returns:
        el: 元素的根节点
    """
    
    if clusters[el][0] == 0:  # 新元素，初始化为[el, 1, 0, False, set(), set()]
        clusters[el][0] = el
        clusters[el][3] = False
        clusters[el][4] = set()  # 初始syndrome点集合
        clusters[el][5] = set()     # syndrome点连接映射
        boundaries[el] = [el]
        return el
    while clusters[el][0] != el:  # 路径压缩：将路径上的所有节点直接指向根节点
        el, clusters[el][0] = clusters[el][0], clusters[clusters[el][0]][0]
    return el

def union(x, y, clusters, code_structure):
    """合并两个簇，按大小合并（小簇合并到大簇）
    Args:
        x, y: 要合并的两个簇的根节点
        clusters: 簇的字典，每个元素存储[父节点, 大小, 奇偶性, 是否到达边界, 初始syndrome点集合, syndrome点连接映射]
    """

    if x == y:  # 同一个簇，不需要合并
        return
    if clusters[x][1] < clusters[y][1]:  # 确保x是大簇
        x, y = y, x

    clusters[y][0] = x  # 将y的父节点设为x
    clusters[x][1] += clusters[y][1]  # 更新大小
    # 使用cast确保类型检查器知道这是int类型
    parity_x = cast(int, clusters[x][2])
    parity_y = cast(int, clusters[y][2])
    clusters[x][2] = parity_x + parity_y  # 更新奇偶性（直接相加，因为奇偶性本身就是0或1）
    clusters[y][2] = clusters[x][2]

    # 合并"是否触边"的状态（只要一个为 True，就为 True）
    clusters[x][3] = clusters[x][3] or clusters[y][3]
    


    
def get_valid_directions(x, y, z, code_structure, error_type='x'):
    """获取有效的生长方向
    Args:
        x, y: 当前点的坐标
        code_structure: 代码结构对象
    Returns:
        list: 有效的生长方向列表（随机顺序）
    """
    # if code_structure.periodic:
    #     directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # toric code所有方向都有效
    #     random.shuffle(directions)
    #     return directions
    
    # if code_structure.code_type == 'rotated':
    #     directions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    #     random.shuffle(directions)
    #     return directions
    # elif code_structure.code_type == 'planar':
    #     directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    #     random.shuffle(directions)
    #     return directions
    if code_structure.code_type == 'rotated':
        if code_structure.repetitions > 1:
            if z == 0:
                directions = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0), (0, 0, 1)]
            elif z == code_structure.repetitions:
                directions = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0), (0, 0, -1)]
            else:
                directions = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0), (0, 0, -1), (0, 0, 1)]
        else:
            directions = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0)]
        # if DECODER_CONFIG['use_uf_peel_forest_list']:
            # random.shuffle(directions)
    else:
        if code_structure.repetitions > 1:
            if z == 0:
                directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1)]
            elif z == code_structure.repetitions:
                directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
            else:
                directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)]
        else:
            directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
        # if DECODER_CONFIG['use_uf_peel_forest_list']:
            # random.shuffle(directions)
    return directions


def is_boundary_point(x, y, z,code_structure, error_type='x'):
    """判断点是否在边界上
    Args:
        x, y: 点的坐标
        code_structure: 代码结构对象
        error_type: 错误类型，'x'或'z'
    Returns:
        bool: 是否在边界上
        tuple: 如果是边界点，返回对应的虚拟stabilizer坐标
    """

    is_boundary = 0
    if code_structure.periodic:
        return is_boundary
    
    
        
    if code_structure.code_type == 'planar':
        #rotated code and planar code are the same
        if error_type == 'x':
            if y < 0 or y > code_structure.L - 2:
                is_boundary = 1            
            if x < 0 or x > code_structure.L:
                is_boundary = 2
        else:
            if x < 0 or x > code_structure.L - 2:
                is_boundary = 1
            if y < 0 or y > code_structure.L:
                is_boundary = 2
        # = 1 means virtual stabilizer, = 2 means rule violation
    elif code_structure.code_type == 'rotated':
        if error_type == 'x':
            if y < 0 or y > code_structure.L - 2:
                is_boundary = 1
            if x < 0 or x > code_structure.L:
                is_boundary = 2
        elif error_type == 'z':
            if x < 0 or x > code_structure.L - 2:
                is_boundary = 1
            if y < 0 or y > code_structure.L:
                is_boundary = 2

    # if DECODER_CONFIG['enable_operation_counting']:
    #     increment_operation('cluster_boundary_checks',1, cluster_list_decoding=code_structure.cluster_list_decoding,
    #                                 peeling_list_decoding=code_structure.peeling_list_decoding)
    return is_boundary
                  
def grow(cluster_roots, boundaries, support, clusters, code_structure, syndrome, error_type='x'):
    """生长和合并簇
    Args:
        cluster_roots: 需要生长的簇的根节点列表
        boundaries: 每个簇的边界点字典
        support: 边的生长状态字典
        clusters: 簇的字典
        code_structure: 代码结构对象
        virtual_stab_need: 虚拟 stabilizer需求标志
        syndrome: syndrome字典
        virtual_syndromes_accumulator_set: 用于累积虚拟syndrome坐标的集合
        error_type: 错误类型，'x'或'z'
    """
    # next_nodes = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    fusion_edges = []
    new_roots = defaultdict(int)  # 新的奇数簇根节点
    found_roots = defaultdict(int)  # 已处理的根节点
    # virtual_stab_add_count = 0
        
    # 生长阶段：所有簇同时向四周生长
    for u in cluster_roots:
        for v in boundaries[u]:
            
            valid_directions = get_valid_directions(v[0], v[1], v[2], code_structure, error_type)
            virtual_stab_add_count = 0
            for n in valid_directions:
                if code_structure.periodic:
                    other = ((v[0] + n[0]) % code_structure.L, (v[1] + n[1]) % code_structure.L, (v[2] + n[2]))
                else:
                    other = (v[0] + n[0], v[1] + n[1], v[2] + n[2])
                
            
                is_boundary = is_boundary_point(other[0], other[1], other[2], code_structure, error_type)
                # if is_boundary:
                #     clusters[find(u, clusters, boundaries)][3] = True  # 设置到达边界标志
                # if DECODER_CONFIG['enable_operation_counting']:
                #     increment_operation('cluster_grow_operations', 3, if_ops_count = True, 
                #     peeling_list_decoding = code_structure.peeling_list_decoding, efficient_decoding = code_structure.efficient_decoding)
                edge = (min(v, other), max(v, other))
                
                if edge not in support:
                    if not is_boundary == 2:
                        support[edge] = 1
                elif support[edge] == 1:
                    if (not code_structure.periodic):
                        if is_boundary == 1: # Touched a growable boundary for virtual stabilizer
                            if virtual_stab_add_count == 0:
                                support[edge] = 2
                                fusion_edges.append(edge)                    
                                # 设置cluster的边界标志为True
                                clusters[find(u, clusters, boundaries, code_structure)][3] = True
                                syndrome[other] = 2
                                # if other not in clusters: # 'other' is the virtual stab coord
                                #     clusters[other][0] = other # Parent is itself
                                #     clusters[other][1] = 0.5     # Size is 1 (or 0.5 to be small)
                                #     clusters[other][2] = 1     # Parity for UF logic is odd
                                #     clusters[other][3] = True  # Virtual node has reached boundary
                                #     boundaries[other] = [] 
                                #     syndrome[other] = 2        # MODIFIED: Mark as type 2 virtual syndrome
                                # virtual_stab_add_count += 1
                            else:
                                support[edge] = 3
                        elif is_boundary == 0:
                            support[edge] = 2
                            fusion_edges.append(edge)
                    else:
                        support[edge] = 2
                        fusion_edges.append(edge)
                elif support[edge] == 2:
                    fusion_edges.append(edge)

    # 合并阶段
    while fusion_edges:
        u, v = fusion_edges.pop()
        u_root = find(u, clusters, boundaries, code_structure)
        v_root = find(v, clusters, boundaries, code_structure)
        
        if u_root != v_root:
            found_roots[u_root] += 1
            found_roots[v_root] += 1
            
            if clusters[u_root][1] < clusters[v_root][1]:
                u_root, v_root = v_root, u_root
                        
            boundaries[u_root].extend(boundaries[v_root])
            
            # 合并簇时，如果任一个簇到达边界，合并后的簇也到达边界
            #clusters[u_root][3] = clusters[u_root][3] or clusters[v_root][3]
            
            union(u_root, v_root, clusters, code_structure)
            
            if new_roots[v_root]:
                new_roots[v_root] = 0
            # 使用cast确保类型检查器知道这是int类型
            parity = cast(int, clusters[u_root][2])
            if parity % 2 == 1:
                new_roots[u_root] += 1


    # 更新边界：移除所有边都已完全生长的边界点
    for u in new_roots:
        for v in boundaries[u]:
            x, y, z = v

            valid_directions = get_valid_directions(x, y, z, code_structure, error_type)

            
            all_directions_complete = True
            for n in valid_directions:
                if code_structure.periodic:
                    other = ((v[0] + n[0]) % code_structure.L, (v[1] + n[1]) % code_structure.L, (v[2] + n[2]))
                else:
                    other = (v[0] + n[0], v[1] + n[1], v[2] + n[2])
                
                edge = (min(v, other), max(v, other))
                # if edge in support and support[edge] in {2, 3}:
                #     all_directions_complete = False
                #     break
                if support.get(edge, 0) in {2, 3}:
                    all_directions_complete = True
                else:
                    all_directions_complete = False
                    break
            if all_directions_complete:
                boundaries[u].remove(v)

    for x in cluster_roots:
        if not found_roots[x]:
            new_roots[x] += 1
            
    final_roots = [x for x in new_roots.keys() if new_roots[x]]
    # print(f"final_roots: {final_roots}")
    return final_roots

def union_find_decoder(syndrome, code_structure, random_seed=None, error_type='x'):
    """list decoding的Union-Find解码器实现
    Args:
        syndrome: syndrome点的列表
        code_structure: 代码结构对象
        random_seed: 随机种子（不使用）
    """
    ###Syndrome data transfer cycle
    # Assuming the freq of IO is 100MHz, 32bit data (decoder works in 1 GHz)
    # 32bit > 15bit (coordinate) x 2, that is 2 syndromes per cycle

    # 初始化数据结构
    support = defaultdict(int)  # 边的生长状态：0=未生长，1=半生长，2=完全生长
    boundaries = defaultdict(list)  # 每个簇的边界点
    clusters = defaultdict(lambda: [0, 1, 0, False, set(), set()])  # [父节点, 大小, 奇偶性, 是否到达边界, 初始syndrome点集合, syndrome点连接映射]
    cluster_roots = []  # 当前所有簇的根节点
    grow_order = []  # 优先队列，控制簇的生长顺序
    entry_num = 1  # 用于确保相同大小的簇按固定顺序生长
    virtual_vertex = set() # 新增：记录虚拟syndrome
    root_vertex = set() # 新增：记录根节点
    
    # 初始化syndrome点
    for g in syndrome:
        clusters[g][0] = g  # 设置父节点为自己
        clusters[g][2] = 1  # 设置奇偶性为1（syndrome点都是奇数）
        clusters[g][3] = False  # 初始时未到达边界
        clusters[g][4] = {g}  # 将初始syndrome点添加到集合中
        clusters[g][5] = set()     # syndrome点连接映射
        cluster_roots.append(g)  # 加入根节点列表
        boundaries[g] = [g]  # 初始边界就是自己
        heapq.heappush(grow_order, [1, entry_num, g])  # 加入生长队列：[boundary size, 簇编号, 簇根节点]
        entry_num += 1



    # 主循环：持续生长和合并簇
    i=0
    current_cluster_size = entry_num - 1 
    next_cluster_size = 0
    iterations_parallel = 0
    while grow_order:

        if DEBUG_UF_GEOMETRY:   
            i += 1
            print(f"grow_order: {grow_order}")
            if i > 10000:
                break
            
        grow_root = heapq.heappop(grow_order)  # 获取最小的簇
        # print(f"Step: {i}")
        # 1 pop + 4x2 comparison and if branch
        if clusters[grow_root[2]][0] != grow_root[2]:  # 不是根节点
            # print(f"Skip root: {grow_root}")
            continue
        # 使用cast确保类型检查器知道这是int类型
        parity = cast(int, clusters[grow_root[2]][2])
        if parity % 2 == 0:  # 偶数簇不需要继续生长
            # print(f"Skip even root: {grow_root}")
            continue
        if len(boundaries[grow_root[2]]) != grow_root[0]:  # 边界已改变
            # print(f"Skip boundary change: {grow_root}")
            continue
        # 如果簇已经到达边界，跳过生长
        if clusters[grow_root[2]][3]:  # 已接触边界
            continue

        # 生长并合并簇
        # print(f"grow_root: {grow_root}")
        new_odd_cluster_roots = grow([grow_root[2]], boundaries, support, clusters, code_structure, syndrome, error_type)
        
        # 将新的奇数簇加入生长队列
        if new_odd_cluster_roots:
            for el in new_odd_cluster_roots:
                heapq.heappush(grow_order, [len(boundaries[el]), entry_num, el])
                entry_num += 1
                if current_cluster_size > 0:
                    next_cluster_size += 1





    # 收集完全生长的边，并标记cluster信息
    erasure = []
    cluster_erasure_info = []  # 存储边和对应的cluster信息
    cluster_syndrome_connections = {}    


    for el in support.keys():
        if support[el] == 2:
            # v1, v2 = el
            erasure.append(el)
            # cluster_erasure_info.append((el, cluster_id_map[cluster_root]))

    
    # 返回erasure和cluster信息
    return erasure, cluster_erasure_info, root_vertex, virtual_vertex, cluster_syndrome_connections





def uf_compression(syndrome, code_structure, error_type='x', list_size = 1):
    # 保存当前随机状态，避免随机种子污染
    original_random_state = np.random.get_state()
    
    # 设置固定种子，确保结果一致性
    np.random.seed(42)
    
    erasure, cluster_erasure_info, root_vertex, \
    virtual_vertex, cluster_syndrome_connections = \
        union_find_decoder(syndrome, code_structure, error_type=error_type)

    all_corrections, all_weights = peeling_decoder_compression(erasure, 
        syndrome, code_structure.num_qubits, code_structure, error_type=error_type, 
        root_vertex=root_vertex, virtual_vertex=virtual_vertex)

    # 恢复原始随机状态
    np.random.set_state(original_random_state)
    
    return all_corrections, all_weights