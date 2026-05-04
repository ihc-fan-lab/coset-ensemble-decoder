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
                
            
                edge = (min(v, other), max(v, other))
                
                if edge not in support:
                    support[edge] = 1
                elif support[edge] == 1:
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













#########################
# Peeling Decoder 部分 #
#########################

def edge_list_to_graph(e, code_structure):
    """从边列表创建邻接表形式的图
    Args:
        e: 图的所有边的列表
    Returns:
        graph: 以顶点为键，相邻顶点列表为值的字典
    """
    graph = defaultdict(list)
    for e1, e2 in e:
        graph[e1].append(e2)
        graph[e2].append(e1)
    return graph

def spanning_tree_dict(graph, visited, tree, vertex, code_structure):
    """创建以vertex为根的生成树
    Args:
        graph: 以顶点为键，相邻顶点列表为值的字典
        visited: 记录顶点访问状态的字典
        tree: 正在构建的树
        vertex: 当前顶点
    Returns:
        tree: 最终的生成树
        visited: 更新后的访问状态字典
    """
    visited[vertex] = 1 ## 访问过的节点为1 (initialize the input vertex to 1)

    # 随机化邻居节点的访问顺序
    neighbours = graph[vertex].copy()
    np.random.shuffle(neighbours)

    for neighbour in neighbours:
        if not visited[neighbour]:
            tree.append((vertex, neighbour))
            tree, visited = spanning_tree_dict(graph, visited, tree, neighbour, code_structure)
    return tree, visited

def spanning_forest_dict(graph,code_structure,roots = None, virtuals = None):
    """创建图的生成森林
    Args:
        graph: 以顶点为键，相邻顶点列表为值的字典，或边列表
        list_decoding: 是否启用列表解码
        used_trees: 已经使用过的树结构集合
    Returns:
        tree: 图的生成森林，作为边的列表
    """
    if type(graph) != dict:
        # 假设输入是边列表
        graph = edge_list_to_graph(graph, code_structure)

    # 检查图是否为空
    if not graph:
        return []

    
    tree = []
    visited = defaultdict(int)
    vertices = list(graph.keys())
    np.random.shuffle(vertices)
    # if roots is None and virtuals is None:
    #     vertices = list(graph.keys())
    #     if code_structure.peeling_list_decoding:
    #         np.random.shuffle(vertices)
    # elif virtuals is not None:
    #     other_vertices = [v for v in graph.keys() if v not in virtuals]
    #     # if code_structure.peeling_list_decoding:
    #     #     random.shuffle(other_vertices)
    #     vertices = list(virtuals) + other_vertices
    # elif roots is not None:
    #     vertices = list(roots) + [v for v in graph.keys() if v not in roots]
        # if code_structure.peeling_list_decoding:
        #     random.shuffle(vertices)
    # 只在需要随机化时打乱顶点顺序
    # if code_structure.peeling_list_decoding:
    #     random.shuffle(vertices)
        # if vertices:
        #     start_vertex = random.choice(vertices)
        #     vertices.remove(start_vertex)
        #     vertices.insert(0, start_vertex)

    for v in vertices:
        if not visited[v]:
            tree, visited = spanning_tree_dict(graph, visited, tree, v, code_structure)

    return tree




def peeling_decoder(erasure, 
        syndrome, num_faults, code_structure, 
        num_candidates=3, error_type='x', 
        root_vertex=None, virtual_vertex=None, 
        cluster_info=None, cluster_superedge_info=None):
    """使用list decoding的peeling解码器实现
    Args:
        erasure: 需要校正的边的列表
        syndrome: syndrome点的列表
        num_faults: 可能的错误数量
        code_structure: 代码结构对象
        virtual_syndromes_sets: 虚拟syndrome集合的列表
        num_candidates: 生成的候选解数量
    Returns:
        corrections: 候选解的列表
        weights: 候选解权重的列表
    """
    all_corrections = []
    all_weights = []
    # print(f"Edges: {erasure}")
    
        
    for i in range(num_candidates):
        F = spanning_forest_dict(erasure, code_structure, roots=root_vertex, virtuals =virtual_vertex)
        F.reverse()
        syndrome_dict_copy = syndrome.copy()
        # 如果 virtual_syndromes_sets 不是 None 且长度足够，则使用它
        correction, weight = peeling(code_structure, F, syndrome_dict_copy, num_faults, error_type)
        all_corrections.append(correction)
        all_weights.append(weight)

    
    return all_corrections, all_weights







def peeling(code_structure, F, syndrome_dict, num_faults, error_type='x'):
    A = []  # 存储需要翻转的边
    vertex_count = defaultdict(int)  # 顶点的度数
    
    
    # 计算顶点度数
    for el in F:
        vertex_count[el[0]] += 1
        vertex_count[el[1]] += 1

    # 主循环：处理叶子节点
    while F:
        # increment_operation('peeling_leaf_operations', 1, peeling_list_decoding=code_structure.peeling_list_decoding)
        # 找到至少有一个端点是叶子节点的边
        leaf_edge = None
        for i, edge in enumerate(F):
            if vertex_count[edge[0]] == 1 or vertex_count[edge[1]] == 1:
                leaf_edge = F.pop(i)
                break
        
        # 如果没有找到叶子节点，说明图已经处理完毕
        if leaf_edge is None:
            break
            
        # 确定叶子节点和另一个端点
        if vertex_count[leaf_edge[0]] == 1 and vertex_count[leaf_edge[1]] == 1:
            if syndrome_dict[leaf_edge[0]] == 2:
                u = leaf_edge[1]
                v = leaf_edge[0]
            else:
                u = leaf_edge[0]
                v = leaf_edge[1]
        elif vertex_count[leaf_edge[0]] == 1:
            u = leaf_edge[0]
            v = leaf_edge[1]
        else:
            u = leaf_edge[1]
            v = leaf_edge[0]
            
        # 更新度数
        vertex_count[u] -= 1
        vertex_count[v] -= 1

        

        if syndrome_dict[u] == 1:
            A.append(leaf_edge)  # 记录需要翻转的边
            syndrome_dict[u] -= 1  # 移除已处理的syndrome点
            if syndrome_dict[v] == 1:
                syndrome_dict[v] -= 1
            else:
                syndrome_dict[v] += 1

    
    # print(f"Peeling Correction: {A}")
    # 将边的列表转换为numpy数组
    correction = np.zeros(num_faults, dtype=int)
    # print("Listdecoding Edges",A)
    for bit in A:
        qubit_idx = edge_to_qubit_index(bit, code_structure,error_type=error_type)
        if qubit_idx != -1 and qubit_idx < num_faults:
            correction[qubit_idx] += 1  # 标记需要翻转的qubit
            correction[qubit_idx] = correction[qubit_idx] % 2

    # 计算单个解的权重
    weight = len(A)  # 这里简单地用边的数量作为权重，你可以根据需要修改权重计算方式
    return correction, weight




def uf_listdecoding(syndrome, code_structure, error_type='x', list_size = 1):
    # 保存当前随机状态，避免随机种子污染
    original_random_state = np.random.get_state()
    
    # 设置固定种子，确保结果一致性
    np.random.seed(42)
    
    erasure, cluster_erasure_info, root_vertex, \
    virtual_vertex, cluster_syndrome_connections = \
        union_find_decoder(syndrome, code_structure, error_type=error_type)

    all_corrections, all_weights = peeling_decoder(erasure, 
        syndrome, code_structure.num_qubits, code_structure, 
        num_candidates= list_size, error_type=error_type, 
        root_vertex=root_vertex, virtual_vertex=virtual_vertex, 
        cluster_info=cluster_erasure_info, cluster_superedge_info=cluster_syndrome_connections)

    # 恢复原始随机状态
    np.random.set_state(original_random_state)
    
    return all_corrections, all_weights