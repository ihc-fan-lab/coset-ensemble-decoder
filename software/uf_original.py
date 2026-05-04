from collections import defaultdict
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import random
from typing import Dict, List, Set, Tuple, Union, Any, cast
from config import DECODER_CONFIG, DEBUG_UF_GEOMETRY
import time
from sympy.abc import D
from operation_counter import increment_operation, increment_operation_efficient
from find_subgraph import build_superedge_graph
from tools.post_precessing import edge_to_qubit_index


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
    clusters[x][3] = clusters[x][3] or clusters[y][3]

    # if code_structure.code_type == 'repetition':




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
            if y < 0 or y > code_structure.L - 1:
                is_boundary = 1
            if x < 0 or x > code_structure.L:
                is_boundary = 2
        elif error_type == 'z':
            if x < 0 or x > code_structure.L - 2:
                is_boundary = 1
            if y < 0 or y > code_structure.L:
                is_boundary = 2
    elif code_structure.code_type == 'repetition':
        if error_type == 'x':
            if y < 0 or y > code_structure.L - 1:
                is_boundary = 1
        elif error_type == 'z':
            if y < 0 or y > code_structure.L - 2:
                is_boundary = 1
    return is_boundary

    


def get_valid_directions(x, y, z, code_structure, error_type='x'):
    """获取有效的生长方向
    Args:
        x, y: 当前点的坐标
        code_structure: 代码结构对象
    Returns:
        list: 有效的生长方向列表（随机顺序）
    """
    if code_structure.repetitions > 1:
        if not code_structure.code_type == 'repetition':
            if z == 0:
                directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1)]
            elif z == code_structure.repetitions:
                directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
            else:
                directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)]
        else:
            if z == 0:
                directions = [(0, 1, 0), (0, -1, 0), (0, 0, 1)]
            elif z == code_structure.repetitions:
                directions = [(0, 1, 0), (0, -1, 0), (0, 0, -1)]
            else:
                directions = [(0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)]
    else:
        if not code_structure.code_type == 'repetition':
            directions = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
        else:
            directions = [(0, 1, 0), (0, -1, 0)]
    return directions


                  
def grow(cluster_roots, boundaries, support, clusters, code_structure, syndrome, virtual_vertex, error_type='x'):
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

    # Grow 6 directions
    for u in cluster_roots:
        for v in boundaries[u]:
            valid_directions = get_valid_directions(v[0], v[1], v[2], code_structure, error_type)
            for n in valid_directions:
                if code_structure.code_type == 'toric':
                    other = ((v[0] + n[0]) % code_structure.L, (v[1] + n[1]) % code_structure.L, (v[2] + n[2]))
                elif code_structure.code_type == 'repetition':
                    other = (v[0] + n[0], (v[1] + n[1]) % code_structure.L, v[2] + n[2])
                else:
                    other = (v[0] + n[0], v[1] + n[1], v[2] + n[2])
                
                is_boundary = is_boundary_point(other[0], other[1], other[2], code_structure, error_type)
                edge = (min(v, other), max(v, other))
                if edge not in support:
                    support[edge] = 1
                elif support[edge] == 1:
                    support[edge] = 2
                    if is_boundary == 1:
                        virtual_vertex.add(other)
                        clusters[find(u, clusters, boundaries, code_structure)][3] = True
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
    return final_roots

def union_find_decoder(syndrome, code_structure, error_type='x', grow_mode='serial'):
    """标准的Union-Find解码器实现
    Args:
        syndrome: syndrome点的列表
        code_structure: 代码结构对象
        random_seed: 随机种子（不使用）
        grow_mode: 生长模式，'serial' 串行按 size 生长，'parallel' 按批次并行生长
    """
    # 初始化数据结构
    support = defaultdict(int)  # 边的生长状态：0=未生长，1=半生长，2=完全生长
    boundaries = defaultdict(list)  # 每个簇的边界点
    clusters = defaultdict(lambda: [0, 1, 0, False])  # [父节点, 大小, 奇偶性, 是否到达边界]
    grow_order = []  # 优先队列，控制簇的生长顺序
    entry_num = 1  # 用于确保相同大小的簇按固定顺序生长
    virtual_vertex = set() # 新增：记录虚拟syndrome
    root_vertex = set() # 新增：记录根节点
    
    # 初始化syndrome点
    for g in syndrome:
        clusters[g][0] = g  # 设置父节点为自己
        clusters[g][2] = 1  # 设置奇偶性为1（syndrome点都是奇数）
        clusters[g][3] = False  # 初始时未到达边界
        boundaries[g] = [g]  # 初始边界就是自己
        heapq.heappush(grow_order, [1, entry_num, g])  # 加入生长队列
        entry_num += 1

    # 主循环：持续生长和合并簇
    i=0
    if grow_mode == 'parallel':
        while grow_order:
            # 一次性取出当前批次中所有仍有效的根，做并行生长
            current_valid_roots = []
            seen_roots = set()
            while grow_order:
                grow_root = heapq.heappop(grow_order)
                root_id = grow_root[2]
                if root_id in seen_roots:
                    continue
                if clusters[root_id][0] != root_id:  # 不是根节点
                    continue
                if len(boundaries[root_id]) != grow_root[0]:  # 边界已改变
                    continue
                parity = cast(int, clusters[root_id][2])
                if parity % 2 == 0:  # 偶数簇不需要继续生长
                    continue
                current_valid_roots.append(root_id)
                seen_roots.add(root_id)

            if not current_valid_roots:
                break

            # 并行生长本轮所有有效簇
            new_odd_cluster_roots = grow(current_valid_roots, boundaries, support, clusters, code_structure, syndrome, virtual_vertex, error_type)

            # 准备下一批 grow_order（下一轮再生长）
            if new_odd_cluster_roots:
                for el in new_odd_cluster_roots:
                    heapq.heappush(grow_order, [len(boundaries[el]), entry_num, el])
                    entry_num += 1
    else:
        # 串行：按最小 size 逐个生长（保留原逻辑）
        while grow_order:
            grow_root = heapq.heappop(grow_order)  # 获取最小的簇
            if clusters[grow_root[2]][0] != grow_root[2]:  # 不是根节点
                continue
            if len(boundaries[grow_root[2]]) != grow_root[0]:  # 边界已改变
                continue
            # 使用cast确保类型检查器知道这是int类型
            parity = cast(int, clusters[grow_root[2]][2])
            if parity % 2 == 0:  # 偶数簇不需要继续生长
                continue

            # 生长并合并簇
            new_odd_cluster_roots = grow([grow_root[2]], boundaries, support, clusters, code_structure, syndrome, virtual_vertex, error_type)

            # 将新的奇数簇加入生长队列
            if new_odd_cluster_roots:
                for el in new_odd_cluster_roots:
                    heapq.heappush(grow_order, [len(boundaries[el]), entry_num, el])
                    entry_num += 1

    
    # 收集完全生长的边，并标记cluster信息
    erasure = []
    
    for el in support.keys():
        if support[el] == 2:
            erasure.append(el)
    
    
    # 返回erasure和cluster信息
    return erasure, None, None, None, virtual_vertex




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
    if code_structure.peeling_list_decoding:
        random.shuffle(neighbours)

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

    for v in vertices:
        if not visited[v]:
            tree, visited = spanning_tree_dict(graph, visited, tree, v, code_structure)

    return tree


def peeling_decoder(erasure, syndrome, num_faults, code_structure, error_type='x', 
                    root_vertex=None, virtual_vertex=None, global_time_info=None):
    """标准的peeling解码器实现
    Args:
        erasure: 需要校正的边的列表
        syndrome: syndrome点的列表
        num_faults: 可能的错误数量
        code_structure: 代码结构对象
        error_type: 错误类型，'x'或'z'
        cluster_peeling_info: cluster的peeling信息，格式为{cluster_root: [peeling_roots]}
    Returns:
        correction: 需要翻转的qubit的numpy数组
        weight: 解的权重
    """
    all_corrections = []
    all_weights = []
    
    
    # Stage 2
    # tic = time.perf_counter()
    F = spanning_forest_dict(erasure, code_structure, roots=root_vertex, virtuals =virtual_vertex)
    # toc = time.perf_counter()
    # global_time_info["Stage 2"].append(toc - tic)
    # End of Stage 2
    F.reverse()
    syndrome_dict_copy = syndrome.copy()

    # Stage 3
    # tic = time.perf_counter()
    correction, weight = peeling(code_structure, F, syndrome_dict_copy, num_faults, virtual_vertex, error_type)
    # toc = time.perf_counter()
    # global_time_info["Stage 3"].append(toc - tic)
    # End of Stage 3
    all_corrections.append(correction)
    all_weights.append(weight)
    return all_corrections, all_weights


def peeling(code_structure, F, syndrome_dict, num_faults, virtual_vertex, error_type='x'):
    A = []  # 存储需要翻转的边
    vertex_count = defaultdict(int)  # 顶点的度数
    virtual_vertex_set = set(virtual_vertex) if virtual_vertex is not None else set()
    
    
    # 计算顶点度数
    for el in F:
        vertex_count[el[0]] += 1
        vertex_count[el[1]] += 1

    # 主循环：处理叶子节点（跳过虚拟叶子，直到其被动消除）
    while F:
        leaf_edge = None
        for i, edge in enumerate(F):
            is_leaf0 = vertex_count[edge[0]] == 1
            is_leaf1 = vertex_count[edge[1]] == 1
            if not (is_leaf0 or is_leaf1):
                continue
            leaf0_non_virtual = is_leaf0 and edge[0] not in virtual_vertex_set
            leaf1_non_virtual = is_leaf1 and edge[1] not in virtual_vertex_set
            if leaf0_non_virtual or leaf1_non_virtual:
                leaf_edge = F.pop(i)
                break
        
        # 如果没有找到叶子节点，说明图已经处理完毕
        if leaf_edge is None:
            break
            
        # 确定叶子节点和另一个端点（优先选择非虚拟叶子为 u）
        if vertex_count[leaf_edge[0]] == 1 and leaf_edge[0] not in virtual_vertex_set:
            u = leaf_edge[0]
            v = leaf_edge[1]
        else:
            u = leaf_edge[1]
            v = leaf_edge[0]
            
        # 更新度数
        vertex_count[u] -= 1
        vertex_count[v] -= 1

        
        # 处理syndrome
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
    for bit in A:
        qubit_idx = edge_to_qubit_index(bit, code_structure,error_type=error_type)
        if qubit_idx != -1 and qubit_idx < num_faults:
            correction[qubit_idx] += 1  # 标记需要翻转的qubit
            correction[qubit_idx] = correction[qubit_idx] % 2

    # 计算单个解的权重
    weight = len(A)  # 这里简单地用边的数量作为权重，你可以根据需要修改权重计算方式
    return correction, weight


def uf_original(syndrome, code_structure, error_type='x', grow_mode='serial'):
    erasure, cluster_peeling_info, \
    cluster_superedge_info, root_vertex, \
    virtual_vertex = union_find_decoder(syndrome, code_structure, error_type=error_type, grow_mode=grow_mode)
    correction, weight = peeling_decoder(erasure, syndrome, code_structure.num_qubits, 
    code_structure, error_type=error_type, root_vertex=root_vertex, 
    virtual_vertex=virtual_vertex, global_time_info=None)
    return correction, weight