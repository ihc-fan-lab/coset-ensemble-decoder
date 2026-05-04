from collections import defaultdict, deque
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import random
from find_subgraph import superedge_reconstruct
from sympy.abc import D
from config import DECODER_CONFIG, DEBUG_UF_GEOMETRY
from operation_counter import increment_operation, increment_operation_efficient
from find_subgraph import build_superedge_graph
from tools.post_precessing import edge_to_qubit_index
from typing import Dict, List, Set, Tuple, Union, Any, cast
from .memory_utils import MB_vertex2cid, MB_3Dedgeweight
from .cluster_utils import sparse_enum
import copy
from .perf_tracker import track_cycles_with_callback, CycleContext


# def group_union(gid_u, gid_v, cid2gid, gid2cids, boundaries, gid_generator, extra_DSU_latency):
#     for i in range(len(cid2gid)):
#         if cid2gid[i+1] == gid_u or cid2gid[i+1] == gid_v:
#             cid2gid[i+1] = gid_generator
#             extra_DSU_latency += 1
#         # if cid2gid[i] == gid_v:
#         #     cid2gid[i] = gid_generator
#     # cid2gid[gid_u] = gid_generator
#     # cid2gid[gid_v] = gid_generator
#     # Initialize new gid2cids entry
#     gid2cids[gid_generator] = [set(), 0, 1]
#     gid2cids[gid_generator][0] = gid2cids[gid_u][0] | gid2cids[gid_v][0] #including clusters
#     gid2cids[gid_generator][1] = gid2cids[gid_u][1] + gid2cids[gid_v][1] #parity
#     gid2cids[gid_generator][2] = gid2cids[gid_u][2] + gid2cids[gid_v][2] #size    
#     gid2cids[gid_u][1] = 0
#     gid2cids[gid_v][1] = 0
#     gid_generator += 1
#     return gid_generator, extra_DSU_latency    
def group_union(gid_u, gid_v, cid2gid, gid2cids, boundaries, new_gid, extra_DSU_latency):
    if gid_u > gid_v:
        new_gid = gid_v
        old_gid = gid_u
    else:
        new_gid = gid_u
        old_gid = gid_v
    for i in range(len(cid2gid)):
        if gid_u > gid_v and cid2gid[i+1] == gid_u:
            extra_DSU_latency += 1
        if gid_v > gid_u and cid2gid[i+1] == gid_v:
            extra_DSU_latency += 1
        if cid2gid[i+1] == gid_u or cid2gid[i+1] == gid_v:
            cid2gid[i+1] = new_gid
    gid2cids[gid_u][2] = 0
    gid2cids[gid_v][2] = 0
    for cid in gid2cids[gid_u][0]:
        gid2cids[gid_u][2] += len(boundaries[cid])
    for cid in gid2cids[gid_v][0]:
        gid2cids[gid_v][2] += len(boundaries[cid])
    gid2cids[gid_u][3] += 1
    gid2cids[gid_v][3] += 1
    gid2cids[new_gid][0] = gid2cids[gid_u][0] | gid2cids[gid_v][0] #including clusters
    gid2cids[new_gid][1] = gid2cids[gid_u][1] + gid2cids[gid_v][1] #parity
    gid2cids[new_gid][2] = gid2cids[gid_u][2] + gid2cids[gid_v][2] #size
    gid2cids[old_gid][1] = 0
    gid2cids[old_gid][2] = 0
    # gid2cids[new_gid][2] = 1
    # gid_generator += 1
    return new_gid, extra_DSU_latency





def add_to_grow_order(grow_order, gid2cids, boundaries, entry_num,new_gid, current_cycle, pipeline_level):
    # cid_size = len(boundaries[new_gid])
    g_size = 0
    for cid in gid2cids[new_gid][0]:
        g_size += len(boundaries[cid])
    # g_size = gid2cids[new_gid][2]
    fusion_number = gid2cids[new_gid][3]
    ready_cycle_in_queue = current_cycle + pipeline_level
    heapq.heappush(grow_order, [g_size, entry_num, new_gid,ready_cycle_in_queue, fusion_number])
    entry_num += 1
    return entry_num



                  
def grow(u, v, boundaries, boundaries_copy, MB_vertex2cid, MB_edgeweight, cid2gid, 
gid2cids, gid_generator, code_structure, cluster_syndrome_connections, 
cid2root, merge_count, erasure=None,
grow_order=None, entry_num=None, extra_DSU_latency = 0, last_new_gid = None, ABLATION_CONFIG=None):
    """Grow and merge clusters
    Args:
        cluster_roots: List of root nodes of clusters that need to grow
        boundaries: Dictionary of boundary points for each cluster
        support: Dictionary of edge growth states
        clusters: Dictionary of clusters
        code_structure: Code structure object
        virtual_stab_need: Virtual stabilizer requirement flag
        syndrome: Syndrome dictionary
        virtual_syndromes_accumulator_set: Set for accumulating virtual syndrome coordinates
        error_type: Error type, 'x' or 'z'
    """

        

    # u = grow_cid
    # # MB_edgeweight.write_with_conflict_check(0,2,0)
    # boundaries_copy = copy.deepcopy(boundaries)
    # for v in boundaries_copy[u]:
    # valid_directions = get_valid_directions(v[0], v[1], v[2], code_structure, error_type)
    fully_grown_label = 0
    all_directions_complete = True
    # last_new_gid = None  # 用于存储循环中的最后一个 new_gid
    ##6 directions growth
    if v[2] == 0:
        edge_wenable = np.uint8(0b101111)
    elif v[2] == code_structure.repetitions:
        edge_wenable = np.uint8(0b011111)
    else:
        edge_wenable = np.uint8(0b111111)
    MB_edgeweight.write_with_conflict_check(v[0], v[1], v[2], edge_wenable, code_structure.L, code_structure.repetitions)
    edge_weight = MB_edgeweight.read_with_conflict_check(v[0], v[1], v[2], code_structure.L, code_structure.repetitions)
    # vertex2cid & cid2gid buffer operations
    all_cids = MB_vertex2cid.read_with_conflict_check(v[0], v[1], v[2], code_structure.L, code_structure.repetitions)
    cid_v1 = all_cids[0]
    # directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    vertex2cid_wenable = np.uint8(0)
    for i,(edge,weight) in enumerate(edge_weight.items()):
        v1, v2 = edge # v1 is the grown vertex, v2 is the new vertex
        if (v[2] == 0 and i == 4) or (v[2] == code_structure.repetitions and i == 5):
            fully_grown_label = fully_grown_label + 1
            continue
        if weight == 2:  
            fully_grown_label = fully_grown_label + 1
            edge = (min(v1, v2), max(v1, v2))
            if edge not in erasure:
                erasure.append(edge)
            all_directions_complete = True
            cid_v2 = all_cids[i+1]
            # component_backup, cid_v1, cid_v2 = sparse_enum(cid_v1_int, cid_v2_int)
            # print(f"component_backup: {component_backup}, cid_v1: {cid_v1}, cid_v2: {cid_v2}")
            if cid_v1 != 0 and cid_v2 == 0: ### Merge a cluster and an unassigned vertex
                vertex2cid_wenable |= (1 << i)
                boundaries[cid_v1].append(v2) 
                # union(u, v, clusters, code_structure)
            elif cid_v1 == 0 and cid_v2 != 0: ### Reverse: unassigned vertex meets existing cluster
                vertex2cid_wenable |= (1 << i)
                boundaries[cid_v2].append(v1)
            elif cid_v1 == 0 and cid_v2 == 0: ### Both unassigned, skip
                pass
            ### Case3. merge a cluster and a cluster (change cid2gid mapping)
            else:
                gid_v1 = cid2gid.get(cid_v1, 0)
                gid_v2 = cid2gid.get(cid_v2, 0)
                v1_root = cid2root[cid_v1]
                v2_root = cid2root[cid_v2]
                boundary_edge = (min(v1, v2),max(v1, v2))
                # if cid_v1 == 0:
                #     print(f"cid_v1 is 0")
                # print(f"v: {v}, v1: {v1}, v2: {v2}, cid_v1: {cid_v1}, cid_v2: {cid_v2}, gid_v1: {gid_v1}, gid_v2: {gid_v2}, v1_root: {v1_root}, v2_root: {v2_root}")
                b2r_v1 = (min(v1, v1_root),max(v1, v1_root))
                b2r_v2 = (min(v2, v2_root),max(v2, v2_root))
                if cid_v1 != cid_v2:
                    ##new superedge detection
                    if boundary_edge not in cluster_syndrome_connections:
                        cluster_syndrome_connections.append(boundary_edge)
                    if v1 != v1_root and b2r_v1 not in cluster_syndrome_connections:
                        cluster_syndrome_connections.append(b2r_v1)
                    if v2 != v2_root and b2r_v2 not in cluster_syndrome_connections:
                        cluster_syndrome_connections.append(b2r_v2)
                if gid_v1 != gid_v2:
                    merge_count += 1
                    # old_gid_generator = gid_generator
                    gid_generator, extra_DSU_latency = group_union(gid_v1, gid_v2, cid2gid, gid2cids, boundaries, gid_generator,extra_DSU_latency)
                    if gid2cids[gid_generator][1] % 2 == 1:
                        last_new_gid = gid_generator # 更新最后一个 new_gid
                        # entry_num = add_to_grow_order(grow_order, gid2cids, boundaries, entry_num, new_gid, current_cycle_num_in_queue, pipeline_level)
        else:
            all_directions_complete = False
    MB_vertex2cid.write_with_conflict_check(v[0], v[1], v[2], vertex2cid_wenable,cid_v1, code_structure.L, code_structure.repetitions)
    if ABLATION_CONFIG['if_grow_skipping']:
        if fully_grown_label == 6:
            boundaries[u].remove(v)
            SKIP_TRIGGER = True
        else:
            SKIP_TRIGGER = False
            # boundaries_copy[u].remove(v)
    else:
        SKIP_TRIGGER = False
        if fully_grown_label == 6:
            boundaries[u].remove(v)
            # boundaries_copy[u].remove(v)


    return last_new_gid, merge_count, entry_num, gid_generator, SKIP_TRIGGER, extra_DSU_latency



@track_cycles_with_callback('cluster',6)
def Module_cluster(syndrome, code_structure, error_type='x',
    cycle_ctx=None, ABLATION_CONFIG=None, _state_collector=None): ##cycle_ctx and _state_collector are for perf tracker
    """Union-Find decoder implementation for list decoding
    Args:
        syndrome: List of syndrome points
        code_structure: Code structure object
        random_seed: Random seed (not used)
    """
    ###Syndrome data transfer cycle
    ###Pipeline cycle counter
    buffer_busy_stall = 7
    ###

    # Initialize
    # support = defaultdict(int)  # Edge growth states: 0=not grown, 1=half grown, 2=fully grown
    boundaries = defaultdict(list)  # Boundary points for each cluster
    # clusters = defaultdict(lambda: [0, 1, 0, 0])  # [parent node, size, parity, cid]
    MB_v2cid = MB_vertex2cid()
    MB_edgeweight = MB_3Dedgeweight()
    cid2gid = defaultdict(int)
    gid2cids = defaultdict(lambda: [set(), 0, 1, 0])
    cid2root = defaultdict(int)
    grow_order = []  # Priority queue, controls cluster growth order
    entry_num = 1  # Used to ensure clusters of same size grow in fixed order
    ready_cycle_in_queue = 0
    cluster_syndrome_connections = []
    # if len(syndrome) > 31:
    #     print("Warning: Large number of syndromes may lead to long decoding times.")
    num_syndrome = len(syndrome)
    # if num_syndrome > 90:
    #     print("Warning: Large number of syndromes may lead to long decoding times.")
    for g in syndrome:
        cid2root[entry_num] = g
        MB_v2cid.write_jit(g[0], g[1], g[2], entry_num, code_structure.L, code_structure.repetitions)
        boundaries[entry_num] = [g]  # Initial boundary is self
        cid2gid[entry_num] = entry_num
        gid2cids[entry_num] = [{entry_num}, 1, 1, 0]  # Initialize gid2cids entry {cids, parity, syndrome_size, fusion number}
        heapq.heappush(grow_order, [1, entry_num, entry_num,ready_cycle_in_queue, 0])  # Add to growth queue: [boundary size, cluster number, cluster root node, cycle_num_in_queue, fusion number]
        entry_num += 1
        ready_cycle_in_queue += 1

    gid_generator = entry_num
    erasure = []
    pipeline_level = _state_collector['pipeline_level']
    # Main loop: continuously grow and merge clusters
    loop_counter = 0
    current_cycle = 0
    # while grow_order:
    time = 0
    non_waiting_cycle = 0
    round_counter = len(grow_order)
    while True:
        if all((info[1] % 2 == 0) for info in gid2cids.values()):
            break
        ##New logic
        hardware_fifo = []
        for item in grow_order:
            if item[3] <= time:
                heapq.heappush(hardware_fifo, item)
        if not hardware_fifo:
            if not grow_order:
                break
            # 推进到下一就绪时间
            time = min(x[3] for x in grow_order)
            continue
        current_entry = heapq.heappop(hardware_fifo)
        grow_order.remove(current_entry)    # O(n)
        heapq.heapify(grow_order)           # O(n)

        ##Old logic
        # round_counter -= 1
        # if round_counter == 0:
        #     time += pipeline_level        
        # current_entry = heapq.heappop(grow_order)  # Get the smallest cluster entry: [boundary_size, entry_num, gid]
        # current_gid = current_entry[2]  # Extract cluster root node ID
        # if current_entry[3] > time:
        #     time = current_entry[3]


        # print(f"Current gid: {current_gid}, boundary size: {len(boundaries)}")
        current_gid = current_entry[2]
        if gid2cids[current_gid][1] % 2 == 0:  # Even clusters don't need to continue growing
            continue
        if gid2cids[current_gid][3] != current_entry[4]:
            continue


        # print(f"grow_root: {grow_root}")
        # for cid in gid2cids[current_gid][0]:
        merge_count = 0
        gid2cids_copy = copy.deepcopy(gid2cids)
        last_new_gid = None
        
        for cid in gid2cids_copy[current_gid][0]: ##u-cid in this group
            boundaries_copy = copy.deepcopy(boundaries)
            SKIP_TRIGGER = False
            for v in boundaries_copy[cid]:
                extra_DSU_latency = 0
                if SKIP_TRIGGER:
                    SKIP_TRIGGER = False
                    continue
                last_new_gid, merge_count, entry_num, gid_generator, SKIP_TRIGGER, extra_DSU_latency  = grow(cid, v, boundaries, boundaries_copy, MB_v2cid, MB_edgeweight,cid2gid, 
                gid2cids, gid_generator, code_structure, cluster_syndrome_connections, cid2root, merge_count, erasure,
                grow_order, entry_num, extra_DSU_latency, last_new_gid, ABLATION_CONFIG)
                # if not ABLATION_CONFIG['if_no_mb_bufffer'] and not ABLATION_CONFIG['if_no_dsu_opt']:
                time += 1
                non_waiting_cycle += 1
                if ABLATION_CONFIG['if_no_mb_bufffer']:
                    time += buffer_busy_stall  
                if ABLATION_CONFIG['if_no_dsu_opt']:
                    time += extra_DSU_latency
            # if new_odd_cluster is not None:
        if last_new_gid is not None:
            entry_num = add_to_grow_order(grow_order, gid2cids, boundaries, entry_num, last_new_gid, time, pipeline_level)        
        if merge_count == 0:
            entry_num = add_to_grow_order(grow_order, gid2cids, boundaries, entry_num, current_gid, time, pipeline_level)
        
        # time += 1



    if gid_generator > 160:
        print(f"Warning: Large number of new gids. : {gid_generator}")
    #perf tracker code
    if _state_collector is not None:
        _state_collector['num_stalls'] = time - non_waiting_cycle
        _state_collector['num_ins'] = time ###+3 is the pipeline latency, +2 is the superedge transfer latency
    # print(f"HW: {erasure}")
    return cluster_syndrome_connections, erasure


#########################
# Peeling Decoder Section #
#########################

def edge_list_to_graph(e, code_structure):
    """Create adjacency list graph from edge list
    Args:
        e: List of all edges in the graph
    Returns:
        graph: Dictionary with vertices as keys and adjacent vertex lists as values
    """
    graph = defaultdict(list)
    for e1, e2 in e:
        graph[e1].append(e2)
        graph[e2].append(e1)
    return graph


def Module_spanning_tree(graph, code_structure, roots=None, root = None):
    """
    Hardware simulation module for spanning tree and forest generation.
    
    This module simulates the hardware behavior of creating spanning trees and forests
    from graph structures, which is essential for the peeling decoder algorithm.
    
    Args:
        graph: Dictionary with vertices as keys and adjacent vertex lists as values, or edge list
        code_structure: Code structure object containing code parameters
        roots: Optional root vertices for tree generation
        virtuals: Optional virtual vertices for tree generation
        
    Returns:
        tree: Spanning forest of the graph, as a list of edges
        
    Hardware Simulation Features:
        - Efficient graph traversal with randomized neighbor selection
        - Memory-efficient tree construction
        - Simulates hardware graph processing units
        - Optimized for quantum error correction decoding
    """

    # Initialize tree construction
    tree = []
    visited = defaultdict(int)
    vertices = list(graph.keys())
    ##Hardware simulator counter
    traverse_count = 0
    hw_token = []
    
    # Hardware randomization for better tree distribution
    if roots is not None:
        spt_fifo_other = [v for v in roots if v != root]
        np.random.shuffle(spt_fifo_other)
        if root is not None:
            vertices = [root] + spt_fifo_other
        else:
            vertices = spt_fifo_other
    else:
        np.random.shuffle(vertices)
        # pass

    # Generate spanning forest by creating trees for each connected component using BFS
    for v in vertices:
        if not visited[v]:
            traverse_count += 1
            tree, visited, traverse_count = _spanning_tree_bfs(graph, visited, tree, v, code_structure, traverse_count)

    return tree, traverse_count



def _spanning_tree_bfs(graph, visited, tree, start_vertex, code_structure, traverse_count):
    """
    Internal helper function for Module_spanning_tree.
    Creates spanning tree rooted at start_vertex using breadth-first search.
    
    Args:
        graph: Dictionary with vertices as keys and adjacent vertex lists as values
        visited: Dictionary recording vertex visit status
        tree: Tree being constructed
        start_vertex: Starting vertex for BFS
        code_structure: Code structure object
        hw_token: Hardware token list storing (vertex, cycle) tuples
        traverse_count: Traverse count
    Returns:
        tuple: (tree, visited, hw_token) - Final spanning tree, updated visit status, and hardware token
    """
    
    # BFS队列，存储(vertex, parent, cycle)元组
    queue = deque([(start_vertex, None)])  # 起始顶点周期为0
    # queue_peel = deque([(start_vertex, None, 0)])  # 起始顶点周期为0
    visited[start_vertex] = 1  # 标记起始顶点为已访问
    
    while queue:
        current_vertex, parent = queue.popleft()
        
        # 如果当前顶点有父节点，添加边到树中
        if parent is not None:
            tree.append((parent, current_vertex))
        
        # 随机化邻居访问顺序以模拟硬件行为
        neighbours = graph[current_vertex].copy()
        np.random.shuffle(neighbours)

        ##Hw sim-when multiple neighbours are available at cycle N, the cycle number of the neighbours should be N+3, N+4,...
        n = 1        
        # 将所有未访问的邻居加入队列
        for neighbour in neighbours:
            traverse_count += 1
            # last_cycle = hw_token[-1][1]
            # if last_cycle + 1 > current_cycle + n:
                # neighbour_cycle = last_cycle + 1
            # else:
                # neighbour_cycle = (current_cycle + n)
            # hw_token.append((neighbour, neighbour_cycle)) 
            n += 1
            if not visited[neighbour]:
                visited[neighbour] = 1  # 立即标记为已访问，避免重复加入队列               
                # 将邻居加入队列，当前顶点作为其父节点
                queue.append((neighbour, current_vertex))
                # queue_peel.append((neighbour, current_vertex, neighbour_cycle))
    
    return tree, visited, traverse_count







def uf_hardware(syndrome, code_structure, error_type='x', list_size = 1, ABLATION_CONFIG=None):
    """
    Hardware implementation of the UF decoder with hardware-optimized features.
    Args:
        syndrome: Syndrome dictionary
        code_structure: Code structure object
        error_type: Error type ('x' or 'z')
        list_size: List size
        ABLATION_CONFIG: Ablation configuration dictionary
    Returns:
        all_corrections: List of candidate solutions
        all_weights: List of candidate solution weights
    """
    
    # Save current random state to avoid random seed contamination
    original_random_state = np.random.get_state()
    # Set fixed seed to ensure result consistency
    np.random.seed(42)
    
    ##perf tracker
    cycle_ctx = CycleContext()
    
    cluster_syndrome_connections, erasure = \
        Module_cluster(syndrome, code_structure, error_type=error_type, 
        cycle_ctx=cycle_ctx, ABLATION_CONFIG=ABLATION_CONFIG)

    # Use hardware module for edge precomputation
    # superedge_to_edges, superedge_to_weight = Module_precompute_edges(erasure, code_structure)
    

    all_corrections, all_weights = peeling_decoder_optimized(syndrome, 
        code_structure.num_qubits, code_structure, 
        num_candidates= list_size, error_type=error_type, erasure=erasure,
        cluster_superedge_info=cluster_syndrome_connections,
        cycle_ctx=cycle_ctx, ABLATION_CONFIG=ABLATION_CONFIG)
    
    # from software.uf_listdecoding import peeling_decoder
    # all_corrections, all_weights = peeling_decoder(erasure, 
    # syndrome, code_structure.num_qubits, code_structure, 
    # num_candidates= list_size, error_type=error_type, 
    # root_vertex=None, virtual_vertex=None, 
    # cluster_info=None, cluster_superedge_info=cluster_syndrome_connections)



    # Restore original random state 
    # print(f"HW Corrections: {all_corrections}")
    np.random.set_state(original_random_state)

    # 获取性能统计信息
    performance_stats = cycle_ctx.get_performance_stats()
    
    return erasure, all_corrections, all_weights, performance_stats





#########################
# Peeling Decoder Section #
#########################

def Module_precompute_edges(cluster_superedge_info, code_structure):
    """
    Hardware simulation module for precomputing edge sets and weights of superedges.
    
    This module simulates the hardware behavior of precomputing the actual edge set
    and weight for each superedge, which is a critical optimization step in the
    peeling decoder to avoid repeated calculations.
    
    Args:
        cluster_superedge_info: List of superedges for clusters
        code_structure: Code structure object containing code parameters
        
    Returns:
        tuple: (superedge_to_edges, superedge_to_weight)
            - superedge_to_edges: Dictionary mapping normalized superedges to actual edge sets
            - superedge_to_weight: Dictionary mapping normalized superedges to weights
            
    Hardware Simulation Features:
        - Edge normalization for consistent representation
        - Lazy computation (only compute when needed)
        - Memory-efficient storage of precomputed results
        - Simulates hardware lookup tables for edge reconstruction
    """
    # Initialize hardware lookup tables
    superedge_to_edges = {}
    superedge_to_weight = {}
    
    # Hardware module: precompute edge set and weight of each superedge
    if cluster_superedge_info:
        for superedge in cluster_superedge_info:
            # Hardware normalization: ensure consistent edge representation
            normalized_superedge = normalize_edge(superedge)
            
            # Hardware optimization: only compute if not already in lookup table
            if normalized_superedge not in superedge_to_edges:
                # Simulate hardware edge reconstruction module
                temp_edges, temp_weight = superedge_reconstruct([superedge], code_structure, 0)
                
                # Store results in hardware lookup tables
                superedge_to_edges[normalized_superedge] = temp_edges
                superedge_to_weight[normalized_superedge] = temp_weight
    
    return superedge_to_edges, superedge_to_weight

def normalize_edge(edge):
    """Normalize edge coordinates to ensure consistency"""
    v1, v2 = edge
    # Use lexicographic order to ensure consistent order
    if v1 <= v2:
        return (v1, v2)
    else:
        return (v2, v1)

@track_cycles_with_callback('peeling',3)
def peeling_decoder_optimized(syndrome, num_faults, code_structure, 
        num_candidates=3, error_type='x', erasure=None,
        cluster_superedge_info=None, superedge_to_edges=None, superedge_to_weight=None,
        cycle_ctx=None, ABLATION_CONFIG=None, _state_collector=None):
    """Peeling decoder with hardware-optimized edge precomputation
    Args:
        syndrome: List of syndrome points
        num_faults: Number of possible errors
        code_structure: Code structure object
        num_candidates: Number of candidate solutions to generate
        error_type: Error type ('x' or 'z')
        cluster_superedge_info: Superedge information for clusters
    Returns:
        corrections: List of candidate solutions
        weights: List of candidate solution weights
    """
    all_corrections = []
    all_weights = []
    ####ABLATION_CONFIG['if_graph_compression']

    if ABLATION_CONFIG['if_graph_compression']:
        graph = cluster_superedge_info
    else:
        graph = erasure

    # # Use hardware module for edge precomputation
    superedge_to_edges, superedge_to_weight = Module_precompute_edges(graph, code_structure)
    if type(graph) != dict:
        graph = edge_list_to_graph(graph, code_structure)

    # single_vertex = [c for c in graph.keys() if len(graph[c]) == 1]
    # # 构造初始树：收集图中所有不重复的无向边
    # seen_undirected = set()
    # init_tree = []
    # for u, nbrs in graph.items():
    #     if len(nbrs) == 1:
    #         key = normalize_edge((u, nbrs[0]))
    #         if key in seen_undirected:
    #             continue
    #         seen_undirected.add(key)
    #         init_tree.append(key)
    # roots = [k for k in syndrome.keys() if syndrome[k] == 1]
    # np.random.shuffle(roots)
    # list_size = min(len(roots), num_candidates)
    # Main loop: generate multiple candidate solutions using hardware modules
    from software.uf_listdecoding import peeling
    max_ST_cycle = 0
    max_Peeling_cycle = 0
    min_ST_cycle = float('inf')
    min_Peeling_cycle = float('inf')
    for i in range(num_candidates):
        # if i < len(roots):
        #     root = roots[i]
        # Use hardware module for spanning tree generation
        F, ST_cycle = Module_spanning_tree(graph, code_structure, roots=None, root = None)
        # F = spanning_forest_dict(graph, code_structure, roots=None, virtuals = None)
        # ST_cycle = 2*len(graph)
        syndrome_dict_copy = syndrome.copy()
        # Use hardware module for peeling decoder
        Peeling_cycle = len(F)
        max_ST_cycle = max(max_ST_cycle, ST_cycle)
        max_Peeling_cycle = max(max_Peeling_cycle, Peeling_cycle)
        min_ST_cycle = min(min_ST_cycle, ST_cycle)
        min_Peeling_cycle = min(min_Peeling_cycle, Peeling_cycle)
        # print(f"HW Forest: {F}")
        # correction, weight = peeling(code_structure, F, syndrome_dict_copy, num_faults, error_type)
        correction, weight = Module_peeling_decoder(code_structure, F, syndrome_dict_copy, num_faults, error_type, superedge_to_edges, superedge_to_weight)
        all_corrections.append(correction)
        all_weights.append(weight)

    
    if _state_collector is not None:
        _state_collector['spanning_tree_cycle'] = max_ST_cycle
        _state_collector['peeling_cycle'] = max_Peeling_cycle
        _state_collector['min_spanning_tree_cycle'] = min_ST_cycle if num_candidates > 0 else 0
        _state_collector['min_peeling_cycle'] = min_Peeling_cycle if num_candidates > 0 else 0
        _state_collector['Peeling_OPs'] = num_candidates * 2* len(graph)
        _state_collector['Baseline_OPs'] = num_candidates * 2* len(erasure)
    return all_corrections, all_weights

def Module_peeling_decoder(code_structure, F, syndrome_dict, num_faults, error_type='x', superedge_to_edges=None, superedge_to_weight=None):
    """
    Hardware simulation module for peeling decoder algorithm.
    Based on 'peeling_decoder_simple' module, Latency = len(F) + 2
    
    Args:
        code_structure: Code structure object
        F: Spanning forest edges
        syndrome_dict: Syndrome dictionary
        num_faults: Number of possible errors
        error_type: Error type ('x' or 'z')
        superedge_to_edges: Pre-computed mapping from superedge to actual edges
        superedge_to_weight: Pre-computed mapping from superedge to weight
        
    Returns:
        tuple: (correction, weight)
            - correction: Correction array for qubit flips
            - weight: Weight of the correction
            
    Hardware Simulation Features:
        - Leaf node detection and processing
        - Syndrome point management
        - Edge weight calculation using precomputed data
        - Qubit index conversion for hardware implementation
    """
    A = []  # Store superedges that need to be flipped
    # vertex_count = defaultdict(int)  # Vertex degrees
    # # Calculate vertex degrees
    # for el in F:
    #     vertex_count[el[0]] += 1
    #     vertex_count[el[1]] += 1

    # Main loop: process leaf nodes
    # i = len(F)
    while F:
        leaf_edge = F.pop()
        v = leaf_edge[0]
        u = leaf_edge[1]
        # Find an edge where at least one endpoint is a leaf node
        # leaf_edge = None
        # for i, edge in enumerate(F):
        #     if vertex_count[edge[0]] == 1 or vertex_count[edge[1]] == 1:
        #         leaf_edge = F.pop(i)
        #         break
        # # If no leaf node is found, the graph has been processed
        # if leaf_edge is None:
        #     break
        # # 确定叶子节点和另一个端点
        # if vertex_count[leaf_edge[0]] == 1 and vertex_count[leaf_edge[1]] == 1:
        #     if syndrome_dict[leaf_edge[0]] == 2:
        #         u = leaf_edge[1]
        #         v = leaf_edge[0]
        #     else:
        #         u = leaf_edge[0]
        #         v = leaf_edge[1]
        # elif vertex_count[leaf_edge[0]] == 1:
        #     u = leaf_edge[0]
        #     v = leaf_edge[1]
        # else:
        #     u = leaf_edge[1]
        #     v = leaf_edge[0]
        # # 更新度数
        # vertex_count[u] -= 1
        # vertex_count[v] -= 1

        if syndrome_dict[u] == 1:
            A.append(leaf_edge)  # Record superedges that need to be flipped
            syndrome_dict[u] -= 1  # Remove processed syndrome points
            if syndrome_dict[v] == 1:
                syndrome_dict[v] -= 1
            else:
                syndrome_dict[v] += 1

    # Collect all actual edges and weights that need to be flipped from precomputed edge set and weight
    all_actual_edges = []
    weight = 0
    
    for superedge in A:
        # Normalize superedge to ensure precomputed value can be found
        normalized_superedge = normalize_edge(superedge)
        if normalized_superedge in superedge_to_edges:
            all_actual_edges.extend(superedge_to_edges[normalized_superedge])
            weight += superedge_to_weight[normalized_superedge]
        else:
            print(f"Warning: normalized superedge {normalized_superedge} not found in precomputed data!")
    
    # Convert edge list to numpy array (Not included in total latency calculation)
    correction = np.zeros(num_faults, dtype=int)
    for bit in all_actual_edges:
        qubit_idx = edge_to_qubit_index(bit, code_structure, error_type=error_type)
        if qubit_idx != -1 and qubit_idx < num_faults:
            correction[qubit_idx] += 1  # Mark qubits that need to be flipped
            correction[qubit_idx] = correction[qubit_idx] % 2
    # weight = len(A)

    return correction, weight