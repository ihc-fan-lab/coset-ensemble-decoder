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



def find(el, clusters, boundaries, code_structure):
    """Find the root node of an element, using path compression optimization
    Args:
        el: The element to find
        clusters: Dictionary of clusters, each element stores [parent node, size, parity, boundary flag, initial syndrome point set, syndrome point connection mapping]
    Returns:
        el: The root node of the element
    """
    
    # if clusters[el][0] == 0:  # New element, initialize as [el, 1, 0]
    #     clusters[el][0] = el
    #     clusters[el][3] = 0
    #     # boundaries[0] = [el]
    #     return 0
    # while clusters[el][0] != el:  # Path compression: directly point all nodes on the path to the root node
    #     el, clusters[el][0] = clusters[el][0], clusters[clusters[el][0]][0]
    # return el
    if clusters[el][3] == 0:  # New element, initialize as [el, 1, 0]
        clusters[el][0] = el
        clusters[el][3] = 0
        # boundaries[0] = [el]
        return 0
    if clusters[el][3] != 0:
        return clusters[el][3]

def union(x, y, clusters, code_structure):
    """Merge two clusters, merge by size (small cluster merges into large cluster)
    Args:
        x, y: Root nodes of the two clusters to merge
        clusters: Dictionary of clusters, each element stores [parent node, size, parity, boundary flag, initial syndrome point set, syndrome point connection mapping]
    """

    if x == y:  # Same cluster, no need to merge
        return
    if clusters[x][3] == 0 and clusters[y][3] != 0:
        x, y = y, x
    # if clusters[x][1] < clusters[y][1]:  # Ensure x is the large cluster
    #     x, y = y, x

    clusters[y][0] = x  # Set y's parent node to x
    clusters[x][1] += clusters[y][1]  # Update size
    # Use cast to ensure type checker knows this is int type
    parity_x = cast(int, clusters[x][2])
    parity_y = cast(int, clusters[y][2])
    clusters[x][2] = parity_x + parity_y  # Update parity (direct addition, since parity is 0 or 1)
    clusters[y][2] = clusters[x][2]
    clusters[y][3] = clusters[x][3]
    
def group_union(gid_u, gid_v, cid2gid, gid2cids, gid_generator):
    for i in range(len(cid2gid)):
        if cid2gid[i] == gid_u:
            cid2gid[i] = gid_generator
        if cid2gid[i] == gid_v:
            cid2gid[i] = gid_generator
    # cid2gid[gid_u] = gid_generator
    # cid2gid[gid_v] = gid_generator
    # Initialize new gid2cids entry
    gid2cids[gid_generator] = [set(), 0]
    gid2cids[gid_generator][0] = gid2cids[gid_u][0] | gid2cids[gid_v][0] #including clusters
    gid2cids[gid_generator][1] = gid2cids[gid_u][1] + gid2cids[gid_v][1] #parity
    gid2cids[gid_u][1] = 0
    gid2cids[gid_v][1] = 0
    gid_generator += 1
    return gid_generator


    
def get_valid_directions(x, y, z, code_structure, error_type='x'):
    """Get valid growth directions
    Args:
        x, y: Current point coordinates
        code_structure: Code structure object
    Returns:
        list: List of valid growth directions (random order)
    """
    # if code_structure.periodic:
    #     directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # All directions are valid for toric code
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



                  
def grow(grow_gid, boundaries, support, clusters, cid2gid, 
gid2cids, gid_generator, code_structure, cluster_syndrome_connections, cid2root, error_type='x'):
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
    # next_nodes = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    fusion_edges = []
    new_roots = defaultdict(int)  # New odd cluster root nodes
    found_roots = defaultdict(int)  # Processed root nodes
    merge_count = 0
    final_gid = []
    # virtual_stab_add_count = 0
        
    # Growth phase: all clusters grow simultaneously in all directions
    for u in gid2cids[grow_gid][0]: ##u-cid in this group
        for v in boundaries[u]:
            valid_directions = get_valid_directions(v[0], v[1], v[2], code_structure, error_type)
            all_directions_complete = True
            for n in valid_directions:
                if code_structure.periodic:
                    other = ((v[0] + n[0]) % code_structure.L, (v[1] + n[1]) % code_structure.L, (v[2] + n[2]))
                else:
                    other = (v[0] + n[0], v[1] + n[1], v[2] + n[2])
                
                edge = (min(v, other), max(v, other))
                
                if edge not in support:
                    support[edge] = 1
                    all_directions_complete = False
                elif support[edge] == 1:
                    support[edge] = 2
                    all_directions_complete = True
                    fusion_edges.append(edge)
                elif support[edge] == 2:
                    fusion_edges.append(edge)
                    all_directions_complete = True
            if all_directions_complete:
                boundaries[u].remove(v)

    # Merge phase
    while fusion_edges:
        u, v = fusion_edges.pop()
        cid_u = find(u, clusters, boundaries, code_structure)
        cid_v = find(v, clusters, boundaries, code_structure)
        # cid_u = clusters[u_root][3]
        # cid_v = clusters[v_root][3]


        if cid_u == 0 and cid_v != 0: ### Case1. merge a unassigned vertex and a cluster
            boundaries[cid_v].append(u)
            union(v, u, clusters, code_structure)
        elif cid_u != 0 and cid_v == 0: ### Case2. merge a unassigned vertex and a cluster
            boundaries[cid_u].append(v) 
            union(u, v, clusters, code_structure)
        ### Case3. merge a cluster and a cluster (change cid2gid mapping)
        else:
            gid_u = cid2gid.get(cid_u, 0)
            gid_v = cid2gid.get(cid_v, 0)
            if gid_u != gid_v:
                u_root = cid2root[cid_u]
                v_root = cid2root[cid_v]
                cluster_syndrome_connections.append((u_root, v_root))
                merge_count += 1
                old_gid_generator = gid_generator
                gid_generator = group_union(gid_u, gid_v, cid2gid, gid2cids, gid_generator)
                # Check if the merged cluster is an odd cluster
                if gid2cids[old_gid_generator][1] % 2 == 1:
                    final_gid.append(old_gid_generator)
        

    # # Update boundaries: remove boundary points where all edges have fully grown
    # for u in new_roots:
    #     for v in boundaries[u]:
    #         x, y, z = v
    #         valid_directions = get_valid_directions(x, y, z, code_structure, error_type)
    #         all_directions_complete = True
    #         for n in valid_directions:
    #             if code_structure.periodic:
    #                 other = ((v[0] + n[0]) % code_structure.L, (v[1] + n[1]) % code_structure.L, (v[2] + n[2]))
    #             else:
    #                 other = (v[0] + n[0], v[1] + n[1], v[2] + n[2])
                
    #             edge = (min(v, other), max(v, other))
    #             if support.get(edge, 0) in {2, 3}:
    #                 all_directions_complete = True
    #             else:
    #                 all_directions_complete = False
    #                 break
    #         if all_directions_complete:
    #             boundaries[u].remove(v)

    if merge_count == 0:
        final_gid.append(grow_gid)
    
    # final_roots = [x for x in new_roots.keys() if new_roots[x]]
    # print(f"final_roots: {final_roots}")
    return final_gid, gid_generator

def union_find_decoder(syndrome, code_structure, random_seed=None, error_type='x'):
    """Union-Find decoder implementation for list decoding
    Args:
        syndrome: List of syndrome points
        code_structure: Code structure object
        random_seed: Random seed (not used)
    """
    ###Syndrome data transfer cycle
    ###Pipeline cycle counter
    pipeline_cycle = 0
    ###

    # Initialize
    support = defaultdict(int)  # Edge growth states: 0=not grown, 1=half grown, 2=fully grown
    boundaries = defaultdict(list)  # Boundary points for each cluster
    clusters = defaultdict(lambda: [0, 1, 0, 0])  # [parent node, size, parity, cid]
    cid2gid = defaultdict(int)
    gid2cids = defaultdict(lambda: [set(), 0])
    cid2root = defaultdict(int)
    # cluster_roots = []  # Current root nodes of all clusters
    grow_order = []  # Priority queue, controls cluster growth order
    entry_num = 1  # Used to ensure clusters of same size grow in fixed order
    virtual_vertex = set() # New: record virtual syndrome
    root_vertex = set() # New: record root nodes
    cluster_syndrome_connections = []
    # Initialize syndrome points
    for g in syndrome:
        cid2root[entry_num] = g
        clusters[g][0] = g  # Set parent node to self
        clusters[g][2] = 1  # Set parity to 1 (syndrome points are all odd)
        clusters[g][3] = entry_num  # cid
        boundaries[entry_num] = [g]  # Initial boundary is self
        cid2gid[entry_num] = entry_num
        gid2cids[entry_num] = [{entry_num}, 1]  # Initialize gid2cids entry
        heapq.heappush(grow_order, [1, entry_num, entry_num])  # Add to growth queue: [boundary size, cluster number, cluster root node]
        entry_num += 1

    gid_generator = entry_num

    # Main loop: continuously grow and merge clusters

    while grow_order:
        current_entry = heapq.heappop(grow_order)  # Get the smallest cluster entry: [boundary_size, entry_num, gid]
        current_gid = current_entry[2]  # Extract cluster root node ID
        if gid2cids[current_gid][1] % 2 == 0:  # Even clusters don't need to continue growing
            continue

        # print(f"grow_root: {grow_root}")
        # for cid in gid2cids[current_gid][0]:
        new_odd_cluster_roots, gid_generator = grow(current_gid, boundaries, support, clusters, cid2gid, 
            gid2cids, gid_generator, code_structure, cluster_syndrome_connections, cid2root, error_type)
        
        # Add new odd clusters to the growth queue
        if new_odd_cluster_roots:
            for new_gid in new_odd_cluster_roots:
                g_size = 0
                for cid in gid2cids[new_gid][0]:
                    g_size += len(boundaries[cid])
                heapq.heappush(grow_order, [g_size, entry_num, new_gid])
                entry_num += 1



    # Collect fully grown edges and mark cluster information
    erasure = []
    cluster_erasure_info = []  # Store edges and corresponding cluster information

    for el in support.keys():
        if support[el] == 2:
            erasure.append(el)


    
    # Return erasure and cluster information
    return erasure, cluster_erasure_info, root_vertex, virtual_vertex, cluster_syndrome_connections













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

def spanning_tree_dict(graph, visited, tree, vertex, code_structure):
    """Create spanning tree rooted at vertex
    Args:
        graph: Dictionary with vertices as keys and adjacent vertex lists as values
        visited: Dictionary recording vertex visit status
        tree: Tree being constructed
        vertex: Current vertex
    Returns:
        tree: Final spanning tree
        visited: Updated visit status dictionary
    """
    visited[vertex] = 1 ## Visited nodes are 1 (initialize the input vertex to 1)

    # Randomize the order of visiting neighbor nodes
    neighbours = graph[vertex].copy()
    np.random.shuffle(neighbours)

    for neighbour in neighbours:
        if not visited[neighbour]:
            tree.append((vertex, neighbour))
            tree, visited = spanning_tree_dict(graph, visited, tree, neighbour, code_structure)
    return tree, visited

def spanning_forest_dict(graph,code_structure,roots = None, virtuals = None):
    """Create spanning forest of the graph
    Args:
        graph: Dictionary with vertices as keys and adjacent vertex lists as values, or edge list
        list_decoding: Whether to enable list decoding
        used_trees: Set of tree structures already used
    Returns:
        tree: Spanning forest of the graph, as a list of edges
    """
    if type(graph) != dict:
        # Assume input is edge list
        graph = edge_list_to_graph(graph, code_structure)

    # Check if graph is empty
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
    # Only shuffle vertex order when randomization is needed
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
    """Peeling decoder implementation using list decoding
    Args:
        erasure: List of edges that need correction
        syndrome: List of syndrome points
        num_faults: Number of possible errors
        code_structure: Code structure object
        virtual_syndromes_sets: List of virtual syndrome sets
        num_candidates: Number of candidate solutions to generate
    Returns:
        corrections: List of candidate solutions
        weights: List of candidate solution weights
    """
    all_corrections = []
    all_weights = []
    # print(f"Edges: {erasure}")
    
        
    for i in range(num_candidates):
        F = spanning_forest_dict(cluster_superedge_info, code_structure, roots=root_vertex, virtuals =virtual_vertex)
        F.reverse()
        syndrome_dict_copy = syndrome.copy()
        # If virtual_syndromes_sets is not None and has sufficient length, use it
        correction, weight = peeling(code_structure, F, syndrome_dict_copy, num_faults, error_type)
        all_corrections.append(correction)
        all_weights.append(weight)

    
    return all_corrections, all_weights







def peeling(code_structure, F, syndrome_dict, num_faults, error_type='x'):
    A = []  # Store edges that need to be flipped
    vertex_count = defaultdict(int)  # Vertex degrees
    
    
    # Calculate vertex degrees
    for el in F:
        vertex_count[el[0]] += 1
        vertex_count[el[1]] += 1

    # Main loop: process leaf nodes
    while F:
        # increment_operation('peeling_leaf_operations', 1, peeling_list_decoding=code_structure.peeling_list_decoding)
        # Find an edge where at least one endpoint is a leaf node
        leaf_edge = None
        for i, edge in enumerate(F):
            if vertex_count[edge[0]] == 1 or vertex_count[edge[1]] == 1:
                leaf_edge = F.pop(i)
                break
        
        # If no leaf node is found, the graph has been processed
        if leaf_edge is None:
            break
            
        # Determine the leaf node and the other endpoint
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
            
        # Update degrees
        vertex_count[u] -= 1
        vertex_count[v] -= 1

        

        if syndrome_dict[u] == 1:
            A.append(leaf_edge)  # Record edges that need to be flipped
            syndrome_dict[u] -= 1  # Remove processed syndrome points
            if syndrome_dict[v] == 1:
                syndrome_dict[v] -= 1
            else:
                syndrome_dict[v] += 1

    weight = 0
    A_reconstruct, weight = superedge_reconstruct(A, code_structure, weight)
    # print(f"Peeling Correction: {A_reconstruct}")
    # Convert edge list to numpy array
    correction = np.zeros(num_faults, dtype=int)
    # print("Listdecoding Edges",A)
    for bit in A_reconstruct:
        qubit_idx = edge_to_qubit_index(bit, code_structure,error_type=error_type)
        if qubit_idx != -1 and qubit_idx < num_faults:
            correction[qubit_idx] += 1  # Mark qubits that need to be flipped
            correction[qubit_idx] = correction[qubit_idx] % 2

    # Calculate weight of single solution
    # weight = len(A)  # Here simply use the number of edges as weight, you can modify the weight calculation method as needed
    return correction, weight




def uf_hardware(syndrome, code_structure, error_type='x', list_size = 1):
    # Save current random state to avoid random seed contamination
    original_random_state = np.random.get_state()
    
    # Set fixed seed to ensure result consistency
    np.random.seed(42)
    
    erasure, cluster_erasure_info, root_vertex, \
    virtual_vertex, cluster_syndrome_connections = \
        union_find_decoder(syndrome, code_structure, error_type=error_type)

    all_corrections, all_weights = peeling_decoder_optimized(erasure, 
        syndrome, code_structure.num_qubits, code_structure, 
        num_candidates= list_size, error_type=error_type, 
        root_vertex=root_vertex, virtual_vertex=virtual_vertex, 
        cluster_info=cluster_erasure_info, cluster_superedge_info=cluster_syndrome_connections)

    # Restore original random state
    np.random.set_state(original_random_state)
    
    return all_corrections, all_weights













#########################
# Peeling Decoder Section #
#########################

def normalize_edge(edge):
    """标准化edge的坐标对顺序，确保一致性"""
    v1, v2 = edge
    # 使用字典序比较，确保一致的顺序
    if v1 <= v2:
        return (v1, v2)
    else:
        return (v2, v1)

def peeling_decoder_optimized(erasure, 
        syndrome, num_faults, code_structure, 
        num_candidates=3, error_type='x', 
        root_vertex=None, virtual_vertex=None, 
        cluster_info=None, cluster_superedge_info=None):
    """优化的Peeling decoder实现，预先计算superedge对应的实际edge set
    Args:
        erasure: List of edges that need correction
        syndrome: List of syndrome points
        num_faults: Number of possible errors
        code_structure: Code structure object
        virtual_syndromes_sets: List of virtual syndrome sets
        num_candidates: Number of candidate solutions to generate
        root_vertex: Root vertex set
        virtual_vertex: Virtual vertex set
        cluster_info: Cluster information
        cluster_superedge_info: Superedge information for clusters
    Returns:
        corrections: List of candidate solutions
        weights: List of candidate solution weights
    """
    all_corrections = []
    all_weights = []
    
    # precompute edge set and weight of each superedge
    superedge_to_edges = {}
    superedge_to_weight = {}
    
    # 遍历所有可能的superedge，预先计算对应的实际edge set和weight
    if cluster_superedge_info:
        for superedge in cluster_superedge_info:
            normalized_superedge = normalize_edge(superedge)
            if normalized_superedge not in superedge_to_edges:
                # 使用superedge_reconstruct计算单条superedge对应的实际edge set和weight
                temp_edges, temp_weight = superedge_reconstruct([superedge], code_structure, 0)
                superedge_to_edges[normalized_superedge] = temp_edges
                superedge_to_weight[normalized_superedge] = temp_weight
    
    # 主循环：生成多个候选解
    for i in range(num_candidates):
        F = spanning_forest_dict(cluster_superedge_info, code_structure, roots=root_vertex, virtuals=virtual_vertex)
        F.reverse()
        syndrome_dict_copy = syndrome.copy()
        
        # 使用预计算的edge set和weight进行peeling
        correction, weight = peeling_optimized(code_structure, F, syndrome_dict_copy, num_faults, error_type, superedge_to_edges, superedge_to_weight)
        all_corrections.append(correction)
        all_weights.append(weight)
    
    return all_corrections, all_weights

def peeling_optimized(code_structure, F, syndrome_dict, num_faults, error_type='x', superedge_to_edges=None, superedge_to_weight=None):
    """优化的peeling函数，使用预计算的edge set和weight
    Args:
        code_structure: Code structure object
        F: Spanning forest edges
        syndrome_dict: Syndrome dictionary
        num_faults: Number of possible errors
        error_type: Error type
        superedge_to_edges: Pre-computed mapping from superedge to actual edges
        superedge_to_weight: Pre-computed mapping from superedge to weight
    Returns:
        correction: Correction array
        weight: Weight of the correction
    """
    A = []  # Store superedges that need to be flipped
    vertex_count = defaultdict(int)  # Vertex degrees
    
    # Calculate vertex degrees
    for el in F:
        vertex_count[el[0]] += 1
        vertex_count[el[1]] += 1

    # Main loop: process leaf nodes
    while F:
        # Find an edge where at least one endpoint is a leaf node
        leaf_edge = None
        for i, edge in enumerate(F):
            if vertex_count[edge[0]] == 1 or vertex_count[edge[1]] == 1:
                leaf_edge = F.pop(i)
                break
        
        # If no leaf node is found, the graph has been processed
        if leaf_edge is None:
            break
            
        # Determine the leaf node and the other endpoint
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
            
        # Update degrees
        vertex_count[u] -= 1
        vertex_count[v] -= 1

        if syndrome_dict[u] == 1:
            A.append(leaf_edge)  # Record superedges that need to be flipped
            syndrome_dict[u] -= 1  # Remove processed syndrome points
            if syndrome_dict[v] == 1:
                syndrome_dict[v] -= 1
            else:
                syndrome_dict[v] += 1

    # 从预计算的edge set和weight中收集所有需要翻转的实际edges和权重
    all_actual_edges = []
    weight = 0
    
    for superedge in A:
        # 标准化superedge以确保能找到预计算的值
        normalized_superedge = normalize_edge(superedge)
        if normalized_superedge in superedge_to_edges:
            all_actual_edges.extend(superedge_to_edges[normalized_superedge])
            # 直接使用预计算的weight
            weight += superedge_to_weight[normalized_superedge]
        else:
            print(f"Warning: normalized superedge {normalized_superedge} not found in precomputed data!")
    
    # Convert edge list to numpy array
    correction = np.zeros(num_faults, dtype=int)
    
    for bit in all_actual_edges:
        qubit_idx = edge_to_qubit_index(bit, code_structure, error_type=error_type)
        if qubit_idx != -1 and qubit_idx < num_faults:
            correction[qubit_idx] += 1  # Mark qubits that need to be flipped
            correction[qubit_idx] = correction[qubit_idx] % 2

    return correction, weight