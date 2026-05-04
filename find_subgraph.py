from collections import defaultdict, deque
from config import DECODER_CONFIG
from operation_counter import increment_operation_efficient


def x_first_reconstruct(v1, v2, E,code_structure):
    code_distance = code_structure.L    
    if abs(v1[0] - v2[0]) > code_distance // 2:
        weight_x = code_distance - abs(v1[0] - v2[0])
    else:
        weight_x = abs(v1[0] - v2[0])
    if abs(v1[1] - v2[1]) > code_distance // 2:
        weight_y = code_distance - abs(v1[1] - v2[1])
    else:
        weight_y = abs(v1[1] - v2[1])
    weight_z = abs(v1[2] - v2[2])
    weight = weight_x + weight_y + weight_z
    x_diff = v1[0] - v2[0]
    # code_distance = code_structure.L
    if x_diff < 0:
        v1_x, v2_x = v2, v1
    else:
        v1_x, v2_x = v1, v2
    end_vertex = v2_x
    if code_structure.periodic and x_diff != 0: 
        if abs(x_diff) > code_distance // 2:
            for i in range(int(code_distance - abs(x_diff))):
                start_vertex = ((v2_x[0] - i) % code_distance, v2_x[1], v2_x[2])
                end_vertex = ((start_vertex[0] - 1) % code_distance, start_vertex[1], start_vertex[2])
                E.append((end_vertex, start_vertex))
        else:
            for i in range(int(abs(x_diff))):
                start_vertex = ((v2_x[0] + i) % code_distance, v2_x[1], v2_x[2])
                end_vertex = ((start_vertex[0] + 1) % code_distance, start_vertex[1], start_vertex[2])
                E.append((start_vertex, end_vertex))
    ###add y direction edge
    y_diff = v1_x[1] - end_vertex[1]
    if y_diff < 0:
        v1_y, v2_y = end_vertex, v1_x
    else:
        v1_y, v2_y = v1_x, end_vertex
    if code_structure.periodic and y_diff != 0:
        if abs(y_diff) > code_distance // 2:
            for i in range(int(code_distance - abs(y_diff))):
                start_vertex = (v2_y[0], (v2_y[1] - i) % code_distance, v2_y[2])
                end_vertex = (start_vertex[0], (start_vertex[1] - 1) % code_distance, start_vertex[2])
                E.append((end_vertex, start_vertex))
        else:
            for i in range(int(abs(y_diff))):
                start_vertex = (v2_y[0], (v2_y[1] + i) % code_distance, v2_y[2])
                end_vertex = (start_vertex[0], (start_vertex[1] + 1) % code_distance, start_vertex[2])
                E.append((start_vertex, end_vertex))  

    return weight


def y_first_reconstruct(v1, v2, E, code_structure):
    code_distance = code_structure.L    
    if abs(v1[0] - v2[0]) > code_distance // 2:
        weight_x = code_distance - abs(v1[0] - v2[0])
    else:
        weight_x = abs(v1[0] - v2[0])
    if abs(v1[1] - v2[1]) > code_distance // 2:
        weight_y = code_distance - abs(v1[1] - v2[1])
    else:
        weight_y = abs(v1[1] - v2[1])    
    weight_z = abs(v1[2] - v2[2])
    weight = weight_x + weight_y + weight_z
    y_diff = v1[1] - v2[1]
    # code_distance = code_structure.L
    if y_diff < 0:
        v1_y, v2_y = v2, v1
    else:
        v1_y, v2_y = v1, v2
    end_vertex = v2_y
    if code_structure.periodic and y_diff != 0: 
        if abs(y_diff) > code_distance // 2:
            for i in range(int(code_distance - abs(y_diff))):
                start_vertex = (v2_y[0], (v2_y[1] - i) % code_distance, v2_y[2])
                end_vertex = (start_vertex[0], (start_vertex[1] - 1) % code_distance, start_vertex[2])
                E.append((end_vertex, start_vertex))
        else:
            for i in range(int(abs(y_diff))):
                start_vertex = (v2_y[0], (v2_y[1] + i) % code_distance, v2_y[2])
                end_vertex = (start_vertex[0], (start_vertex[1] + 1) % code_distance, start_vertex[2])
                E.append((start_vertex, end_vertex))
    
    ###add x direction edge
    x_diff = v1_y[0] - end_vertex[0]
    if x_diff < 0:
        v1_x, v2_x = end_vertex, v1_y
    else:
        v1_x, v2_x = v1_y, end_vertex
    if code_structure.periodic and x_diff != 0:
        if abs(x_diff) > code_distance // 2:
            for i in range(int(code_distance - abs(x_diff))):
                start_vertex = ((v2_x[0] - i) % code_distance, v2_x[1], v2_x[2])
                end_vertex = ((start_vertex[0] - 1) % code_distance, start_vertex[1], start_vertex[2])
                E.append((end_vertex, start_vertex))
        else:
            for i in range(int(abs(x_diff))):
                start_vertex = ((v2_x[0] + i) % code_distance, v2_x[1], v2_x[2])
                end_vertex = ((start_vertex[0] + 1) % code_distance, start_vertex[1], start_vertex[2])
                E.append((start_vertex, end_vertex))
    return weight

def superedge_reconstruct(A, code_structure, weight):
    E = []
    import random
    ####superedge reconstruction
    for super_edge_info in A:
        v1, v2 = super_edge_info
        # x_first_reconstruct(v1, v2, E, code_structure)
        # 随机选择x优先或y优先
        if random.choice([True, False]):
            weight += x_first_reconstruct(v1, v2, E, code_structure)
        else:
            weight += y_first_reconstruct(v1, v2, E, code_structure)
                        
    return E, weight

def build_superedge_graph(superedge_info):
    """
    Build a graph from superedge information.
    Returns a dictionary where each edge maps to (direction, distance) information.
    Direction: 0=x, 1=y, 2=z
    Distance: direct value (vertex2 - vertex1), not absolute
    """
    graph = {}  # Store edges as coordinate pairs with direction and distance info
    for root in superedge_info.keys():
        for pairs in superedge_info[root]:
            if DECODER_CONFIG['enable_operation_counting']: # 2 assign + 3add + 3abs + 1sum
                increment_operation_efficient('extra_ops', count = 9, if_ops_count = True)

            v1, v2 = pairs
            x_diff = v2[0] - v1[0]  # Direct difference, not absolute
            y_diff = v2[1] - v1[1]
            z_diff = v2[2] - v1[2]
            
            # Count how many coordinates are different
            diff_count = sum([abs(x_diff) > 0, abs(y_diff) > 0, abs(z_diff) > 0])
            
            if DECODER_CONFIG['enable_operation_counting']: # 2 in if + 2 assign
                increment_operation_efficient('extra_ops', count = 4, if_ops_count = True)
            if diff_count == 1:
                # Case 1: Only one coordinate is different
                # Direct edge between the two vertices
                if abs(x_diff) > 0:
                    direction = 0  # x direction
                    distance = x_diff
                elif abs(y_diff) > 0:
                    direction = 1  # y direction
                    distance = y_diff
                else:  # abs(z_diff) > 0
                    direction = 2  # z direction
                    distance = z_diff
                
                graph[(v1, v2)] = (direction, distance)
                
            elif diff_count == 2:
                # Case 2: Two coordinates are different
                # Add intermediate edges to create a path
                if abs(x_diff) > 0 and abs(y_diff) > 0:
                    # xy different: add (x0,y0,z0) -> (x1,y0,z0) -> (x1,y1,z0)
                    intermediate = (v2[0], v1[1], v1[2])
                    graph[(v1, intermediate)] = (0, x_diff)  # x direction
                    graph[(intermediate, v2)] = (1, y_diff)  # y direction
                    
                elif abs(x_diff) > 0 and abs(z_diff) > 0:
                    # xz different: add (x0,y0,z0) -> (x1,y0,z0) -> (x1,y0,z1)
                    intermediate = (v2[0], v1[1], v1[2])
                    graph[(v1, intermediate)] = (0, x_diff)  # x direction
                    graph[(intermediate, v2)] = (2, z_diff)  # z direction
                    
                elif abs(y_diff) > 0 and abs(z_diff) > 0:
                    # yz different: add (x0,y0,z0) -> (x0,y1,z0) -> (x0,y1,z1)
                    intermediate = (v1[0], v2[1], v1[2])
                    graph[(v1, intermediate)] = (1, y_diff)  # y direction
                    graph[(intermediate, v2)] = (2, z_diff)  # z direction
                    
            elif diff_count == 3:
                # Case 3: All three coordinates are different
                # Create a cuboid with 12 edges connecting the two diagonal vertices
                x0, y0, z0 = v1
                x1, y1, z1 = v2
                
                # Define all 8 vertices of the cuboid
                vertices = [
                    (x0, y0, z0),  # v1
                    (x1, y0, z0),
                    (x0, y1, z0),
                    (x1, y1, z0),
                    (x0, y0, z1),
                    (x1, y0, z1),
                    (x0, y1, z1),
                    (x1, y1, z1)   # v2
                ]
                
                # Add all 12 edges of the cuboid with direction and distance info
                edges_info = [
                    # Bottom face (4 edges)
                    ((x0, y0, z0), (x1, y0, z0), 0, x_diff),  # x direction
                    ((x0, y0, z0), (x0, y1, z0), 1, y_diff),  # y direction
                    ((x1, y0, z0), (x1, y1, z0), 1, y_diff),  # y direction
                    ((x0, y1, z0), (x1, y1, z0), 0, x_diff),  # x direction
                    
                    # Top face (4 edges)
                    ((x0, y0, z1), (x1, y0, z1), 0, x_diff),  # x direction
                    ((x0, y0, z1), (x0, y1, z1), 1, y_diff),  # y direction
                    ((x1, y0, z1), (x1, y1, z1), 1, y_diff),  # y direction
                    ((x0, y1, z1), (x1, y1, z1), 0, x_diff),  # x direction
                    
                    # Vertical edges (4 edges)
                    ((x0, y0, z0), (x0, y0, z1), 2, z_diff),  # z direction
                    ((x1, y0, z0), (x1, y0, z1), 2, z_diff),  # z direction
                    ((x0, y1, z0), (x0, y1, z1), 2, z_diff),  # z direction
                    ((x1, y1, z0), (x1, y1, z1), 2, z_diff)   # z direction
                ]
                
                # Add all edges to the graph
                for v1_edge, v2_edge, direction, distance in edges_info:
                    graph[(v1_edge, v2_edge)] = (direction, distance)
    
    return graph

def build_connected_subgraph(clustered_edges, syndrome):
    """
    Process each cluster independently to find minimal subgraph connecting weight=1 nodes.
    
    Args:
        clustered_edges: List of tuples: ((u, v), cluster_id)
        syndrome: defaultdict(int), only nodes with weight=1 need to be connected
    
    Returns:
        used_edges: list of ((u, v), cluster_id) used to connect syndrome points
        used_syndrome: defaultdict(int), only contains weight=1 nodes that were connected
    """

    # Group edges by cluster
    cluster_to_edges = defaultdict(list)
    for (u, v), cluster in clustered_edges:
        cluster_to_edges[cluster].append((u, v))

    final_edges = []
    final_syndrome = defaultdict(int)

    for cluster_id, edges in cluster_to_edges.items():
        # Extract all nodes in this cluster
        nodes_in_cluster = set()
        for u, v in edges:
            nodes_in_cluster.update([u, v])

        # Build local syndrome: only nodes with weight=1 in this cluster
        local_syndrome = defaultdict(int, {v: syndrome[v] for v in nodes_in_cluster if syndrome[v] == 1})
        terminals = list(local_syndrome.keys())

        if len(terminals) < 2:
            # Nothing to connect or only one node; keep if it exists
            for t in terminals:
                final_syndrome[t] = 1
            continue

        # Build graph
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # All-pairs shortest paths using BFS
        paths = {}
        weights = []
        for i, start in enumerate(terminals):
            dist = {start: 0}
            prev = {}
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in graph[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
            for j, end in enumerate(terminals):
                if j <= i or end not in dist:
                    continue
                weights.append((dist[end], i, j))
                # Reconstruct path
                path = []
                cur = end
                while cur != start:
                    path.append(cur)
                    cur = prev[cur]
                path.append(start)
                path.reverse()
                paths[(i, j)] = path

        # # ORIGINAL KRUSKAL ALGORITHM (COMMENTED OUT - FOR FUTURE USE)
        # # Kruskal's MST on terminal graph
        # parent = list(range(len(terminals)))
        # def find(u):
        #     while parent[u] != u:
        #         parent[u] = parent[parent[u]]
        #         u = parent[u]
        #     return u
        # def union(u, v):
        #     parent[find(u)] = find(v)

        # weights.sort()
        # used_edge_set = set()
        # for w, i, j in weights:
        #     if find(i) != find(j):
        #         union(i, j)
        #         path = paths[(i, j)]
        #         for k in range(len(path) - 1):
        #             edge = tuple(sorted((path[k], path[k + 1])))
        #             used_edge_set.add(edge)

        # Directly use all BFS paths without Kruskal filtering
        used_edge_set = set()
        for w, i, j in weights:
            path = paths[(i, j)]
            for k in range(len(path) - 1):
                edge = tuple(sorted((path[k], path[k + 1])))
                used_edge_set.add(edge)

        # Filter only edges in original edge list
        allowed_edges = {tuple(sorted((u, v))) for u, v in edges}
        final_cluster_edges = [
            ((u, v), cluster_id)
            for (u, v) in used_edge_set
            if (u, v) in allowed_edges or (v, u) in allowed_edges
        ]

        final_edges.extend(final_cluster_edges)
        for u, v in used_edge_set:
            if syndrome[u] == 1:
                final_syndrome[u] = 1
            if syndrome[v] == 1:
                final_syndrome[v] = 1

    return final_edges, final_syndrome
