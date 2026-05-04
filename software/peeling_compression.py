# -*- coding: utf-8 -*-
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional
import hashlib
import random
import math
import numpy as np
from tools.post_precessing import edge_to_qubit_index
# ============ 你的环境中应已存在的函数 ============
# - peeling(code_structure, F, syndrome_dict, num_faults, error_type='x')
# - edge_to_qubit_index(...) 由 peeling 调用
# 我这里不重新定义，直接调用。

Vertex = Tuple[Any, ...]         # 顶点坐标元组，如 (4.0, 0.0, 3.0)
Edge    = Tuple[Vertex, Vertex]  # 无向边（坐标对）

# ----------------------------
# 基础工具
# ----------------------------
def canon_edge(u: Vertex, v: Vertex) -> Edge:
    """无向边规范化：字典序小者在前"""
    return (u, v) if u <= v else (v, u)

def manhattan(a: Vertex, b: Vertex) -> int:
    """曼哈顿距离（坐标维度任意；float 视为数值）"""
    return int(sum(abs(float(x) - float(y)) for x, y in zip(a, b)))

def stable_hash_bits(u: Vertex, v: Vertex, seed: int, bits: int) -> int:
    """稳定低位随机：与运行无关（不用 Python 内置 hash）"""
    u_, v_ = canon_edge(u, v)
    payload = f"{u_}|{v_}|{seed}".encode("utf-8")
    h = hashlib.blake2s(payload, digest_size=8).digest()  # 64-bit
    val = int.from_bytes(h, "little")
    mask = (1 << bits) - 1
    return val & mask

# ----------------------------
# 并查集
# ----------------------------
class DSU:
    __slots__ = ("parent", "rank", "comp_count")
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n
        self.comp_count = n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.comp_count -= 1
        return True

# ----------------------------
# Insight A：共享 peeling（保存“公共边”，缩小到含环核心）
# ----------------------------
def graph_peeling_common(erasure_edges: Sequence[Edge]) -> Tuple[List[Edge], List[Edge]]:
    """
    对整张图做叶剥离：
      返回 (common_edges, residual_edges)
      - common_edges：所有生成树都必然包含的边（桥/叶链）
      - residual_edges：剥离后剩下的边（含环核心）
    """
    # 规范化 & 建邻接
    graph = defaultdict(set)
    for u, v in erasure_edges:
        u, v = canon_edge(u, v)
        graph[u].add(v)
        graph[v].add(u)

    deg = {v: len(ns) for v, ns in graph.items()}
    q = deque([v for v, d in deg.items() if d == 1])
    common_edges: List[Edge] = []

    while q:
        u = q.popleft()
        if deg.get(u, 0) != 1:
            continue
        # 找唯一邻居
        nbrs = [w for w in graph[u] if deg.get(w, 0) > 0]
        if not nbrs:
            continue
        v = nbrs[0]
        # 记录公共边
        common_edges.append(canon_edge(u, v))
        # “删除” u，并更新 v 的度
        deg[u]  = 0
        deg[v]  = deg.get(v, 0) - 1
        if deg[v] == 1:
            q.append(v)

    # residual：度数>0 的边才保留
    residual_edges: List[Edge] = []
    seen = set()
    for u, ns in graph.items():
        if deg.get(u, 0) == 0:
            continue
        for v in ns:
            if deg.get(v, 0) == 0:
                continue
            e = canon_edge(u, v)
            if e not in seen:
                residual_edges.append(e)
                seen.add(e)
    return common_edges, residual_edges

# ----------------------------
# Insight B + C：多分支 Borůvka（一次边流、K 套权重）
# ----------------------------
@dataclass
class BranchCfg:
    root: Vertex
    seed: int
    b: int = 20     # 同层随机低位宽度
    alpha: int = 1  # near 的比例（建议 2^k；=1 也行）

def boruvka_multi_roots(
    residual_edges: Sequence[Edge],
    branches: Sequence[BranchCfg],
) -> List[List[Edge]]:
    """
    对“剩余含环部分”做 K 分支 Borůvka：
      - 每个分支一套权重：W = (alpha * near)<<b | hash
      - 产出 K 份森林（若连通则是树）
    """
    # 顶点编号
    vertices: List[Vertex] = []
    vid: Dict[Vertex, int] = {}
    def get_vid(p: Vertex) -> int:
        if p in vid: return vid[p]
        i = len(vertices)
        vertices.append(p)
        vid[p] = i
        return i

    edges_idx: List[Tuple[int, int, Vertex, Vertex]] = []
    seen = set()
    for u, v in residual_edges:
        u, v = canon_edge(u, v)
        if (u, v) in seen:  # 去重
            continue
        seen.add((u, v))
        iu, iv = get_vid(u), get_vid(v)
        edges_idx.append((iu, iv, u, v))

    n = len(vertices)
    k = len(branches)
    dsus   = [DSU(n) for _ in range(k)]
    forests: List[List[Edge]] = [[] for _ in range(k)]
    done   = [False] * k

    if n == 0 or not edges_idx:
        return forests

    # 多轮
    while True:
        minEdge: List[Dict[int, Tuple[int, Tuple[int, int]]]] = [dict() for _ in range(k)]
        progressed = False

        # 一次遍历，服务 K 分支
        for iu, iv, u, v in edges_idx:
            for j, br in enumerate(branches):
                if done[j]:
                    continue
                dsu = dsus[j]
                ru, rv = dsu.find(iu), dsu.find(iv)
                if ru == rv:
                    continue
                near = min(manhattan(u, br.root), manhattan(v, br.root))
                W = ((br.alpha * near) << br.b) | stable_hash_bits(u, v, br.seed, br.b)
                # 更新两侧分量候选
                cur = minEdge[j].get(ru)
                if (cur is None) or (W < cur[0]):
                    minEdge[j][ru] = (W, (iu, iv))
                cur = minEdge[j].get(rv)
                if (cur is None) or (W < cur[0]):
                    minEdge[j][rv] = (W, (iu, iv))

        # 轮末批量合并
        for j in range(k):
            if done[j]:
                continue
            dsu = dsus[j]
            picked = 0
            for _comp, (_, (a, b)) in minEdge[j].items():
                if dsu.union(a, b):
                    forests[j].append(canon_edge(vertices[a], vertices[b]))
                    picked += 1
            if picked == 0:
                done[j] = True
            else:
                progressed = True

        if all(done) or not progressed:
            break

    return forests

# ----------------------------
# 构造分支配置（从 root_vertex 参数）
# ----------------------------
def build_branches_from_root_vertex(
    root_vertex: Optional[Sequence[Vertex]],
    candidate_vertices: Sequence[Vertex],
    seeds: Optional[Sequence[int]] = None,
    k_default: int = 1,
    b: int = 20,
    alpha: int = 1,
) -> List[BranchCfg]:
    """
    root_vertex: None / 单个 root / root 列表
    若 None：从 candidate_vertices 随机取 k_default 个
    """
    rnd = random.Random(20250831)
    # 归一化 roots
    if root_vertex is None:
        # 默认取一个根；你也可以把 k_default 设大些来一次生成多棵
        roots = [candidate_vertices[0]] if candidate_vertices else []
    else:
        if isinstance(root_vertex, tuple):
            roots = [root_vertex]
        else:
            roots = list(root_vertex)

    if seeds is None:
        seeds = [rnd.getrandbits(64) for _ in roots]
    assert len(seeds) == len(roots)
    return [BranchCfg(root=r, seed=s, b=b, alpha=alpha) for r, s in zip(roots, seeds)]



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











# =========================================================
# 顶层：替换版 peeling_decoder（保持你的接口）
# =========================================================
def peeling_decoder(erasure, syndrome, num_faults, code_structure, error_type='x',
                    root_vertex=None, virtual_vertex=None, global_time_info=None,
                    boruvka_b: int = 20, boruvka_alpha: int = 1,
                    enable_shared_peeling: bool = True):
    """
    顶层保持你的接口：
      - 直接使用 erasure（坐标对边集合）
      - 先做 Insight A（共享 peeling）得到 common_edges 与 residual_edges
      - 对 residual_edges 做 Insight B/C（多分支 Borůvka），每个 root 产一棵树
      - 把 common_edges 拼回每棵树 → 作为 F 喂给你的 peeling()
      - 返回每棵树对应的 (correction, weight)
    """
    all_corrections: List[np.ndarray] = []
    all_weights: List[int] = []

    # --- Insight A：共享 peeling ---
    if enable_shared_peeling:
        common_edges, residual_edges = graph_peeling_common(erasure)
    else:
        common_edges, residual_edges = [], list({canon_edge(*e) if len(e)==2 else canon_edge(e[0], e[1]) for e in erasure})

    # 候选顶点：从 erasure 中提取
    vertices = []
    seen_v = set()
    for u, v in erasure:
        if u not in seen_v:
            vertices.append(u); seen_v.add(u)
        if v not in seen_v:
            vertices.append(v); seen_v.add(v)

    root_vertex = [v for v in syndrome.keys() if syndrome[v] == 1]
    # 构造分支（一个 root 一棵树；root_vertex 可为 None/单个/列表）
    branches = build_branches_from_root_vertex(
        root_vertex=root_vertex,
        candidate_vertices=vertices,
        seeds=None,
        k_default=1,
        b=boruvka_b,
        alpha=boruvka_alpha,
    )

    # --- Insight B + C：多分支 Borůvka（在含环核心上）---
    forests_residual = boruvka_multi_roots(residual_edges, branches)

    # 每棵树 = common_edges + residual_forest
    forests = []
    for j, fr in enumerate(forests_residual):
        F = list(common_edges) + list(fr)
        # 你的 Stage 3：peeling（注意：F 是边列表）
        F_rev = list(reversed(F))  # 与你原版保持一致（peeling 前 reverse）
        syndrome_dict_copy = syndrome.copy()
        correction, weight = peeling(code_structure, F_rev, syndrome_dict_copy, num_faults, error_type)
        all_corrections.append(correction)
        all_weights.append(weight)
        forests.append(F)

    return all_corrections, all_weights


