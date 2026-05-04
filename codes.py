import numpy as np
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
from qecsim.models.planar import PlanarCode
from qecsim.models.rotatedplanar import RotatedPlanarCode

### Toric Code
def repetition_code(n):
    """构造重复码的校验矩阵
    Args:
        n: 码长
    Returns:
        csc_matrix: 校验矩阵
    """
    row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
    data = np.ones(2*n, dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)))


def repetition_chain_stabilisers(L):
    """Construct periodic-boundary repetition-code parity checks.

    The parity-check matrix has shape (L, L), where each row checks
    neighboring bits i and (i+1) % L.
    """
    if L < 2:
        raise ValueError("Repetition code requires L >= 2")
    row_ind, col_ind = zip(*((i, j) for i in range(L) for j in (i, (i + 1) % L)))
    data = np.ones(2 * L, dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)), shape=(L, L), dtype=np.uint8)


def repetition_code_logicals(L):
    """Construct the single repetition-code logical operator."""
    if L < 2:
        raise ValueError("Repetition code requires L >= 2")
    return csc_matrix(np.ones((1, L), dtype=np.uint8))


def repetition_code_extract(L):
    """Return (Hx, Hz, Lx, Lz) compatible with current decoder pipeline.

    We duplicate the same repetition parity checks/logicals on X/Z channels so
    the existing channel-agnostic plumbing can initialize cleanly.
    """
    H = repetition_chain_stabilisers(L)
    logical = repetition_code_logicals(L)
    return H, H.copy(), logical, logical.copy()

def toric_code_x_stabilisers(L):
    """构造toric code的X稳定子矩阵
    Args:
        L: 格子大小
    Returns:
        csc_matrix: X稳定子矩阵
    """
    Hr = repetition_code(L)
    H = hstack(
        [kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)],
        dtype=np.uint8
    )
    H.data = H.data % 2
    H.eliminate_zeros()
    return csc_matrix(H)

def toric_code_z_stabilisers(L):
    """构造toric code的Z稳定子矩阵
    Args:
        L: 格子大小
    Returns:
        csc_matrix: Z稳定子矩阵
    """
    Hr = repetition_code(L)
    Hz = hstack([
        kron(eye(Hr.shape[1], dtype=np.uint8), Hr),
        kron(Hr.T, eye(Hr.shape[0], dtype=np.uint8))
    ], dtype=np.uint8)
    Hz.data %= 2
    Hz.eliminate_zeros()
    return csc_matrix(Hz)



def toric_code_x_logicals(L):
    """构造toric code的X逻辑算符矩阵
    Args:
        L: 格子大小
    Returns:
        csc_matrix: X逻辑算符矩阵
    """
    H1 = csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
    H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
    x_logicals = block_diag([kron(H1, H0), kron(H0, H1)])
    x_logicals.data = x_logicals.data % 2
    x_logicals.eliminate_zeros()
    return csc_matrix(x_logicals)

def toric_code_z_logicals(L):
    """构造toric code的Z逻辑算符矩阵
    Args:
        L: 格子大小
    Returns:
        csc_matrix: Z逻辑算符矩阵
    """
    H1 = csc_matrix(([1], ([0], [0])), shape=(1, L), dtype=np.uint8)
    H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
    z_logicals = block_diag([kron(H0, H1), kron(H1, H0)])
    z_logicals.data %= 2
    z_logicals.eliminate_zeros()
    return csc_matrix(z_logicals)




def planar_code_extract(L):
    """
    从qecsim的PlanarCode中提取Hx, Hz, Lx, Lz矩阵
    
    Args:
        code: qecsim.models.planar.PlanarCode实例
    
    Returns:
        Hx: X稳定子矩阵，形状为(L(L-1), L^2+(L-1)^2)
        Hz: Z稳定子矩阵，形状为(L(L-1), L^2+(L-1)^2)
        Lx: X逻辑算子矩阵，形状为(1, L^2+(L-1)^2)
        Lz: Z逻辑算子矩阵，形状为(1, L^2+(L-1)^2)
    """
    # 创建5x5的planar code
    code = PlanarCode(L, L)
    # 获取稳定子矩阵
    M = code.stabilizers  # shape=(#checks, 2*n_phys)
    n = code.n_k_d[0]    # 

   
    # 提取X稳定子矩阵 (Hx)
    # X稳定子只在前n列有非零元素
    Hx = M[L*(L-1):, :n] #% 2  
    Hz = M[:L*(L-1), n:] #% 2  
    
    # 提取Z稳定子矩阵 (Hz)
    # Z稳定子只在后n列有非零元素
    
    # print(f"Hx shape: {Hx.shape}")
    # print(f"Hz shape: {Hz.shape}")    
    # 提取逻辑算子
    Lx = code.logical_xs[:,:n] # X逻辑算子
    Lz = code.logical_zs[:,n:] # Z逻辑算子
    
    # 转换为稀疏矩阵
    Hx = csc_matrix(Hx)
    Hz = csc_matrix(Hz)
    Lx = csc_matrix(Lx)
    Lz = csc_matrix(Lz)
    
    return Hx, Hz, Lx, Lz



######rotated surface code
def rotated_code_extract(L):
    code = RotatedPlanarCode(L, L)
    M = code.stabilizers
    n = code.n_k_d[0]
    num_rows = M.shape[0]
    Hz = csc_matrix((M[:num_rows//2, n:] % 2).astype(np.uint8))
    Hx =  csc_matrix((M[num_rows//2:, :n] % 2).astype(np.uint8))
    Lz = csc_matrix((code.logical_zs[:,n:] % 2).astype(np.uint8))    
    Lx = csc_matrix((code.logical_xs[:,:n] % 2).astype(np.uint8))

    return Hx, Hz, Lx, Lz



