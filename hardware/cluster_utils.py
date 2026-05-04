import numpy as np
from numba import njit, types
from numba.typed import List

# @njit(cache=True)
def _tz_u32(x):  # count trailing zeros for uint32, x != 0
    c = 0
    while (x & np.uint32(1)) == np.uint32(0):
        c += 1
        x >>= np.uint32(1)
    return c

# @njit(cache=True)
def sparse_enum(v1_bits: int, v2_bits: int):
    # 强制用固定位宽无符号整型，避免符号扩展/类型提升的坑
    v1 = np.uint32(v1_bits)
    v2_orig = np.uint32(v2_bits)

    out = []

    v1w = v1
    cid_v1 = 0
    cid_v2 = 0
    while v1w != np.uint32(0):
        # 取 v1 的最低 1 位（不使用 -v1）
        low1 = v1w & (~v1w + np.uint32(1))
        i = _tz_u32(low1)
        cid_v1 = np.int32(i)

        v2w = v2_orig
        while v2w != np.uint32(0):
            low2 = v2w & (~v2w + np.uint32(1))
            j = _tz_u32(low2)
            cid_v2 = np.int32(j)
            out.append((np.int32(i), np.int32(j)))
            # 清掉 v2 的最低 1 位
            v2w &= v2w - np.uint32(1)

        # 清掉 v1 的最低 1 位
        v1w &= v1w - np.uint32(1)

    return out, cid_v1, cid_v2

        
def lowest_set_bit_index(x: int) -> int:
    return (x & -x).bit_length() - 1