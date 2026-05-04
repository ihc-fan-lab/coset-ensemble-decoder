import numpy as np
from abc import ABC, abstractmethod

# =========================
# 全局：bank 映射与紧凑地址 LUT
# =========================

# bank 哈希：mod 22，xy 周期、z 非周期
V_M = 22
V_ALPHA, V_BETA, V_GAMMA = 1, 3, 5
V_L_MAX = 16
V_R_MAX = 16

# 预计算每个 bank 的紧凑地址映射：ADDR_LUT[bank, x, y, z] -> 紧凑地址 (或 -1)
V_ADDR_LUT = -np.ones((V_L_MAX, V_L_MAX, V_R_MAX), dtype=np.int32)
_v_bank_counts = np.zeros(V_M, dtype=np.int32)

for x0 in range(V_L_MAX):
    for y0 in range(V_L_MAX):
        for z0 in range(V_R_MAX):
            b = (V_ALPHA * x0 + V_BETA * y0 + V_GAMMA * z0) % V_M
            idx = _v_bank_counts[b]
            V_ADDR_LUT[x0, y0, z0] = idx
            _v_bank_counts[b] += 1

V_MAX_PER_BANK = int(_v_bank_counts.max())  # 188

E_M = 9

E_ADDR_LUT = -np.ones((V_L_MAX, V_L_MAX, V_R_MAX), dtype=np.int32)
_e_bank_counts = np.zeros(E_M, dtype=np.int32)
for x0 in range(V_L_MAX):
    for y0 in range(V_L_MAX):
        for z0 in range(V_R_MAX):
            b = x0 % E_M
            idx = _e_bank_counts[b]
            E_ADDR_LUT[x0, y0, z0] = idx
            _e_bank_counts[b] += 1
E_MAX_PER_BANK = int(_e_bank_counts.max())

try:
    from numba import njit

    @njit(inline='always')
    def bank_hash(xw, yw, zw):
        # 保证传入的是整数
        return int((V_ALPHA * int(xw) + V_BETA * int(yw) + V_GAMMA * int(zw)) % V_M)

    # @njit(cache=True)
    def has_duplicate_banks(bnums, n):
        """bnums: 长度>=n 的int数组；返回是否有重复"""
        seen = np.zeros(V_M, np.uint8)
        for i in range(n):
            b = int(bnums[i])
            if seen[b] != 0:
                return True
            seen[b] = 1
        return False

    # 读：中心 + 6 邻居（x、y 周期，z 越界返回 0）
    # @njit(cache=True)
    def vertex_read_with_neighbors_jit(banks, x, y, z, L, R):
        """x,y 周期 (mod L)；z 越界邻居返回 cid=0（未分配）"""
        results = np.zeros(7, dtype=np.uint32)
        bank_numbers = np.empty(7, dtype=np.int64)

        # wrap x,y；z 直接用；显式转 int
        xw = int(x % L)
        yw = int(y % L)
        zw = int(z)

        # 中心
        bank = bank_hash(xw, yw, zw)
        addr = int(V_ADDR_LUT[xw, yw, zw])
        results[0] = banks[bank, addr]
        bank_numbers[0] = bank

        # 六邻居
        directions = ((1,0,0), (0,1,0),
                      (-1,0,0), (0,-1,0),
                      (0,0,-1), (0,0,1))
        n = 1
        for dx, dy, dz in directions:
            nx = int((xw + dx) % L)
            ny = int((yw + dy) % L)
            nz = int(zw + dz)

            if nz < 0 or nz >= V_R_MAX:
                results[n] = 0
                bank_numbers[n] = 0
            else:
                nbank = bank_hash(nx, ny, nz)
                naddr = int(V_ADDR_LUT[nx, ny, nz])
                results[n] = banks[nbank, naddr]
                bank_numbers[n] = nbank
            n += 1

        return results, n

    # 写：中心点（x、y 周期；z 不处理边界）
    @njit(cache=True)
    def vertex_write_jit(banks, x, y, z, cid_in, L, R):
        xw = int(x % L)
        yw = int(y % L)
        zw = int(z)

        bank = bank_hash(xw, yw, zw)
        addr = int(V_ADDR_LUT[xw, yw, zw])
        banks[bank, addr] = cid_in

    # 写：按 wenable 对 6 邻居写入（x、y 周期；z 越界跳过）
    # @njit(cache=True)
    def vertex_write_with_neighbors_jit(banks, x, y, z, wenable, cid_in, L, R):
        directions = ((1,0,0), (0,1,0),
                      (-1,0,0), (0,-1,0),
                      (0,0,-1), (0,0,1))
        bank_numbers = np.empty(6, dtype=np.int64)

        xw = int(x % L)
        yw = int(y % L)
        zw = int(z)

        n = 0
        for i, d in enumerate(directions):
            dx, dy, dz = d
            if (int(wenable) & (1 << i)) == 0:
                continue

            nx = int((xw + dx) % L)
            ny = int((yw + dy) % L)
            nz = int(zw + dz)

            if nz < 0 or nz >= V_R_MAX:
                continue

            nbank = bank_hash(nx, ny, nz)
            naddr = int(V_ADDR_LUT[nx, ny, nz])
            banks[nbank, naddr] = cid_in
            if n < bank_numbers.shape[0]:
                bank_numbers[n] = nbank
                n += 1

    # ============== 边权重（保持你原有 6 向） ==============
    # @njit(cache=True)
    def edge_read_with_directions_jit(x_banks, y_banks, z_banks, x:int, y:int, z:int, L:int, R:int):
        """边权重读取 - 大JIT函数 (周期性边界)"""
        results = np.zeros(6, dtype=np.uint8)
        nx = int((x - 1) % L)
        ny = int((y - 1) % L)
        nz = int((z - 1) % R)

        bank_x = int(nx % E_M)
        bank_x_plus = int(x % E_M)
        bank_y = int(ny % E_M)
        bank_y_plus = int(y % E_M)
        bank_z = int(nz % 2)
        bank_z_plus = int(z % 2)

        address_x = int(E_ADDR_LUT[nx, y, z])
        address_x_plus = int(E_ADDR_LUT[x, y, z])
        address_y = int(E_ADDR_LUT[ny, x, z])
        address_y_plus = int(E_ADDR_LUT[y, x, z])
        address_z = int(x * 128 + y * 8 + nz//2)
        address_z_plus = int(x * 128 + y * 8 + z//2)



        results[2] = x_banks[bank_x,address_x]
        results[0] = x_banks[bank_x_plus,address_x_plus]
        results[3] = y_banks[bank_y,address_y]
        results[1] = y_banks[bank_y_plus,address_y_plus]
        results[4] = z_banks[bank_z,address_z]
        results[5] = z_banks[bank_z_plus,address_z_plus]

        return results

    # @njit(cache=True)
    def edge_write_with_directions_jit(x_banks, y_banks, z_banks, x:int, y:int, z:int, wenable, L:int, R:int):
        """边权重写入 - 大JIT函数 (周期性边界)"""
        nx = int((x - 1) % L)
        ny = int((y - 1) % L)
        nz = int((z - 1) % R)

        bank_x = int(nx % E_M)
        bank_x_plus = int(x % E_M)
        bank_y = int(ny % E_M)
        bank_y_plus = int(y % E_M)
        bank_z = int(nz % 2)
        bank_z_plus = int(z % 2)

        address_x = int(E_ADDR_LUT[nx, y, z])
        address_x_plus = int(E_ADDR_LUT[x, y, z])
        address_y = int(E_ADDR_LUT[ny, x, z])
        address_y_plus = int(E_ADDR_LUT[y, x, z])
        address_z = int(x * 128 + y * 8 + nz // 2)
        address_z_plus = int(x * 128 + y * 8 + z // 2)

        if x_banks[bank_x,address_x] < 2 and (int(wenable) & (1 << 0)) != 0:
            x_banks[bank_x,address_x] += 1
        if x_banks[bank_x_plus,address_x_plus] < 2 and (int(wenable) & (1 << 1)) != 0:
            x_banks[bank_x_plus,address_x_plus] += 1
        if y_banks[bank_y,address_y] < 2 and (int(wenable) & (1 << 2)) != 0:
            y_banks[bank_y,address_y] += 1
        if y_banks[bank_y_plus,address_y_plus] < 2 and (int(wenable) & (1 << 3)) != 0:
            y_banks[bank_y_plus,address_y_plus] += 1
        if z_banks[bank_z,address_z] < 2 and (int(wenable) & (1 << 4)) != 0:
            z_banks[bank_z,address_z] += 1
        if z_banks[bank_z_plus,address_z_plus] < 2 and (int(wenable) & (1 << 5)) != 0:
            z_banks[bank_z_plus,address_z_plus] += 1


except ImportError as e:
    print(f"Error in Memory Class Generation: {str(e)}")


# =================================================================
# Multi-Bank RAM 硬件仿真功能
# =================================================================

class MultiBankRAM(ABC):
    def __init__(self):
        self.L = 16  # 固定L=16（运行时可传 <=15 用作周期）
        self.R = 16  # 固定R=16
        self._create_data()

    @abstractmethod
    def _create_data(self):
        pass

    def reset(self):
        self.banks.fill(0)


class MB_vertex2cid(MultiBankRAM):
    """
    - 22 个 bank，每个 bank 使用紧凑地址（最大 188）
    - bank: (x + 3*y + 5*z) % 22
    - 地址映射：ADDR_LUT[bank,x,y,z] 给出 bank 内线性地址
    - x,y 周期；z 不 wrap（由上层控制）
    """
    def __init__(self):
        self.NUM_BANKS = V_M                      # 22
        self.ADDRESSES_PER_BANK = V_MAX_PER_BANK  # 188
        super().__init__()

    def _create_data(self):
        self.banks = np.zeros((self.NUM_BANKS, self.ADDRESSES_PER_BANK), dtype=np.uint8)

    def read_with_conflict_check(self, x, y, z, L, R):
        results, _ = vertex_read_with_neighbors_jit(self.banks, x, y, z, L, R)
        return results

    def write_jit(self, x, y, z, cid_in, L, R):
        vertex_write_jit(self.banks, x, y, z, cid_in, L, R)

    def write_with_conflict_check(self, x, y, z, wenable, cid_in, L, R):
        vertex_write_with_neighbors_jit(self.banks, x, y, z, wenable, cid_in, L, R)

    def read_all_values(self):
        all_values = {}
        for x0 in range(self.L):
            for y0 in range(self.L):
                for z0 in range(self.R):
                    b = (V_ALPHA*x0 + V_BETA*y0 + V_GAMMA*z0) % V_M
                    a = int(V_ADDR_LUT[x0, y0, z0])
                    all_values[(x0, y0, z0)] = self.banks[b, a]
        return all_values


# 3D 边权重存储（保持你原来的三组 bank 方案）
class MB_3Dedgeweight(MultiBankRAM):
    """
    MB_3Dedgeweight
    - 3 组方向 bank，每组 2 个子 bank（奇偶分离），按你的原有布局
    """
    def __init__(self):
        self.NUM_EDGE_BANKS = 9
        self.ADDRESSES_PER_BANK = E_MAX_PER_BANK
        super().__init__()

    def _create_data(self):
        self.x_banks = np.zeros((self.NUM_EDGE_BANKS, self.ADDRESSES_PER_BANK), dtype=int)
        self.y_banks = np.zeros((self.NUM_EDGE_BANKS, self.ADDRESSES_PER_BANK), dtype=int)
        self.z_banks = np.zeros((2, self.L * self.L * self.R//2), dtype=int)


    def read_with_conflict_check(self, x, y, z, L, R):
        results = edge_read_with_directions_jit(self.x_banks, self.y_banks, self.z_banks, int(x), int(y), int(z), L, R)
        result_dict = {}
        
        result_dict[((x, y, z), ((x+1)%L, y, z))] = results[0]
        result_dict[((x, y, z), (x, (y+1)%L, z))] = results[1]
        result_dict[((x, y, z), ((x-1)%L, y, z))] = results[2]
        result_dict[((x, y, z), (x, (y-1)%L, z))] = results[3]
        result_dict[((x, y, z), (x, y, (z-1)%(R+1)))] = results[4]
        result_dict[((x, y, z), (x, y, (z+1)%(R+1)))] = results[5]
        return result_dict

    def write_with_conflict_check(self, x, y, z, wenable, L, R):
        edge_write_with_directions_jit(self.x_banks, self.y_banks, self.z_banks, int(x), int(y), int(z), wenable, L, R)
        return None

    def read_all_values(self):
        # 这里保留你的原逻辑（如需调试）
        all_values = {}
        # 可按需实现
        return all_values
