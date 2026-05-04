import numpy as np
from collections import defaultdict

from config import DEBUG_UF_GEOMETRY

##Syndrome preprocessing
def process_syndrome(syndrome_array, code_structure, error_channel='x'):
    """将2D syndrome数组转换为defaultdict格式的syndrome
    Args:
        syndrome_array: 2D数组，第一维是稳定子位置，第二维是不同round的测量结果
        code_structure: 代码结构对象，包含格子大小和类型信息
    Returns:
        syndrome_dict: defaultdict(int)，键为坐标，值为1表示有syndrome
    """
    if DEBUG_UF_GEOMETRY:
        print(f"[DEBUG process_syndrome] Input syndrome_array shape: {syndrome_array.shape}, "
              f"L: {code_structure.L}, code_type: {code_structure.code_type}, error_type: {error_channel}, syndrome: {syndrome_array}")

    syndrome_dict = defaultdict(int)
    
    for round_idx in range(syndrome_array.shape[1]):
        # 遍历稳定子位置（先列后行）
        for stab_idx in range(syndrome_array.shape[0]):
            # 计算稳定子的坐标
            row, col = -1, -1 # Initialize for safety
            if code_structure.code_type == 'toric':
                col = stab_idx // code_structure.L  # 先列
                row = stab_idx % code_structure.L   # 后行
            elif code_structure.code_type == 'rotated':
                if error_channel == 'z': # X stabilizers for Z error
                    # For Z errors, X stabilizers are used.
                    # Their layout might be on a grid related to L for rotated codes.
                    # Example: (L*L-1)/2 X stabilizers.
                    # The mapping (2*stab_idx)//(L-1) etc. was from your commented code.
                    # This part needs to be accurate for your specific rotated code construction.
                    # Placeholder for potentially complex mapping:
                    # For rotated L=3, num_stabs_x = (3*3-1)/2 = 4.
                    # A common mapping for X stabilizers (faces for Z errors):
                    # (0,0) (0,2) (2,0) (2,2) on a 2L-1 x 2L-1 effective grid or similar for visualization
                    # Or map to a L x L grid of faces.
                    # Using your previous commented logic for rotated X stabs (error_type 'z' for Z-error means X-stabs)
                    col = (2*stab_idx) // (code_structure.L - 1)
                    row = (2*stab_idx) % (code_structure.L - 1) + (col+1)%2

                else: # Z stabilizers for X error (error_type 'x')
                    # For X errors, Z stabilizers are used.
                    # Their layout might be on a grid related to L.
                    # Using your previous commented logic for rotated Z stabs (error_type 'x' for X-error means Z-stabs)
                    col = (2*stab_idx) // (code_structure.L + 1) # This was for error_type 'x' (Z stabs)
                    row = (2*stab_idx) % (code_structure.L + 1) + col%2


            elif code_structure.code_type == 'planar':  # planar
                if error_channel == 'z': # X stabilizers for Z error, size = L x (L-1)
                    # num_cols_X_stabs_planar = code_structure.L # if L*(L-1) stabs, L-1 rows, L cols
                    # Assuming Hx for planar is (L*(L-1)) x N, Z stabilizers are (L*(L-1)) x N
                    # If X stabs for Z errors: num_stabs_x = L*(L-1)
                    # stab_idx for X-stabs:
                    # num_cols_for_Xstabs = code_structure.L # Based on your Hx from planar_code_extract giving L*(L-1) stabs
                                                        # e.g. L rows of (L-1) stabs, or L-1 rows of L stabs.
                                                        # codes.py: planar_code_extract -> Hx is L(L-1) x N
                                                        # Hz is (L-1)L x N
                    # If error_type is 'z', we are looking at X stabilizers.
                    # Their count is L*(L-1). Let's assume they form a grid of (L-1) rows and L columns.
                    num_stabilizers_per_row = code_structure.L
                    row = stab_idx // num_stabilizers_per_row
                    col = stab_idx % num_stabilizers_per_row

                else: # Z stabilizers for X error, size = (L-1) x L
                    # If error_type is 'x', we are looking at Z stabilizers.
                    # Their count is (L-1)*L. Let's assume they form a grid of L rows and (L-1) columns.
                    num_stabilizers_per_row = code_structure.L - 1
                    row = stab_idx // num_stabilizers_per_row
                    col = stab_idx % num_stabilizers_per_row
            else: ### repetition code
                if error_channel == 'x':
                    row = 0
                    col = stab_idx
                else:
                    row = 0
                    col = stab_idx
            # syndrome_sum = 0
            # for round_idx in range(syndrome_array.shape[1]):
            #     syndrome_sum += syndrome_array[stab_idx, round_idx]
                        
            if syndrome_array[stab_idx, round_idx] % 2 == 1:
                syndrome_dict[(row, col, round_idx)] = 1
                if DEBUG_UF_GEOMETRY:
                    print(f"[DEBUG process_syndrome] Activated: round_idx={round_idx}, stab_idx={stab_idx} -> coord=({row},{col})")   

                # if code_structure.code_type == 'toric':
                #     syndrome_dict[(row, col, round_idx)] = 1
                # elif code_structure.code_type == 'rotated':
                #     #if 0 <= row < code_structure.L-1 and 0 <= col < code_structure.L-1:
                #     syndrome_dict[(row, col, round_idx)] = 1
                # else:  # planar
                #     #if 0 <= row < code_structure.L and 0 <= col < code_structure.L:
                #     syndrome_dict[(row, col, round_idx)] = 1
 

    if DEBUG_UF_GEOMETRY and not syndrome_dict:
        print(f"[DEBUG process_syndrome] No syndromes activated.")
    elif DEBUG_UF_GEOMETRY:
        print(f"[DEBUG process_syndrome] Final syndrome_dict (first 5 items): {dict(list(syndrome_dict.items())[:5])}")
    return syndrome_dict



def coo2syndrome(coo_array, code_structure, error_type='x'):
    """将2D syndrome数组转换为defaultdict格式的syndrome
    Args:
        coo_array: defaultdict格式的syndrome，键为坐标元组，值为1表示有syndrome
        code_structure: 代码结构对象，包含格子大小和类型信息
        error_type: 错误类型 ('x' 或 'z')
    Returns:
        syndrome_array: numpy数组，表示syndrome
    """
    if DEBUG_UF_GEOMETRY:
        print(f"[DEBUG coo2syndrome] Input coo_array: {coo_array}, "
              f"L: {code_structure.L}, code_type: {code_structure.code_type}, error_type: {error_type}")

    if error_type == 'x':
        syndrome_array = np.zeros((code_structure.num_stabs_x,code_structure.repetitions+1), dtype=int)
    else:
        syndrome_array = np.zeros((code_structure.num_stabs_z,code_structure.repetitions+1), dtype=int)

    for coord, value in coo_array.items():
        if value == 1:
            # 确保坐标值是整数
            row, col, round_idx = int(coord[0]), int(coord[1]), int(coord[2])
            
            if DEBUG_UF_GEOMETRY:
                print(f"[DEBUG coo2syndrome] Processing coord: ({row}, {col}, {round_idx}), types: {type(row)}, {type(col)}, {type(round_idx)}")
            
            if code_structure.code_type == 'toric':
                stab_idx = col * code_structure.L + row
                if DEBUG_UF_GEOMETRY:
                    print(f"[DEBUG coo2syndrome] Toric: stab_idx = {col} * {code_structure.L} + {row} = {stab_idx}")
                syndrome_array[stab_idx, round_idx] = 1
            elif code_structure.code_type == 'rotated':
                if error_type == 'z': # X stabilizers for Z error
                    stab_idx = col * ((code_structure.L - 1)//2) + row // 2
                    if DEBUG_UF_GEOMETRY:
                        print(f"[DEBUG coo2syndrome] Rotated Z: stab_idx = {col} * {((code_structure.L - 1)//2)} + {row // 2} = {stab_idx}")
                    syndrome_array[stab_idx, round_idx] = 1
                else: # Z stabilizers for X error (error_type 'x')
                    stab_idx = col * ((code_structure.L - 1)//2 + 1) + row // 2
                    if DEBUG_UF_GEOMETRY:
                        print(f"[DEBUG coo2syndrome] Rotated X: stab_idx = {col} * {((code_structure.L - 1)//2 + 1)} + {row // 2} = {stab_idx}")
                    if stab_idx >= code_structure.num_stabs_x:
                        print(f"[DEBUG coo2syndrome] Rotated X: stab_idx = {stab_idx} >= {code_structure.num_stabs_x}")
                    syndrome_array[stab_idx, round_idx] = 1
            elif code_structure.code_type == 'repetition':
                # Repetition code uses 1D stabilizer indexing along col dimension.
                syndrome_array[col, round_idx] = 1
            # else:  # planar
            #     if error_type == 'z': # X stabilizers for Z error, size = L x (L-1)
            #         num_stabilizers_per_row = code_structure.L
            #         row = stab_idx // num_stabilizers_per_row
            #         col = stab_idx % num_stabilizers_per_row
            #     else: # Z stabilizers for X error, size = (L-1) x L
            #         num_stabilizers_per_row = code_structure.L - 1
            #         row = stab_idx // num_stabilizers_per_row
            #         col = stab_idx % num_stabilizers_per_row

    return syndrome_array