from config import DECODER_CONFIG
from operation_counter import increment_operation

def edge_to_qubit_index(edge, code_structure, error_type='x'):
    """将边映射到对应的qubit索引
    Args:
        edge: (stab1, stab2) 稳定子坐标对
        code_structure: 代码结构对象
    Returns:
        qubit_indices: qubit的索引列表（H矩阵中的列索引）
    """
    stab1, stab2 = edge
    indices = -1
    L = code_structure.L


    ##3D extention
    if stab1[2] != stab2[2] and stab1[0] == stab2[0] and stab1[1] == stab2[1]:
        if DECODER_CONFIG['enable_operation_counting']: # 4 in if
            increment_operation('peeling_post_precessing', count = 4, if_ops_count = True,
                                peeling_list_decoding=code_structure.peeling_list_decoding,
                                efficient_decoding=code_structure.efficient_decoding)
        return indices

    if DECODER_CONFIG['enable_operation_counting']: # 6 in if + 5 + 4
        increment_operation('peeling_post_precessing', count = 15, if_ops_count = True,
                            peeling_list_decoding=code_structure.peeling_list_decoding,
                            efficient_decoding=code_structure.efficient_decoding)    
    if code_structure.code_type == 'rotated':
        if error_type == 'z':
            if stab1[0] > stab2[0] and stab1[1] > stab2[1]:
                right_down = stab1
                left_up = stab2
                indices = right_down[0] + (right_down[1] - 1) * L
            elif stab1[0] < stab2[0] and stab1[1] < stab2[1]:
                right_down = stab2
                left_up = stab1
                indices = right_down[0] + (right_down[1] - 1) * L
            elif stab1[0] < stab2[0] and stab1[1] > stab2[1]:
                right_up = stab1
                left_down = stab2
                indices = left_down[0] + left_down[1] * L
            else:
                right_up = stab2
                left_down = stab1
                indices = left_down[0] + left_down[1] * L
        else: # stabilizer_type == 'z'
            if stab1[0] > stab2[0] and stab1[1] > stab2[1]:
                right_down = stab1
                left_up = stab2
                indices = right_down[0] - 1 + right_down[1] * L
            elif stab1[0] < stab2[0] and stab1[1] < stab2[1]:
                right_down = stab2
                left_up = stab1
                indices = right_down[0] - 1 + right_down[1] * L
            elif stab1[0] < stab2[0] and stab1[1] > stab2[1]:
                right_up = stab1
                left_down = stab2
                indices = right_up[0] + right_up[1] * L
            else:
                right_up = stab2
                left_down = stab1
                indices = right_up[0] + right_up[1] * L

    elif code_structure.code_type == 'planar':
        if error_type == 'z':
            if stab1[0] == stab2[0]: # 水平边
                row = stab1[0]
                col1, col2 = stab1[1], stab2[1]
                left_col = min(col1, col2)
                right_col = max(col1, col2)
                indices = L * L + row * (L-1) + right_col
            else: # 垂直边
                col = stab1[1]
                row1, row2 = stab1[0], stab2[0]
                up_row = min(row1, row2)
                low_row = max(row1, row2)
                indices = low_row * L + col
        else: # stabilizer_type == 'z'
            if stab1[0] == stab2[0]: # 水平边
                row = stab1[0]
                col1, col2 = stab1[1], stab2[1]
                left_col = min(col1, col2)
                right_col = max(col1, col2)
                indices =  row * L + right_col
            else: # 垂直边
                col = stab1[1]
                row1, row2 = stab1[0], stab2[0]
                up_row = min(row1, row2)
                low_row = max(row1, row2)
                indices = L * L + up_row * (L-1) + col

    elif code_structure.code_type == 'repetition': ## repetition code
        col1, col2 = stab1[1], stab2[1]
        left_col = min(col1, col2)
        right_col = max(col1, col2)
        # For periodic repetition code, the wrap-around edge (L-1, 0)
        # should map to qubit index 0 consistently.
        if right_col - left_col > 1:
            indices = 0
        else:
            indices = right_col
    else: # toric
        if stab1[0] == stab2[0]:  # 水平边
            row = stab1[0]
            col1, col2 = stab1[1], stab2[1]
            left_col = min(col1, col2)
            right_col = max(col1, col2)
            if right_col - left_col > 1:  # edge case
                bias = 0
            else:
                bias = right_col * L
            indices = row + bias
        else:  # 垂直边
            col = stab1[1]
            row1, row2 = stab1[0], stab2[0]
            up_row = min(row1, row2)
            low_row = max(row1, row2)
            if low_row - up_row > 1:  # edge case
                bias = L - 1
            else:
                bias = up_row
            indices = L * L + col * L + bias
    
    return int(indices)
