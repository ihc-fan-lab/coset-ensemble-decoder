from collections import defaultdict
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import random
import time
from config import DECODER_CONFIG, DEBUG_UF_GEOMETRY
from calculate_score import multi_solution
from software.uf_listdecoding import uf_listdecoding
from software.uf_original import uf_original
from software.uf_efficient import uf_efficient
from hardware.QuliD_hardware import uf_hardware
from software.uf_compression import uf_compression
from hardware.perf_tracker import CycleContext
#######################
# UF Decoder auxiliary functions #
#######################

class CodeStructure:
    def __init__(self, H_x, H_z, logicals_x, logicals_z, L, repetitions=1, cluster_list_decoding=False, peeling_list_decoding=False, efficient_decoding=False, hardware_goldenmodel=False):
        """initialize code structure
        Args:
            H_x: X stabilizer check matrix
            H_z: Z stabilizer check matrix
            logicals_x: X logical operator matrix
            logicals_z: Z logical operator matrix
            L: lattice size
        """
        # ensure input parameters are valid
        if not isinstance(L, int) or L <= 0:
            raise ValueError("L must be a positive integer")
            
        # convert input matrices to csr_matrix
        self.H_x = csr_matrix(H_x)
        self.H_z = csr_matrix(H_z)
        self.logicals_x = csr_matrix(logicals_x)
        self.logicals_z = csr_matrix(logicals_z)
        
        # infer code parameters from matrix dimensions
        self.num_stabs_x, self.num_qubits = H_x.shape
        self.num_stabs_z, _ = H_z.shape
        
        # set lattice size
        self.L = L
        self.repetitions = repetitions
        self.cluster_list_decoding = cluster_list_decoding
        self.peeling_list_decoding = peeling_list_decoding
        self.efficient_decoding = efficient_decoding
        self.hardware_goldenmodel = hardware_goldenmodel
        
        # infer code type
        self._infer_code_type()


    def _infer_code_type(self):
        X = self.num_stabs_x
        Z = self.num_stabs_z
        L = self.L

        if self.num_qubits == L:
            self.code_type = 'repetition'
        elif X==L*L and Z==L*L:
            self.code_type = 'toric'
        elif X==L*(L-1) and Z==L*(L-1):
            self.code_type = 'planar'
        else:
            self.code_type = 'rotated'
        # else:
        #     raise ValueError(f"Unrecognized stabilizer counts X={X}, Z={Z} for L={L}")
        self.periodic = (self.code_type in ('toric', 'repetition'))
        # if self.code_type == 'planar':
        #     self.num_qubits = L*L + (L-1)**2
        # elif self.code_type == 'rotated':
        #     self.num_qubits = L*L
        # else:
        #     self.num_qubits = 2*L*L
        

##pre compute syndrome array from syndrome dict
# def syndrome_array2coo(syndrome_shot, code_structure, channel):

#     Hx = code_structure.H_x
#     Hz = code_structure.H_z

#     syndrome_dict = defaultdict(int)
#     # syndrome_dict_z = defaultdict(int)
#     syndrome_array = np.zeros((Hx.shape[0], repetitions+1))
#     # syndrome_array_z = np.zeros((Hz.shape[0], repetitions+1))

#     for j in range(len(detector_coords)):
#         if syndrome_shot[j] == 1:
#             if code_type == 'rotated':
#                 col = detector_coords[j][0]
#                 row = detector_coords[j][1]
#                 time_step = detector_coords[j][2]
#                 if (col + row) % 2 == 1:
#                     syndrome_dict[(row, col - 1, time_step)] = 1
#                 else:
#                     syndrome_dict[(row - 1, col, time_step)] = 1
#             elif code_type == 'toric':
#                 col = detector_coords[j][0] // 2
#                 row = detector_coords[j][1] // 2
#                 time_step = detector_coords[j][2]
#                 if detector_coords[j][1] % 2 == 0:
#                     syndrome_dict[(row, col, time_step)] = 1
#                     syndrome_array[int(col * L + row), int(time_step)] = 1
#                 else:
#                     syndrome_dict[(row, col, time_step)] = 1
#                     syndrome_array[int(col * L + row), int(time_step)] = 1

#     return syndrome_dict, syndrome_array

#########################
# UF Decoder interface   #
#########################



def decode(syndrome_dict, syndrome_array,code_structure, 
           run_branch, ##=0-normal UF, =1-listdecoding UF, =2-efficient UF
           list_size=1,
           actual_logicals=None,
           channel = 'x',
           ABLATION_CONFIG=None,
           random_seed=None):
    try:

        # syndrome_dict, syndrome_array = syndrome_array2coo(syndrome_shot, code_structure, channel)


        # syndrome_array = coo2syndrome(syndrome_dict, code_structure, channel)
        # if DEBUG_UF_GEOMETRY:
        #     print(f"[DEBUG decode] Syndrome_array: {syndrome_array}")
        # create deep copy of syndrome_dict to avoid modifying original data
        syndrome_dict_copy = defaultdict(int)
        for k, v in syndrome_dict.items():
            syndrome_dict_copy[k] = v

        if not any(syndrome_dict_copy.values()):
            # if syndrome is all 0, return all 0 solution
            zero_solution = np.zeros(code_structure.logicals_x.shape[0], dtype=int)
            # create num_candidates all 0 solutions
            zero_solutions = [zero_solution] * list_size
            if run_branch == 1:
                return zero_solutions,zero_solutions, zero_solution, zero_solution, zero_solution, zero_solution, zero_solution
            elif run_branch == 2:
                # 创建全零的性能统计
                zero_performance_stats = CycleContext.create_zero_stats()
                return zero_solutions, zero_solutions, zero_solution, zero_solution, zero_solution, zero_solution, zero_solution, zero_performance_stats
            else:
                return zero_solution


        if run_branch == 1:
            erasure, all_corrections, all_weights = uf_listdecoding(
                syndrome_dict_copy,
                code_structure,
                channel,
                list_size=list_size,
                random_seed=random_seed,
            )
            # print(f"List decoding: {all_corrections}")
        elif run_branch == 2:
            # all_corrections, all_weights = uf_hardware(syndrome_dict_copy, code_structure, channel, list_size=list_size)
            erasure, all_corrections, all_weights, performance_stats = uf_hardware(syndrome_dict_copy, code_structure, channel, list_size=list_size, ABLATION_CONFIG=ABLATION_CONFIG)
            # print(f"Hardware: {all_corrections}")
        elif run_branch == 0:
            all_corrections, all_weights = uf_original(syndrome_dict_copy, code_structure, channel, grow_mode='parallel')
            # print(f"UF: {all_corrections}")

        
        if run_branch in [1,2]:
            # multiple results
            predicted_logicals, weight_candidate, min_vote_logicals, \
            max_vote_logicals, syndrome_candidate, topological_candidate \
            = multi_solution(all_corrections, all_weights, code_structure, syndrome_dict, syndrome_array, actual_logicals)
            
            if run_branch == 2:
                # 硬件模块返回性能统计信息
                return erasure, predicted_logicals, weight_candidate, min_vote_logicals, max_vote_logicals, syndrome_candidate, topological_candidate, performance_stats
            else:
                return erasure, predicted_logicals, weight_candidate, min_vote_logicals, max_vote_logicals, syndrome_candidate, topological_candidate
        else:
            # Ensure all_corrections[0] is not None and is a numpy array
            if all_corrections is not None and len(all_corrections) > 0 and all_corrections[0] is not None:
                 # Determine if logicals_x or logicals_z should be used based on channel
                logicals_to_use = code_structure.logicals_x if channel == 'x' else code_structure.logicals_z
                single_predicted_logicals = (all_corrections[0] @ logicals_to_use.T) % 2
            else: # Fallback if no correction is found or invalid
                logicals_to_use = code_structure.logicals_x if channel == 'x' else code_structure.logicals_z
                single_predicted_logicals = np.zeros(logicals_to_use.shape[0], dtype=int)
            
            # if DEBUG_UF_GEOMETRY:
            #     print(f"[DEBUG decode] Standard path. Predicted logicals: {single_predicted_logicals}")
            #     print(f"[DEBUG decode] Standard path. Actual logicals: {actual_logicals}")
            

            
            return single_predicted_logicals
        
    except Exception as e:
        print(f"Error in decode: {str(e)}")
        # Optionally, if debugging, re-raise to get a full traceback
        if DEBUG_UF_GEOMETRY:
             import traceback
             print(traceback.format_exc())
        raise









