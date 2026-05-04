# from ldpc import bposd_decoder
import numpy as np
import matplotlib.pyplot as plt
# from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
from scipy.sparse import hstack, eye, block_diag, csr_matrix
from pymatching import Matching
# from data_collector import ClusterDataCollector
from config import DECODER_CONFIG, DEBUG_UF_GEOMETRY
import uf_decoder
import os
from codes import *
from ldpc.bposd_decoder import BpOsdDecoder
import time
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from functools import partial
# import stim
from operation_counter import reset_counter, get_counter, create_counter
import gc
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
try:
    import stimcircuits
except ImportError:
    stimcircuits = None
# from panqec.decoders import BeliefPropagationOSDDecoder
# from operation_counter import (
#     create_counter, get_counter, reset_counter, 
#     increment_operation, print_all_summaries,
#     get_global_counter, reset_global_counter
# )
# Global debug flag, enabled when UF decoder encounters errors
# _enable_detailed_debug = False
from tools.experiment_noise import (
    batchify_noise_generation as _batchify_noise_generation,
    build_syndrome_from_detector_shot,
    generate_random_error_and_syndrome as _generate_random_error_and_syndrome,
    generate_random_error_and_syndrome_simple as _generate_random_error_and_syndrome_simple,
    get_noise_generator as _get_noise_generator,
    simple_batchify as _simple_batchify,
)
from tools import experiment_metrics as _experiment_metrics
from tools import experiment_plotting as _experiment_plotting
from collections import defaultdict
import signal
import tempfile

# ===== Pure-function task support for loky backend with per-process lazy cache =====
_WORKER_LAZY_CTX = {}
batchify_noise_generation = _batchify_noise_generation
generate_random_error_and_syndrome = _generate_random_error_and_syndrome
simple_batchify = _simple_batchify
generate_random_error_and_syndrome_simple = _generate_random_error_and_syndrome_simple
get_noise_generator = _get_noise_generator

def _build_multiround_pcm(pcm, repetitions, format="csr"):
    """Build space-time PCM so BPOSD can decode multiround syndrome."""
    if not isinstance(pcm, csr_matrix):
        pcm = csr_matrix(pcm)

    pcm_rows, _ = pcm.shape
    # repetitions here means the code repetitions; syndrome has repetitions+1 rounds.
    H_3DPCM = block_diag([pcm] * (repetitions + 1), format=format)
    H_3DID_diag = block_diag([eye(pcm_rows, format=format)] * (repetitions + 1), format=format)
    H_3DID_offdiag = eye(pcm_rows * (repetitions + 1), k=-pcm_rows, format=format)
    H_3DID = H_3DID_diag + H_3DID_offdiag
    return hstack([H_3DPCM, H_3DID], format=format)


def _build_unweighted_matching_from_dem(dem):
    """Construct an unweighted matching graph from a detector error model."""
    weighted = Matching.from_detector_error_model(dem)
    unweighted = Matching()
    for node_a, node_b, data in weighted.edges():
        fault_ids = set(data.get('fault_ids', set()))
        if node_a is None or node_b is None:
            boundary_node = node_b if node_a is None else node_a
            unweighted.add_boundary_edge(
                boundary_node,
                fault_ids=fault_ids,
                weight=1.0,
                error_probability=0.5,
            )
        else:
            unweighted.add_edge(
                node_a,
                node_b,
                fault_ids=fault_ids,
                weight=1.0,
                error_probability=0.5,
            )
    return unweighted


def _is_mwpm_prediction_error(prediction, actual, noise_source):
    """Return True when MWPM prediction mismatches actual observable."""
    if noise_source == 'stim':
        if DECODER_CONFIG.get('stim_compare_full_observables', False):
            return not np.array_equal(np.asarray(prediction), np.asarray(actual))
        return not np.array_equal([prediction[0]], actual)
    return not np.array_equal(np.asarray(prediction), np.asarray(actual))


def _normalize_code_type(code_type):
    mapping = {
        'toric': 'toric',
        'toric_code': 'toric',
        'surface_code': 'surface_code',
        'planar': 'surface_code',
        'rotated_surface_code': 'rotated_surface_code',
        'rotated': 'rotated_surface_code',
        'repetition': 'repetition_code',
        'repetition_code': 'repetition_code',
    }
    if code_type not in mapping:
        raise ValueError(f"Unsupported code_type: {code_type}")
    return mapping[code_type]


def _build_code_matrices(code_type, L):
    normalized = _normalize_code_type(code_type)
    if normalized == 'toric':
        Hx = toric_code_x_stabilisers(L)
        logX = toric_code_x_logicals(L)
        Hz = toric_code_z_stabilisers(L)
        logZ = toric_code_z_logicals(L)
    elif normalized == 'surface_code':
        Hx, Hz, logX, logZ = planar_code_extract(L)
    elif normalized == 'rotated_surface_code':
        Hx, Hz, logX, logZ = rotated_code_extract(L)
    elif normalized == 'repetition_code':
        Hx, Hz, logX, logZ = repetition_code_extract(L)
    else:
        raise ValueError(f"Unsupported code_type: {code_type}")
    return normalized, Hx, Hz, logX, logZ

def _get_or_build_ctx(code_type, L, p, channel, repetitions):
    """Build and cache heavy decoder context in the worker process.

    Key: (code_type, L, p, channel, repetitions)
    Returns dict with matching, bp_osd_decoder, and code structures.
    """
    normalized_code_type = _normalize_code_type(code_type)
    noise_source = DECODER_CONFIG.get('noise_source', 'custom')
    mwpm_backend = DECODER_CONFIG.get('mwpm_backend', 'dem_unweighted')
    mwpm_report_dual = bool(DECODER_CONFIG.get('mwpm_report_dual', True))
    mwpm_report_dem_weighted = bool(DECODER_CONFIG.get('mwpm_report_dem_weighted', True))
    bposd_backend = DECODER_CONFIG.get('bposd_backend', 'dem_graph')
    bposd_report_dual = bool(DECODER_CONFIG.get('bposd_report_dual', True))
    key = (
        normalized_code_type,
        int(L),
        float(p),
        channel,
        int(repetitions),
        noise_source,
        mwpm_backend,
        mwpm_report_dual,
        mwpm_report_dem_weighted,
        bposd_backend,
        bposd_report_dual,
    )
    if key in _WORKER_LAZY_CTX:
        return _WORKER_LAZY_CTX[key]

    normalized_code_type, Hx, Hz, logX, logZ = _build_code_matrices(normalized_code_type, L)

    q_eff = p
    weight_value = np.log((1 - p) / p) if p > 0 else 1
    timelike_weight = np.log((1 - q_eff) / q_eff) if q_eff > 0 else 1

    if channel == 'x':
        if noise_source == 'stim':
            matching_hx_manual = Matching(
                Hx,
                weights=1.0,
                repetitions=repetitions + 1,
                timelike_weights=1.0,
                faults_matrix=logX
            )
        else:
            matching_hx_manual = Matching(
                Hx,
                weights=weight_value,
                repetitions=repetitions+1,
                timelike_weights=timelike_weight,
                faults_matrix=logX
            )
        if normalized_code_type == 'toric':
            code_name = "toric_code:unrotated_memory_x"
        elif normalized_code_type == 'surface_code':
            code_name = "surface_code:unrotated_memory_x"
        elif normalized_code_type == 'rotated_surface_code':
            code_name = "surface_code:rotated_memory_x"
        elif normalized_code_type == 'repetition_code':
            code_name = None
        else:
            raise ValueError(f"Unsupported code_type: {code_type}")
    else:
        if noise_source == 'stim':
            matching_hx_manual = Matching(
                Hz,
                weights=1.0,
                repetitions=repetitions + 1,
                timelike_weights=1.0,
                faults_matrix=logZ
            )
        else:
            matching_hx_manual = Matching(
                Hz,
                weights=weight_value,
                repetitions=repetitions+1,
                timelike_weights=timelike_weight,
                faults_matrix=logZ
            )
        if normalized_code_type == 'toric':
            code_name = "toric_code:unrotated_memory_z"
        elif normalized_code_type == 'surface_code':
            code_name = "surface_code:unrotated_memory_z"
        elif normalized_code_type == 'rotated_surface_code':
            code_name = "surface_code:rotated_memory_z"
        elif normalized_code_type == 'repetition_code':
            code_name = None
        else:
            raise ValueError(f"Unsupported code_type: {code_type}")

    # For stim multiround decoding, build BPOSD directly from detector error model matrices.
    bp_osd_decoder = None
    bposd_observables_matrix = None
    bposd_decoder_dem_graph = None
    bposd_decoder_shared_graph = None
    matching_dem_unweighted = None
    matching_dem_weighted = None
    if code_name is not None and stimcircuits is not None:
        circuit = stimcircuits.generate_circuit(
            code_name,
            distance=L,
            rounds=repetitions,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            after_reset_flip_probability=0,
            before_measure_flip_probability=p,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        dem_mats = detector_error_model_to_check_matrices(dem)
        if noise_source == 'stim' and (mwpm_backend == 'dem_unweighted' or mwpm_report_dual):
            matching_dem_unweighted = _build_unweighted_matching_from_dem(dem)
        if noise_source == 'stim' and (mwpm_backend == 'dem_weighted' or mwpm_report_dem_weighted):
            matching_dem_weighted = Matching.from_detector_error_model(dem)
        bposd_observables_matrix = dem_mats.observables_matrix
        if DECODER_CONFIG.get('use_dem_priors_for_bposd', False):
            bposd_decoder_dem_graph = BpOsdDecoder(
                dem_mats.check_matrix,
                error_channel=list(dem_mats.priors),
                bp_method='product_sum',
                max_iter=7,
                schedule='serial',
                osd_method='osd_cs',
                osd_order=3,
            )
        else:
            bposd_decoder_dem_graph = BpOsdDecoder(
                dem_mats.check_matrix,
                error_rate=float(p),
                bp_method='product_sum',
                max_iter=7,
                schedule='serial',
                osd_method='osd_cs',
                osd_order=3,
            )

    if noise_source == 'stim' and (bposd_backend == 'shared_graph' or bposd_report_dual):
        pcm = Hx if channel == 'x' else Hz
        H3D = _build_multiround_pcm(pcm, repetitions)
        bposd_decoder_shared_graph = BpOsdDecoder(
            H3D,
            error_rate=float(p),
            bp_method='product_sum',
            max_iter=7,
            schedule='serial',
            osd_method='osd_cs',
            osd_order=3,
        )

    if noise_source == 'stim':
        if bposd_backend == 'shared_graph' and bposd_decoder_shared_graph is not None:
            bp_osd_decoder = bposd_decoder_shared_graph
        else:
            bp_osd_decoder = bposd_decoder_dem_graph

    code_structure_uf = uf_decoder.CodeStructure(
        H_x=Hx, H_z=Hz, logicals_x=logX, logicals_z=logZ, L=L, repetitions=repetitions,
        cluster_list_decoding=False, peeling_list_decoding=False
    )
    # code_structure_peeling = uf_decoder.CodeStructure(
    #     H_x=Hx, H_z=Hz, logicals_x=logX, logicals_z=logZ, L=L, repetitions=repetitions,
    #     cluster_list_decoding=False, peeling_list_decoding=True
    # )
    # code_structure_efficient = uf_decoder.CodeStructure(
    #     H_x=Hx, H_z=Hz, logicals_x=logX, logicals_z=logZ, L=L, repetitions=repetitions,
    #     cluster_list_decoding=False, peeling_list_decoding=False, efficient_decoding=True
    # )

    if noise_source == 'stim' and mwpm_backend == 'dem_unweighted' and matching_dem_unweighted is not None:
        matching = matching_dem_unweighted
    elif noise_source == 'stim' and mwpm_backend == 'dem_weighted' and matching_dem_weighted is not None:
        matching = matching_dem_weighted
    else:
        matching = matching_hx_manual

    ctx = dict(
        matching=matching,
        matching_hx_manual_unweighted=matching_hx_manual if noise_source == 'stim' else None,
        matching_dem_unweighted=matching_dem_unweighted,
        matching_dem_weighted=matching_dem_weighted,
        bp_osd_decoder=bp_osd_decoder,
        bposd_observables_matrix=bposd_observables_matrix,
        bposd_decoder_dem_graph=bposd_decoder_dem_graph,
        bposd_decoder_shared_graph=bposd_decoder_shared_graph,
        code_structure_uf=code_structure_uf
    )
    _WORKER_LAZY_CTX[key] = ctx
    return ctx

def _simulate_single_shot_core(list_size, channel,
                               syndrome_dict_x, syndrome_dict_z,
                               syndrome_array_x, syndrome_array_z,
                               actual_x, actual_z,
                               matching, bp_osd_decoder,
                               code_structure_uf,
                               stim_detector_shot=None,
                               bposd_observables_matrix=None,
                               matching_hx_manual_unweighted=None,
                               matching_dem_unweighted=None,
                               matching_dem_weighted=None,
                               bposd_decoder_dem_graph=None,
                               bposd_decoder_shared_graph=None,
                               shot_id=None):
    """Functional core adapted from simulate_single_shot for channel 'x'."""
    performance_stats = None
    error_flags = {
        'mwpm': False,
        'uf': False,
        'uf_peel_list': False,
        'uf_peel_minweight': False,
        'uf_peel_votemax': False,
        'uf_peel_syndrome': False,
        'uf_peel_efficient_list': False,
        'uf_peel_efficient_minweight': False,
        'uf_peel_efficient_votemax': False,
        'uf_peel_efficient_syndrome': False,
        'bposd': False,
        'uf_ablation_baseline_votemax': False,
        'uf_ablation_mbuffer_only_votemax': False,
        'uf_ablation_dsuopt_only_votemax': False,
        'uf_ablation_graphcompression_votemax': False,
        'uf_ablation_growskipping_votemax': False,
        'mwpm_dem_unweighted': False,
        'mwpm_dem_weighted': False,
        'mwpm_hx_manual_unweighted': False,
        'mwpm_disagree': False,
        'mwpm_disagree_hx_vs_dem_weighted': False,
        'mwpm_disagree_dem_unweighted_vs_dem_weighted': False,
        'bposd_dem_graph': False,
        'bposd_shared_graph': False,
        'bposd_disagree': False,
    }

    if channel in ['x', 'both']:
        if DECODER_CONFIG['use_mwpm']:
            if DECODER_CONFIG['noise_source'] == 'stim':
                mwpm_backend = DECODER_CONFIG.get('mwpm_backend', 'dem_unweighted')
                report_dual = bool(DECODER_CONFIG.get('mwpm_report_dual', True))

                pred_dem = None
                pred_dem_weighted = None
                pred_hx = None
                if matching_dem_unweighted is not None and stim_detector_shot is not None:
                    pred_dem = matching_dem_unweighted.decode(stim_detector_shot)
                    error_flags['mwpm_dem_unweighted'] = _is_mwpm_prediction_error(
                        pred_dem, actual_x, noise_source='stim'
                    )
                if matching_dem_weighted is not None and stim_detector_shot is not None:
                    pred_dem_weighted = matching_dem_weighted.decode(stim_detector_shot)
                    error_flags['mwpm_dem_weighted'] = _is_mwpm_prediction_error(
                        pred_dem_weighted, actual_x, noise_source='stim'
                    )
                if matching_hx_manual_unweighted is not None:
                    pred_hx = matching_hx_manual_unweighted.decode(syndrome_array_x)
                    error_flags['mwpm_hx_manual_unweighted'] = _is_mwpm_prediction_error(
                        pred_hx, actual_x, noise_source='stim'
                    )

                if mwpm_backend == 'dem_unweighted' and pred_dem is not None:
                    mwpm_pred_logX = pred_dem
                elif mwpm_backend == 'dem_weighted' and pred_dem_weighted is not None:
                    mwpm_pred_logX = pred_dem_weighted
                elif mwpm_backend == 'hx_manual_unweighted' and pred_hx is not None:
                    mwpm_pred_logX = pred_hx
                else:
                    mwpm_pred_logX = matching.decode(syndrome_array_x)

                error_flags['mwpm'] = _is_mwpm_prediction_error(
                    mwpm_pred_logX, actual_x, noise_source='stim'
                )
                if report_dual and pred_dem is not None and pred_hx is not None:
                    error_flags['mwpm_disagree'] = not np.array_equal(
                        np.asarray(pred_dem), np.asarray(pred_hx)
                    )
                if report_dual and pred_dem_weighted is not None and pred_hx is not None:
                    error_flags['mwpm_disagree_hx_vs_dem_weighted'] = not np.array_equal(
                        np.asarray(pred_dem_weighted), np.asarray(pred_hx)
                    )
                if report_dual and pred_dem_weighted is not None and pred_dem is not None:
                    error_flags['mwpm_disagree_dem_unweighted_vs_dem_weighted'] = not np.array_equal(
                        np.asarray(pred_dem), np.asarray(pred_dem_weighted)
                    )
            else:
                mwpm_pred_logX = matching.decode(syndrome_array_x)
                error_flags['mwpm'] = _is_mwpm_prediction_error(
                    mwpm_pred_logX, actual_x, noise_source='custom'
                )

        if DECODER_CONFIG['use_uf']:
            # if DECODER_CONFIG['enable_timing']:
            #     start_time = time.time()
            uf_pred_logX = uf_decoder.decode(
                syndrome_dict_x, syndrome_array_x, code_structure_uf,
                run_branch=0,
                list_size=list_size,
                actual_logicals=actual_x,
                channel='x'
            )

            if DECODER_CONFIG['noise_source'] == 'stim':
                error_flags['uf'] = not np.array_equal([uf_pred_logX[0]], actual_x)
            else:
                error_flags['uf'] = not np.array_equal(uf_pred_logX, actual_x)

        if DECODER_CONFIG['use_peel_listdecoding']:
            corrections_uf,uf_peel_x, uf_peel_minweight_x, uf_peel_votemin_x, uf_peel_votemax_x, \
            uf_peel_syndromemin_x, uf_peel_topological_x = uf_decoder.decode(
                syndrome_dict_x, syndrome_array_x, code_structure_uf,
                run_branch=1,
                list_size=list_size,
                actual_logicals=actual_x,
                channel='x',
                random_seed=shot_id,
            )
            # if DECODER_CONFIG['enable_timing']:
            #     error_flags['uf_peel_time'] = time.time() - start_time
            if DECODER_CONFIG['noise_source'] == 'stim':
                if not np.array_equal([uf_peel_minweight_x[0]], actual_x):
                    error_flags['uf_peel_minweight'] = True
                if not np.array_equal([uf_peel_votemax_x[0]], actual_x):
                    error_flags['uf_peel_votemax'] = True
                if not np.array_equal([uf_peel_syndromemin_x[0]], actual_x):
                    error_flags['uf_peel_syndrome'] = True
                all_incorrect_peeling = all(not np.array_equal([pred[0]], actual_x) for pred in uf_peel_x)
                if all_incorrect_peeling:
                    error_flags['uf_peel_list'] = True
            else:
                if not np.array_equal(uf_peel_minweight_x, actual_x):
                    error_flags['uf_peel_minweight'] = True
                if not np.array_equal(uf_peel_votemax_x, actual_x):
                    error_flags['uf_peel_votemax'] = True
                if not np.array_equal(uf_peel_syndromemin_x, actual_x):
                    error_flags['uf_peel_syndrome'] = True
                all_incorrect_peeling = all(not np.array_equal(pred, actual_x) for pred in uf_peel_x)
                if all_incorrect_peeling:
                    error_flags['uf_peel_list'] = True


        if DECODER_CONFIG['use_peel_efficient']:
            ABLATION_CONFIG = {
                'if_graph_compression': True,
                'if_grow_skipping': False,
                'if_no_dsu_opt': False,
                'if_no_mb_bufffer': False,
            }
            corrections_uf_eff, uf_eff_x, uf_eff_min_x, uf_eff_vmin_x, uf_eff_vmax_x, \
            uf_eff_syn_x, uf_eff_topo_x, performance_stats = uf_decoder.decode(
                syndrome_dict_x, syndrome_array_x, code_structure_uf,
                run_branch=2,
                list_size=list_size,
                actual_logicals=actual_x,
                channel='x',
                ABLATION_CONFIG=ABLATION_CONFIG
            )
            if DECODER_CONFIG['noise_source'] == 'stim':
                if not np.array_equal([uf_eff_min_x[0]], actual_x):
                    error_flags['uf_peel_efficient_minweight'] = True
                if not np.array_equal([uf_eff_vmax_x[0]], actual_x):
                    error_flags['uf_peel_efficient_votemax'] = True
                if not np.array_equal([uf_eff_syn_x[0]], actual_x):
                    error_flags['uf_peel_efficient_syndrome'] = True
                all_incorrect_eff = all(not np.array_equal([pred[0]], actual_x) for pred in uf_eff_x)
                if all_incorrect_eff:
                    error_flags['uf_peel_efficient_list'] = True
            else:
                if not np.array_equal(uf_eff_min_x, actual_x):
                    error_flags['uf_peel_efficient_minweight'] = True
                if not np.array_equal(uf_eff_vmax_x, actual_x):
                    error_flags['uf_peel_efficient_votemax'] = True
                if not np.array_equal(uf_eff_syn_x, actual_x):
                    error_flags['uf_peel_efficient_syndrome'] = True
                all_incorrect_eff = all(not np.array_equal(pred, actual_x) for pred in uf_eff_x)
                if all_incorrect_eff:
                    error_flags['uf_peel_efficient_list'] = True


        
        ablation_performance_stats = {}
        if performance_stats is not None:
            ablation_performance_stats['peel_efficient'] = performance_stats
        if DECODER_CONFIG['use_ablation_baseline']:
            ABLATION_CONFIG = {
                'if_graph_compression': False,
                'if_grow_skipping': False,
                'if_no_dsu_opt': True,
                'if_no_mb_bufffer': True,
            }
            corrections_uf_eff, uf_eff_x, uf_eff_min_x, uf_eff_vmin_x, uf_eff_vmax_x, \
            uf_eff_syn_x, uf_eff_topo_x, perf_stats = uf_decoder.decode(
                syndrome_dict_x, syndrome_array_x, code_structure_uf,
                run_branch=2,
                list_size=list_size,
                actual_logicals=actual_x,
                channel='x',
                ABLATION_CONFIG=ABLATION_CONFIG
            )
            ablation_performance_stats['baseline'] = perf_stats
            if performance_stats is None:
                performance_stats = perf_stats
            if DECODER_CONFIG['noise_source'] == 'stim':
                if not np.array_equal([uf_eff_vmax_x[0]], actual_x):
                    error_flags['uf_ablation_baseline_votemax'] = True

        if DECODER_CONFIG['use_ablation_mbuffer_only']:
            ABLATION_CONFIG = {
                'if_graph_compression': False,
                'if_grow_skipping': False,
                'if_no_dsu_opt': True,
                'if_no_mb_bufffer': False,
            }
            corrections_uf_eff, uf_eff_x, uf_eff_min_x, uf_eff_vmin_x, uf_eff_vmax_x, \
            uf_eff_syn_x, uf_eff_topo_x, perf_stats = uf_decoder.decode(
                syndrome_dict_x, syndrome_array_x, code_structure_uf,
                run_branch=2,
                list_size=list_size,
                actual_logicals=actual_x,
                channel='x',
                ABLATION_CONFIG=ABLATION_CONFIG
            )
            ablation_performance_stats['mbuffer_only'] = perf_stats
            if performance_stats is None:
                performance_stats = perf_stats
            if DECODER_CONFIG['noise_source'] == 'stim':
                if not np.array_equal([uf_eff_vmax_x[0]], actual_x):
                    error_flags['uf_ablation_mbuffer_only_votemax'] = True

        if DECODER_CONFIG['use_ablation_dsuopt_only']:
            ABLATION_CONFIG = {
                'if_graph_compression': False,
                'if_grow_skipping': False,
                'if_no_dsu_opt': False,
                'if_no_mb_bufffer': True,
            }
            corrections_uf_eff, uf_eff_x, uf_eff_min_x, uf_eff_vmin_x, uf_eff_vmax_x, \
            uf_eff_syn_x, uf_eff_topo_x, perf_stats = uf_decoder.decode(
                syndrome_dict_x, syndrome_array_x, code_structure_uf,
                run_branch=2,
                list_size=list_size,
                actual_logicals=actual_x,
                channel='x',
                ABLATION_CONFIG=ABLATION_CONFIG
            )
            ablation_performance_stats['dsuopt_only'] = perf_stats
            if performance_stats is None:
                performance_stats = perf_stats
            if DECODER_CONFIG['noise_source'] == 'stim':
                if not np.array_equal([uf_eff_vmax_x[0]], actual_x):
                    error_flags['uf_ablation_dsuopt_only_votemax'] = True

        if DECODER_CONFIG['use_ablation_graphcompression']:
            ABLATION_CONFIG = {
                'if_graph_compression': True,
                'if_grow_skipping': False,
                'if_no_dsu_opt': True,
                'if_no_mb_bufffer': True,
            }
            corrections_uf_eff, uf_eff_x, uf_eff_min_x, uf_eff_vmin_x, uf_eff_vmax_x, \
            uf_eff_syn_x, uf_eff_topo_x, perf_stats = uf_decoder.decode(
                syndrome_dict_x, syndrome_array_x, code_structure_uf,
                run_branch=2,
                list_size=list_size,
                actual_logicals=actual_x,
                channel='x',
                ABLATION_CONFIG=ABLATION_CONFIG
            )
            ablation_performance_stats['graphcompression'] = perf_stats
            if performance_stats is None:
                performance_stats = perf_stats
            if DECODER_CONFIG['noise_source'] == 'stim':
                if not np.array_equal([uf_eff_vmax_x[0]], actual_x):
                    error_flags['uf_ablation_graphcompression_votemax'] = True

        if DECODER_CONFIG['use_ablation_growskipping']:
            ABLATION_CONFIG = {
                'if_graph_compression': False,
                'if_grow_skipping': True,
                'if_no_dsu_opt': True,
                'if_no_mb_bufffer': True,
            }
            corrections_uf_eff, uf_eff_x, uf_eff_min_x, uf_eff_vmin_x, uf_eff_vmax_x, \
            uf_eff_syn_x, uf_eff_topo_x, perf_stats = uf_decoder.decode(
                syndrome_dict_x, syndrome_array_x, code_structure_uf,
                run_branch=2,
                list_size=list_size,
                actual_logicals=actual_x,
                channel='x',
                ABLATION_CONFIG=ABLATION_CONFIG
            )
            ablation_performance_stats['growskipping'] = perf_stats
            if performance_stats is None:
                performance_stats = perf_stats
            if DECODER_CONFIG['noise_source'] == 'stim':
                if not np.array_equal([uf_eff_vmax_x[0]], actual_x):
                    error_flags['uf_ablation_growskipping_votemax'] = True

        # 汇总各 ablation 的性能统计，嵌入到返回的 performance_stats 中，便于后续画图
        if ablation_performance_stats:
            if performance_stats is not None and isinstance(performance_stats, dict):
                performance_stats['ablation_variants'] = ablation_performance_stats
            else:
                performance_stats = {'ablation_variants': ablation_performance_stats}

        # 忽略顺序进行比较；坐标不重复时，用 set 更高效
        # def _to_immutable(x):
        #     if isinstance(x, np.ndarray):
        #         return tuple(x.tolist())
        #     if isinstance(x, (list, tuple)):
        #         return tuple(_to_immutable(e) for e in x)
        #     return x

        # left_set = set(_to_immutable(e) for e in list(corrections_uf))
        # right_set = set(_to_immutable(e) for e in list(corrections_uf_eff))
        # if left_set != right_set:
        #     print(f"Cluster generation different")
        #     print(f"Syndrome: {syndrome_dict_x}")
        #     print(f"UF Peel Efficient Corrections: {corrections_uf_eff}")
        #     print(f"UF Corrections: {corrections_uf}")
        # else:
        #     if all_incorrect_eff and not all_incorrect_peeling:
        #         print(f"UF Peel Efficient Correct, UF Peel List Incorrect")
        #         print(f"Syndrome: {syndrome_dict_x}")
        #         print(f"UF HW: {corrections_uf_eff}")
        #         print(f"UF SW: {corrections_uf}")
        #     else:
        #     # print(f"Cluster generation same")
        #         pass
        if DECODER_CONFIG['use_bposd']:
            if bp_osd_decoder is None:
                raise ValueError("BPOSD is not initialized for this code/noise configuration.")
            if DECODER_CONFIG['noise_source'] == 'stim':
                bposd_backend = DECODER_CONFIG.get('bposd_backend', 'dem_graph')
                bposd_report_dual = bool(DECODER_CONFIG.get('bposd_report_dual', True))

                pred_dem = None
                pred_shared = None
                if (
                    bposd_decoder_dem_graph is not None
                    and stim_detector_shot is not None
                    and bposd_observables_matrix is not None
                ):
                    syndrome_bposd_dem = np.asarray(stim_detector_shot, dtype=np.uint8)
                    decoding_dem = bposd_decoder_dem_graph.decode(syndrome_bposd_dem)
                    pred_dem = (bposd_observables_matrix @ decoding_dem) % 2
                    if DECODER_CONFIG.get('stim_compare_full_observables', False):
                        error_flags['bposd_dem_graph'] = not np.array_equal(
                            np.asarray(pred_dem, dtype=np.int8),
                            np.asarray(actual_x, dtype=np.int8),
                        )
                    else:
                        error_flags['bposd_dem_graph'] = not np.array_equal(
                            np.asarray([pred_dem[0]], dtype=np.int8),
                            np.asarray(actual_x, dtype=np.int8),
                        )

                if bposd_decoder_shared_graph is not None:
                    syndrome_bposd_shared = np.asarray(syndrome_array_x, dtype=np.int8).flatten("F")
                    decoding_shared = bposd_decoder_shared_graph.decode(syndrome_bposd_shared)
                    num_data_qubits = code_structure_uf.H_x.shape[1]
                    num_rounds = int(code_structure_uf.repetitions) + 1
                    space_correction = decoding_shared[: num_data_qubits * num_rounds].reshape((num_rounds, num_data_qubits)).T
                    final_data_correction = (np.cumsum(space_correction, axis=1) % 2)[:, -1]
                    pred_shared = code_structure_uf.logicals_x @ final_data_correction % 2
                    if DECODER_CONFIG.get('stim_compare_full_observables', False):
                        error_flags['bposd_shared_graph'] = not np.array_equal(
                            np.asarray(pred_shared, dtype=np.int8),
                            np.asarray(actual_x, dtype=np.int8),
                        )
                    else:
                        error_flags['bposd_shared_graph'] = not np.array_equal(
                            np.asarray([pred_shared[0]], dtype=np.int8),
                            np.asarray(actual_x, dtype=np.int8),
                        )

                if bposd_backend == 'shared_graph' and pred_shared is not None:
                    error_flags['bposd'] = error_flags['bposd_shared_graph']
                elif bposd_backend == 'dem_graph' and pred_dem is not None:
                    error_flags['bposd'] = error_flags['bposd_dem_graph']
                elif pred_dem is not None:
                    error_flags['bposd'] = error_flags['bposd_dem_graph']
                elif pred_shared is not None:
                    error_flags['bposd'] = error_flags['bposd_shared_graph']
                else:
                    raise ValueError("No valid BPOSD prediction is available in stim mode.")

                if bposd_report_dual and pred_dem is not None and pred_shared is not None:
                    error_flags['bposd_disagree'] = not np.array_equal(
                        np.asarray(pred_dem, dtype=np.int8),
                        np.asarray(pred_shared, dtype=np.int8),
                    )
            else:
                # BPOSD expects a 1D syndrome whose length equals the decoder PCM row count.
                syndrome_bposd = np.asarray(syndrome_array_x, dtype=np.int8).flatten("F")
                decoding = bp_osd_decoder.decode(syndrome_bposd)

                # Decode output contains space vars first, then time vars; only space vars affect logicals.
                num_data_qubits = code_structure_uf.H_x.shape[1]
                num_rounds = int(code_structure_uf.repetitions) + 1
                space_correction = decoding[: num_data_qubits * num_rounds].reshape((num_rounds, num_data_qubits)).T
                final_data_correction = (np.cumsum(space_correction, axis=1) % 2)[:, -1]
                bposd_pred_logX = code_structure_uf.logicals_x @ final_data_correction % 2
                error_flags['bposd'] = not np.array_equal(bposd_pred_logX, actual_x)

        # if not error_flags['uf_peel_efficient_votemax'] and error_flags['uf_peel_votemax']:
        #     print(f"UF Peel Efficient Votemax False, UF True")
        #     uf_peel_x, uf_peel_minweight_x, uf_peel_votemin_x, uf_peel_votemax_x, \
        #     uf_peel_syndromemin_x, uf_peel_topological_x = uf_decoder.decode(
        #         syndrome_dict_x, syndrome_array_x, code_structure_uf,
        #         run_branch=1,
        #         list_size=list_size,
        #         actual_logicals=actual_x,
        #         channel='x',
        #     )
        #     a, b, c, d, e, f,performance_stats = uf_decoder.decode(
        #     syndrome_dict_x, syndrome_array_x, code_structure_uf,
        #     run_branch=2,
        #     list_size=list_size,
        #     actual_logicals=actual_x,
        #     channel='x',
        #     ABLATION_CONFIG=ABLATION_CONFIG
        #     )

    # 返回error_flags和performance_stats
    if 'performance_stats' in locals():
        return error_flags, performance_stats
    else:
        return error_flags, None

def _process_single_shot_task(shot_id,
                              code_type, L, p, repetitions, channel, list_size,
                              detector_coords,
                              syndrome_shot, actual_observables_shot):
    """Top-level task for loky: build ctx lazily and simulate one shot."""
    import sys
    if sys.getrecursionlimit() < 10000:
        sys.setrecursionlimit(10000)
    ctx = _get_or_build_ctx(code_type, L, p, channel, repetitions)

    np.random.seed(42 + int(shot_id))

    Hx = ctx['code_structure_uf'].H_x
    Hz = ctx['code_structure_uf'].H_z

    actual_x = actual_observables_shot
    actual_z = actual_observables_shot

    syndrome_dict_x, syndrome_dict_z, syndrome_array_x, syndrome_array_z = build_syndrome_from_detector_shot(
        detector_coords=detector_coords,
        syndrome_shot=syndrome_shot,
        code_type=code_type,
        L=L,
        num_x_stabs=Hx.shape[0],
        num_z_stabs=Hz.shape[0],
        repetitions=repetitions,
    )

    error_flags, performance_stats = _simulate_single_shot_core(
        list_size, channel,
        syndrome_dict_x, syndrome_dict_z,
        syndrome_array_x, syndrome_array_z,
        actual_x, actual_z,
        ctx['matching'], ctx['bp_osd_decoder'],
        ctx['code_structure_uf'],
        syndrome_shot,
        ctx.get('bposd_observables_matrix'),
        ctx.get('matching_hx_manual_unweighted'),
        ctx.get('matching_dem_unweighted'),
        ctx.get('matching_dem_weighted'),
        ctx.get('bposd_decoder_dem_graph'),
        ctx.get('bposd_decoder_shared_graph'),
        shot_id,
    )
    
    return error_flags, performance_stats
class UFTester:
    def __init__(self, save_dir='cluster_data'):
        """Initialize tester
        Args:
            save_dir: Root directory for data storage
        """
        self.save_dir = save_dir
        # self.data_collector = ClusterDataCollector(save_dir=save_dir)
    
    # def _enable_debug_if_needed(self):
    #     """Enable DEBUG_UF_GEOMETRY if global debug flag is set"""
    #     global _enable_detailed_debug
    #     if _enable_detailed_debug:
    #         import config
    #         if not config.DEBUG_3D_CODE:
    #             print(f"[DEBUG] Enabling DEBUG_3D_CODE debug mode")
    #             config.DEBUG_3D_CODE = True
    #             # Re-import DEBUG_UF_GEOMETRY to related modules
    #             import importlib
    #             import uf_decoder
    #             importlib.reload(uf_decoder)
    #             print(f"[DEBUG] Debug mode enabled, subsequent decoder calls will show detailed debug information")
    
    # def simulate_single_shot(self, list_size, channel, 
    #                          syndrome_dict_x, syndrome_dict_z, syndrome_array_x, syndrome_array_z, 
    #                          actual_x, actual_z, matching, bp_osd_decoder, code_structure_uf):
    #     """Simulate single experiment
    #     Args:
    #         L: Grid size
    #         p: Error probability
    #         num_candidates_cluster: Number of cluster candidate solutions
    #         num_candidates_peeling: Number of peeling candidate solutions
    #         channel: Channel type ('x', 'z', 'both')
    #         noisy_syndrome_x: Noisy syndrome for X channel
    #         noisy_syndrome_z: Noisy syndrome for Z channel
    #         noise_total_x: Total noise for X channel (actual X error)
    #         noise_total_z: Total noise for Z channel (actual Z error)
    #         matching_x: Matching decoder for X channel
    #         matching_z: Matching decoder for Z channel
    #         bp_osd_x: BP-OSD decoder for X channel
    #         bp_osd_z: BP-OSD decoder for Z channel
    #         logX: Logical operator for X channel
    #         logZ: Logical operator for Z channel
    #         code_structure_uf: Code structure object used by UF decoder
    #         code_structure_cluster: Code structure object used by cluster decoder
    #         code_structure_peeling: Code structure object used by peeling decoder

    #     Returns:
    #         dict: Dictionary containing results from various decoders
    #     """

        
    #     error_flags = {
    #         'mwpm': False,
    #         'uf': False,

    #         'uf_peel_list': False,
    #         'uf_peel_minweight': False,
    #         'uf_peel_votemax': False,
    #         'uf_peel_syndrome': False,

    #         'uf_peel_efficient_list': False,
    #         'uf_peel_efficient_minweight': False,
    #         'uf_peel_efficient_votemax': False,
    #         'uf_peel_efficient_syndrome': False,

    #         ###
    #         'bposd': False,
    #     }
        
    #     if channel in ['x', 'both']:
    #         # actual_logX = (noise_total_x @ code_structure_uf.logicals_x.T) % 2
            
    #         if DECODER_CONFIG['use_mwpm']:# and DECODER_CONFIG['noise_source'] == 'custom':
    #             mwpm_pred_logX = matching.decode(syndrome_array_x)
    #             # mwpm_pred_logX = matching.decode_batch(syndrome)
    #             if DECODER_CONFIG['noise_source'] == 'stim':
    #                 error_flags['mwpm'] = not np.array_equal([mwpm_pred_logX[0]], actual_x)
    #             else:
    #                 error_flags['mwpm'] = not np.array_equal(mwpm_pred_logX, actual_x)
            
    #         if DECODER_CONFIG['use_uf']:
    #             # print(f"UF")
    #             # if DEBUG_UF_GEOMETRY:
    #             #     print(f"UF: syndrome_dict_x: {syndrome_dict_x}")
    #             # if DECODER_CONFIG['enable_timing']:
    #             #     start_time = time.time()
    #             uf_pred_logX = uf_decoder.decode(
    #                 syndrome_dict_x, syndrome_array_x, code_structure_uf,
    #                 run_branch=0,
    #                 list_size=list_size,
    #                 actual_logicals=actual_x,
    #                 channel='x'
    #             )
    #             # if DECODER_CONFIG['enable_timing']:
    #             #     end_time = time.time()
    #             #     error_flags['uf_time'] = end_time - start_time
                


    #             if DECODER_CONFIG['noise_source'] == 'stim':
    #                 error_flags['uf'] = not np.array_equal([uf_pred_logX[0]], actual_x)
    #             else:
    #                 error_flags['uf'] = not np.array_equal(uf_pred_logX, actual_x)
            


    #         # if DECODER_CONFIG['use_bposd']:
    #         #     # print(f"Syndrome: {noisy_syndrome_x}")
    #         #     decoding = bp_osd_decoder.decode(syndrome_array_x)
    #         #     bposd_pred_logX = logX@decoding % 2
    #         #     error_flags['bposd'] = not np.array_equal(bposd_pred_logX, actual_x)




    #         if DECODER_CONFIG['use_peel_listdecoding']:
    #             # print(f"Peeling List Decoding")
    #             # if DECODER_CONFIG['enable_timing']:
    #             #     start_time = time.time()
                
    #             # Check if debug mode needs to be enabled
    #             # self._enable_debug_if_needed()
                
    #             # if DEBUG_UF_GEOMETRY:
    #             #     print(f"Peeling List Decoding: syndrome_dict_x: {syndrome_dict_x}")


    #             uf_peel_x, uf_peel_minweight_x, uf_peel_votemin_x, uf_peel_votemax_x, \
    #             uf_peel_syndromemin_x, uf_peel_topological_x = uf_decoder.decode(
    #                 syndrome_dict_x, syndrome_array_x, code_structure_uf,
    #                 run_branch=1,
    #                 list_size=list_size,
    #                 actual_logicals=actual_x,
    #                 channel='x',
    #             )
    #             # if DECODER_CONFIG['enable_timing']:
    #             #     end_time = time.time()
    #             #     error_flags['uf_peel_time'] = end_time - start_time

    #             if DECODER_CONFIG['noise_source'] == 'stim':
    #                 if not np.array_equal([uf_peel_minweight_x[0]], actual_x):
    #                     error_flags['uf_peel_minweight'] = True  
    #                 if not np.array_equal([uf_peel_votemax_x[0]], actual_x):
    #                     error_flags['uf_peel_votemax'] = True
    #                 if not np.array_equal([uf_peel_syndromemin_x[0]], actual_x):
    #                     error_flags['uf_peel_syndrome'] = True              
    #                 all_incorrect_peeling = all(not np.array_equal([pred[0]], actual_x) for pred in uf_peel_x)
    #                 if all_incorrect_peeling:
    #                     error_flags['uf_peel_list'] = True                    
    #             else:
    #                 if not np.array_equal(uf_peel_minweight_x, actual_x):
    #                     error_flags['uf_peel_minweight'] = True 
    #                 if not np.array_equal(uf_peel_votemax_x, actual_x):
    #                     error_flags['uf_peel_votemax'] = True
    #                 if not np.array_equal(uf_peel_syndromemin_x, actual_x):
    #                     error_flags['uf_peel_syndrome'] = True              
    #                 all_incorrect_peeling = all(not np.array_equal(pred, actual_x) for pred in uf_peel_x)
    #                 if all_incorrect_peeling:
    #                     error_flags['uf_peel_list'] = True



    #         # if not error_flags['mwpm'] and error_flags['uf'] and not error_flags['uf_peel_syndrome']:
    #         #     print(f"MWPM Correct, UF Error, UF Peel Syndrome Correct")

    #         if DECODER_CONFIG['use_peel_efficient']:
    #             uf_peel_efficient_x, uf_peel_efficient_minweight_x, uf_peel_efficient_votemin_x, uf_peel_efficient_votemax_x, \
    #             uf_peel_efficient_syndromemin_x, uf_peel_efficient_topological_x, performance_stats \
    #             = uf_decoder.decode(
    #                 syndrome_dict_x, syndrome_array_x, code_structure_uf,
    #                 run_branch=2,
    #                 list_size=list_size,
    #                 actual_logicals=actual_x,
    #                 channel='x'
    #             )
                
    #             # 收集性能统计信息
    #             # 性能统计信息已通过返回值传递
    #             # if DECODER_CONFIG['enable_timing']:
    #             #     end_time = time.time()
    #             #     error_flags['uf_peel_efficient_time'] = end_time - start_time
    #             if DECODER_CONFIG['noise_source'] == 'stim':
    #                 if not np.array_equal([uf_peel_efficient_minweight_x[0]], actual_x):
    #                     error_flags['uf_peel_efficient_minweight'] = True
    #                 if not np.array_equal([uf_peel_efficient_votemax_x[0]], actual_x):
    #                     error_flags['uf_peel_efficient_votemax'] = True
    #                 if not np.array_equal([uf_peel_efficient_syndromemin_x[0]], actual_x):
    #                     error_flags['uf_peel_efficient_syndrome'] = True
    #                 all_incorrect_peeling_efficient = all(not np.array_equal([pred[0]], actual_x) for pred in uf_peel_efficient_x)
    #                 if all_incorrect_peeling_efficient:
    #                     error_flags['uf_peel_efficient_list'] = True
    #             else:
    #                 if not np.array_equal(uf_peel_efficient_minweight_x, actual_x):
    #                     error_flags['uf_peel_efficient_minweight'] = True
    #                 if not np.array_equal(uf_peel_efficient_votemax_x, actual_x):
    #                     error_flags['uf_peel_efficient_votemax'] = True
    #                 if not np.array_equal(uf_peel_efficient_syndromemin_x, actual_x):
    #                     error_flags['uf_peel_efficient_syndrome'] = True
    #                 all_incorrect_peeling_efficient = all(not np.array_equal(pred, actual_x) for pred in uf_peel_efficient_x)
    #                 if all_incorrect_peeling_efficient:
    #                     error_flags['uf_peel_efficient_list'] = True
    #             # error_flags['uf_peel_efficient_time'] = end_time - start_time

    #         # if error_flags['uf_peel_efficient_list'] and not error_flags['uf_peel_list']:
    #         #     print(f"UF Peel Efficient Votemax False, UF list decoding True")
    #         #     a, b, c, d, e, f, performance_stats = uf_decoder.decode(
    #         #         syndrome_dict_x, syndrome_array_x, code_structure_uf,
    #         #         run_branch=2,
    #         #         list_size=list_size,
    #         #         actual_logicals=actual_x,
    #         #         channel='x'
    #         #     )
            
    #         # 性能统计信息已通过返回值传递


     
            
    #     # 返回error_flags和performance_stats
    #     if 'performance_stats' in locals():
    #         return error_flags, performance_stats
    #     else:
    #         return error_flags, None
    
    def run_experiments(self, code_type, Ls, before_round_data_error_rate, 
                        before_measure_error_rate, num_shots=1000, 
                        list_size=4, 
                        channel='x', if_repetitions=True):
        """Run experiments and collect results"""
        # global_time_info = {"Stage 1": [], "Stage 2": [], "Stage 3": [], "Stage 4": []  }
        ps = before_round_data_error_rate
        qs = before_measure_error_rate
        


        log_errors_all_L_mwpm = []
        log_errors_all_L_uf = []
        log_errors_all_L_uf_peel_list = []
        log_errors_all_L_uf_peel_minweight = []
        log_errors_all_L_uf_peel_votemax = []
        log_errors_all_L_uf_peel_syndrome = []
        ###
        log_errors_all_L_uf_peel_efficient_list = []
        log_errors_all_L_uf_peel_efficient_minweight = []
        log_errors_all_L_uf_peel_efficient_votemax = []
        log_errors_all_L_uf_peel_efficient_syndrome = []
        ###
        log_errors_all_L_bposd = []
        raw_latency_all_L = []
        # ablation variants (errors over L)
        log_errors_all_L_uf_ablation_baseline_votemax = []
        log_errors_all_L_uf_ablation_mbuffer_only_votemax = []
        log_errors_all_L_uf_ablation_dsuopt_only_votemax = []
        log_errors_all_L_uf_ablation_graphcompression_votemax = []
        log_errors_all_L_uf_ablation_growskipping_votemax = []
        # latency per L for efficient and ablations
        raw_latency_all_L_peel_efficient = []
        raw_latency_all_L_ablation_baseline = []
        raw_latency_all_L_ablation_mbuffer_only = []
        raw_latency_all_L_ablation_dsuopt_only = []
        raw_latency_all_L_ablation_graphcompression = []
        raw_latency_all_L_ablation_growskipping = []

        for L in Ls:
            # Start new experiment record
            # self.data_collector.start_new_experiment(L=L, noise_level=ps[0])
            if DECODER_CONFIG.get('verbose_top', False):
                print(f"Simulating L={L}...")
            
            log_errors_mwpm = []
            log_errors_uf = []
            log_errors_uf_peel_list = []
            log_errors_uf_peel_minweight = []
            log_errors_uf_peel_votemax = []
            log_errors_uf_peel_syndrome = []
            log_errors_uf_peel_efficient_list = []
            log_errors_uf_peel_efficient_minweight = []
            log_errors_uf_peel_efficient_votemax = []
            log_errors_uf_peel_efficient_syndrome = []
            ###
            log_errors_bposd = []
            # ablation variants (errors per L, to be appended per p later)
            log_errors_uf_ablation_baseline_votemax = []
            log_errors_uf_ablation_mbuffer_only_votemax = []
            log_errors_uf_ablation_dsuopt_only_votemax = []
            log_errors_uf_ablation_graphcompression_votemax = []
            log_errors_uf_ablation_growskipping_votemax = []

            # Add operation counter data collection
            # all_ops_counter_peel_data = []  # For collecting operation counter data for each p value
            raw_latency_all_p_values = []   # For collecting all p values (overall)
            raw_latency_all_p_values_peel_efficient = []
            raw_latency_all_p_values_ablation_baseline = []
            raw_latency_all_p_values_ablation_mbuffer_only = []
            raw_latency_all_p_values_ablation_dsuopt_only = []
            raw_latency_all_p_values_ablation_graphcompression = []
            raw_latency_all_p_values_ablation_growskipping = []
            
            

            # Construct code
            normalized_code_type, Hx, Hz, logX, logZ = _build_code_matrices(code_type, L)


            repetitions = L if if_repetitions else 1

            # Create code structure objects
            code_structure_uf = uf_decoder.CodeStructure(
                H_x= Hx,
                H_z=Hz,
                logicals_x=logX,
                logicals_z=logZ,
                L=L,
                repetitions=repetitions,
                cluster_list_decoding=False,
                peeling_list_decoding=False
            )


            code_structure_peeling = uf_decoder.CodeStructure(
                H_x= Hx,
                H_z=Hz,
                logicals_x=logX,
                logicals_z=logZ,
                L=L,
                repetitions=repetitions,
                cluster_list_decoding=False,
                peeling_list_decoding=True,
                hardware_goldenmodel=False
            )            

            code_structure_efficient = uf_decoder.CodeStructure(
                H_x= Hx,
                H_z=Hz,
                logicals_x=logX,
                logicals_z=logZ,
                L=L,
                repetitions=repetitions,
                cluster_list_decoding=False,
                peeling_list_decoding=False,
                efficient_decoding=True
            )   

            code_structure_hardware = uf_decoder.CodeStructure(
                H_x= Hx,
                H_z=Hz,
                logicals_x=logX,
                logicals_z=logZ,
                L=L,
                repetitions=repetitions,
                cluster_list_decoding=False,
                peeling_list_decoding=False,
                hardware_goldenmodel=True
            )   

            for p in ps:
                # print(f"Start p={p:.3f}...")
                # Start new experiment record, called each time p value changes
                # self.data_collector.start_new_experiment(L=L, noise_level=p)
            
                # 初始化当前p值的延迟数据收集
                raw_latency_for_p_values = []
                raw_latency_for_p_values_peel_efficient = []
                raw_latency_for_p_values_ablation_baseline = []
                raw_latency_for_p_values_ablation_mbuffer_only = []
                raw_latency_for_p_values_ablation_dsuopt_only = []
                raw_latency_for_p_values_ablation_graphcompression = []
                raw_latency_for_p_values_ablation_growskipping = []
                
                # Set measurement noise parameters
                q_eff = p
                # q_eff_z = 0.0  # Z channel doesn't consider measurement noise

                # Calculate timelike_weights, avoid division by 0:
                timelike_weight_x = np.log((1 - q_eff) / q_eff) if q_eff > 0 else 1
                timelike_weight_z = np.log((1 - q_eff) / q_eff) if q_eff > 0 else 1  # Fixed to 0

                # Calculate spatial weights, avoid division by 0 when p_data_effective is 0:
                if p == 0:
                    weight_value = 1.0
                else:
                    weight_value = np.log((1 - p) / p)


                noise_config = DECODER_CONFIG.get('noise_source', 'custom')  # Get noise/matching from configuration
                if normalized_code_type == 'repetition_code' and noise_config == 'stim':
                    if DECODER_CONFIG.get('verbose_top', False):
                        print("Repetition code currently uses custom noise path; overriding noise_source=stim.")
                    noise_config = 'custom'

                if channel == 'x':
                    H3D_x = _build_multiround_pcm(Hx, repetitions)
                    bp_osd_decoder = BpOsdDecoder(
                        H3D_x,
                        error_rate = float(p),
                        bp_method = 'product_sum',
                        max_iter = 7,
                        schedule = 'serial',
                        osd_method = 'osd_cs', #set to OSD_0 for fast solve
                        osd_order = 2
                    )
                else:
                    H3D_z = _build_multiround_pcm(Hz, repetitions)
                    bp_osd_decoder = BpOsdDecoder(
                        H3D_z,
                        error_rate = float(p),
                        bp_method = 'product_sum', #minimum_sum
                        max_iter = 20,
                        schedule = 'serial', #parallel
                        osd_method = 'osd_cs', #set to OSD_0 for fast solve
                        osd_order = 2
                    )

                # Initialize error counters
                num_errors_mwpm = 0
                num_errors_uf = 0
                num_errors_uf_peel_list = 0
                num_errors_uf_peel_minweight = 0
                num_errors_uf_peel_votemax = 0
                num_errors_uf_peel_syndrome = 0
                num_errors_uf_peel_efficient_list = 0
                num_errors_uf_peel_efficient_minweight = 0
                num_errors_uf_peel_efficient_votemax = 0
                num_errors_uf_peel_efficient_syndrome = 0
                ##
                num_errors_bposd = 0
                raw_latency_for_p_values = []  # For collecting raw latency data for each p value
                # ablation counters
                num_errors_uf_ablation_baseline_votemax = 0
                num_errors_uf_ablation_mbuffer_only_votemax = 0
                num_errors_uf_ablation_dsuopt_only_votemax = 0
                num_errors_uf_ablation_graphcompression_votemax = 0
                num_errors_uf_ablation_growskipping_votemax = 0





                #########################################################################################################################################################
                # Option 1: Use unified noise generator interface

                noise_generator = get_noise_generator(
                    config_type=noise_config,
                    L=L, Hx=Hx, Hz=Hz,
                    repetitions=repetitions,
                    after_clifford_depolarization=0,
                    before_round_data_error_rate=p,
                    before_measure_error_rate=q_eff,
                    num_shots=num_shots,
                    code_structure=code_structure_uf,
                    channel=channel
                )

                
                # Initialize empty syndrome_shots list for dynamically adding syndrome data
                syndrome_shots = []
                actual_observables_shots = []
                for shot_id, (actual_x, actual_z, syndrome_dict_x, syndrome_dict_z, syndrome_array_x, syndrome_array_z, matching) in enumerate(noise_generator):
                
                    results, performance_stats = _simulate_single_shot_core(list_size, channel, 
                                                                syndrome_dict_x, syndrome_dict_z, syndrome_array_x, syndrome_array_z, actual_x, actual_z,
                                                                matching, bp_osd_decoder, code_structure_uf,
                                                                shot_id=shot_id)
            
                
                    
                    # Use get method to safely get results, return False if key doesn't exist
                    if results.get('mwpm', False):
                        num_errors_mwpm += 1
                    if results.get('uf', False):
                        num_errors_uf += 1

                    if results.get('uf_peel_list', False):
                        num_errors_uf_peel_list += 1
                    if results.get('uf_peel_minweight', False):
                        num_errors_uf_peel_minweight += 1
                    # if results.get('uf_peel_votemin', False):
                    #     num_errors_uf_peel_votemin += 1
                    if results.get('uf_peel_votemax', False):
                        num_errors_uf_peel_votemax += 1
                    if results.get('uf_peel_syndrome', False):
                        num_errors_uf_peel_syndrome += 1
                    if results.get('uf_peel_efficient_list', False):
                        num_errors_uf_peel_efficient_list += 1
                    if results.get('uf_peel_efficient_minweight', False):
                        num_errors_uf_peel_efficient_minweight += 1
                    if results.get('uf_peel_efficient_votemax', False):
                        num_errors_uf_peel_efficient_votemax += 1
                    if results.get('uf_peel_efficient_syndrome', False):
                        num_errors_uf_peel_efficient_syndrome += 1
                    ##
                    if results.get('bposd', False):
                        num_errors_bposd += 1
                    # ablation variants (only votemax)
                    if results.get('uf_ablation_baseline_votemax', False):
                        num_errors_uf_ablation_baseline_votemax += 1
                    if results.get('uf_ablation_mbuffer_only_votemax', False):
                        num_errors_uf_ablation_mbuffer_only_votemax += 1
                    if results.get('uf_ablation_dsuopt_only_votemax', False):
                        num_errors_uf_ablation_dsuopt_only_votemax += 1
                    if results.get('uf_ablation_graphcompression_votemax', False):
                        num_errors_uf_ablation_graphcompression_votemax += 1
                    if results.get('uf_ablation_growskipping_votemax', False):
                        num_errors_uf_ablation_growskipping_votemax += 1

                    # 采集当前 shot 的性能统计（overall + ablation variants）
                    if performance_stats:
                        def _extract(ps):
                            return {
                                'cluster_operations': ps.get('cluster_latency', 0),
                                'tree_operations': 0,
                                'peeling_operations': ps.get('peeling_latency', 0),
                                'total_cycles': ps.get('total_cycles', 0),
                                'estimated_baseline_latency': ps.get('Estimated_Baseline_latency'),
                                'Peeling_OPs': ps.get('Peeling_OPs', 0),
                                'Baseline_OPs': ps.get('Baseline_OPs', 0)
                            }
                        raw_latency_for_p_values.append(_extract(performance_stats))
                        variants = performance_stats.get('ablation_variants') or {}
                        if 'peel_efficient' in variants:
                            raw_latency_for_p_values_peel_efficient.append(_extract(variants['peel_efficient']))
                        if 'baseline' in variants:
                            raw_latency_for_p_values_ablation_baseline.append(_extract(variants['baseline']))
                        if 'mbuffer_only' in variants:
                            raw_latency_for_p_values_ablation_mbuffer_only.append(_extract(variants['mbuffer_only']))
                        if 'dsuopt_only' in variants:
                            raw_latency_for_p_values_ablation_dsuopt_only.append(_extract(variants['dsuopt_only']))
                        if 'graphcompression' in variants:
                            raw_latency_for_p_values_ablation_graphcompression.append(_extract(variants['graphcompression']))
                        if 'growskipping' in variants:
                            raw_latency_for_p_values_ablation_growskipping.append(_extract(variants['growskipping']))






                # Collect results
                if DECODER_CONFIG['use_mwpm']:
                    log_errors_mwpm.append(num_errors_mwpm/num_shots)
                if DECODER_CONFIG['use_uf']:
                    log_errors_uf.append(num_errors_uf/num_shots)
                if DECODER_CONFIG['use_peel_listdecoding']:
                    log_errors_uf_peel_list.append(num_errors_uf_peel_list/num_shots)
                    log_errors_uf_peel_minweight.append(num_errors_uf_peel_minweight/num_shots)
                    # log_errors_uf_peel_votemin.append(num_errors_uf_peel_votemin/num_shots)
                    log_errors_uf_peel_votemax.append(num_errors_uf_peel_votemax/num_shots)
                    log_errors_uf_peel_syndrome.append(num_errors_uf_peel_syndrome/num_shots)             
                # Calculate average operations for current p value

           
                if DECODER_CONFIG['use_peel_efficient']:
                    log_errors_uf_peel_efficient_list.append(num_errors_uf_peel_efficient_list/num_shots)
                    log_errors_uf_peel_efficient_minweight.append(num_errors_uf_peel_efficient_minweight/num_shots)
                    log_errors_uf_peel_efficient_votemax.append(num_errors_uf_peel_efficient_votemax/num_shots)
                    log_errors_uf_peel_efficient_syndrome.append(num_errors_uf_peel_efficient_syndrome/num_shots)

                # ablation variants 错误率（只统计 votemax）
                if DECODER_CONFIG.get('use_ablation_baseline'):
                    log_errors_uf_ablation_baseline_votemax.append(num_errors_uf_ablation_baseline_votemax/num_shots)
                if DECODER_CONFIG.get('use_ablation_mbuffer_only'):
                    log_errors_uf_ablation_mbuffer_only_votemax.append(num_errors_uf_ablation_mbuffer_only_votemax/num_shots)
                if DECODER_CONFIG.get('use_ablation_dsuopt_only'):
                    log_errors_uf_ablation_dsuopt_only_votemax.append(num_errors_uf_ablation_dsuopt_only_votemax/num_shots)
                if DECODER_CONFIG.get('use_ablation_graphcompression'):
                    log_errors_uf_ablation_graphcompression_votemax.append(num_errors_uf_ablation_graphcompression_votemax/num_shots)
                if DECODER_CONFIG.get('use_ablation_growskipping'):
                    log_errors_uf_ablation_growskipping_votemax.append(num_errors_uf_ablation_growskipping_votemax/num_shots)



                ##
                if DECODER_CONFIG['use_bposd']:
                    log_errors_bposd.append(num_errors_bposd/num_shots)
                
                # 添加当前p值的延迟数据
                raw_latency_all_p_values.append(raw_latency_for_p_values)
                raw_latency_all_p_values_peel_efficient.append(raw_latency_for_p_values_peel_efficient)
                raw_latency_all_p_values_ablation_baseline.append(raw_latency_for_p_values_ablation_baseline)
                raw_latency_all_p_values_ablation_mbuffer_only.append(raw_latency_for_p_values_ablation_mbuffer_only)
                raw_latency_all_p_values_ablation_dsuopt_only.append(raw_latency_for_p_values_ablation_dsuopt_only)
                raw_latency_all_p_values_ablation_graphcompression.append(raw_latency_for_p_values_ablation_graphcompression)
                raw_latency_all_p_values_ablation_growskipping.append(raw_latency_for_p_values_ablation_growskipping)
            
            
            # 保存当前L的结果
            if DECODER_CONFIG['use_mwpm']:
                log_errors_all_L_mwpm.append(np.array(log_errors_mwpm))
            if DECODER_CONFIG['use_uf']:
                log_errors_all_L_uf.append(np.array(log_errors_uf))
            if DECODER_CONFIG['use_peel_listdecoding']:
                log_errors_all_L_uf_peel_list.append(np.array(log_errors_uf_peel_list))
                log_errors_all_L_uf_peel_minweight.append(np.array(log_errors_uf_peel_minweight))
                log_errors_all_L_uf_peel_votemax.append(np.array(log_errors_uf_peel_votemax))
                log_errors_all_L_uf_peel_syndrome.append(np.array(log_errors_uf_peel_syndrome))
               

            if DECODER_CONFIG['use_peel_efficient']:
                log_errors_all_L_uf_peel_efficient_list.append(np.array(log_errors_uf_peel_efficient_list))
                log_errors_all_L_uf_peel_efficient_minweight.append(np.array(log_errors_uf_peel_efficient_minweight))
                log_errors_all_L_uf_peel_efficient_votemax.append(np.array(log_errors_uf_peel_efficient_votemax))
                log_errors_all_L_uf_peel_efficient_syndrome.append(np.array(log_errors_uf_peel_efficient_syndrome))
            # 保存 ablation 错误率
            if DECODER_CONFIG.get('use_ablation_baseline'):
                log_errors_all_L_uf_ablation_baseline_votemax.append(np.array(log_errors_uf_ablation_baseline_votemax))
            if DECODER_CONFIG.get('use_ablation_mbuffer_only'):
                log_errors_all_L_uf_ablation_mbuffer_only_votemax.append(np.array(log_errors_uf_ablation_mbuffer_only_votemax))
            if DECODER_CONFIG.get('use_ablation_dsuopt_only'):
                log_errors_all_L_uf_ablation_dsuopt_only_votemax.append(np.array(log_errors_uf_ablation_dsuopt_only_votemax))
            if DECODER_CONFIG.get('use_ablation_graphcompression'):
                log_errors_all_L_uf_ablation_graphcompression_votemax.append(np.array(log_errors_uf_ablation_graphcompression_votemax))
            if DECODER_CONFIG.get('use_ablation_growskipping'):
                log_errors_all_L_uf_ablation_growskipping_votemax.append(np.array(log_errors_uf_ablation_growskipping_votemax))


            ##
            if DECODER_CONFIG['use_bposd']:
                log_errors_all_L_bposd.append(np.array(log_errors_bposd))
            
            # 添加当前L的延迟数据
            raw_latency_all_L.append(raw_latency_all_p_values)
            raw_latency_all_L_peel_efficient.append(raw_latency_all_p_values_peel_efficient)
            raw_latency_all_L_ablation_baseline.append(raw_latency_all_p_values_ablation_baseline)
            raw_latency_all_L_ablation_mbuffer_only.append(raw_latency_all_p_values_ablation_mbuffer_only)
            raw_latency_all_L_ablation_dsuopt_only.append(raw_latency_all_p_values_ablation_dsuopt_only)
            raw_latency_all_L_ablation_graphcompression.append(raw_latency_all_p_values_ablation_graphcompression)
            raw_latency_all_L_ablation_growskipping.append(raw_latency_all_p_values_ablation_growskipping)

        # 返回所有结果
        return {
            'ps': ps,
            'num_shots': num_shots,
            'log_errors_all_L_mwpm': log_errors_all_L_mwpm,
            'log_errors_all_L_uf': log_errors_all_L_uf,
            'log_errors_all_L_uf_peel_list': log_errors_all_L_uf_peel_list,
            'log_errors_all_L_uf_peel_minweight': log_errors_all_L_uf_peel_minweight,
            'log_errors_all_L_uf_peel_votemax': log_errors_all_L_uf_peel_votemax,
            'log_errors_all_L_uf_peel_syndrome': log_errors_all_L_uf_peel_syndrome,
            'log_errors_all_L_uf_peel_efficient_list': log_errors_all_L_uf_peel_efficient_list,
            'log_errors_all_L_uf_peel_efficient_minweight': log_errors_all_L_uf_peel_efficient_minweight,
            'log_errors_all_L_uf_peel_efficient_votemax': log_errors_all_L_uf_peel_efficient_votemax,
            'log_errors_all_L_uf_peel_efficient_syndrome': log_errors_all_L_uf_peel_efficient_syndrome,
            'log_errors_all_L_bposd': log_errors_all_L_bposd,
            'raw_latency_all_L': raw_latency_all_L,
            # ablation errors per L
            'log_errors_all_L_uf_ablation_baseline_votemax': log_errors_all_L_uf_ablation_baseline_votemax,
            'log_errors_all_L_uf_ablation_mbuffer_only_votemax': log_errors_all_L_uf_ablation_mbuffer_only_votemax,
            'log_errors_all_L_uf_ablation_dsuopt_only_votemax': log_errors_all_L_uf_ablation_dsuopt_only_votemax,
            'log_errors_all_L_uf_ablation_graphcompression_votemax': log_errors_all_L_uf_ablation_graphcompression_votemax,
            'log_errors_all_L_uf_ablation_growskipping_votemax': log_errors_all_L_uf_ablation_growskipping_votemax,
            # latency per L for variants
            'raw_latency_all_L_peel_efficient': raw_latency_all_L_peel_efficient,
            'raw_latency_all_L_ablation_baseline': raw_latency_all_L_ablation_baseline,
            'raw_latency_all_L_ablation_mbuffer_only': raw_latency_all_L_ablation_mbuffer_only,
            'raw_latency_all_L_ablation_dsuopt_only': raw_latency_all_L_ablation_dsuopt_only,
            'raw_latency_all_L_ablation_graphcompression': raw_latency_all_L_ablation_graphcompression,
            'raw_latency_all_L_ablation_growskipping': raw_latency_all_L_ablation_growskipping,
        }
    
    def _format_error_label(self, value):
        """格式化错误率标签为科学计数法
        Args:
            value: 错误率值
        Returns:
            str: 格式化后的字符串
        """
        return _experiment_metrics.format_error_label(value)
    
    def plot_results(self, Ls, ps, results, num_candidates):
        """绘制实验结果
        Args:
            Ls: 格子大小列表
            ps: 错误概率列表
            results: 实验结果字典
            num_candidates: 候选解数量
        """
        return _experiment_plotting.plot_results(self, Ls, ps, results, num_candidates)

    def get_fidelity(self, LER, decoding_latency, d=1):
        """计算系统保真度
        Args:
            LER: 逻辑错误率
            decoding_latency: 解码延迟（周期数）
            d: 距离参数，默认为1
        Returns:
            float: 系统保真度
        """
        return _experiment_metrics.get_fidelity(LER, decoding_latency, d=d)
    
    def _get_helios_cycles(self, L, p):
        """获取helios (UF) 解码器的周期数
        Args:
            L: 代码距离
            p: 物理错误率
        Returns:
            float: 周期数
        """
        return _experiment_metrics.get_helios_cycles(L, p)
    
    def _get_micro_blossom_cycles(self, L, p):
        """获取micro-blossom (MWPM) 解码器的周期数
        Args:
            L: 代码距离
            p: 物理错误率
        Returns:
            float: 周期数
        """
        return _experiment_metrics.get_micro_blossom_cycles(L, p)

    def plot_latency(self, Ls, ps, results, code_type = 'toric_code'):
        """绘制平均解码延迟与物理错误率的关系图，并使用小提琴图显示数据分布
        Args:
            Ls: 格子大小列表
            ps: 错误概率列表
            results: 实验结果字典，包含 'raw_latency_all_L'
            code_type: 代码类型
        """
        return _experiment_plotting.plot_latency(self, Ls, ps, results, code_type)



    def _plot_performance_tracker_latency(self, Ls, ps, results, code_type):
        """使用性能追踪器数据绘制延迟图表，按L值和p值分别分析"""
        return _experiment_plotting._plot_performance_tracker_latency(self, Ls, ps, results, code_type)

    def plot_results_with_varying_styles(self, Ls, ps, results, num_candidates, code_type = 'toric_code'):
        """绘制实验结果，根据不同的L值使用不同的线型和颜色
        Args:
            Ls: 格子大小列表
            ps: 错误概率列表
            results: 实验结果字典
            num_candidates: 候选解数量
        """
        return _experiment_plotting.plot_results_with_varying_styles(self, Ls, ps, results, num_candidates, code_type=code_type)

    def run_experiments_parallel(self, code_type, Ls, before_round_data_error_rate, before_measure_error_rate, 
                                num_shots=1000, list_size=4, 
                                channel='x', if_repetitions=True, n_jobs=-1, parallel_level='shots'):
        """并行运行实验并收集结果
        
        Args:
            code_type: 代码类型 ('toric_code', 'surface_code', 'rotated_surface_code')
            Ls: 格子大小列表
            before_round_data_error_rate: 数据错误概率列表
            before_measure_error_rate: 测量错误概率列表
            num_shots: 每个实验的shot数量
            num_candidates_cluster: cluster候选解数量
            num_candidates_peeling: peeling候选解数量
            channel: 通道类型 ('x', 'z', 'both')
            if_repetitions: 是否使用重复测量
            n_jobs: 并行进程数，-1表示使用所有CPU核心
            parallel_level: 并行级别 ('shots', 'p', 'L')
        
        Returns:
            dict: 包含所有结果的字典
        """
        # global_time_info = {"Stage 1": [], "Stage 2": [], "Stage 3": [], "Stage 4": []  }
        ps = before_round_data_error_rate
        qs = before_measure_error_rate
        
        # 智能选择进程数 - 减少进程数以避免内存问题
        if n_jobs == -1:
            # 使用更保守的进程数，避免内存溢出
            n_jobs = int(4*mp.cpu_count()/5)  # 从7/8减少到1/2
        
        print(f"Using {n_jobs} processes for parallel computation, parallel level: {parallel_level}")
        
        if parallel_level == 'shots':
            return self._run_experiments_parallel_shots(
                code_type, Ls, ps, qs, num_shots, list_size, 
                channel, if_repetitions, n_jobs
            )
        else:
            raise ValueError("parallel_level must be 'shots'")

    def _run_experiments_parallel_shots(self, code_type, Ls, ps, qs, num_shots, 
                                      list_size, 
                                      channel, if_repetitions, n_jobs):
        """在shot级别进行并行化"""
        results = {
            'ps': ps,
            'num_shots': num_shots,
            'log_errors_all_L_mwpm': [],
            'log_errors_all_L_mwpm_dem_unweighted': [],
            'log_errors_all_L_mwpm_dem_weighted': [],
            'log_errors_all_L_mwpm_hx_manual_unweighted': [],
            'log_errors_all_L_mwpm_disagree_rate': [],
            'log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate': [],
            'log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate': [],
            'mwpm_fairness_diff_all_L': [],
            'mwpm_fairness_se_all_L': [],
            'mwpm_fairness_z_all_L': [],
            'mwpm_fairness_diff_hx_vs_dem_weighted_all_L': [],
            'mwpm_fairness_se_hx_vs_dem_weighted_all_L': [],
            'mwpm_fairness_z_hx_vs_dem_weighted_all_L': [],
            'mwpm_fairness_diff_dem_unweighted_vs_dem_weighted_all_L': [],
            'mwpm_fairness_se_dem_unweighted_vs_dem_weighted_all_L': [],
            'mwpm_fairness_z_dem_unweighted_vs_dem_weighted_all_L': [],
            'mwpm_fairness_protocol': DECODER_CONFIG.get('mwpm_fairness_protocol', {}),
            'log_errors_all_L_uf': [],
            'log_errors_all_L_uf_peel_list': [],
            'log_errors_all_L_uf_peel_minweight': [],
            'log_errors_all_L_uf_peel_votemax': [],
            'log_errors_all_L_uf_peel_syndrome': [],
            'log_errors_all_L_uf_peel_efficient_list': [],
            'log_errors_all_L_uf_peel_efficient_minweight': [],
            'log_errors_all_L_uf_peel_efficient_votemax': [],
            'log_errors_all_L_uf_peel_efficient_syndrome': [],
            'log_errors_all_L_bposd': [],
            'log_errors_all_L_bposd_dem_graph': [],
            'log_errors_all_L_bposd_shared_graph': [],
            'log_errors_all_L_bposd_disagree_rate': [],
            # ablation variants
            'log_errors_all_L_uf_ablation_baseline_votemax': [],
            'log_errors_all_L_uf_ablation_mbuffer_only_votemax': [],
            'log_errors_all_L_uf_ablation_dsuopt_only_votemax': [],
            'log_errors_all_L_uf_ablation_graphcompression_votemax': [],
            'log_errors_all_L_uf_ablation_growskipping_votemax': [],
            # latency collections (all L)
            'raw_latency_all_L': [],
            'raw_latency_all_L_peel_efficient': [],
            'raw_latency_all_L_ablation_baseline': [],
            'raw_latency_all_L_ablation_mbuffer_only': [],
            'raw_latency_all_L_ablation_dsuopt_only': [],
            'raw_latency_all_L_ablation_graphcompression': [],
            'raw_latency_all_L_ablation_growskipping': [],
        }

        # 添加延迟统计数据结构（总览）
        raw_latency_all_L = []
        raw_latency_all_L_peel_efficient = []
        raw_latency_all_L_ablation_baseline = []
        raw_latency_all_L_ablation_mbuffer_only = []
        raw_latency_all_L_ablation_dsuopt_only = []
        raw_latency_all_L_ablation_graphcompression = []
        raw_latency_all_L_ablation_growskipping = []

        for L in Ls:
            if DECODER_CONFIG.get('verbose_top', False):
                print(f"Simulating L={L}...")
            # self.data_collector.start_new_experiment(L=L, noise_level=ps[0])
            
            # 构造代码
            normalized_code_type, Hx, Hz, logX, logZ = _build_code_matrices(code_type, L)

            repetitions = L if if_repetitions else 1
            code_structure_uf = uf_decoder.CodeStructure(
                H_x=Hx, H_z=Hz, logicals_x=logX, logicals_z=logZ, L=L, repetitions=repetitions, 
                cluster_list_decoding=False, peeling_list_decoding=False
            )
            code_structure_peeling = uf_decoder.CodeStructure(
                H_x=Hx, H_z=Hz, logicals_x=logX, logicals_z=logZ, L=L, repetitions=repetitions,
                cluster_list_decoding=False, peeling_list_decoding=True
            )
            code_structure_efficient = uf_decoder.CodeStructure(
                H_x=Hx, H_z=Hz, logicals_x=logX, logicals_z=logZ, L=L, repetitions=repetitions,
                cluster_list_decoding=False, peeling_list_decoding=False, efficient_decoding=True
            )

            log_errors_mwpm = []
            log_errors_mwpm_dem_unweighted = []
            log_errors_mwpm_dem_weighted = []
            log_errors_mwpm_hx_manual_unweighted = []
            log_errors_mwpm_disagree_rate = []
            log_errors_mwpm_disagree_hx_vs_dem_weighted_rate = []
            log_errors_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate = []
            mwpm_fairness_diff = []
            mwpm_fairness_se = []
            mwpm_fairness_z = []
            mwpm_fairness_diff_hx_vs_dem_weighted = []
            mwpm_fairness_se_hx_vs_dem_weighted = []
            mwpm_fairness_z_hx_vs_dem_weighted = []
            mwpm_fairness_diff_dem_unweighted_vs_dem_weighted = []
            mwpm_fairness_se_dem_unweighted_vs_dem_weighted = []
            mwpm_fairness_z_dem_unweighted_vs_dem_weighted = []
            log_errors_uf = []
            log_errors_uf_peel_list = []
            log_errors_uf_peel_minweight = []
            log_errors_uf_peel_votemax = []
            log_errors_uf_peel_syndrome = []
            log_errors_uf_peel_efficient_list = []
            log_errors_uf_peel_efficient_minweight = []
            log_errors_uf_peel_efficient_votemax = []
            log_errors_uf_peel_efficient_syndrome = []
            log_errors_bposd = []
            log_errors_bposd_dem_graph = []
            log_errors_bposd_shared_graph = []
            log_errors_bposd_disagree_rate = []
            # ablation variants
            log_errors_uf_ablation_baseline_votemax = []
            log_errors_uf_ablation_mbuffer_only_votemax = []
            log_errors_uf_ablation_dsuopt_only_votemax = []
            log_errors_uf_ablation_graphcompression_votemax = []
            log_errors_uf_ablation_growskipping_votemax = []

            # 添加延迟统计变量（按 p 聚合）
            raw_latency_all_p_values = []   # 总体
            raw_latency_all_p_values_peel_efficient = []
            raw_latency_all_p_values_ablation_baseline = []
            raw_latency_all_p_values_ablation_mbuffer_only = []
            raw_latency_all_p_values_ablation_dsuopt_only = []
            raw_latency_all_p_values_ablation_graphcompression = []
            raw_latency_all_p_values_ablation_growskipping = []

            # 检查是否使用stim噪声生成器
            noise_config = DECODER_CONFIG.get('noise_source', 'custom')
            if normalized_code_type == 'repetition_code' and noise_config == 'stim':
                if DECODER_CONFIG.get('verbose_top', False):
                    print("Repetition code currently uses custom noise path; overriding noise_source=stim.")
                noise_config = 'custom'



            if noise_config == 'stim':
                # 预生成stim数据
                stim_data = self._pre_generate_stim_data(L, ps, num_shots, code_structure_uf, channel)
                
                for p in ps:
                    # print(f"Start p={p:.3f}...")
                    # self.data_collector.start_new_experiment(L=L, noise_level=p)
                    # 创建matching对象
                    q_eff = p
                    # weight_value = np.log((1 - p) / p) if p > 0 else 1
                    # timelike_weight = np.log((1 - q_eff) / q_eff) if q_eff > 0 else 1
                    if channel == 'x':
                        H3D_x = _build_multiround_pcm(Hx, repetitions)
                        bp_osd_decoder = BpOsdDecoder(H3D_x, error_rate=float(p), bp_method='product_sum',
                                                max_iter=7, schedule='serial', osd_method='osd_cs', osd_order=2)                
                    else:
                        H3D_z = _build_multiround_pcm(Hz, repetitions)
                        bp_osd_decoder = BpOsdDecoder(H3D_z, error_rate=float(p), bp_method='product_sum',
                                                max_iter=20, schedule='serial', osd_method='osd_cs', osd_order=2)
                    # 使用预生成数据的shot级别并行（仅传当前 p 的数据，避免复制整份 stim_data）
                    p_key = f"{p:.6f}"
                    if p_key not in stim_data:
                        raise ValueError(f"预生成数据中没有p={p}的数据")
                    p_data = stim_data[p_key]
                    detector_coords = p_data['detector_coords']
                    code_type = p_data['code_type']
                    syndrome = p_data['syndrome']
                    actual_observables = p_data['actual_observables']

                    shot_results = Parallel(
                        n_jobs=n_jobs,
                        backend="loky",
                        verbose=0,
                        max_nbytes="200M",  # 增加内存限制从50M到200M
                        mmap_mode="r",
                        pre_dispatch=n_jobs,  # 减少预调度，从2*n_jobs到n_jobs
                        batch_size="auto",
                        prefer="processes"
                    )(
                        delayed(_process_single_shot_task)(
                            shot_id,
                            code_type, L, p, repetitions, channel, list_size,
                            detector_coords,
                            syndrome[shot_id, :],
                            actual_observables[shot_id],
                        ) for shot_id in range(num_shots)
                    )

                    # shot_results = Parallel(
                    #     n_jobs=n_jobs,
                    #     verbose=0,
                    #     max_nbytes=None,
                    #     mmap_mode=None,
                    #     backend="loky",
                    #     # temp_folder="/dev/shm",  # disable shm to avoid large state copy to shm
                    #     initializer=_init_ufdecoder_worker_ctx,
                    #     initargs=(self, code_type,
                    #               L, p, repetitions, channel, list_size),
                    # )(
                    #     delayed(_run_shot_with_ctx)(shot_id) for shot_id in range(num_shots)
                    # )
                    
                    # 分离error_flags和performance_stats
                    error_flags_list = [r[0] for r in shot_results]
                    performance_stats_list = [r[1] for r in shot_results if r[1] is not None]
                    
                    # 统计结果
                    num_errors_mwpm = sum(1 for r in error_flags_list if r.get('mwpm', False))
                    num_errors_mwpm_dem_unweighted = sum(1 for r in error_flags_list if r.get('mwpm_dem_unweighted', False))
                    num_errors_mwpm_dem_weighted = sum(1 for r in error_flags_list if r.get('mwpm_dem_weighted', False))
                    num_errors_mwpm_hx_manual_unweighted = sum(1 for r in error_flags_list if r.get('mwpm_hx_manual_unweighted', False))
                    num_errors_mwpm_disagree = sum(1 for r in error_flags_list if r.get('mwpm_disagree', False))
                    num_errors_mwpm_disagree_hx_vs_dem_weighted = sum(1 for r in error_flags_list if r.get('mwpm_disagree_hx_vs_dem_weighted', False))
                    num_errors_mwpm_disagree_dem_unweighted_vs_dem_weighted = sum(1 for r in error_flags_list if r.get('mwpm_disagree_dem_unweighted_vs_dem_weighted', False))
                    num_errors_uf = sum(1 for r in error_flags_list if r.get('uf', False))
                    num_errors_uf_peel_list = sum(1 for r in error_flags_list if r.get('uf_peel_list', False))
                    num_errors_uf_peel_minweight = sum(1 for r in error_flags_list if r.get('uf_peel_minweight', False))
                    num_errors_uf_peel_votemax = sum(1 for r in error_flags_list if r.get('uf_peel_votemax', False))
                    num_errors_uf_peel_syndrome = sum(1 for r in error_flags_list if r.get('uf_peel_syndrome', False))
                    num_errors_uf_peel_efficient_list = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_list', False))
                    num_errors_uf_peel_efficient_minweight = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_minweight', False))
                    num_errors_uf_peel_efficient_votemin = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_votemin', False))
                    num_errors_uf_peel_efficient_votemax = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_votemax', False))
                    num_errors_uf_peel_efficient_syndrome = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_syndrome', False))
                    num_errors_bposd = sum(1 for r in error_flags_list if r.get('bposd', False))
                    num_errors_bposd_dem_graph = sum(1 for r in error_flags_list if r.get('bposd_dem_graph', False))
                    num_errors_bposd_shared_graph = sum(1 for r in error_flags_list if r.get('bposd_shared_graph', False))
                    num_errors_bposd_disagree = sum(1 for r in error_flags_list if r.get('bposd_disagree', False))
                    # ablation variants
                    num_errors_uf_ablation_baseline_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_baseline_votemax', False))
                    num_errors_uf_ablation_mbuffer_only_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_mbuffer_only_votemax', False))
                    num_errors_uf_ablation_dsuopt_only_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_dsuopt_only_votemax', False))
                    num_errors_uf_ablation_graphcompression_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_graphcompression_votemax', False))
                    num_errors_uf_ablation_growskipping_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_growskipping_votemax', False))
                    
                    # 收集性能统计数据（总体 + 各 ablation 变体）
                    raw_latency_for_p_values = []
                    raw_latency_for_p_values_peel_efficient = []
                    raw_latency_for_p_values_ablation_baseline = []
                    raw_latency_for_p_values_ablation_mbuffer_only = []
                    raw_latency_for_p_values_ablation_dsuopt_only = []
                    raw_latency_for_p_values_ablation_graphcompression = []
                    raw_latency_for_p_values_ablation_growskipping = []
                    for perf_stats in performance_stats_list:
                        if not perf_stats:
                            continue
                        def _extract(ps):
                            return {
                                'cluster_operations': ps.get('cluster_latency', 0),
                                'tree_operations': 0,  # peeling 模块没有 tree_operations
                                'peeling_operations': ps.get('peeling_latency', 0),
                                'total_cycles': ps.get('total_cycles', 0),
                                'estimated_baseline_latency': ps.get('Estimated_Baseline_latency'),
                                'Peeling_OPs': ps.get('Peeling_OPs', 0),
                                'Baseline_OPs': ps.get('Baseline_OPs', 0)
                            }
                        # 总体（顶层 stats）
                        raw_latency_for_p_values.append(_extract(perf_stats))
                        # 分变体（若存在）
                        variants = perf_stats.get('ablation_variants') or {}
                        if 'peel_efficient' in variants:
                            raw_latency_for_p_values_peel_efficient.append(_extract(variants['peel_efficient']))
                        if 'baseline' in variants:
                            raw_latency_for_p_values_ablation_baseline.append(_extract(variants['baseline']))
                        if 'mbuffer_only' in variants:
                            raw_latency_for_p_values_ablation_mbuffer_only.append(_extract(variants['mbuffer_only']))
                        if 'dsuopt_only' in variants:
                            raw_latency_for_p_values_ablation_dsuopt_only.append(_extract(variants['dsuopt_only']))
                        if 'graphcompression' in variants:
                            raw_latency_for_p_values_ablation_graphcompression.append(_extract(variants['graphcompression']))
                        if 'growskipping' in variants:
                            raw_latency_for_p_values_ablation_growskipping.append(_extract(variants['growskipping']))



                    # 收集结果
                    if DECODER_CONFIG['use_mwpm']:
                        log_errors_mwpm.append(num_errors_mwpm/num_shots)
                        dem_rate = num_errors_mwpm_dem_unweighted / num_shots
                        dem_weighted_rate = num_errors_mwpm_dem_weighted / num_shots
                        hx_rate = num_errors_mwpm_hx_manual_unweighted / num_shots
                        disagree_rate = num_errors_mwpm_disagree / num_shots
                        disagree_hx_demw_rate = num_errors_mwpm_disagree_hx_vs_dem_weighted / num_shots
                        disagree_demu_demw_rate = num_errors_mwpm_disagree_dem_unweighted_vs_dem_weighted / num_shots
                        log_errors_mwpm_dem_unweighted.append(dem_rate)
                        log_errors_mwpm_dem_weighted.append(dem_weighted_rate)
                        log_errors_mwpm_hx_manual_unweighted.append(hx_rate)
                        log_errors_mwpm_disagree_rate.append(disagree_rate)
                        log_errors_mwpm_disagree_hx_vs_dem_weighted_rate.append(disagree_hx_demw_rate)
                        log_errors_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate.append(disagree_demu_demw_rate)
                        diff = hx_rate - dem_rate
                        se_diff = np.sqrt(
                            dem_rate * (1 - dem_rate) / num_shots +
                            hx_rate * (1 - hx_rate) / num_shots
                        )
                        mwpm_fairness_diff.append(diff)
                        mwpm_fairness_se.append(se_diff)
                        mwpm_fairness_z.append(diff / se_diff if se_diff > 0 else 0.0)
                        diff_hx_demw = hx_rate - dem_weighted_rate
                        se_hx_demw = np.sqrt(
                            dem_weighted_rate * (1 - dem_weighted_rate) / num_shots +
                            hx_rate * (1 - hx_rate) / num_shots
                        )
                        mwpm_fairness_diff_hx_vs_dem_weighted.append(diff_hx_demw)
                        mwpm_fairness_se_hx_vs_dem_weighted.append(se_hx_demw)
                        mwpm_fairness_z_hx_vs_dem_weighted.append(diff_hx_demw / se_hx_demw if se_hx_demw > 0 else 0.0)
                        diff_demu_demw = dem_rate - dem_weighted_rate
                        se_demu_demw = np.sqrt(
                            dem_weighted_rate * (1 - dem_weighted_rate) / num_shots +
                            dem_rate * (1 - dem_rate) / num_shots
                        )
                        mwpm_fairness_diff_dem_unweighted_vs_dem_weighted.append(diff_demu_demw)
                        mwpm_fairness_se_dem_unweighted_vs_dem_weighted.append(se_demu_demw)
                        mwpm_fairness_z_dem_unweighted_vs_dem_weighted.append(diff_demu_demw / se_demu_demw if se_demu_demw > 0 else 0.0)
                    if DECODER_CONFIG['use_uf']:
                        log_errors_uf.append(num_errors_uf/num_shots)
                    if DECODER_CONFIG['use_peel_listdecoding']:
                        log_errors_uf_peel_list.append(num_errors_uf_peel_list/num_shots)
                        log_errors_uf_peel_minweight.append(num_errors_uf_peel_minweight/num_shots)
                        log_errors_uf_peel_votemax.append(num_errors_uf_peel_votemax/num_shots)
                        log_errors_uf_peel_syndrome.append(num_errors_uf_peel_syndrome/num_shots)
                    if DECODER_CONFIG['use_peel_efficient']:
                        log_errors_uf_peel_efficient_list.append(num_errors_uf_peel_efficient_list/num_shots)
                        log_errors_uf_peel_efficient_minweight.append(num_errors_uf_peel_efficient_minweight/num_shots)
                        log_errors_uf_peel_efficient_votemax.append(num_errors_uf_peel_efficient_votemax/num_shots)
                        log_errors_uf_peel_efficient_syndrome.append(num_errors_uf_peel_efficient_syndrome/num_shots)
                    if DECODER_CONFIG.get('use_ablation_baseline'):
                        log_errors_uf_ablation_baseline_votemax.append(num_errors_uf_ablation_baseline_votemax/num_shots)
                    if DECODER_CONFIG.get('use_ablation_mbuffer_only'):
                        log_errors_uf_ablation_mbuffer_only_votemax.append(num_errors_uf_ablation_mbuffer_only_votemax/num_shots)
                    if DECODER_CONFIG.get('use_ablation_dsuopt_only'):
                        log_errors_uf_ablation_dsuopt_only_votemax.append(num_errors_uf_ablation_dsuopt_only_votemax/num_shots)
                    if DECODER_CONFIG.get('use_ablation_graphcompression'):
                        log_errors_uf_ablation_graphcompression_votemax.append(num_errors_uf_ablation_graphcompression_votemax/num_shots)
                    if DECODER_CONFIG.get('use_ablation_growskipping'):
                        log_errors_uf_ablation_growskipping_votemax.append(num_errors_uf_ablation_growskipping_votemax/num_shots)
                    if DECODER_CONFIG['use_bposd']:
                        log_errors_bposd.append(num_errors_bposd/num_shots)
                        log_errors_bposd_dem_graph.append(num_errors_bposd_dem_graph/num_shots)
                        log_errors_bposd_shared_graph.append(num_errors_bposd_shared_graph/num_shots)
                        log_errors_bposd_disagree_rate.append(num_errors_bposd_disagree/num_shots)
                    
                    # 添加当前 p 的延迟数据到各自 all_p_values
                    raw_latency_all_p_values.append(raw_latency_for_p_values)
                    raw_latency_all_p_values_peel_efficient.append(raw_latency_for_p_values_peel_efficient)
                    raw_latency_all_p_values_ablation_baseline.append(raw_latency_for_p_values_ablation_baseline)
                    raw_latency_all_p_values_ablation_mbuffer_only.append(raw_latency_for_p_values_ablation_mbuffer_only)
                    raw_latency_all_p_values_ablation_dsuopt_only.append(raw_latency_for_p_values_ablation_dsuopt_only)
                    raw_latency_all_p_values_ablation_graphcompression.append(raw_latency_for_p_values_ablation_graphcompression)
                    raw_latency_all_p_values_ablation_growskipping.append(raw_latency_for_p_values_ablation_growskipping)

            else:
                # 原有的自定义噪声生成器逻辑
                for p in ps:
                    # print(f"Start p={p:.3f}...")
                    # self.data_collector.start_new_experiment(L=L, noise_level=p)
                    
                    # 为每个p值创建解码器
                    q_eff = p
                    # timelike_weight_x = np.log((1 - q_eff) / q_eff) if q_eff > 0 else 1
                    # timelike_weight_z = np.log((1 - q_eff) / q_eff) if q_eff > 0 else 1
                    # weight_value = 1.0 if p == 0 else np.log((1 - p) / p)

                    if channel == 'x':
                        H3D_x = _build_multiround_pcm(Hx, repetitions)
                        bp_osd_decoder = BpOsdDecoder(H3D_x, error_rate=float(p), bp_method='product_sum',
                                               max_iter=7, schedule='serial', osd_method='osd_cs', osd_order=2)
                    else:
                        H3D_z = _build_multiround_pcm(Hz, repetitions)
                        bp_osd_decoder = BpOsdDecoder(H3D_z, error_rate=float(p), bp_method='product_sum',
                                               max_iter=20, schedule='serial', osd_method='osd_cs', osd_order=2)

                    # 并行处理shots - 每个子进程只处理一个shot
                    shot_results = Parallel(n_jobs=n_jobs, verbose=0)(
                        delayed(self._process_single_shot)(
                            L, p, list_size, channel,
                            repetitions, Hx, Hz, code_structure_uf,
                            shot_id  # 传递shot_id确保每个进程处理不同的shot
                        ) for shot_id in range(num_shots)
                    )

                    # 分离error_flags和performance_stats
                    error_flags_list = []
                    performance_stats_list = []
                    
                    for result in shot_results:
                        if isinstance(result, tuple) and len(result) == 2:
                            # 返回值是(error_flags, performance_stats)
                            error_flags, performance_stats = result
                            error_flags_list.append(error_flags)
                            if performance_stats is not None:
                                performance_stats_list.append(performance_stats)
                        else:
                            # 返回值只是error_flags
                            error_flags_list.append(result)

                    # 统计结果
                    num_errors_mwpm = sum(1 for r in error_flags_list if r.get('mwpm', False))
                    num_errors_mwpm_dem_unweighted = sum(1 for r in error_flags_list if r.get('mwpm_dem_unweighted', False))
                    num_errors_mwpm_dem_weighted = sum(1 for r in error_flags_list if r.get('mwpm_dem_weighted', False))
                    num_errors_mwpm_hx_manual_unweighted = sum(1 for r in error_flags_list if r.get('mwpm_hx_manual_unweighted', False))
                    num_errors_mwpm_disagree = sum(1 for r in error_flags_list if r.get('mwpm_disagree', False))
                    num_errors_mwpm_disagree_hx_vs_dem_weighted = sum(1 for r in error_flags_list if r.get('mwpm_disagree_hx_vs_dem_weighted', False))
                    num_errors_mwpm_disagree_dem_unweighted_vs_dem_weighted = sum(1 for r in error_flags_list if r.get('mwpm_disagree_dem_unweighted_vs_dem_weighted', False))
                    num_errors_uf = sum(1 for r in error_flags_list if r.get('uf', False))
                    num_errors_uf_peel_list = sum(1 for r in error_flags_list if r.get('uf_peel_list', False))
                    num_errors_uf_peel_minweight = sum(1 for r in error_flags_list if r.get('uf_peel_minweight', False))
                    num_errors_uf_peel_votemax = sum(1 for r in error_flags_list if r.get('uf_peel_votemax', False))
                    num_errors_uf_peel_syndrome = sum(1 for r in error_flags_list if r.get('uf_peel_syndrome', False))
                    num_errors_uf_peel_efficient_list = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_list', False))
                    num_errors_uf_peel_efficient_minweight = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_minweight', False))
                    num_errors_uf_peel_efficient_votemax = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_votemax', False))
                    num_errors_uf_peel_efficient_syndrome = sum(1 for r in error_flags_list if r.get('uf_peel_efficient_syndrome', False))
                    num_errors_bposd = sum(1 for r in error_flags_list if r.get('bposd', False))
                    num_errors_bposd_dem_graph = sum(1 for r in error_flags_list if r.get('bposd_dem_graph', False))
                    num_errors_bposd_shared_graph = sum(1 for r in error_flags_list if r.get('bposd_shared_graph', False))
                    num_errors_bposd_disagree = sum(1 for r in error_flags_list if r.get('bposd_disagree', False))
                    num_errors_uf_ablation_baseline_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_baseline_votemax', False))
                    num_errors_uf_ablation_mbuffer_only_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_mbuffer_only_votemax', False))
                    num_errors_uf_ablation_dsuopt_only_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_dsuopt_only_votemax', False))
                    num_errors_uf_ablation_graphcompression_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_graphcompression_votemax', False))
                    num_errors_uf_ablation_growskipping_votemax = sum(1 for r in error_flags_list if r.get('uf_ablation_growskipping_votemax', False))

                    # 收集延迟数据
                    raw_latency_for_p_values = []
                    raw_latency_for_p_values_peel_efficient = []
                    raw_latency_for_p_values_ablation_baseline = []
                    raw_latency_for_p_values_ablation_mbuffer_only = []
                    raw_latency_for_p_values_ablation_dsuopt_only = []
                    raw_latency_for_p_values_ablation_graphcompression = []
                    raw_latency_for_p_values_ablation_growskipping = []
                    for perf_stats in performance_stats_list:
                        if not perf_stats:
                            continue
                        def _extract(ps):
                            return {
                                'cluster_operations': ps.get('cluster_latency', 0),
                                'tree_operations': 0,
                                'peeling_operations': ps.get('peeling_latency', 0),
                                'total_cycles': ps.get('total_cycles', 0),
                                'estimated_baseline_latency': ps.get('Estimated_Baseline_latency'),
                                'Peeling_OPs': ps.get('Peeling_OPs', 0),
                                'Baseline_OPs': ps.get('Baseline_OPs', 0)
                            }
                        raw_latency_for_p_values.append(_extract(perf_stats))
                        variants = perf_stats.get('ablation_variants') or {}
                        if 'peel_efficient' in variants:
                            raw_latency_for_p_values_peel_efficient.append(_extract(variants['peel_efficient']))
                        if 'baseline' in variants:
                            raw_latency_for_p_values_ablation_baseline.append(_extract(variants['baseline']))
                        if 'mbuffer_only' in variants:
                            raw_latency_for_p_values_ablation_mbuffer_only.append(_extract(variants['mbuffer_only']))
                        if 'dsuopt_only' in variants:
                            raw_latency_for_p_values_ablation_dsuopt_only.append(_extract(variants['dsuopt_only']))
                        if 'graphcompression' in variants:
                            raw_latency_for_p_values_ablation_graphcompression.append(_extract(variants['graphcompression']))
                        if 'growskipping' in variants:
                            raw_latency_for_p_values_ablation_growskipping.append(_extract(variants['growskipping']))


                    # 收集结果
                    if DECODER_CONFIG['use_mwpm']:
                        log_errors_mwpm.append(num_errors_mwpm/num_shots)
                        dem_rate = num_errors_mwpm_dem_unweighted / num_shots
                        dem_weighted_rate = num_errors_mwpm_dem_weighted / num_shots
                        hx_rate = num_errors_mwpm_hx_manual_unweighted / num_shots
                        disagree_rate = num_errors_mwpm_disagree / num_shots
                        disagree_hx_demw_rate = num_errors_mwpm_disagree_hx_vs_dem_weighted / num_shots
                        disagree_demu_demw_rate = num_errors_mwpm_disagree_dem_unweighted_vs_dem_weighted / num_shots
                        log_errors_mwpm_dem_unweighted.append(dem_rate)
                        log_errors_mwpm_dem_weighted.append(dem_weighted_rate)
                        log_errors_mwpm_hx_manual_unweighted.append(hx_rate)
                        log_errors_mwpm_disagree_rate.append(disagree_rate)
                        log_errors_mwpm_disagree_hx_vs_dem_weighted_rate.append(disagree_hx_demw_rate)
                        log_errors_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate.append(disagree_demu_demw_rate)
                        diff = hx_rate - dem_rate
                        se_diff = np.sqrt(
                            dem_rate * (1 - dem_rate) / num_shots +
                            hx_rate * (1 - hx_rate) / num_shots
                        )
                        mwpm_fairness_diff.append(diff)
                        mwpm_fairness_se.append(se_diff)
                        mwpm_fairness_z.append(diff / se_diff if se_diff > 0 else 0.0)
                        diff_hx_demw = hx_rate - dem_weighted_rate
                        se_hx_demw = np.sqrt(
                            dem_weighted_rate * (1 - dem_weighted_rate) / num_shots +
                            hx_rate * (1 - hx_rate) / num_shots
                        )
                        mwpm_fairness_diff_hx_vs_dem_weighted.append(diff_hx_demw)
                        mwpm_fairness_se_hx_vs_dem_weighted.append(se_hx_demw)
                        mwpm_fairness_z_hx_vs_dem_weighted.append(diff_hx_demw / se_hx_demw if se_hx_demw > 0 else 0.0)
                        diff_demu_demw = dem_rate - dem_weighted_rate
                        se_demu_demw = np.sqrt(
                            dem_weighted_rate * (1 - dem_weighted_rate) / num_shots +
                            dem_rate * (1 - dem_rate) / num_shots
                        )
                        mwpm_fairness_diff_dem_unweighted_vs_dem_weighted.append(diff_demu_demw)
                        mwpm_fairness_se_dem_unweighted_vs_dem_weighted.append(se_demu_demw)
                        mwpm_fairness_z_dem_unweighted_vs_dem_weighted.append(diff_demu_demw / se_demu_demw if se_demu_demw > 0 else 0.0)
                    if DECODER_CONFIG['use_uf']:
                        log_errors_uf.append(num_errors_uf/num_shots)
                    if DECODER_CONFIG['use_peel_listdecoding']:
                        log_errors_uf_peel_list.append(num_errors_uf_peel_list/num_shots)
                        log_errors_uf_peel_minweight.append(num_errors_uf_peel_minweight/num_shots)
                        # log_errors_uf_peel_votemin.append(num_errors_uf_peel_votemin/num_shots)
                        log_errors_uf_peel_votemax.append(num_errors_uf_peel_votemax/num_shots)
                        log_errors_uf_peel_syndrome.append(num_errors_uf_peel_syndrome/num_shots)
                    if DECODER_CONFIG['use_peel_efficient']:
                        log_errors_uf_peel_efficient_list.append(num_errors_uf_peel_efficient_list/num_shots)
                        log_errors_uf_peel_efficient_minweight.append(num_errors_uf_peel_efficient_minweight/num_shots)
                        log_errors_uf_peel_efficient_votemax.append(num_errors_uf_peel_efficient_votemax/num_shots)
                        log_errors_uf_peel_efficient_syndrome.append(num_errors_uf_peel_efficient_syndrome/num_shots)
                    if DECODER_CONFIG.get('use_ablation_baseline'):
                        log_errors_uf_ablation_baseline_votemax.append(num_errors_uf_ablation_baseline_votemax/num_shots)
                    if DECODER_CONFIG.get('use_ablation_mbuffer_only'):
                        log_errors_uf_ablation_mbuffer_only_votemax.append(num_errors_uf_ablation_mbuffer_only_votemax/num_shots)
                    if DECODER_CONFIG.get('use_ablation_dsuopt_only'):
                        log_errors_uf_ablation_dsuopt_only_votemax.append(num_errors_uf_ablation_dsuopt_only_votemax/num_shots)
                    if DECODER_CONFIG.get('use_ablation_graphcompression'):
                        log_errors_uf_ablation_graphcompression_votemax.append(num_errors_uf_ablation_graphcompression_votemax/num_shots)
                    if DECODER_CONFIG.get('use_ablation_growskipping'):
                        log_errors_uf_ablation_growskipping_votemax.append(num_errors_uf_ablation_growskipping_votemax/num_shots)
                    if DECODER_CONFIG['use_bposd']:
                        log_errors_bposd.append(num_errors_bposd/num_shots)
                        log_errors_bposd_dem_graph.append(num_errors_bposd_dem_graph/num_shots)
                        log_errors_bposd_shared_graph.append(num_errors_bposd_shared_graph/num_shots)
                        log_errors_bposd_disagree_rate.append(num_errors_bposd_disagree/num_shots)
                    
                    # 添加当前p值的延迟数据到all_p_values
                    raw_latency_all_p_values.append(raw_latency_for_p_values)
                    raw_latency_all_p_values_peel_efficient.append(raw_latency_for_p_values_peel_efficient)
                    raw_latency_all_p_values_ablation_baseline.append(raw_latency_for_p_values_ablation_baseline)
                    raw_latency_all_p_values_ablation_mbuffer_only.append(raw_latency_for_p_values_ablation_mbuffer_only)
                    raw_latency_all_p_values_ablation_dsuopt_only.append(raw_latency_for_p_values_ablation_dsuopt_only)
                    raw_latency_all_p_values_ablation_graphcompression.append(raw_latency_for_p_values_ablation_graphcompression)
                    raw_latency_all_p_values_ablation_growskipping.append(raw_latency_for_p_values_ablation_growskipping)

            # 保存当前L的结果
            if DECODER_CONFIG['use_mwpm']:
                results['log_errors_all_L_mwpm'].append(np.array(log_errors_mwpm))
                results['log_errors_all_L_mwpm_dem_unweighted'].append(np.array(log_errors_mwpm_dem_unweighted))
                results['log_errors_all_L_mwpm_dem_weighted'].append(np.array(log_errors_mwpm_dem_weighted))
                results['log_errors_all_L_mwpm_hx_manual_unweighted'].append(np.array(log_errors_mwpm_hx_manual_unweighted))
                results['log_errors_all_L_mwpm_disagree_rate'].append(np.array(log_errors_mwpm_disagree_rate))
                results['log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'].append(np.array(log_errors_mwpm_disagree_hx_vs_dem_weighted_rate))
                results['log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'].append(np.array(log_errors_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate))
                results['mwpm_fairness_diff_all_L'].append(np.array(mwpm_fairness_diff))
                results['mwpm_fairness_se_all_L'].append(np.array(mwpm_fairness_se))
                results['mwpm_fairness_z_all_L'].append(np.array(mwpm_fairness_z))
                results['mwpm_fairness_diff_hx_vs_dem_weighted_all_L'].append(np.array(mwpm_fairness_diff_hx_vs_dem_weighted))
                results['mwpm_fairness_se_hx_vs_dem_weighted_all_L'].append(np.array(mwpm_fairness_se_hx_vs_dem_weighted))
                results['mwpm_fairness_z_hx_vs_dem_weighted_all_L'].append(np.array(mwpm_fairness_z_hx_vs_dem_weighted))
                results['mwpm_fairness_diff_dem_unweighted_vs_dem_weighted_all_L'].append(np.array(mwpm_fairness_diff_dem_unweighted_vs_dem_weighted))
                results['mwpm_fairness_se_dem_unweighted_vs_dem_weighted_all_L'].append(np.array(mwpm_fairness_se_dem_unweighted_vs_dem_weighted))
                results['mwpm_fairness_z_dem_unweighted_vs_dem_weighted_all_L'].append(np.array(mwpm_fairness_z_dem_unweighted_vs_dem_weighted))
            if DECODER_CONFIG['use_uf']:
                results['log_errors_all_L_uf'].append(np.array(log_errors_uf))
            if DECODER_CONFIG['use_peel_listdecoding']:
                results['log_errors_all_L_uf_peel_list'].append(np.array(log_errors_uf_peel_list))
                results['log_errors_all_L_uf_peel_minweight'].append(np.array(log_errors_uf_peel_minweight))
                results['log_errors_all_L_uf_peel_votemax'].append(np.array(log_errors_uf_peel_votemax))
                results['log_errors_all_L_uf_peel_syndrome'].append(np.array(log_errors_uf_peel_syndrome))
            if DECODER_CONFIG['use_peel_efficient']:
                results['log_errors_all_L_uf_peel_efficient_list'].append(np.array(log_errors_uf_peel_efficient_list))
                results['log_errors_all_L_uf_peel_efficient_minweight'].append(np.array(log_errors_uf_peel_efficient_minweight))
                results['log_errors_all_L_uf_peel_efficient_votemax'].append(np.array(log_errors_uf_peel_efficient_votemax))
                results['log_errors_all_L_uf_peel_efficient_syndrome'].append(np.array(log_errors_uf_peel_efficient_syndrome))
            if DECODER_CONFIG.get('use_ablation_baseline'):
                results['log_errors_all_L_uf_ablation_baseline_votemax'].append(np.array(log_errors_uf_ablation_baseline_votemax))
            if DECODER_CONFIG.get('use_ablation_mbuffer_only'):
                results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'].append(np.array(log_errors_uf_ablation_mbuffer_only_votemax))
            if DECODER_CONFIG.get('use_ablation_dsuopt_only'):
                results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'].append(np.array(log_errors_uf_ablation_dsuopt_only_votemax))
            if DECODER_CONFIG.get('use_ablation_graphcompression'):
                results['log_errors_all_L_uf_ablation_graphcompression_votemax'].append(np.array(log_errors_uf_ablation_graphcompression_votemax))
            if DECODER_CONFIG.get('use_ablation_growskipping'):
                results['log_errors_all_L_uf_ablation_growskipping_votemax'].append(np.array(log_errors_uf_ablation_growskipping_votemax))
            if DECODER_CONFIG['use_bposd']:
                results['log_errors_all_L_bposd'].append(np.array(log_errors_bposd))
                results['log_errors_all_L_bposd_dem_graph'].append(np.array(log_errors_bposd_dem_graph))
                results['log_errors_all_L_bposd_shared_graph'].append(np.array(log_errors_bposd_shared_graph))
                results['log_errors_all_L_bposd_disagree_rate'].append(np.array(log_errors_bposd_disagree_rate))
            
            # 添加当前 L 的延迟数据
            raw_latency_all_L.append(raw_latency_all_p_values)
            raw_latency_all_L_peel_efficient.append(raw_latency_all_p_values_peel_efficient)
            raw_latency_all_L_ablation_baseline.append(raw_latency_all_p_values_ablation_baseline)
            raw_latency_all_L_ablation_mbuffer_only.append(raw_latency_all_p_values_ablation_mbuffer_only)
            raw_latency_all_L_ablation_dsuopt_only.append(raw_latency_all_p_values_ablation_dsuopt_only)
            raw_latency_all_L_ablation_graphcompression.append(raw_latency_all_p_values_ablation_graphcompression)
            raw_latency_all_L_ablation_growskipping.append(raw_latency_all_p_values_ablation_growskipping)

        results['raw_latency_all_L'] = raw_latency_all_L
        results['raw_latency_all_L_peel_efficient'] = raw_latency_all_L_peel_efficient
        results['raw_latency_all_L_ablation_baseline'] = raw_latency_all_L_ablation_baseline
        results['raw_latency_all_L_ablation_mbuffer_only'] = raw_latency_all_L_ablation_mbuffer_only
        results['raw_latency_all_L_ablation_dsuopt_only'] = raw_latency_all_L_ablation_dsuopt_only
        results['raw_latency_all_L_ablation_graphcompression'] = raw_latency_all_L_ablation_graphcompression
        results['raw_latency_all_L_ablation_growskipping'] = raw_latency_all_L_ablation_growskipping
        return results


    def _process_single_shot(self, L, p, list_size, channel,
                           repetitions, Hx, Hz, code_structure_uf, shot_id):
        """处理单个shot的辅助函数，用于并行化"""
        # 在每个子进程中重置计数器，确保每个shot从零开始统计
        # from operation_counter import reset_counter, get_counter, create_counter
        
            
        q_eff = p
        
        # 使用shot_id设置随机种子，确保每个shot有不同的随机数
        np.random.seed(42 + shot_id)
        
        # 使用统一的噪声生成器接口
        noise_config = DECODER_CONFIG.get('noise_source', 'custom')
        if code_structure_uf.code_type == 'repetition' and noise_config == 'stim':
            noise_config = 'custom'
        
        # 为每个shot创建噪声生成器
        noise_generator = get_noise_generator(
            config_type=noise_config,
            L=L, Hx=Hx, Hz=Hz,
            repetitions=repetitions,
            after_clifford_depolarization=0,
            before_round_data_error_rate=p,
            before_measure_error_rate=q_eff,
            num_shots=1,  # 每个进程只生成一个shot
            code_structure=code_structure_uf,
            channel=channel
        )
        
        # 获取单个shot的数据
        actual_x, actual_z, syndrome_dict_x, syndrome_dict_z, syndrome_array_x, syndrome_array_z, matching = next(noise_generator)

        # 在每个工作进程中重新创建解码器对象（避免pickle BpOsdDecoder失败）
        if channel == 'x':
            H3D_x = _build_multiround_pcm(Hx, repetitions)
            bp_osd_decoder = BpOsdDecoder(
                H3D_x, error_rate=float(p), bp_method='product_sum',
                max_iter=7, schedule='serial', osd_method='osd_cs', osd_order=2
            )
        else:
            H3D_z = _build_multiround_pcm(Hz, repetitions)
            bp_osd_decoder = BpOsdDecoder(
                H3D_z, error_rate=float(p), bp_method='product_sum',
                max_iter=20, schedule='serial', osd_method='osd_cs', osd_order=2
            )

        return _simulate_single_shot_core(
            list_size, channel,
            syndrome_dict_x, syndrome_dict_z,
            syndrome_array_x, syndrome_array_z,
            actual_x, actual_z,
            matching, bp_osd_decoder, code_structure_uf,
            shot_id=shot_id,
        )

    def _pre_generate_stim_data(self, L, ps, num_shots, code_structure, channel):
        """预生成所有p值的stim数据，避免在子进程中重复创建"""
        if DECODER_CONFIG.get('verbose_top', False):
            print("Pre-generating stim data...")
        stim_data = {}
        
        for p in ps:
            if DECODER_CONFIG.get('verbose_top', False):
                print(f"Generating stim data for p={p:.3f}...")
            
            # 确定代码名称
            if code_structure.code_type == 'toric':
                if channel == 'x':
                    code_name = "toric_code:unrotated_memory_x"
                elif channel == 'z':
                    code_name = "toric_code:unrotated_memory_z"
            elif code_structure.code_type == 'planar':
                if channel == 'x':
                    code_name = "surface_code:unrotated_memory_x"
                elif channel == 'z':
                    code_name = "surface_code:unrotated_memory_z"
            elif code_structure.code_type == 'rotated':
                if channel == 'x':
                    code_name = "surface_code:rotated_memory_x"
                elif channel == 'z':
                    code_name = "surface_code:rotated_memory_z"
            else:
                raise ValueError(f"未知 code_type: {code_structure.code_type}")
            
            # 创建电路和采样器（只创建一次）
            circuit = stimcircuits.generate_circuit(code_name,
                                            distance=L,
                                            rounds=code_structure.repetitions,
                                            after_clifford_depolarization=p,
                                            before_round_data_depolarization=p,
                                            after_reset_flip_probability=0,
                                            before_measure_flip_probability=p)
            
            detector_coords = circuit.get_detector_coordinates()
            dem = circuit.detector_error_model(decompose_errors=True)
            sampler = circuit.compile_detector_sampler()
            
            # 一次性采样所有shots
            syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
            
            # 存储数据
            p_key = f"{p:.6f}"  # 使用相同的精度
            stim_data[p_key] = {
                'syndrome': syndrome,
                'actual_observables': actual_observables,
                'detector_coords': detector_coords,
                'code_type': code_structure.code_type,
                'L': int(L),  # 确保L是整数
                'repetitions': int(code_structure.repetitions)  # 确保repetitions是整数
            }
        
        if DECODER_CONFIG.get('verbose_top', False):
            print("Stim data pre-generation completed")
        return stim_data

    def _process_single_shot_with_pre_generated_stim(self, L, p, list_size, channel,
                                                   repetitions, Hx, Hz, logX, logZ, bp_osd_decoder, code_structure_uf, 
                                                   code_structure_peeling, code_structure_efficient, shot_id,
                                                   syndrome_shot, actual_observables_shot, detector_coords, code_type):
        """使用预生成的stim数据处理单个shot"""
        

        q_eff = p
        weight_value = np.log((1 - p) / p) if p > 0 else 1
        timelike_weight = np.log((1 - q_eff) / q_eff) if q_eff > 0 else 1
        if channel == 'x':
            matching = Matching(
                Hx,
                weights=weight_value,
                repetitions=repetitions+1,
                timelike_weights=timelike_weight,
                faults_matrix=logX
            )             
        else:
            matching = Matching(
                Hz,
                weights=weight_value,
                repetitions=repetitions+1,
                timelike_weights=timelike_weight,
                faults_matrix=logZ
            )

        q_eff = p
        
        # 使用shot_id设置随机种子，确保每个shot有不同的随机数
        np.random.seed(42 + shot_id)
        # 从 code_structure 获取校验矩阵尺寸
        Hx = code_structure_uf.H_x
        Hz = code_structure_uf.H_z

        # 构造当前 shot 的观测与综合
        actual_x = actual_observables_shot
        actual_z = actual_observables_shot

        syndrome_dict_x = defaultdict(int)
        syndrome_dict_z = defaultdict(int)

        syndrome_array_x = np.zeros((Hx.shape[0], repetitions+1))
        syndrome_array_z = np.zeros((Hz.shape[0], repetitions+1))

        for j in range(len(detector_coords)):
            if syndrome_shot[j] == 1:
                if code_type == 'rotated':
                    col = detector_coords[j][0]
                    row = detector_coords[j][1]
                    time = detector_coords[j][2]
                    if (col + row) % 2 == 1:
                        syndrome_dict_x[(row, col - 1, time)] = 1
                    else:
                        syndrome_dict_z[(row - 1, col, time)] = 1
                elif code_type == 'toric':
                    col = detector_coords[j][0] // 2
                    row = detector_coords[j][1] // 2
                    time = detector_coords[j][2]
                    if detector_coords[j][1] % 2 == 0:
                        syndrome_dict_x[(row, col, time)] = 1
                        syndrome_array_x[int(col * L + row), int(time)] = 1
                    else:
                        syndrome_dict_z[(row, col, time)] = 1
                        syndrome_array_z[int(col * L + row), int(time)] = 1

        # 调用原有的simulate_single_shot方法
        result = self.simulate_single_shot(
            list_size,
            channel,
            syndrome_dict_x,
            syndrome_dict_z,
            syndrome_array_x,
            syndrome_array_z,
            actual_x,
            actual_z,
            matching,
            bp_osd_decoder,
            code_structure_uf,
            code_structure_peeling,
            code_structure_efficient,
        )
        
        return result

    def run_mwpm_fairness_protocol(self, code_type='toric_code', channel='x', list_size=4, if_repetitions=True,
                                   batch_size=10000, verbose_batches=True):
        """Run the recommended MWPM fairness protocol from DECODER_CONFIG."""
        protocol = DECODER_CONFIG.get('mwpm_fairness_protocol', {})
        Ls = protocol.get('Ls', [3, 5])
        ps = protocol.get('ps', [5e-3, 1e-2])
        num_shots = int(protocol.get('num_shots', 8000))
        key_points_num_shots = int(protocol.get('key_points_num_shots', num_shots))

        # Use fixed p for q to keep the protocol deterministic and easy to reproduce.
        qs = [float(p) for p in ps]
        results = self.run_experiments_batched(
            code_type=code_type,
            Ls=Ls,
            before_round_data_error_rate=ps,
            before_measure_error_rate=qs,
            num_shots=num_shots,
            list_size=list_size,
            channel=channel,
            if_repetitions=if_repetitions,
            batch_size=batch_size,
            verbose_batches=verbose_batches,
        )
        results['mwpm_fairness_protocol'] = {
            'Ls': Ls,
            'ps': ps,
            'num_shots': num_shots,
            'key_points_num_shots': key_points_num_shots,
            'note': 'Primary runs use num_shots; key_points_num_shots is the recommended high-confidence setting.',
        }
        return results

    def run_experiments_batched(self, code_type, Ls, before_round_data_error_rate, 
                                before_measure_error_rate, num_shots=1000, 
                                list_size=4, 
                                channel='x', if_repetitions=True, 
                                batch_size=10000, verbose_batches=True):
        """使用批处理方式运行实验，将大的shot数量分解为小的批次循环执行
        
        Args:
            code_type: 代码类型 ('toric_code', 'surface_code', 'rotated_surface_code')
            Ls: 格子大小列表
            before_round_data_error_rate: 数据错误概率列表
            before_measure_error_rate: 测量错误概率列表
            num_shots: 总的shot数量
            list_size: 候选解数量
            channel: 通道类型 ('x', 'z', 'both')
            if_repetitions: 是否使用重复测量
            batch_size: 每批次的shot数量，建议设置为10000或更小
            verbose_batches: 是否显示详细的批次进度信息（默认True）
        
        Returns:
            dict: 包含所有结果的字典
        """
        # global_time_info = {"Stage 1": [], "Stage 2": [], "Stage 3": [], "Stage 4": []}
        ps = before_round_data_error_rate
        qs = before_measure_error_rate
        
        # 计算需要多少个批次
        num_batches = (num_shots + batch_size - 1) // batch_size  # 向上取整
        actual_batch_size = min(batch_size, num_shots)  # 最后一批可能更小
        
        if verbose_batches:
            print(f"Total number of shots: {num_shots}")
            print(f"Batch size: {batch_size}")
            print(f"Total number of batches: {num_batches}")
            print(f"Last batch size: {num_shots % batch_size if num_shots % batch_size != 0 else batch_size}")
        else:
            print(f"Batch processing configuration: {num_shots} shots → {num_batches} batches × {batch_size}")
        
        # 初始化累计结果
        cumulative_results = {
            'ps': ps,
            'num_shots': num_shots,
            'log_errors_all_L_mwpm': [],
            'log_errors_all_L_mwpm_dem_unweighted': [],
            'log_errors_all_L_mwpm_dem_weighted': [],
            'log_errors_all_L_mwpm_hx_manual_unweighted': [],
            'log_errors_all_L_mwpm_disagree_rate': [],
            'log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate': [],
            'log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate': [],
            'mwpm_fairness_diff_all_L': [],
            'mwpm_fairness_se_all_L': [],
            'mwpm_fairness_z_all_L': [],
            'mwpm_fairness_diff_hx_vs_dem_weighted_all_L': [],
            'mwpm_fairness_se_hx_vs_dem_weighted_all_L': [],
            'mwpm_fairness_z_hx_vs_dem_weighted_all_L': [],
            'mwpm_fairness_diff_dem_unweighted_vs_dem_weighted_all_L': [],
            'mwpm_fairness_se_dem_unweighted_vs_dem_weighted_all_L': [],
            'mwpm_fairness_z_dem_unweighted_vs_dem_weighted_all_L': [],
            'mwpm_fairness_protocol': DECODER_CONFIG.get('mwpm_fairness_protocol', {}),
            'log_errors_all_L_uf': [],
            'log_errors_all_L_uf_peel_list': [],
            'log_errors_all_L_uf_peel_minweight': [],
            'log_errors_all_L_uf_peel_votemax': [],
            'log_errors_all_L_uf_peel_syndrome': [],
            'log_errors_all_L_uf_peel_efficient_list': [],
            'log_errors_all_L_uf_peel_efficient_minweight': [],
            'log_errors_all_L_uf_peel_efficient_votemax': [],
            'log_errors_all_L_uf_peel_efficient_syndrome': [],
            'log_errors_all_L_bposd': [],
            'log_errors_all_L_bposd_dem_graph': [],
            'log_errors_all_L_bposd_shared_graph': [],
            'log_errors_all_L_bposd_disagree_rate': [],
            'raw_latency_all_L': [],
            # ablation errors per L (accumulated across batches)
            'log_errors_all_L_uf_ablation_baseline_votemax': [],
            'log_errors_all_L_uf_ablation_mbuffer_only_votemax': [],
            'log_errors_all_L_uf_ablation_dsuopt_only_votemax': [],
            'log_errors_all_L_uf_ablation_graphcompression_votemax': [],
            'log_errors_all_L_uf_ablation_growskipping_votemax': [],
            # latency variants per L (lists of p → shots dicts)
            'raw_latency_all_L_peel_efficient': [],
            'raw_latency_all_L_ablation_baseline': [],
            'raw_latency_all_L_ablation_mbuffer_only': [],
            'raw_latency_all_L_ablation_dsuopt_only': [],
            'raw_latency_all_L_ablation_graphcompression': [],
            'raw_latency_all_L_ablation_growskipping': [],
        }
        
        # 为每个L初始化结果列表
        for L in Ls:
            cumulative_results['log_errors_all_L_mwpm'].append([])
            cumulative_results['log_errors_all_L_mwpm_dem_unweighted'].append([])
            cumulative_results['log_errors_all_L_mwpm_dem_weighted'].append([])
            cumulative_results['log_errors_all_L_mwpm_hx_manual_unweighted'].append([])
            cumulative_results['log_errors_all_L_mwpm_disagree_rate'].append([])
            cumulative_results['log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'].append([])
            cumulative_results['log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'].append([])
            cumulative_results['mwpm_fairness_diff_all_L'].append([])
            cumulative_results['mwpm_fairness_se_all_L'].append([])
            cumulative_results['mwpm_fairness_z_all_L'].append([])
            cumulative_results['mwpm_fairness_diff_hx_vs_dem_weighted_all_L'].append([])
            cumulative_results['mwpm_fairness_se_hx_vs_dem_weighted_all_L'].append([])
            cumulative_results['mwpm_fairness_z_hx_vs_dem_weighted_all_L'].append([])
            cumulative_results['mwpm_fairness_diff_dem_unweighted_vs_dem_weighted_all_L'].append([])
            cumulative_results['mwpm_fairness_se_dem_unweighted_vs_dem_weighted_all_L'].append([])
            cumulative_results['mwpm_fairness_z_dem_unweighted_vs_dem_weighted_all_L'].append([])
            cumulative_results['log_errors_all_L_uf'].append([])
            cumulative_results['log_errors_all_L_uf_peel_list'].append([])
            cumulative_results['log_errors_all_L_uf_peel_minweight'].append([])
            cumulative_results['log_errors_all_L_uf_peel_votemax'].append([])
            cumulative_results['log_errors_all_L_uf_peel_syndrome'].append([])
            cumulative_results['log_errors_all_L_uf_peel_efficient_list'].append([])
            cumulative_results['log_errors_all_L_uf_peel_efficient_minweight'].append([])
            cumulative_results['log_errors_all_L_uf_peel_efficient_votemax'].append([])
            cumulative_results['log_errors_all_L_uf_peel_efficient_syndrome'].append([])
            cumulative_results['log_errors_all_L_bposd'].append([])
            cumulative_results['log_errors_all_L_bposd_dem_graph'].append([])
            cumulative_results['log_errors_all_L_bposd_shared_graph'].append([])
            cumulative_results['log_errors_all_L_bposd_disagree_rate'].append([])
            # 为每个L值初始化一个包含所有p值的列表
            cumulative_results['raw_latency_all_L'].append([[] for _ in ps])
            # 初始化 ablation 错误率容器
            cumulative_results['log_errors_all_L_uf_ablation_baseline_votemax'].append([])
            cumulative_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'].append([])
            cumulative_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'].append([])
            cumulative_results['log_errors_all_L_uf_ablation_graphcompression_votemax'].append([])
            cumulative_results['log_errors_all_L_uf_ablation_growskipping_votemax'].append([])
            # 初始化各变体的延迟容器
            cumulative_results['raw_latency_all_L_peel_efficient'].append([[] for _ in ps])
            cumulative_results['raw_latency_all_L_ablation_baseline'].append([[] for _ in ps])
            cumulative_results['raw_latency_all_L_ablation_mbuffer_only'].append([[] for _ in ps])
            cumulative_results['raw_latency_all_L_ablation_dsuopt_only'].append([[] for _ in ps])
            cumulative_results['raw_latency_all_L_ablation_graphcompression'].append([[] for _ in ps])
            cumulative_results['raw_latency_all_L_ablation_growskipping'].append([[] for _ in ps])
        
        # 逐批次执行
        # 控制批次打印频率：每固定若干批打印一次（默认1000，可在DECODER_CONFIG['batch_progress_every']中覆盖）
        print_every_batches = int(DECODER_CONFIG.get('batch_progress_every', 1000))
        if print_every_batches <= 0:
            print_every_batches = 1000
        use_progress_bar = bool(DECODER_CONFIG.get('progress_bar', True))
        progress_step_percent = int(DECODER_CONFIG.get('progress_step_percent', 2))
        if progress_step_percent <= 0:
            progress_step_percent = 2
        last_progress_percent = -1
        for batch_idx in range(num_batches):
            current_batch_size = actual_batch_size if batch_idx < num_batches - 1 else (num_shots - batch_idx * batch_size)
            
            if use_progress_bar:
                done = batch_idx
                total = num_batches
                percent = int(done * 100 / total) if total > 0 else 100
                if percent != last_progress_percent and (last_progress_percent < 0 or percent >= last_progress_percent + progress_step_percent or done == 0 or done == total):
                    width = 30
                    filled = int(width * percent / 100)
                    bar = '#' * filled + '-' * (width - filled)
                    print(f"\rBatches: |{bar}| {percent}% ({done}/{total})", end='', flush=True)
                    last_progress_percent = percent
            elif verbose_batches and (((batch_idx + 1) % print_every_batches == 0) or (batch_idx == 0) or (batch_idx == num_batches - 1)):
                print(f"\n=== Executing Batch {batch_idx + 1}/{num_batches} (shots: {current_batch_size}) ===")
            
            # 执行当前批次
            _auto_njobs = int(os.environ.get('UF_N_JOBS', '-1'))
            batch_results = self.run_experiments_parallel(
                code_type, Ls, ps, qs, current_batch_size, list_size,
                channel, if_repetitions, n_jobs=_auto_njobs, parallel_level='shots'
            )
            
            
            # 累积结果
            for L_idx, L in enumerate(Ls):
                # 累积错误率数据
                if DECODER_CONFIG['use_mwpm'] and len(batch_results['log_errors_all_L_mwpm']) > 0:
                    if len(cumulative_results['log_errors_all_L_mwpm'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_mwpm'][L_idx] = batch_results['log_errors_all_L_mwpm'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_mwpm'][L_idx] += batch_results['log_errors_all_L_mwpm'][L_idx] * current_batch_size
                if DECODER_CONFIG['use_mwpm'] and len(batch_results.get('log_errors_all_L_mwpm_dem_unweighted', [])) > 0:
                    if len(cumulative_results['log_errors_all_L_mwpm_dem_unweighted'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_mwpm_dem_unweighted'][L_idx] = (
                            batch_results['log_errors_all_L_mwpm_dem_unweighted'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_dem_weighted'][L_idx] = (
                            batch_results['log_errors_all_L_mwpm_dem_weighted'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_hx_manual_unweighted'][L_idx] = (
                            batch_results['log_errors_all_L_mwpm_hx_manual_unweighted'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_disagree_rate'][L_idx] = (
                            batch_results['log_errors_all_L_mwpm_disagree_rate'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'][L_idx] = (
                            batch_results['log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'][L_idx] = (
                            batch_results['log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'][L_idx] * current_batch_size
                        )
                    else:
                        cumulative_results['log_errors_all_L_mwpm_dem_unweighted'][L_idx] += (
                            batch_results['log_errors_all_L_mwpm_dem_unweighted'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_dem_weighted'][L_idx] += (
                            batch_results['log_errors_all_L_mwpm_dem_weighted'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_hx_manual_unweighted'][L_idx] += (
                            batch_results['log_errors_all_L_mwpm_hx_manual_unweighted'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_disagree_rate'][L_idx] += (
                            batch_results['log_errors_all_L_mwpm_disagree_rate'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'][L_idx] += (
                            batch_results['log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'][L_idx] * current_batch_size
                        )
                        cumulative_results['log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'][L_idx] += (
                            batch_results['log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'][L_idx] * current_batch_size
                        )
                
                if DECODER_CONFIG['use_uf'] and len(batch_results['log_errors_all_L_uf']) > 0:
                    if len(cumulative_results['log_errors_all_L_uf'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_uf'][L_idx] = batch_results['log_errors_all_L_uf'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_uf'][L_idx] += batch_results['log_errors_all_L_uf'][L_idx] * current_batch_size
                
                if DECODER_CONFIG['use_peel_listdecoding'] and len(batch_results['log_errors_all_L_uf_peel_list']) > 0:
                    if len(cumulative_results['log_errors_all_L_uf_peel_list'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_uf_peel_list'][L_idx] = batch_results['log_errors_all_L_uf_peel_list'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_minweight'][L_idx] = batch_results['log_errors_all_L_uf_peel_minweight'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_votemax'][L_idx] = batch_results['log_errors_all_L_uf_peel_votemax'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_syndrome'][L_idx] = batch_results['log_errors_all_L_uf_peel_syndrome'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_uf_peel_list'][L_idx] += batch_results['log_errors_all_L_uf_peel_list'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_minweight'][L_idx] += batch_results['log_errors_all_L_uf_peel_minweight'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_votemax'][L_idx] += batch_results['log_errors_all_L_uf_peel_votemax'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_syndrome'][L_idx] += batch_results['log_errors_all_L_uf_peel_syndrome'][L_idx] * current_batch_size
                
                if DECODER_CONFIG['use_peel_efficient'] and len(batch_results['log_errors_all_L_uf_peel_efficient_list']) > 0:
                    if len(cumulative_results['log_errors_all_L_uf_peel_efficient_list'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_uf_peel_efficient_list'][L_idx] = batch_results['log_errors_all_L_uf_peel_efficient_list'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_efficient_minweight'][L_idx] = batch_results['log_errors_all_L_uf_peel_efficient_minweight'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_efficient_votemax'][L_idx] = batch_results['log_errors_all_L_uf_peel_efficient_votemax'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_efficient_syndrome'][L_idx] = batch_results['log_errors_all_L_uf_peel_efficient_syndrome'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_uf_peel_efficient_list'][L_idx] += batch_results['log_errors_all_L_uf_peel_efficient_list'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_efficient_minweight'][L_idx] += batch_results['log_errors_all_L_uf_peel_efficient_minweight'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_efficient_votemax'][L_idx] += batch_results['log_errors_all_L_uf_peel_efficient_votemax'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_uf_peel_efficient_syndrome'][L_idx] += batch_results['log_errors_all_L_uf_peel_efficient_syndrome'][L_idx] * current_batch_size
                # 累积 ablation 错误率
                if DECODER_CONFIG.get('use_ablation_baseline') and 'log_errors_all_L_uf_ablation_baseline_votemax' in batch_results and len(batch_results['log_errors_all_L_uf_ablation_baseline_votemax']) > 0:
                    if len(cumulative_results['log_errors_all_L_uf_ablation_baseline_votemax'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_uf_ablation_baseline_votemax'][L_idx] = batch_results['log_errors_all_L_uf_ablation_baseline_votemax'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_uf_ablation_baseline_votemax'][L_idx] += batch_results['log_errors_all_L_uf_ablation_baseline_votemax'][L_idx] * current_batch_size
                if DECODER_CONFIG.get('use_ablation_mbuffer_only') and 'log_errors_all_L_uf_ablation_mbuffer_only_votemax' in batch_results and len(batch_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax']) > 0:
                    if len(cumulative_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'][L_idx] = batch_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'][L_idx] += batch_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'][L_idx] * current_batch_size
                if DECODER_CONFIG.get('use_ablation_dsuopt_only') and 'log_errors_all_L_uf_ablation_dsuopt_only_votemax' in batch_results and len(batch_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax']) > 0:
                    if len(cumulative_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'][L_idx] = batch_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'][L_idx] += batch_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'][L_idx] * current_batch_size
                if DECODER_CONFIG.get('use_ablation_graphcompression') and 'log_errors_all_L_uf_ablation_graphcompression_votemax' in batch_results and len(batch_results['log_errors_all_L_uf_ablation_graphcompression_votemax']) > 0:
                    if len(cumulative_results['log_errors_all_L_uf_ablation_graphcompression_votemax'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_uf_ablation_graphcompression_votemax'][L_idx] = batch_results['log_errors_all_L_uf_ablation_graphcompression_votemax'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_uf_ablation_graphcompression_votemax'][L_idx] += batch_results['log_errors_all_L_uf_ablation_graphcompression_votemax'][L_idx] * current_batch_size
                if DECODER_CONFIG.get('use_ablation_growskipping') and 'log_errors_all_L_uf_ablation_growskipping_votemax' in batch_results and len(batch_results['log_errors_all_L_uf_ablation_growskipping_votemax']) > 0:
                    if len(cumulative_results['log_errors_all_L_uf_ablation_growskipping_votemax'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_uf_ablation_growskipping_votemax'][L_idx] = batch_results['log_errors_all_L_uf_ablation_growskipping_votemax'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_uf_ablation_growskipping_votemax'][L_idx] += batch_results['log_errors_all_L_uf_ablation_growskipping_votemax'][L_idx] * current_batch_size
                
                if DECODER_CONFIG['use_bposd'] and len(batch_results['log_errors_all_L_bposd']) > 0:
                    if len(cumulative_results['log_errors_all_L_bposd'][L_idx]) == 0:
                        cumulative_results['log_errors_all_L_bposd'][L_idx] = batch_results['log_errors_all_L_bposd'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_bposd_dem_graph'][L_idx] = batch_results['log_errors_all_L_bposd_dem_graph'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_bposd_shared_graph'][L_idx] = batch_results['log_errors_all_L_bposd_shared_graph'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_bposd_disagree_rate'][L_idx] = batch_results['log_errors_all_L_bposd_disagree_rate'][L_idx] * current_batch_size
                    else:
                        cumulative_results['log_errors_all_L_bposd'][L_idx] += batch_results['log_errors_all_L_bposd'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_bposd_dem_graph'][L_idx] += batch_results['log_errors_all_L_bposd_dem_graph'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_bposd_shared_graph'][L_idx] += batch_results['log_errors_all_L_bposd_shared_graph'][L_idx] * current_batch_size
                        cumulative_results['log_errors_all_L_bposd_disagree_rate'][L_idx] += batch_results['log_errors_all_L_bposd_disagree_rate'][L_idx] * current_batch_size
                
                # 累积延迟数据 - 直接累积batch_results的结构
                if 'raw_latency_all_L' in batch_results and batch_results['raw_latency_all_L']:
                    if L_idx < len(batch_results['raw_latency_all_L']):
                        batch_latency_for_current_L = batch_results['raw_latency_all_L'][L_idx]
                        if batch_latency_for_current_L:  # 如果当前L值有延迟数据
                            # batch_results的结构是[L_idx][p_idx]，直接按p_idx累积
                            for p_idx, p in enumerate(ps):
                                if p_idx < len(batch_latency_for_current_L) and batch_latency_for_current_L[p_idx]:
                                    # 直接累积batch中的延迟数据到对应的p_idx
                                    cumulative_results['raw_latency_all_L'][L_idx][p_idx].extend(batch_latency_for_current_L[p_idx])
                # 累积各变体的延迟数据
                for key_src, key_dst in [
                    ('raw_latency_all_L_peel_efficient', 'raw_latency_all_L_peel_efficient'),
                    ('raw_latency_all_L_ablation_baseline', 'raw_latency_all_L_ablation_baseline'),
                    ('raw_latency_all_L_ablation_mbuffer_only', 'raw_latency_all_L_ablation_mbuffer_only'),
                    ('raw_latency_all_L_ablation_dsuopt_only', 'raw_latency_all_L_ablation_dsuopt_only'),
                    ('raw_latency_all_L_ablation_graphcompression', 'raw_latency_all_L_ablation_graphcompression'),
                    ('raw_latency_all_L_ablation_growskipping', 'raw_latency_all_L_ablation_growskipping'),
                ]:
                    if key_src in batch_results and batch_results[key_src]:
                        if L_idx < len(batch_results[key_src]):
                            batch_latency_L = batch_results[key_src][L_idx]
                            if batch_latency_L:
                                for p_idx, p in enumerate(ps):
                                    if p_idx < len(batch_latency_L) and batch_latency_L[p_idx]:
                                        cumulative_results[key_dst][L_idx][p_idx].extend(batch_latency_L[p_idx])
                
            # 清理内存
            del batch_results
            gc.collect()
            
            if use_progress_bar:
                done = batch_idx + 1
                total = num_batches
                percent = int(done * 100 / total) if total > 0 else 100
                if percent != last_progress_percent and (percent >= last_progress_percent + progress_step_percent or done == total):
                    width = 30
                    filled = int(width * percent / 100)
                    bar = '#' * filled + '-' * (width - filled)
                    print(f"\rBatches: |{bar}| {percent}% ({done}/{total})", end='', flush=True)
                    last_progress_percent = percent
                if done == total:
                    print()
            elif verbose_batches and (((batch_idx + 1) % print_every_batches == 0) or (batch_idx == num_batches - 1)):
                print(f"Batch {batch_idx + 1} completed, processed {min((batch_idx + 1) * batch_size, num_shots)}/{num_shots} shots")
        
        # 计算最终的错误率（除以总shot数量）
        if verbose_batches:
            print("\n=== Calculating Final Results ===")
        else:
            print("\nCalculating final results...")
        for L_idx, L in enumerate(Ls):
            if DECODER_CONFIG['use_mwpm'] and len(cumulative_results['log_errors_all_L_mwpm'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_mwpm'][L_idx] = np.array(cumulative_results['log_errors_all_L_mwpm'][L_idx]) / num_shots
            if DECODER_CONFIG['use_mwpm'] and len(cumulative_results['log_errors_all_L_mwpm_dem_unweighted'][L_idx]) > 0:
                dem_rates = np.array(cumulative_results['log_errors_all_L_mwpm_dem_unweighted'][L_idx]) / num_shots
                dem_weighted_rates = np.array(cumulative_results['log_errors_all_L_mwpm_dem_weighted'][L_idx]) / num_shots
                hx_rates = np.array(cumulative_results['log_errors_all_L_mwpm_hx_manual_unweighted'][L_idx]) / num_shots
                disagree_rates = np.array(cumulative_results['log_errors_all_L_mwpm_disagree_rate'][L_idx]) / num_shots
                disagree_hx_demw_rates = np.array(cumulative_results['log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'][L_idx]) / num_shots
                disagree_demu_demw_rates = np.array(cumulative_results['log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'][L_idx]) / num_shots
                diff = hx_rates - dem_rates
                se_diff = np.sqrt(
                    dem_rates * (1 - dem_rates) / num_shots +
                    hx_rates * (1 - hx_rates) / num_shots
                )
                z_scores = np.divide(diff, se_diff, out=np.zeros_like(diff), where=se_diff > 0)
                diff_hx_demw = hx_rates - dem_weighted_rates
                se_hx_demw = np.sqrt(
                    dem_weighted_rates * (1 - dem_weighted_rates) / num_shots +
                    hx_rates * (1 - hx_rates) / num_shots
                )
                z_hx_demw = np.divide(diff_hx_demw, se_hx_demw, out=np.zeros_like(diff_hx_demw), where=se_hx_demw > 0)
                diff_demu_demw = dem_rates - dem_weighted_rates
                se_demu_demw = np.sqrt(
                    dem_weighted_rates * (1 - dem_weighted_rates) / num_shots +
                    dem_rates * (1 - dem_rates) / num_shots
                )
                z_demu_demw = np.divide(diff_demu_demw, se_demu_demw, out=np.zeros_like(diff_demu_demw), where=se_demu_demw > 0)
                cumulative_results['log_errors_all_L_mwpm_dem_unweighted'][L_idx] = dem_rates
                cumulative_results['log_errors_all_L_mwpm_dem_weighted'][L_idx] = dem_weighted_rates
                cumulative_results['log_errors_all_L_mwpm_hx_manual_unweighted'][L_idx] = hx_rates
                cumulative_results['log_errors_all_L_mwpm_disagree_rate'][L_idx] = disagree_rates
                cumulative_results['log_errors_all_L_mwpm_disagree_hx_vs_dem_weighted_rate'][L_idx] = disagree_hx_demw_rates
                cumulative_results['log_errors_all_L_mwpm_disagree_dem_unweighted_vs_dem_weighted_rate'][L_idx] = disagree_demu_demw_rates
                cumulative_results['mwpm_fairness_diff_all_L'][L_idx] = diff
                cumulative_results['mwpm_fairness_se_all_L'][L_idx] = se_diff
                cumulative_results['mwpm_fairness_z_all_L'][L_idx] = z_scores
                cumulative_results['mwpm_fairness_diff_hx_vs_dem_weighted_all_L'][L_idx] = diff_hx_demw
                cumulative_results['mwpm_fairness_se_hx_vs_dem_weighted_all_L'][L_idx] = se_hx_demw
                cumulative_results['mwpm_fairness_z_hx_vs_dem_weighted_all_L'][L_idx] = z_hx_demw
                cumulative_results['mwpm_fairness_diff_dem_unweighted_vs_dem_weighted_all_L'][L_idx] = diff_demu_demw
                cumulative_results['mwpm_fairness_se_dem_unweighted_vs_dem_weighted_all_L'][L_idx] = se_demu_demw
                cumulative_results['mwpm_fairness_z_dem_unweighted_vs_dem_weighted_all_L'][L_idx] = z_demu_demw
            
            if DECODER_CONFIG['use_uf'] and len(cumulative_results['log_errors_all_L_uf'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_uf'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf'][L_idx]) / num_shots
            
            if DECODER_CONFIG['use_peel_listdecoding'] and len(cumulative_results['log_errors_all_L_uf_peel_list'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_uf_peel_list'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_peel_list'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_uf_peel_minweight'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_peel_minweight'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_uf_peel_votemax'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_peel_votemax'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_uf_peel_syndrome'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_peel_syndrome'][L_idx]) / num_shots
               
            if DECODER_CONFIG['use_peel_efficient'] and len(cumulative_results['log_errors_all_L_uf_peel_efficient_list'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_uf_peel_efficient_list'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_peel_efficient_list'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_uf_peel_efficient_minweight'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_peel_efficient_minweight'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_uf_peel_efficient_votemax'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_peel_efficient_votemax'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_uf_peel_efficient_syndrome'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_peel_efficient_syndrome'][L_idx]) / num_shots
            
            if DECODER_CONFIG['use_bposd'] and len(cumulative_results['log_errors_all_L_bposd'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_bposd'][L_idx] = np.array(cumulative_results['log_errors_all_L_bposd'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_bposd_dem_graph'][L_idx] = np.array(cumulative_results['log_errors_all_L_bposd_dem_graph'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_bposd_shared_graph'][L_idx] = np.array(cumulative_results['log_errors_all_L_bposd_shared_graph'][L_idx]) / num_shots
                cumulative_results['log_errors_all_L_bposd_disagree_rate'][L_idx] = np.array(cumulative_results['log_errors_all_L_bposd_disagree_rate'][L_idx]) / num_shots
            # 归一化 ablation 错误率
            if DECODER_CONFIG.get('use_ablation_baseline') and len(cumulative_results['log_errors_all_L_uf_ablation_baseline_votemax'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_uf_ablation_baseline_votemax'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_ablation_baseline_votemax'][L_idx]) / num_shots
            if DECODER_CONFIG.get('use_ablation_mbuffer_only') and len(cumulative_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_ablation_mbuffer_only_votemax'][L_idx]) / num_shots
            if DECODER_CONFIG.get('use_ablation_dsuopt_only') and len(cumulative_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_ablation_dsuopt_only_votemax'][L_idx]) / num_shots
            if DECODER_CONFIG.get('use_ablation_graphcompression') and len(cumulative_results['log_errors_all_L_uf_ablation_graphcompression_votemax'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_uf_ablation_graphcompression_votemax'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_ablation_graphcompression_votemax'][L_idx]) / num_shots
            if DECODER_CONFIG.get('use_ablation_growskipping') and len(cumulative_results['log_errors_all_L_uf_ablation_growskipping_votemax'][L_idx]) > 0:
                cumulative_results['log_errors_all_L_uf_ablation_growskipping_votemax'][L_idx] = np.array(cumulative_results['log_errors_all_L_uf_ablation_growskipping_votemax'][L_idx]) / num_shots
        
        if verbose_batches:
            print("Batch processing completed!")
        else:
            print("Batch processing completed!")
        return cumulative_results
