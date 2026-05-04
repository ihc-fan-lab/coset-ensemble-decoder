import os
import sys
import argparse

sys.setrecursionlimit(10000)
import json
from typing import Any, Dict, List

sys.setrecursionlimit(100000)

import numpy as np

# Ensure imports work when this script is run from `rebuttal/`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from uf_test_utils import UFTester
from plotting.ablation_plots import plot_and_save_all


def parse_list_of_ints(value: str) -> List[int]:
    return [int(x) for x in value.split(',') if x.strip()]


def parse_list_of_floats(value: str) -> List[float]:
    return [float(x) for x in value.split(',') if x.strip()]


def parse_list_of_ints_flexible(tokens: List[str]) -> List[int]:
    """
    Parse integer lists from either:
    - one comma-separated token: ["500000,300000,200000"]
    - spaced comma tokens: ["500000,", "300000,", "200000"]
    - plain spaced tokens: ["500000", "300000", "200000"]
    """
    merged = ",".join(tokens)
    return [int(x.strip()) for x in merged.split(",") if x.strip()]


def _extract_first_numeric_or_zero(value) -> float:
    """Return first numeric entry from list/ndarray/scalar; fallback to 0.0."""
    if value is None:
        return 0.0
    if np.isscalar(value):
        try:
            return float(value)
        except Exception:
            return 0.0
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return 0.0
        return float(arr.reshape(-1)[0])
    except Exception:
        return 0.0


def _jsonable(obj):
    """Recursively convert numpy types to JSON-serializable python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Run UF experiments (batched) and generate plots with parameterized inputs.")
    parser.add_argument('--code_type', type=str, default='toric_code', help="Code type: toric_code | surface_code | rotated_surface_code")
    parser.add_argument('--Ls', type=parse_list_of_ints, required=True, help="Comma-separated list of code distances, e.g., 3,5,7")
    parser.add_argument('--ps', type=parse_list_of_floats, required=True, help="Comma-separated list of physical error rates, e.g., 0.001,0.00125,0.0015")
    parser.add_argument('--num_shots', type=int, default=10000, help="Total number of shots")
    parser.add_argument(
        '--num_shots_per_p',
        nargs='+',
        type=str,
        default=None,
        help="Optional per-p shots, e.g., 200000,120000,80000 (must match --ps length)",
    )
    parser.add_argument('--list_size', type=int, default=24, help="Candidate list size")
    parser.add_argument('--channel', type=str, default='x', help="Channel: x | z | both")
    parser.add_argument('--if_repetitions', action='store_true', help="Use repetitions = L if set (default False)")
    parser.add_argument('--batch_size', type=int, default=10000, help="Shots per batch for batched execution")
    parser.add_argument('--verbose_batches', action='store_true', help="Print verbose batch progress")
    parser.add_argument('--output_root', type=str, default='analysis_outputs', help="Directory to write plots and aggregated data")
    parser.add_argument('--output_format', type=str, default='pdf', choices=['png', 'pdf', 'svg'], help="Figure file format")
    parser.add_argument('--with_error_bars', action='store_true', help="Plot LER error bars using binomial standard error")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint when --num_shots_per_p is used")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Optional checkpoint path override")

    args = parser.parse_args()

    code_type = args.code_type
    Ls = args.Ls
    ps = args.ps
    num_shots = int(args.num_shots)
    list_size = args.list_size
    channel = args.channel
    if_repetitions = bool(args.if_repetitions)
    batch_size = args.batch_size
    verbose_batches = bool(args.verbose_batches)
    output_format = args.output_format
    with_error_bars = bool(args.with_error_bars)

    # Build output directory that encodes distances and list_size.
    if len(Ls) == 1:
        L_tag = f"L{Ls[0]}"
    else:
        L_tag = "Ls_" + "-".join(str(L) for L in Ls)
    out_dir = os.path.join(args.output_root, f"{L_tag}_list{list_size}")
    os.makedirs(out_dir, exist_ok=True)

    # Prepare tester
    tester = UFTester(save_dir='savedata')

    if args.num_shots_per_p is not None and len(args.num_shots_per_p) > 0:
        shots_per_p = parse_list_of_ints_flexible(args.num_shots_per_p)
        if len(shots_per_p) != len(ps):
            raise ValueError("--num_shots_per_p length must equal --ps length.")
        if any(v <= 0 for v in shots_per_p):
            raise ValueError("--num_shots_per_p must be positive integers.")

        # Run each p with its own shot budget, then stitch results.
        decoder_result_keys = [
            "log_errors_all_L_mwpm",
            "log_errors_all_L_uf",
            "log_errors_all_L_uf_peel_list",
            "log_errors_all_L_uf_peel_votemax",
            "log_errors_all_L_uf_peel_minweight",
            "log_errors_all_L_uf_peel_syndrome",
            "log_errors_all_L_uf_peel_efficient_list",
            "log_errors_all_L_uf_peel_efficient_votemax",
            "log_errors_all_L_uf_peel_efficient_minweight",
            "log_errors_all_L_uf_peel_efficient_syndrome",
            "log_errors_all_L_bposd",
            "log_errors_all_L_ablation_baseline",
            "log_errors_all_L_ablation_mbuffer_only",
            "log_errors_all_L_ablation_dsuopt_only",
            "log_errors_all_L_ablation_graphcompression",
            "log_errors_all_L_ablation_growskipping",
            "raw_latency_all_L",
            "raw_latency_all_L_peel_efficient",
            "raw_latency_all_L_ablation_baseline",
            "raw_latency_all_L_ablation_mbuffer_only",
            "raw_latency_all_L_ablation_dsuopt_only",
            "raw_latency_all_L_ablation_graphcompression",
            "raw_latency_all_L_ablation_growskipping",
        ]
        checkpoint_path = args.checkpoint_path or os.path.join(
            out_dir, f"checkpoint_{L_tag}_list{list_size}.json"
        )
        results = {k: [[] for _ in Ls] for k in decoder_result_keys}
        start_idx = 0
        if args.resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                meta = ckpt.get("meta", {})
                if (
                    meta.get("code_type") == code_type
                    and meta.get("Ls") == [int(x) for x in Ls]
                    and np.allclose(meta.get("ps", []), ps)
                    and meta.get("shots_per_p") == [int(x) for x in shots_per_p]
                    and int(meta.get("list_size", -1)) == int(list_size)
                    and meta.get("channel") == channel
                    and bool(meta.get("if_repetitions")) == bool(if_repetitions)
                ):
                    loaded_results = ckpt.get("results", {})
                    if isinstance(loaded_results, dict):
                        results.update(loaded_results)
                    start_idx = int(ckpt.get("completed_p_count", 0))
                    start_idx = max(0, min(start_idx, len(ps)))
                    print(f"[resume] Loaded checkpoint: {checkpoint_path} (completed {start_idx}/{len(ps)} p points)")
                else:
                    print(f"[resume] Checkpoint metadata mismatch, starting fresh: {checkpoint_path}")
            except Exception as e:
                print(f"[resume] Failed to load checkpoint ({e}), starting fresh.")

        for p_idx in range(start_idx, len(ps)):
            p = ps[p_idx]
            shots = shots_per_p[p_idx]
            print(f"[progress] Running p={p} ({p_idx + 1}/{len(ps)}), shots={shots}")
            one = tester.run_experiments_batched(
                code_type, Ls, [p],
                [p],  # before_measure_error_rate uses same p
                num_shots=int(shots),
                list_size=list_size,
                channel=channel,
                if_repetitions=if_repetitions,
                batch_size=batch_size,
                verbose_batches=verbose_batches,
            )
            for key in decoder_result_keys:
                arr_l = one.get(key, [])
                for l_idx in range(len(Ls)):
                    vals = arr_l[l_idx] if l_idx < len(arr_l) else []
                    if key.startswith("raw_latency_all_L"):
                        # Keep per-p raw shot stats structure expected by plotting:
                        # results[key][l_idx][p_idx] -> list[dict]
                        if isinstance(vals, list) and len(vals) > 0:
                            per_p_raw = vals[0]
                            if isinstance(per_p_raw, list):
                                results[key][l_idx].append(per_p_raw)
                            else:
                                results[key][l_idx].append([])
                        else:
                            results[key][l_idx].append([])
                    else:
                        results[key][l_idx].append(_extract_first_numeric_or_zero(vals))
            if args.resume:
                checkpoint_payload = {
                    "meta": {
                        "code_type": code_type,
                        "Ls": [int(x) for x in Ls],
                        "ps": [float(x) for x in ps],
                        "shots_per_p": [int(x) for x in shots_per_p],
                        "list_size": int(list_size),
                        "channel": channel,
                        "if_repetitions": bool(if_repetitions),
                    },
                    "completed_p_count": int(p_idx + 1),
                    "results": results,
                }
                _save_checkpoint(checkpoint_path, checkpoint_payload)
                print(f"[resume] Checkpoint updated ({p_idx + 1}/{len(ps)}): {checkpoint_path}")
        # Keep scalar num_shots for downstream plotting API compatibility.
        num_shots_for_plot = int(max(shots_per_p))
    else:
        results = tester.run_experiments_batched(
            code_type, Ls, ps,
            ps,  # before_measure_error_rate uses same ps as notebook
            num_shots=num_shots,
            list_size=list_size,
            channel=channel,
            if_repetitions=if_repetitions,
            batch_size=batch_size,
            verbose_batches=verbose_batches,
        )
        num_shots_for_plot = num_shots

    # Generate plots + aggregated data (L_index=0 by default)
    plot_and_save_all(
        results, Ls, ps,
        L_index=0,
        output_root=out_dir,
        file_ext=output_format,
        include_error_bars=with_error_bars,
        num_shots=num_shots_for_plot,
    )

    # Rename output files to include list_size (and L for data json) in file names
    # Figures produced: fig1..fig6 with suffix _L{L}.{ext}; add _list{list_size}
    if Ls:
        L0 = Ls[0]
        name_map = {
            1: 'ler_mwpm_uf_ufe',
            2: 'ler_mwpm_uf_peel',
            3: 'ler_efficient_ablations',
            4: 'latency_efficient_ablations',
            5: 'ops_mbuffer_only_stages',  # updated fig5 name
            6: 'fidelity_mwpm_uf_efficient',
        }
        ext = output_format
        for i in range(1, 7):
            base = name_map[i]
            src = os.path.join(out_dir, f"fig{i}_{base}_L{L0}.{ext}")
            if os.path.exists(src):
                dst = os.path.join(out_dir, f"fig{i}_{base}_L{L0}_list{list_size}.{ext}")
                try:
                    os.replace(src, dst)
                except Exception:
                    pass
        # Rename plots_data.json as well
        data_src = os.path.join(out_dir, 'plots_data.json')
        data_dst = os.path.join(out_dir, f'plots_data_L{L0}_list{list_size}.json')
        if os.path.exists(data_src):
            try:
                os.replace(data_src, data_dst)
            except Exception:
                pass

    print(f"Done. Outputs written under: {out_dir}")


if __name__ == '__main__':
    main()


