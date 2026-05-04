"""
Shared constants, tables, functions and style settings for all rebuttal notebooks.
Extracted from compose_plots.ipynb to allow each notebook to run independently.
"""

import os
import json
import glob
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogFormatterSciNotation
from matplotlib import patheffects as pe

__all__ = [
    # re-exported modules & types
    'os', 'json', 'glob', 'np', 'plt', 'pe',
    'Dict', 'List', 'Any', 'Optional', 'Tuple',
    'MaxNLocator', 'LogFormatterSciNotation',
    # paths
    'TORIC_CODE_ROOT', 'L_TARGETS', 'FIGPLOT_DIR',
    # constants
    'SCALE_MICRO_BLOSSOM', 'SCALE_HELIOS', 'SCALE_TOTAL_TO_CYCLES', 'D_EXPONENT',
    'PLOT_AS_SCATTER', 'SCATTER_SIZE', 'MARKER_EDGE_WIDTH', 'X_JITTER',
    # tables
    '_HELIOS_TABLE', '_MICRO_BLOSSOM_TABLE',
    # palettes
    'palette',
    # style
    'setup_plot_style',
    # cycles & fidelity
    'get_helios_cycles', 'get_micro_blossom_cycles', 'compute_fidelity', 'calc_infid',
    # data loading
    '_load_latest_plots_data_for_L', 'find_plots_data_under',
    'find_plots_data_for_targets', 'load_toric_code_data',
    # plot helpers
    'get_series', 'get_err_series', '_safe_err', 'light_xticks', 'ensure_dir',
    '_finite_positive', '_set_cluster_ylim', '_scatter_with_oob',
]

# ============================================================
# Paths & Targets
# ============================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TORIC_CODE_ROOT = os.path.join(_THIS_DIR, "rebuttal_outputs", "toric_code")
L_TARGETS = [3, 5, 7, 9, 11]
FIGPLOT_DIR = os.path.join(_THIS_DIR, "figplot")

# ============================================================
# Constants
# ============================================================
SCALE_MICRO_BLOSSOM = 0.0232
SCALE_HELIOS = 0.0133
SCALE_TOTAL_TO_CYCLES = 0.0061
D_EXPONENT = 1

PLOT_AS_SCATTER = True
SCATTER_SIZE = 160
MARKER_EDGE_WIDTH = 1.8
X_JITTER = 1e-5

# ============================================================
# Tables
# ============================================================
_HELIOS_TABLE = {
    3:  {0.0005: 11.1527, 0.00075: 11.2768, 0.001: 11.3173, 0.00125: 11.4417, 0.0015: 11.4872},
    5:  {0.0005: 11.9882, 0.00075: 12.4421, 0.001: 12.9897, 0.00125: 13.5605, 0.0015: 13.9666},
    7:  {0.0005: 14.398,  0.00075: 16.2252, 0.001: 17.6502, 0.00125: 19.2076, 0.0015: 20.7541},
    9:  {0.0005: 19.4252, 0.00075: 23.245,  0.001: 26.3648, 0.00125: 30.2762, 0.0015: 33.038},
    11: {0.0005: 27.0971, 0.00075: 33.5048, 0.001: 39.7536, 0.00125: 44.6698, 0.0015: 48.9028},
}

_MICRO_BLOSSOM_TABLE = {
    3:  {0.0005: 5.6302,   0.00075: 6.3692,   0.001: 7.3894,   0.00125: 8.0564,   0.0015: 9.0906},
    5:  {0.0005: 13.1732,  0.00075: 17.6634,  0.001: 23.1238,  0.00125: 27.4426,  0.0015: 33.1738},
    7:  {0.0005: 31.4794,  0.00075: 45.1838,  0.001: 59.2154,  0.00125: 73.0846,  0.0015: 86.0702},
    9:  {0.0005: 62.9546,  0.00075: 92.9466,  0.001: 121.5564, 0.00125: 151.5874, 0.0015: 181.8154},
    11: {0.0005: 112.723,  0.00075: 166.1346, 0.001: 219.101,  0.00125: 274.604,  0.0015: 331.2032},
}

# ============================================================
# Palettes
# ============================================================
palette = {
    'mwpm': '#1f77b4',
    'uf': '#ff7f0e',
    'hardware': '#2ca02c',
    'peel': '#d62728',
    'baseline': '#1f77b4',
    'dsuopt': '#9467bd',
    'mbuffer': '#ff7f0e',
    'growskipping': '#8c564b',
    'graphcompression': '#d62728',
}

# ============================================================
# Style
# ============================================================
def setup_plot_style():
    """Apply ISCA/HPCA-style rcParams."""
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'lines.linewidth': 3.0,
        'lines.markersize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
    })

# ============================================================
# Functions: Cycles & Fidelity
# ============================================================
def get_helios_cycles(L_val: int, p: float) -> float:
    table = _HELIOS_TABLE.get(L_val, {})
    return float(table.get(p, 20.0))


def get_micro_blossom_cycles(L_val: int, p: float) -> float:
    table = _MICRO_BLOSSOM_TABLE.get(L_val, {})
    return float(table.get(p, 50.0))


def compute_fidelity(ler_val: float, latency: float, d: int = 1) -> float:
    if ler_val <= 0.0:
        return 1.0
    if latency > d:
        system_fidelity = (1.0 - 2.0 * ler_val) ** (latency / float(d))
    else:
        system_fidelity = (1.0 - 2.0 * ler_val)
    return max(0.0, system_fidelity)


def calc_infid(ler_arr: List[float], cycles_arr: List[float], d: int = D_EXPONENT) -> List[float]:
    return [1.0 - compute_fidelity(float(l), float(c), d=d) for l, c in zip(ler_arr, cycles_arr)]

# ============================================================
# Functions: Data Loading
# ============================================================
def _load_latest_plots_data_for_L(root: str, L: int) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    candidates.extend(glob.glob(os.path.join(root, f"L{L}_*", "plots_data*.json")))
    if not candidates:
        candidates.extend(glob.glob(os.path.join(root, "**", f"plots_data_L{L}_*.json"), recursive=True))
        candidates.extend(glob.glob(os.path.join(root, "**", "plots_data*.json"), recursive=True))
    if not candidates:
        return None
    items: List[Tuple[float, str]] = []
    for path in candidates:
        try:
            with open(path, "r") as f:
                d = json.load(f)
            if int(d.get("L")) == L:
                items.append((os.path.getmtime(path), path))
        except Exception:
            continue
    if not items:
        return None
    items.sort(key=lambda x: x[0], reverse=True)
    with open(items[0][1], "r") as f:
        return json.load(f)


def find_plots_data_under(root: str) -> Dict[int, Dict[str, Any]]:
    patterns = [os.path.join(root, "**", "plots_data*.json")]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    by_L: Dict[int, Dict[str, Any]] = {}
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
            L_val = int(data.get('L'))
            by_L[L_val] = data
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    return by_L


def find_plots_data_for_targets(root: str, L_targets: List[int]) -> Dict[int, Dict[str, Any]]:
    res: Dict[int, Dict[str, Any]] = {}
    for L in L_targets:
        patterns = [
            os.path.join(root, f"L{L}_*", f"plots_data_L{L}_*.json"),
            os.path.join(root, f"L{L}_*", "plots_data*.json"),
        ]
        candidates: List[str] = []
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            f = candidates[0]
            try:
                with open(f, 'r') as fp:
                    res[L] = json.load(fp)
            except Exception as e:
                print(f"Warning: failed to read {f}: {e}")
    if len(res) < len(L_targets):
        fallback = find_plots_data_under(root)
        for L in L_targets:
            if L not in res and L in fallback:
                res[L] = fallback[L]
    return res


def load_toric_code_data(root: str = TORIC_CODE_ROOT,
                         L_targets: List[int] = L_TARGETS) -> Dict[int, Dict[str, Any]]:
    return find_plots_data_for_targets(root, L_targets)

# ============================================================
# Functions: Plot Helpers
# ============================================================
def get_series(data: Dict[str, Any], path: List[str]) -> List[float]:
    cur = data
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return []
        cur = cur[k]
    return cur if isinstance(cur, list) else []


def get_err_series(data: Dict[str, Any], key: str) -> List[float]:
    ler_err = data.get('ler_err') or {}
    v = ler_err.get(key)
    return v if isinstance(v, list) else []


def _safe_err(y: List[float], e: List[float]):
    try:
        return e if isinstance(e, list) and len(e) == len(y) and len(y) > 0 else None
    except Exception:
        return None


def light_xticks(ax, max_ticks: int = 4):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _finite_positive(vals):
    return [float(v) for v in vals if isinstance(v, (float, int)) and np.isfinite(v) and v > 0]


def _set_cluster_ylim(ax, cluster_vals, pad_decades: float = 0.25):
    v = _finite_positive(cluster_vals)
    if not v:
        return
    lo, hi = min(v), max(v)
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    ax.set_ylim(10 ** (log_lo - pad_decades), 10 ** (log_hi + pad_decades))


def _scatter_with_oob(ax, x, y, **scatter_kwargs):
    """When a point is outside the y-axis range, place it at the edge with an annotation."""
    ymin, ymax = ax.get_ylim()
    pad = 1.05
    y_bottom = ymin * pad
    y_top = ymax / pad
    color = scatter_kwargs.get('color', 'k')
    if y < ymin:
        ax.scatter([x], [y_bottom], **scatter_kwargs, zorder=6, clip_on=False)
        ax.annotate('QUEKUF', xy=(x, y_bottom), xytext=(0, -12), textcoords='offset points',
                    ha='center', va='top', fontsize=14,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5))
    elif y > ymax:
        ax.scatter([x], [y_top], **scatter_kwargs, zorder=6, clip_on=False)
        ax.annotate('QUEKUF', xy=(x, y_top), xytext=(0, -12), textcoords='offset points',
                    ha='center', va='top', fontsize=14,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5))
    else:
        ax.scatter([x], [y], **scatter_kwargs, zorder=6, clip_on=False)
        ax.annotate('QUEKUF', xy=(x, y), xytext=(0, -12), textcoords='offset points',
                    ha='center', va='top', fontsize=14)
