# Coset Ensemble Decoder for QEC with Algorithm-Hardware Co-Design

**Paper:** [Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design](https://arxiv.org/abs/2606.11076) (ISCA 2026)

This repository provides the software implementation and experiment scripts for the Coset Ensemble Decoder. It includes the complete decoder pipeline, a cycle-accurate hardware simulator, and all scripts needed to reproduce the main results.

> **Algorithm–hardware co-design:** This project couples decoder algorithms with a dedicated hardware architecture. The current release includes software implementations and a **cycle-accurate hardware simulator** (`hardware/`) that models the proposed design's behavior. **Verilog RTL will be released in `hardware_code/` soon.**

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Getting Started (~15 min)](#2-getting-started-15-min)
3. [Reproduction Modes](#3-reproduction-modes)
4. [Mapping to Paper Figures](#4-mapping-to-paper-figures)
5. [Understanding Results and Statistical Variability](#5-understanding-results-and-statistical-variability)
6. [Project Structure](#6-project-structure)
7. [Citation](#7-citation)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Hardware Requirements

| Mode | CPU Cores | RAM | Estimated Time |
|------|-----------|-----|----------------|
| Plot-only | Any | 4 GB | ~1 min |
| Minimal (L=3,5,7) | 8+ | 32 GB | ~4-6 hours |
| Lightweight (L=3-11) | 16+ | 64 GB | ~1-2 days |
| Full | 32+ | 128 GB | ~1 week |

- **Architecture:** x86-64 (tested on Intel Xeon)
- **No GPU, FPGA, or quantum hardware required.** The hardware architecture is evaluated via a cycle-accurate software simulator included in this artifact (`hardware/`).
- **Verilog RTL:** The Verilog hardware implementation will be added to `hardware_code/` soon.

---

## 2. Getting Started (~15 min)

### Prerequisites
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Git, build-essential, cmake
- A Linux system (tested on Ubuntu 20.04/22.04)

### Step 1: Enter the artifact directory

```bash
cd ae_artifact
```

### Step 2: Run the setup script

```bash
bash setup.sh
```

This creates a Conda environment `qec_ae` with Python 3.10 and all dependencies.

### Step 3: Activate and smoke test

```bash
conda activate qec_ae
python -c "import sys; sys.path.insert(0,'.'); from uf_test_utils import UFTester; print('OK')"
```

---

## 3. Reproduction Modes

Four bash scripts are provided, from fastest to most thorough. **All scripts are one-command runs.**

### Mode 1: Plot-Only (~1 min) — `run_plot_only.sh`

Regenerates all 4 Key Result figures from **pre-computed data** included in the artifact. No experiments are re-run.

```bash
bash run_plot_only.sh
```

**Output:** `rebuttal/figplot/` with Fig. 8, 10, 12, 16 in paper-identical format.

This verifies the plotting pipeline and allows inspection of the pre-computed experimental data.

---

### Mode 2: Minimal (~4-6 hours on 8 cores) — `run_minimal.sh`

Runs experiments for **L=3, 5, 7 only** (skips the expensive L=9, 11). Uses paper-identical shot counts with graduated p-points:

| L | p-points | Total shots |
|---|----------|-------------|
| 3 | 5/5 (all) | 860K |
| 5 | 4/5 | 2.2M |
| 7 | 3/5 | 3.5M |

```bash
bash run_minimal.sh
```

**Output:** `rebuttal/figplot_minimal/fig8_minimal.pdf`, etc. Figures will show data for L=3,5,7 subplots; L=9,11 subplots will be empty.

---

### Mode 3: Lightweight (~1-2 days on 16 cores) — `run_lightweight.sh`

Runs all code distances L=3 to 11 with graduated coverage and paper-identical shot counts where applicable:

| L | p-points | Total shots | Notes |
|---|----------|-------------|-------|
| 3 | 5/5 (all) | 860K | Paper-identical |
| 5 | 4/5 | 2.2M | Paper-identical |
| 7 | 3/5 | 3.5M | Paper-identical |
| 9 | 2/5 | 5M | Paper-identical |
| 11 | 1/5 | 600K | 1/10 of paper (to save time) |

```bash
bash run_lightweight.sh
```

**Output:** `rebuttal/figplot_lightweight/fig8_lightweight.pdf`, etc. All 5 subplots populated with graduated data density.

---

### Mode 4: Full Reproduction (~1 week on 32 cores) — `run_full.sh`

Reproduces **all data points** with paper-identical shot counts across all 5 physical error rates and all code distances.

```bash
bash run_full.sh
```

| L | p=0.0005 | p=0.00075 | p=0.001 | p=0.00125 | p=0.0015 |
|---|----------|-----------|---------|-----------|----------|
| 3 | 400K | 200K | 120K | 80K | 60K |
| 5 | 1.5M | 1M | 600K | 400K | 200K |
| 7 | 5M | 3M | 2M | 1M | 500K |
| 9 | 15M | 8M | 5M | 3M | 2M |
| 11 | 40M | 15M | 10M | 8M | 6M |

Supports **checkpoint/resume**: if interrupted, re-run the same command to continue.

### Memory considerations

- L=9 with 24 workers: ~60 GB RAM
- L=11 with 17 workers: ~90 GB RAM
- Reduce `--n_jobs` in the script if you have less RAM

---

## 4. Mapping to Paper Figures

| Paper Figure | Description | Notebook | Output PDF |
|---|---|---|---|
| **Fig. 8** | LER: MWPM vs UF vs Ours (d=3-11) | `nb_combined_ler.ipynb` | `combined_ler_mwpm_uf_ours_row.pdf` |
| **Fig. 10** | Decoding latency comparison | `nb_latency_comparison.ipynb` | `combined_latency_comparison_row.pdf` |
| **Fig. 12** | System infidelity comparison | `nb_fidelity_multi.ipynb` | `on_the_fly_fidelity_row.pdf` |
| **Fig. 16** | Repetition code LER (d=5,7) | `1_plots_acc_repetition.ipynb` | `combined_ler_mwpm_uf_uflist_1x2.pdf` |

### Key claims to verify

1. **Fig. 8:** Our decoder ("Ours") consistently achieves **lower LER than UF** across all code distances. At larger d, Ours approaches MWPM accuracy.
2. **Fig. 10:** Our decoder achieves **the lowest latency** among all three decoders at all code distances, maintaining sub-microsecond latency.
3. **Fig. 12:** Our decoder achieves **the lowest system infidelity** (lower is better), with the advantage growing at larger code distances.
4. **Fig. 16:** The accuracy advantage of our decoder over UF persists on repetition codes under a different (phenomenological) noise model.

### Data flow

```
run_ablation_pipeline.py
  -> UFTester.run_experiments_batched()
     -> Stim (circuit-level noise generation)
     -> MWPM decoder (PyMatching baseline)
     -> UF decoder + Coset Ensemble (our implementation)
     -> Hardware cycle simulator (QuliD_hardware.py)
  -> plots_data_L{L}_list24.json

Notebooks (nb_combined_ler.ipynb, etc.)
  -> load plots_data via plot_utils.py
  -> combine with Helios/Micro-Blossom baseline tables (from published papers)
  -> generate PDF figures
```

---

## 5. Understanding Results and Statistical Variability

**Exact numerical reproduction is not expected.** There are two independent sources of randomness in the experiments, and both are by design:

### Source 1: Monte Carlo noise sampling

QEC experiments use Monte Carlo sampling: random noise is injected and the decoder attempts correction over many independent trials. The logical error rate (LER) is the fraction of trials where decoding fails. Different random seeds produce slightly different LER estimates.

### Source 2: Algorithmic randomness in tree generation

Our coset ensemble decoder uses **randomized priority functions** during the ensemble forest exploration phase (Algorithm 2 in the paper, `PriorityForests`). Each candidate in the ensemble is generated with independently sampled random priorities, producing different spanning forests and therefore different candidate corrections. This randomness is fundamental to the algorithm's design — it is how the decoder explores multiple cosets to approximate maximum-likelihood decoding.

As a result, even with identical noise inputs, the decoder's output depends on the random seeds used for tree generation. This means **two runs with identical parameters will generally produce slightly different LER values**, due to both noise sampling and the algorithmic randomness.

### What to check

Rather than matching exact numbers, verify these **qualitative trends** which hold robustly regardless of random seeds:

- **Decoder ordering in Fig. 8:** At every (L, p) point, the LER ordering should be: **UF > Ours >= MWPM**. Our decoder's LER should be significantly lower than UF and close to MWPM. The gap between UF and Ours grows with code distance.
- **Latency ordering in Fig. 10:** Our decoder should have the **lowest  latency** among most cases, followed by Helios, then Micro-Blossom (which grows steeply with d).
- **Generality in Fig. 16:** The same LER ordering (UF > Ours >= MWPM) should hold on repetition codes under a different noise model.

### Expected variability

Even with paper-identical shot counts, numerical differences on the order of 10-30% for individual LER data points are normal, especially at low error rates where fewer decoding failures are observed. The paper's data was generated with the same code and parameters under one particular random seed realization.

**The key claims of the paper are about trends and relative ordering, not exact numerical values.** These trends are statistically robust and should be clearly visible in any reproduction run.

### Latency results

The **decoding latency** has less variability than LER. The hardware cycle model is deterministic given a syndrome input, and the average latency over many shots converges quickly. Small differences in average latency between runs reflect differences in the syndrome distribution (from noise sampling), not algorithmic randomness.

---

## 6. Project Structure

```
ae_artifact/
├── README.md                 # This file
├── requirements_ae.txt       # Python dependencies
├── setup.sh                  # Environment setup
├── run_plot_only.sh          # Mode 1: Plot from pre-computed data
├── run_minimal.sh            # Mode 2: L=3,5,7 only
├── run_lightweight.sh        # Mode 3: L=3-11 graduated
├── run_full.sh               # Mode 4: Full reproduction
│
├── config.py                 # Decoder configuration
├── uf_test_utils.py          # Core experiment orchestrator
├── uf_decoder.py             # Union-Find decoder
├── software/                 # Decoder implementations
│   ├── uf_efficient.py       # Optimized UF with graph compression
│   ├── uf_listdecoding.py    # Ensemble forest exploration (Sec. III-A)
│   ├── peeling_efficient.py  # Reverse-order elimination (Sec. III-B)
│   └── ...
├── hardware_code/            # Verilog RTL (to be released soon)
├── hardware/                 # Cycle-accurate hardware simulator (software golden model)
│   ├── QuliD_hardware.py     # Main hardware model (Sec. IV)
│   └── ...
├── tools/                    # Noise generation & metrics
├── plotting/                 # Plot generation utilities
├── stimcircuits/             # Stim circuit definitions
│
└── rebuttal/                 # Experiment scripts & notebooks
    ├── run_ablation_pipeline.py  # Main experiment runner
    ├── plot_utils.py             # Shared plotting utilities
    ├── nb_combined_ler.ipynb     # Figure 8
    ├── nb_latency_comparison.ipynb # Figure 10
    ├── nb_fidelity_multi.ipynb   # Figure 12
    ├── 1_plots_acc_repetition.ipynb # Figure 16
    └── rebuttal_outputs/         # Pre-computed reference data
```

---

## 7. Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{liang2026cosetensembledecoderquantum,
      title={Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design}, 
      author={Shuang Liang and Jubo Xu and Giulio Bassanino and Qianzhou Wang and Yidong Zhou and Yuncheng Lu and Zhiwen Mo and Paul H. J. Kelly and Bo Yuan and Wayne Luk and Hongxiang Fan},
      year={2026},
      eprint={2606.11076},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2606.11076}, 
}
```

**Paper:** [arXiv:2606.11076](https://arxiv.org/abs/2606.11076) · Accepted at ISCA 2026

---

## 8. Troubleshooting

### Import errors

Ensure the Conda environment is activated (`conda activate qec_ae`) and you are running from the `ae_artifact/` directory. If needed: `export PYTHONPATH=$(pwd):$PYTHONPATH`.

### ldpc/localuf installation fails

These packages require a C compiler:
```bash
sudo apt-get install build-essential cmake
```

### Out of memory during L=9 or L=11

Reduce `--n_jobs` in the run script. Rule of thumb: each L=11 worker needs ~5.4 GB RAM.

### Notebook execution fails

Run notebooks manually for better diagnostics:
```bash
jupyter lab   # then open the notebook in browser
```

### Results differ from paper

This is expected due to Monte Carlo variability. Verify the **qualitative trends** (decoder ordering) rather than exact numbers. See Section 5 for details.
