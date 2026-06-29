# HotCRP Submission Fields (Copy-Paste Ready)

## Title
Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

## Abstract
This artifact provides the software implementation of the Coset Ensemble Decoder described in the paper. It includes the complete decoder pipeline (Union-Find clustering, ensemble forest exploration, reverse-order elimination, graph compression), a cycle-accurate hardware simulator, and all experiment scripts and plotting notebooks needed to reproduce the main results (Figures 8, 10, 12, 16). The artifact runs on a standard multi-core CPU server; no FPGA or quantum hardware is required. The Verilog RTL hardware implementation will be released in `hardware_code/` soon; the current release uses a software golden model (`hardware/`) to evaluate the proposed architecture. Dependencies: Python 3.10, Stim (circuit-level noise), PyMatching (MWPM baseline), NumPy, SciPy, Matplotlib, Jupyter, joblib.

## Badges Applied For
- [x] Artifact Available
- [x] Artifacts Evaluated - Functional
- [x] Results Reproduced

## Key Results to be Reproduced
- **Figure 8** (Sec. VI-A): Logical error rate (LER) comparison among MWPM-based decoders, UF-based decoders, and our coset ensemble decoder across code distances d=3 to d=11 under circuit-level depolarizing noise. The key trend to verify is that our decoder consistently achieves lower LER than baseline UF, approaching MWPM accuracy.
- **Figure 10** (Sec. VI-B): Decoding latency comparison against Micro-Blossom (MWPM hardware) and Helios (UF hardware) across code distances d=3 to d=11. The key trend to verify is that our decoder achieves the lowest latency among the three at all code distances.
- **Figure 12** (Sec. VI-C): System infidelity comparison combining LER and hardware latency. The key trend to verify is that our decoder achieves the lowest system infidelity, with the advantage growing at larger code distances.
- **Figure 16** (Sec. VIII-B): Logical error rate comparison on the repetition code under phenomenological noise. The key trend to verify is that the accuracy advantage of our decoder over UF generalizes to a different code family and noise model.

Four reproduction modes are provided: (1) **Plot-only** (~1 min): regenerates all figures from pre-computed data included in the artifact; (2) **Minimal** (~4--6 hours on 8 cores): runs L=3,5,7 with paper-identical shot counts; (3) **Lightweight** (~1--2 days on 16 cores): runs all code distances L=3--11 with graduated coverage; (4) **Full reproduction** (~1 week on 32 cores): regenerates all data points with paper-identical parameters. Due to the Monte Carlo nature of the experiments, exact numerical reproduction is not expected; evaluators should verify the qualitative trends (decoder ordering: UF > Ours >= MWPM in LER).

## Hardware Dependencies
- Minimum: x86-64 CPU, 8 cores, 32 GB RAM (sufficient for lightweight validation mode)
- Recommended: x86-64 CPU, 32+ cores, 128 GB RAM (for full reproduction; the largest experiment at d=11 with 40M Monte Carlo shots requires approximately 90 GB RAM when using 16 parallel workers)
- No GPU, FPGA, or special hardware is required. The proposed hardware architecture is evaluated through a cycle-accurate software simulator included in the artifact (`hardware/`).
- Verilog RTL will be added to `hardware_code/` soon.

## Software Dependencies
- Operating system: Ubuntu 20.04+ or equivalent Linux distribution
- Python 3.10 (tested with Conda/Miniconda)
- Key Python packages: stim 1.15.0 (circuit-level noise generation), PyMatching 2.2.2 (MWPM baseline decoder), numpy, scipy, matplotlib, joblib (parallelization), numba, networkx
- Custom packages installed from git: ldpc, localuf, qLDPC, StimCircuits
- A complete setup script (`setup.sh`) is provided that installs all dependencies automatically via Conda and pip
- No proprietary software is required

## Data Dependencies
No external benchmarks, datasets, or device models are required. All quantum noise models are generated programmatically using the Stim library (circuit-level depolarizing noise). Baseline hardware decoder latency numbers for Helios and Micro-Blossom are taken from their respective published papers and embedded as lookup tables in the artifact source code.

## ISCA Paper Number
1199

## Topics
- Accelerators
- Architectural Support for Quantum Computing
- Evaluation and measurement of real systems
