#!/usr/bin/env bash
# ============================================================
# Lightweight Validation Script
# Paper: Coset Ensemble Decoder for QEC with Algorithm-Hardware Co-Design
# ISCA 2025 - Paper #1199
#
# Uses paper-identical shot counts but fewer p-points for large L:
#   L=3:  all 5 p-points   (fast)
#   L=5:  4 p-points       (skip lowest p)
#   L=7:  3 p-points       (skip 2 lowest p)
#   L=9:  2 p-points       (highest 2 only)
#   L=11: 1 p-point        (highest only, 1/10 paper shots)
#
# Estimated runtime: ~1-2 days on 8-core, ~12-18 hours on 32-core
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

TORIC_OUT="rebuttal/rebuttal_outputs/toric_code_lightweight"
REP_OUT="rebuttal/rebuttal_outputs/repetition_code_lightweight"
mkdir -p "${TORIC_OUT}" "${REP_OUT}"

echo "============================================"
echo "  Lightweight Validation (paper shot counts)"
echo "============================================"
echo ""

# -------------------------------------------------------
# Fig. 8, 10, 12: Toric code
# Shot counts match the paper exactly for each (L, p).
# -------------------------------------------------------

echo "[1/7] L=3: all 5 p-points (860K total shots) ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 3 \
  --ps 0.0005,0.00075,0.001,0.00125,0.0015 \
  --num_shots_per_p 400000,200000,120000,80000,60000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --with_error_bars \
  --resume \
  --output_root "${TORIC_OUT}"

echo "[2/7] L=5: 4 p-points (2.2M total shots) ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 5 \
  --ps 0.00075,0.001,0.00125,0.0015 \
  --num_shots_per_p 1000000,600000,400000,200000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --with_error_bars \
  --resume \
  --output_root "${TORIC_OUT}"

echo "[3/7] L=7: 3 p-points (3.5M total shots) ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 7 \
  --ps 0.001,0.00125,0.0015 \
  --num_shots_per_p 2000000,1000000,500000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --with_error_bars \
  --resume \
  --output_root "${TORIC_OUT}"

echo "[4/7] L=9: 2 p-points (5M total shots) ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 9 \
  --ps 0.00125,0.0015 \
  --num_shots_per_p 3000000,2000000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --with_error_bars \
  --resume \
  --output_root "${TORIC_OUT}"

echo "[5/7] L=11: 1 p-point (600K shots, 1/10 of paper) ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 11 \
  --ps 0.0015 \
  --num_shots_per_p 600000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --with_error_bars \
  --resume \
  --output_root "${TORIC_OUT}"

# -------------------------------------------------------
# Fig. 16: Repetition code
#   L=5: all 7 p-points (fast)
#   L=7: 4 highest p-points
# -------------------------------------------------------

echo "[6/7] Fig.16 L=5: all 7 p-points (700K shots) ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type repetition_code \
  --Ls 5 \
  --ps 0.005,0.0075,0.01,0.015,0.02,0.025,0.03 \
  --num_shots 100000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --with_error_bars \
  --output_root "${REP_OUT}"

echo "[7/7] Fig.16 L=7: 4 highest p-points (400K shots) ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type repetition_code \
  --Ls 7 \
  --ps 0.015,0.02,0.025,0.03 \
  --num_shots 100000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --with_error_bars \
  --output_root "${REP_OUT}"

# -------------------------------------------------------
# Swap data so notebooks read the new experiment results
# -------------------------------------------------------
echo ""
echo "Preparing data for plotting ..."
cd rebuttal
mv rebuttal_outputs/toric_code rebuttal_outputs/toric_code_precomputed
mv rebuttal_outputs/repetition_code rebuttal_outputs/repetition_code_precomputed 2>/dev/null || true
ln -s toric_code_lightweight rebuttal_outputs/toric_code
ln -s repetition_code_lightweight rebuttal_outputs/repetition_code

# -------------------------------------------------------
# Generate plots from new experiment data
# -------------------------------------------------------
echo "Generating lightweight plots ..."
mkdir -p figplot figplot/repetition_code

jupyter nbconvert --to notebook --execute nb_combined_ler.ipynb \
    --output nb_combined_ler_lightweight.ipynb 2>&1 || echo "  Warning: nb_combined_ler failed"
jupyter nbconvert --to notebook --execute nb_latency_comparison.ipynb \
    --output nb_latency_comparison_lightweight.ipynb 2>&1 || echo "  Warning: nb_latency_comparison failed"
jupyter nbconvert --to notebook --execute nb_fidelity_multi.ipynb \
    --output nb_fidelity_multi_lightweight.ipynb 2>&1 || echo "  Warning: nb_fidelity_multi failed"
jupyter nbconvert --to notebook --execute 1_plots_acc_repetition.ipynb \
    --output 1_plots_acc_repetition_lightweight.ipynb 2>&1 || echo "  Warning: 1_plots_acc_repetition failed"

# Copy outputs with clear naming
mkdir -p figplot_lightweight
cp figplot/combined_ler_mwpm_uf_ours_row.pdf figplot_lightweight/fig8_lightweight.pdf 2>/dev/null || true
cp figplot/combined_latency_comparison_row.pdf figplot_lightweight/fig10_lightweight.pdf 2>/dev/null || true
cp figplot/on_the_fly_fidelity_row.pdf figplot_lightweight/fig12_lightweight.pdf 2>/dev/null || true
cp figplot/repetition_code/combined_ler_mwpm_uf_uflist_1x2.pdf figplot_lightweight/fig16_lightweight.pdf 2>/dev/null || true

# Restore pre-computed data
rm -f rebuttal_outputs/toric_code rebuttal_outputs/repetition_code
mv rebuttal_outputs/toric_code_precomputed rebuttal_outputs/toric_code
mv rebuttal_outputs/repetition_code_precomputed rebuttal_outputs/repetition_code 2>/dev/null || true
cd ..

echo ""
echo "============================================"
echo "  Lightweight validation complete!"
echo ""
echo "  Figures: rebuttal/figplot_lightweight/"
echo "    fig8_lightweight.pdf   (LER comparison)"
echo "    fig10_lightweight.pdf  (Latency comparison)"
echo "    fig12_lightweight.pdf  (System infidelity)"
echo "    fig16_lightweight.pdf  (Repetition code LER)"
echo ""
echo "  Data points verified per L:"
echo "    L=3: 5/5  L=5: 4/5  L=7: 3/5  L=9: 2/5  L=11: 1/5"
echo "============================================"
