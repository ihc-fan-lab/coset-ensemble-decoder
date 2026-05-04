#!/usr/bin/env bash
# ============================================================
# Full Reproduction Script (estimated: days on 32-core server)
# Paper: Coset Ensemble Decoder for QEC with Algorithm-Hardware Co-Design
# ISCA 2025 - Paper #1199
#
# This script reproduces all data points for Figures 8, 10, 12.
# Shot budgets are set to achieve CV <= 10% on LER estimates.
#
# WARNING: L=11 with 40M shots at p=0.0005 requires ~90GB RAM
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

OUTPUT_ROOT="rebuttal/rebuttal_outputs/toric_code"

echo "============================================"
echo "  Full Reproduction"
echo "  Output: ${OUTPUT_ROOT}"
echo "============================================"
echo ""
echo "WARNING: This will take a very long time."
echo "  L=3:  ~minutes   | L=5:  ~tens of minutes"
echo "  L=7:  ~hours     | L=9:  ~many hours"
echo "  L=11: ~days      |"
echo ""
echo "Press Ctrl+C within 10s to abort..."
sleep 10

# -------------------------------------------------------
# L=3: LER range 2.7e-4 ~ 2.1e-3
# -------------------------------------------------------
echo "[1/5] Running L=3 ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 3 \
  --ps 0.0005,0.00075,0.001,0.00125,0.0015 \
  --num_shots_per_p 400000,200000,120000,80000,60000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --verbose_batches \
  --with_error_bars \
  --resume \
  --output_root "${OUTPUT_ROOT}"

# -------------------------------------------------------
# L=5: LER range 3.7e-5 ~ 6.8e-4
# -------------------------------------------------------
echo "[2/5] Running L=5 ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 5 \
  --ps 0.0005,0.00075,0.001,0.00125,0.0015 \
  --num_shots_per_p 1500000,1000000,600000,400000,200000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --verbose_batches \
  --with_error_bars \
  --resume \
  --output_root "${OUTPUT_ROOT}"

# -------------------------------------------------------
# L=7: LER range 4e-6 ~ 2.2e-4
# -------------------------------------------------------
echo "[3/5] Running L=7 ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 7 \
  --ps 0.0005,0.00075,0.001,0.00125,0.0015 \
  --num_shots_per_p 5000000,3000000,2000000,1000000,500000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --verbose_batches \
  --with_error_bars \
  --resume \
  --output_root "${OUTPUT_ROOT}"

# -------------------------------------------------------
# L=9: LER range 8.7e-7 ~ 6e-5
# -------------------------------------------------------
echo "[4/5] Running L=9 ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 9 \
  --ps 0.0005,0.00075,0.001,0.00125,0.0015 \
  --num_shots_per_p 15000000,8000000,5000000,3000000,2000000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --verbose_batches \
  --with_error_bars \
  --resume \
  --output_root "${OUTPUT_ROOT}"

# -------------------------------------------------------
# L=11: LER range 5e-8 ~ 1.7e-5
# -------------------------------------------------------
echo "[5/5] Running L=11 ..."
python rebuttal/run_ablation_pipeline.py \
  --code_type toric_code \
  --Ls 11 \
  --ps 0.0005,0.00075,0.001,0.00125,0.0015 \
  --num_shots_per_p 40000000,15000000,10000000,8000000,6000000 \
  --list_size 24 \
  --channel x \
  --if_repetitions \
  --batch_size 20000 \
  --verbose_batches \
  --with_error_bars \
  --resume \
  --output_root "${OUTPUT_ROOT}"

# -------------------------------------------------------
# Fig. 16: Repetition code (L=5,7)
# -------------------------------------------------------
echo "[6/6] Running Fig.16 experiments ..."
for L in 5 7; do
  python rebuttal/run_ablation_pipeline.py \
    --code_type repetition_code \
    --Ls ${L} \
    --ps 0.005,0.0075,0.01,0.015,0.02,0.025,0.03 \
    --num_shots 100000 \
    --list_size 24 \
    --channel x \
    --if_repetitions \
    --batch_size 20000 \
    --verbose_batches \
    --with_error_bars \
    --output_root rebuttal/rebuttal_outputs/repetition_code
done

# -------------------------------------------------------
# Generate plots
# -------------------------------------------------------
echo ""
echo "Generating all plots ..."
cd rebuttal
mkdir -p figplot figplot/repetition_code

jupyter nbconvert --to notebook --execute nb_combined_ler.ipynb --output nb_combined_ler_executed.ipynb
jupyter nbconvert --to notebook --execute nb_latency_comparison.ipynb --output nb_latency_comparison_executed.ipynb
jupyter nbconvert --to notebook --execute nb_fidelity_multi.ipynb --output nb_fidelity_multi_executed.ipynb
jupyter nbconvert --to notebook --execute 1_plots_acc_repetition.ipynb --output 1_plots_acc_repetition_executed.ipynb

# Copy with clear naming
mkdir -p figplot_full
cp figplot/combined_ler_mwpm_uf_ours_row.pdf figplot_full/fig8_full.pdf 2>/dev/null || true
cp figplot/combined_latency_comparison_row.pdf figplot_full/fig10_full.pdf 2>/dev/null || true
cp figplot/on_the_fly_fidelity_row.pdf figplot_full/fig12_full.pdf 2>/dev/null || true
cp figplot/repetition_code/combined_ler_mwpm_uf_uflist_1x2.pdf figplot_full/fig16_full.pdf 2>/dev/null || true
cd ..

echo ""
echo "============================================"
echo "  Full reproduction complete!"
echo ""
echo "  Figures: rebuttal/figplot_full/"
echo "    fig8_full.pdf   (LER comparison)"
echo "    fig10_full.pdf  (Latency comparison)"
echo "    fig12_full.pdf  (System infidelity)"
echo "    fig16_full.pdf  (Repetition code LER)"
echo "============================================"
