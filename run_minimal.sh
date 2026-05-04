#!/usr/bin/env bash
# ============================================================
# Minimal Validation Script (~4-6 hours on 8-core, ~1-2 hours on 32-core)
# Paper: Coset Ensemble Decoder for QEC with Algorithm-Hardware Co-Design
# ISCA 2025 - Paper #1199
#
# Runs L=3,5,7 only (skips expensive L=9,11).
# Uses paper-identical shot counts with graduated p-points:
#   L=3: all 5 p-points
#   L=5: 4 p-points
#   L=7: 3 p-points
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

TORIC_OUT="rebuttal/rebuttal_outputs/toric_code_minimal"
REP_OUT="rebuttal/rebuttal_outputs/repetition_code_minimal"
mkdir -p "${TORIC_OUT}" "${REP_OUT}"

echo "============================================"
echo "  Minimal Validation (L=3,5,7 only)"
echo "============================================"
echo ""

# -------------------------------------------------------
# Toric code
# -------------------------------------------------------

echo "[1/5] L=3: all 5 p-points (860K shots) ..."
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

echo "[2/5] L=5: 4 p-points (2.2M shots) ..."
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

echo "[3/5] L=7: 3 p-points (3.5M shots) ..."
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

# -------------------------------------------------------
# Repetition code
# -------------------------------------------------------

echo "[4/5] Fig.16 L=5: all 7 p-points (700K shots) ..."
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

echo "[5/5] Fig.16 L=7: 4 p-points (400K shots) ..."
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
# Backup pre-computed data
mv rebuttal_outputs/toric_code rebuttal_outputs/toric_code_precomputed
mv rebuttal_outputs/repetition_code rebuttal_outputs/repetition_code_precomputed 2>/dev/null || true
# Point notebooks to our new experiment data
ln -s toric_code_minimal rebuttal_outputs/toric_code
ln -s repetition_code_minimal rebuttal_outputs/repetition_code

# -------------------------------------------------------
# Generate plots from new experiment data
# -------------------------------------------------------
echo "Generating plots ..."
mkdir -p figplot figplot/repetition_code

jupyter nbconvert --to notebook --execute nb_combined_ler.ipynb \
    --output nb_combined_ler_minimal.ipynb 2>&1 || echo "  Warning: nb_combined_ler failed"
jupyter nbconvert --to notebook --execute nb_latency_comparison.ipynb \
    --output nb_latency_comparison_minimal.ipynb 2>&1 || echo "  Warning: nb_latency_comparison failed"
jupyter nbconvert --to notebook --execute nb_fidelity_multi.ipynb \
    --output nb_fidelity_multi_minimal.ipynb 2>&1 || echo "  Warning: nb_fidelity_multi failed"
jupyter nbconvert --to notebook --execute 1_plots_acc_repetition.ipynb \
    --output 1_plots_acc_repetition_minimal.ipynb 2>&1 || echo "  Warning: 1_plots_acc_repetition failed"

# Copy with clear naming
mkdir -p figplot_minimal
cp figplot/combined_ler_mwpm_uf_ours_row.pdf figplot_minimal/fig8_minimal.pdf 2>/dev/null || true
cp figplot/combined_latency_comparison_row.pdf figplot_minimal/fig10_minimal.pdf 2>/dev/null || true
cp figplot/on_the_fly_fidelity_row.pdf figplot_minimal/fig12_minimal.pdf 2>/dev/null || true
cp figplot/repetition_code/combined_ler_mwpm_uf_uflist_1x2.pdf figplot_minimal/fig16_minimal.pdf 2>/dev/null || true

# Restore pre-computed data
rm -f rebuttal_outputs/toric_code rebuttal_outputs/repetition_code
mv rebuttal_outputs/toric_code_precomputed rebuttal_outputs/toric_code
mv rebuttal_outputs/repetition_code_precomputed rebuttal_outputs/repetition_code 2>/dev/null || true
cd ..

echo ""
echo "============================================"
echo "  Minimal validation complete!"
echo ""
echo "  Figures: rebuttal/figplot_minimal/"
echo "    fig8_minimal.pdf   (LER, L=3,5,7)"
echo "    fig10_minimal.pdf  (Latency, L=3,5,7)"
echo "    fig12_minimal.pdf  (Infidelity, L=3,5,7)"
echo "    fig16_minimal.pdf  (Repetition code, L=5,7)"
echo "============================================"
