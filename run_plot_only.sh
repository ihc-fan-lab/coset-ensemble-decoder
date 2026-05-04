#!/usr/bin/env bash
# ============================================================
# Plot-Only Script (uses pre-computed reference data, ~1 min)
# Paper: Coset Ensemble Decoder for QEC with Algorithm-Hardware Co-Design
# ISCA 2025 - Paper #1199
#
# Regenerates all 4 Key Result figures from pre-computed data.
# No experiments are re-run.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "============================================"
echo "  Plot-Only Mode (pre-computed data)"
echo "============================================"
echo ""

# Verify pre-computed data exists
echo "Checking pre-computed data ..."
for L in 3 5 7 9 11; do
    F="rebuttal/rebuttal_outputs/toric_code/L${L}_list24/plots_data_L${L}_list24.json"
    if [ -f "${F}" ]; then
        echo "  [OK] Toric L=${L}"
    else
        echo "  [MISSING] ${F}"
        echo "  ERROR: Pre-computed data not found. Run run_full.sh first."
        exit 1
    fi
done
for L in 5 7; do
    F="rebuttal/rebuttal_outputs/repetition_code/L${L}_list24/acc_ler_L${L}_list24.json"
    if [ -f "${F}" ]; then
        echo "  [OK] Repetition L=${L}"
    else
        echo "  [MISSING] ${F}"
    fi
done
echo ""

# Create output directories
mkdir -p rebuttal/figplot rebuttal/figplot/repetition_code

# Execute notebooks
cd rebuttal

echo "Generating Figure 8 (LER comparison) ..."
jupyter nbconvert --to notebook --execute nb_combined_ler.ipynb \
    --output nb_combined_ler_executed.ipynb \
    --ExecutePreprocessor.timeout=120 2>&1 | tail -1
echo "  -> figplot/combined_ler_mwpm_uf_ours_row.pdf"
echo ""

echo "Generating Figure 10 (Latency comparison) ..."
jupyter nbconvert --to notebook --execute nb_latency_comparison.ipynb \
    --output nb_latency_comparison_executed.ipynb \
    --ExecutePreprocessor.timeout=120 2>&1 | tail -1
echo "  -> figplot/combined_latency_comparison_row.pdf"
echo ""

echo "Generating Figure 12 (System infidelity) ..."
jupyter nbconvert --to notebook --execute nb_fidelity_multi.ipynb \
    --output nb_fidelity_multi_executed.ipynb \
    --ExecutePreprocessor.timeout=120 2>&1 | tail -1
echo "  -> figplot/on_the_fly_fidelity_row.pdf"
echo ""

echo "Generating Figure 16 (Repetition code LER) ..."
mkdir -p figplot/repetition_code
jupyter nbconvert --to notebook --execute 1_plots_acc_repetition.ipynb \
    --output 1_plots_acc_repetition_executed.ipynb \
    --ExecutePreprocessor.timeout=120 2>&1 | tail -1
echo "  -> figplot/repetition_code/combined_ler_mwpm_uf_uflist_1x2.pdf"
echo ""

cd ..

echo "============================================"
echo "  All plots generated successfully!"
echo "  Output directory: rebuttal/figplot/"
echo "============================================"
echo ""
echo "Generated figures:"
ls -la rebuttal/figplot/*.pdf rebuttal/figplot/repetition_code/*.pdf 2>/dev/null || echo "  (no PDFs found)"
