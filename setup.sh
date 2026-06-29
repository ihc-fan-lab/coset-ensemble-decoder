#!/usr/bin/env bash
# ============================================================
# Setup Script
# Paper: Coset Ensemble Decoder for QEC with Algorithm-Hardware Co-Design
# https://arxiv.org/abs/2606.11076
# Verilog RTL to be released in hardware_code/
# ============================================================
set -e

ENV_NAME="${QEC_AE_ENV:-qec_ae}"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  Coset Ensemble Decoder - Environment Setup"
echo "============================================"
echo ""
echo "This script will:"
echo "  1. Create a Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
echo "  2. Install all required dependencies"
echo "  3. Verify the installation"
echo ""

# -----------------------------------------------------------
# Step 1: Create Conda environment
# -----------------------------------------------------------
echo "[1/4] Creating Conda environment '${ENV_NAME}' ..."

if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "  Environment '${ENV_NAME}' already exists. Activating..."
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "  Python: $(python --version)"
echo "  Location: $(which python)"
echo ""

# -----------------------------------------------------------
# Step 2: Install pip dependencies
# -----------------------------------------------------------
echo "[2/4] Installing pip dependencies ..."
pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/requirements_ae.txt"
echo ""

# -----------------------------------------------------------
# Step 3: Install custom git-based packages
# -----------------------------------------------------------
echo "[3/4] Installing custom packages from git ..."

echo "  Installing ldpc ..."
pip install "ldpc==0.1.50" --quiet 2>/dev/null || pip install "git+https://github.com/quantumgizmos/ldpc.git" --quiet

echo "  Installing localuf ..."
pip install "git+https://github.com/timchan0/localuf.git" --quiet

echo "  Installing qLDPC ..."
pip install "git+https://github.com/qLDPCOrg/qLDPC.git" --quiet

echo "  Installing StimCircuits ..."
pip install "git+https://github.com/oscarhiggott/StimCircuits.git" --quiet

echo ""

# -----------------------------------------------------------
# Step 4: Verify installation
# -----------------------------------------------------------
echo "[4/4] Verifying installation ..."

python -c "
import sys
errors = []

modules = [
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('matplotlib', 'matplotlib'),
    ('stim', 'stim'),
    ('pymatching', 'PyMatching'),
    ('joblib', 'joblib'),
    ('numba', 'numba'),
    ('networkx', 'networkx'),
    ('ldpc', 'ldpc'),
    ('localuf', 'localuf'),
]

for mod, name in modules:
    try:
        __import__(mod)
        print(f'  [OK] {name}')
    except ImportError as e:
        print(f'  [FAIL] {name}: {e}')
        errors.append(name)

if errors:
    print(f'\nWARNING: {len(errors)} package(s) failed to import: {errors}')
    sys.exit(1)
else:
    print('\n  All packages imported successfully!')
"

# Quick smoke test of the decoder
echo ""
echo "  Running smoke test ..."
cd "${SCRIPT_DIR}"
python -c "
import sys
sys.path.insert(0, '.')
from uf_test_utils import UFTester
print('  [OK] UFTester imported successfully')
print('')
print('============================================')
print('  Setup complete!')
print('  Activate with: conda activate ${ENV_NAME}')
print('============================================')
"
