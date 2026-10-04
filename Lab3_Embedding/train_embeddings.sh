#!/usr/bin/env bash
# ==============================================================================
# train_embeddings.sh
# 
# Bash script to train 6 word embedding models:
# 1. Word2Vec CBOW on QUAERO_FrenchMed
# 2. Word2Vec Skip-gram on QUAERO_FrenchMed
# 3. FastText CBOW on QUAERO_FrenchMed
# 4. Word2Vec CBOW on QUAERO_FrenchPress
# 5. Word2Vec Skip-gram on QUAERO_FrenchPress
# 6. FastText CBOW on QUAERO_FrenchPress
#
# Hyperparameters:
#   dim = 100
#   min_count = 1
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_MED="data/QUAERO_FrenchMed/QUAERO_FrenchMed_traindev.ospl"
DATA_PRESS="data/QUAERO_FrenchPress/QUAERO_FrenchPress_traindev.ospl"
MODELS_DIR="models"

mkdir -p "$MODELS_DIR"

echo "=========================================================="
echo "Starting Word Embedding Training Pipeline"
echo "Medical Corpus: $DATA_MED"
echo "Press Corpus:   $DATA_PRESS"
echo "Models Dir:     $MODELS_DIR"
echo "=========================================================="

# Check if Python is available
PYTHON_CMD="python"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
fi

# Run the complete Python training pipeline using Gensim
echo ""
echo ">>> Training 6 models via Python Gensim pipeline..."
$PYTHON_CMD train_embeddings.py \
    --corpus all \
    --model all \
    --dim 100 \
    --min_count 1 \
    --window 5 \
    --epochs 30 \
    --output_dir "$MODELS_DIR"

echo ""
echo "=========================================================="
echo "Training completed successfully! Saved models in $MODELS_DIR:"
ls -lh "$MODELS_DIR"
echo "=========================================================="
