#!/bin/bash
# PetRWKV 训练准备脚本

set -e

echo "=== PetRWKV Training Preparation ==="

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required"
    exit 1
fi

# 检查依赖
echo "Checking dependencies..."
python3 -c "import torch" 2>/dev/null || {
    echo "Error: PyTorch is required. Install with: pip install torch"
    exit 1
}

python3 -c "import pytorch_lightning" 2>/dev/null || {
    echo "Error: pytorch-lightning is required. Install with: pip install pytorch-lightning==1.9.5"
    exit 1
}

# 创建输出目录
OUTPUT_DIR="output/pet_rwkv_0.4b"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo "Output directory: $OUTPUT_DIR"

# 检查训练数据
TRAIN_DATA="data/train.jsonl"
VAL_DATA="data/val.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Warning: Training data not found at $TRAIN_DATA"
    echo "Please prepare your training data in JSONL format:"
    echo '  {"instruction": "用户输入", "output": "期望回复"}'
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Warning: Validation data not found at $VAL_DATA"
    echo "Creating empty validation file..."
    touch "$VAL_DATA"
fi

echo "=== Preparation Complete ==="
echo ""
echo "Next steps:"
echo "1. Prepare training data in data/train.jsonl"
echo "2. Run: bash scripts/train.sh"
