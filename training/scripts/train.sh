#!/bin/bash
# PetRWKV LoRA 微调训练脚本

set -e

echo "=== PetRWKV LoRA Training ==="

# 配置
TRAIN_DATA="data/train.jsonl"
VAL_DATA="data/val.jsonl"
OUTPUT_DIR="output/pet_rwkv_0.4b"
BATCH_SIZE=4
MAX_EPOCHS=10
LEARNING_RATE=1e-4

# 检查数据
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    exit 1
fi

echo "Training data: $TRAIN_DATA"
echo "Validation data: $VAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $MAX_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo ""

# 运行训练
python3 -c "
from src.trainer import train_pet_model
from src.model import PetRWKVConfig, LoRAConfig

config = PetRWKVConfig()
lora_config = LoRAConfig(
    r=8,
    alpha=16,
    target_modules=['receptance', 'key', 'value', 'output']
)

train_pet_model(
    train_data_path='$TRAIN_DATA',
    val_data_path='$VAL_DATA',
    output_dir='$OUTPUT_DIR',
    config=config,
    lora_config=lora_config,
    batch_size=$BATCH_SIZE,
    max_epochs=$MAX_EPOCHS,
    learning_rate=$LEARNING_RATE,
)
"

echo ""
echo "=== Training Complete ==="
echo "Model saved to: $OUTPUT_DIR/final"
