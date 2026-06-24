#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/install_into_llamafactory.sh /path/to/LLaMA-Factory" >&2
  exit 1
fi

TARGET_ROOT="$1"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$TARGET_ROOT/src/llamafactory/train/sft" ]; then
  echo "Target does not look like a LLaMA-Factory checkout: $TARGET_ROOT" >&2
  exit 1
fi

mkdir -p "$TARGET_ROOT/data"
mkdir -p "$TARGET_ROOT/config"
mkdir -p "$TARGET_ROOT/src/llamafactory/train/sft"

cp "$SOURCE_ROOT/data/train_data.json" "$TARGET_ROOT/data/train_data.json"
cp "$SOURCE_ROOT/data/val_data.json" "$TARGET_ROOT/data/val_data.json"
cp "$SOURCE_ROOT/data/test_data.json" "$TARGET_ROOT/data/test_data.json"
cp "$SOURCE_ROOT/data/dataset_info.json" "$TARGET_ROOT/data/dataset_info.json"

cp "$SOURCE_ROOT/config/train_qwen2_5_vl_lora.yaml" "$TARGET_ROOT/config/train_qwen2_5_vl_lora.yaml"

cp "$SOURCE_ROOT/src/llamafactory/train/sft/__init__.py" "$TARGET_ROOT/src/llamafactory/train/sft/__init__.py"
cp "$SOURCE_ROOT/src/llamafactory/train/sft/my_collator.py" "$TARGET_ROOT/src/llamafactory/train/sft/my_collator.py"
cp "$SOURCE_ROOT/src/llamafactory/train/sft/trainer.py" "$TARGET_ROOT/src/llamafactory/train/sft/trainer.py"
cp "$SOURCE_ROOT/src/llamafactory/train/sft/workflow.py" "$TARGET_ROOT/src/llamafactory/train/sft/workflow.py"

echo "Installed visual comfort assessment files into $TARGET_ROOT"
