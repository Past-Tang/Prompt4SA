#!/usr/bin/env bash
set -euo pipefail

# 可选：在调用本脚本前激活你的 Conda 环境
# conda activate mars2-vqa

# 允许外部预设 CUDA_VISIBLE_DEVICES，否则使用默认值
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

python bast.py
