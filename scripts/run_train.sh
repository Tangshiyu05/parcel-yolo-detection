#!/usr/bin/env bash
# 一键训练脚本（本地 / Kaggle 通用）
set -e
cd "$(dirname "$0")/.."

python main.py "$@"
