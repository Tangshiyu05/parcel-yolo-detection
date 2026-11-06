# -*- coding: utf-8 -*-
"""
训练总入口
用法：
    python main.py                # 按 config 训练
    python main.py --no-attention # 关闭 TAM，训练基线用于对照
    python main.py --epochs 100
"""
import argparse
import json
import sys
from pathlib import Path

# 保证从任意工作目录运行都能导入项目内模块
sys.path.append(str(Path(__file__).resolve().parent))

from configs.config import SETTINGS, summary, LOGS_DIR
from utils.seed import set_seed
from utils.logger import get_logger
from core.trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description="快递包装类型检测训练入口")
    parser.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    parser.add_argument("--batch", type=int, default=SETTINGS["batch"])
    parser.add_argument("--no-attention", action="store_true", help="关闭类型感知注意力")
    parser.add_argument("--device", default=SETTINGS["device"])
    return parser.parse_args()


def main():
    args = parse_args()
    SETTINGS["epochs"] = args.epochs
    SETTINGS["batch"] = args.batch
    SETTINGS["use_attention"] = not args.no_attention
    SETTINGS["device"] = args.device

    logger = get_logger(LOGS_DIR / "train.log")
    set_seed(SETTINGS["seed"])

    logger.info("运行配置：\n" + json.dumps(summary(), ensure_ascii=False, indent=2))
    model, results = train()
    logger.info("训练完成，权重与评估图已保存至 outputs/weights/parcel_yolov8s")


if __name__ == "__main__":
    main()
