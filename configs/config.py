# -*- coding: utf-8 -*-
"""
全局配置文件
自动识别当前运行环境（Kaggle / 本地），切换数据路径与输出路径，
并集中管理训练超参数。所有模块统一从这里读取配置，避免硬编码。
"""
import os
from pathlib import Path


def detect_runtime() -> str:
    """判断当前运行环境：Kaggle 内核返回 'kaggle'，否则返回 'local'。"""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or Path("/kaggle").exists():
        return "kaggle"
    return "local"


RUNTIME = detect_runtime()

# ----------------------------------------------------------------------
# 路径配置：本地与 Kaggle 自动切换
# ----------------------------------------------------------------------
if RUNTIME == "kaggle":
    # Kaggle 上挂载的输入数据集（在 Kaggle -> Add Input 中添加你的数据集）
    DATA_ROOT = Path("/kaggle/input/parcel-conveyor-dataset")
    # Kaggle 仅 /kaggle/working 可写，输出权重与日志都放这里
    OUTPUT_ROOT = Path("/kaggle/working/outputs")
    # 克隆下来的代码根目录（Kaggle 中通常为 /kaggle/working/parcel-yolo-detection）
    PROJECT_ROOT = Path("/kaggle/working/parcel-yolo-detection")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = PROJECT_ROOT / "datasets" / "parcel_conveyor"
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"

WEIGHTS_DIR = OUTPUT_ROOT / "weights"
LOGS_DIR = OUTPUT_ROOT / "logs"
for _d in (WEIGHTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 数据集 yaml（Ultralytics 训练所需，内含 train/val 路径与类别名）
DATA_YAML = DATA_ROOT / "data.yaml"

# ----------------------------------------------------------------------
# 模型与训练超参数
# ----------------------------------------------------------------------
SETTINGS = {
    # 模型：基线权重 / 自定义结构 yaml（启用 TAM 时使用）
    "base_model": "yolov8s.pt",
    "custom_model_yaml": str(PROJECT_ROOT / "core" / "yolov8s_tam.yaml"),
    "use_attention": True,          # 是否启用类型感知注意力 TAM
    # 训练参数
    "epochs": 150,
    "imgsz": 640,
    "batch": 16,
    "workers": 4,
    "device": 0,                    # Kaggle GPU 用 0；无 GPU 时改为 "cpu"
    "optimizer": "AdamW",
    "lr0": 0.001,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 30,                 # 早停耐心值
    "seed": 42,
    # 数据增强
    "mosaic": 1.0,
    "mixup": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    # 评估
    "conf_thres": 0.25,
    "iou_thres": 0.6,
}

# 类别定义（与 data.yaml 保持一致）
CLASSES = ["carton", "soft_bag", "envelope"]

# 高风险混淆对（误判方向 -> 风险权重），用于 HRMR / CWER 计算
# 下标对应 CLASSES：0 硬纸箱, 1 软包, 2 信封
HIGH_RISK_PAIRS = {
    (1, 0): 5,   # 软包 -> 硬纸箱
    (2, 0): 5,   # 信封 -> 硬纸箱
    (0, 1): 3,   # 硬纸箱 -> 软包
    (2, 1): 3,   # 信封 -> 软包
}


def summary() -> dict:
    """返回一份完整配置快照，便于写入日志保证实验可复现。"""
    return {
        "runtime": RUNTIME,
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "data_yaml": str(DATA_YAML),
        "classes": CLASSES,
        "settings": SETTINGS,
    }
