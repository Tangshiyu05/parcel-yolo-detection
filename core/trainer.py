# -*- coding: utf-8 -*-
"""
训练器：基于 Ultralytics YOLO 接口封装，统一从 configs/config.py 读取参数。
自定义 TAM 的注册方式见 README「自定义模块接入」一节。
"""
from ultralytics import YOLO

from configs.config import SETTINGS, DATA_YAML, WEIGHTS_DIR, LOGS_DIR


def build_model(use_attention: bool):
    """加载基线或带 TAM 的自定义结构。"""
    if use_attention:
        # 自定义结构需在 core/yolov8s_tam.yaml 中声明并完成模块注册
        model = YOLO(SETTINGS["custom_model_yaml"])
        else:
        model = YOLO(SETTINGS["base_model"])
    return model


def train():
    model = build_model(SETTINGS["use_attention"])

    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=SETTINGS["epochs"],
        imgsz=SETTINGS["imgsz"],
        batch=SETTINGS["batch"],
        workers=SETTINGS["workers"],
        device=SETTINGS["device"],
        optimizer=SETTINGS["optimizer"],
        lr0=SETTINGS["lr0"],
        weight_decay=SETTINGS["weight_decay"],
        warmup_epochs=SETTINGS["warmup_epochs"],
        patience=SETTINGS["patience"],
        seed=SETTINGS["seed"],
        project=str(WEIGHTS_DIR),
        name="parcel_yolov8s",
        exist_ok=True,
        # 数据增强
        mosaic=SETTINGS["mosaic"],
        mixup=SETTINGS["mixup"],
        hsv_h=SETTINGS["hsv_h"],
        hsv_s=SETTINGS["hsv_s"],
        hsv_v=SETTINGS["hsv_v"],
        fliplr=SETTINGS["fliplr"],
        # 训练后自动在验证集评估，并产出混淆矩阵
        verbose=True,
        plots=True,
    )
    results = model.train(**train_kwargs)
    return model, results
