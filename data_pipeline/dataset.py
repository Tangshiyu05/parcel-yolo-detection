# -*- coding: utf-8 -*-
"""
数据集信息工具：读取 data.yaml、统计类别分布、生成划分。
实际训练的 Dataset 由 Ultralytics 内部构建，这里用于数据分布核查。
"""
from collections import Counter
from pathlib import Path

import yaml


def load_data_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_distribution(label_dir: str):
    """统计各类别标注数量，判断是否存在类别不平衡。"""
    counter = Counter()
    for label in Path(label_dir).glob("*.txt"):
        for line in label.read_text().splitlines():
            counter[int(line.split()[0])] += 1
    return dict(counter)


def make_split(image_dir: str, ratios=(0.8, 0.1, 0.1), seed=42):
    """生成 train/val/test 文件名划分（Ultralytics 已自动划分时可不用）。"""
    import random
    files = sorted(p.name for p in Path(image_dir).glob("*.jpg"))
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }
