# -*- coding: utf-8 -*-
"""
数据预处理：清洗去重、标注校验、尺寸归一化。
Ultralytics 训练时会自动完成 resize/归一化，这里用于入库前的数据治理。
"""
import hashlib
from pathlib import Path

import cv2

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def drop_duplicates(image_dir: str):
    """基于文件哈希去重，返回被判定为重复的文件列表。"""
    seen, duplicates = set(), []
    for p in Path(image_dir).iterdir():
        if p.suffix.lower() not in VALID_EXT:
            continue
        h = file_md5(p)
        if h in seen:
            duplicates.append(str(p))
        else:
            seen.add(h)
    return duplicates


def check_corrupt(image_dir: str):
    """检测无法读取或分辨率过低的损坏图像。"""
    bad = []
    for p in Path(image_dir).iterdir():
        if p.suffix.lower() not in VALID_EXT:
            continue
        img = cv2.imread(str(p))
        if img is None or min(img.shape[:2]) < 100:
            bad.append(str(p))
    return bad


def verify_labels(image_dir: str, label_dir: str, num_classes: int = 3):
    """校验 YOLO 标注：类别越界、坐标超出 [0,1]、缺标签文件。"""
    issues = []
    for img in Path(image_dir).glob("*.jpg"):
        label = Path(label_dir) / (img.stem + ".txt")
        if not label.exists():
            issues.append(f"缺少标签: {img.name}")
            continue
        for line in label.read_text().splitlines():
            cid, x, y, w, h = map(float, line.split()[:5])
            if not (0 <= cid < num_classes) or not all(0 <= v <= 1 for v in (x, y, w, h)):
                issues.append(f"非法标注: {label.name}")
    return issues
