# -*- coding: utf-8 -*-
"""
推理脚本：加载训练好的权重，对单张图 / 文件夹 / 视频进行预测。
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from configs.config import WEIGHTS_DIR, SETTINGS, CLASSES


def load_model(weight_path: str = None) -> YOLO:
    """加载训练好的权重，默认读取 outputs 下的最佳权重。"""
    weight = weight_path or str(WEIGHTS_DIR / "parcel_yolov8s" / "weights" / "best.pt")
    return YOLO(weight)


def predict(model: YOLO, source, save: bool = True):
    """预测函数框架，返回检测结果（类别、置信度、坐标）。"""
    results = model.predict(
        source=source,
        conf=SETTINGS["conf_thres"],
        iou=SETTINGS["iou_thres"],
        save=save,
        verbose=False,
    )
    detections = []
    for res in results:
        for box in res.boxes:
            detections.append({
                "class_id": int(box.cls.item()),
                "class_name": CLASSES[int(box.cls.item())],
                "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(v, 2) for v in box.xyxy.squeeze().tolist()],
            })
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="快递包装类型检测推理")
    parser.add_argument("--source", required=True, help="图片/文件夹/视频路径")
    parser.add_argument("--weight", default=None)
    args = parser.parse_args()

    net = load_model(args.weight)
    out = predict(net, args.source)
    print(f"共检测到 {len(out)} 个目标：")
    for item in out:
        print(item)
