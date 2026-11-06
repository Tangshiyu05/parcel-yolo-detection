# -*- coding: utf-8 -*-
"""
评价指标：在常规检测指标之外，独立量化高风险误判。
输入为真实标签与预测标签（已由检测结果按 IoU 匹配得到），输出：
混淆矩阵、高风险误判率 HRMR、代价加权错误率 CWER、分方向危险错误率。
"""
import numpy as np
from sklearn.metrics import confusion_matrix

from configs.config import HIGH_RISK_PAIRS, CLASSES


def build_confusion(y_true, y_pred, num_classes: int = None):
    num_classes = num_classes or len(CLASSES)
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def high_risk_misclassification_rate(cm: np.ndarray):
    """
    HRMR：高风险混淆对错误数占总样本的比例，
    以及其占全部误判的比例。
    """
    total = cm.sum()
    errors = total - np.trace(cm)
    hr_errors = sum(cm[true, pred] for (true, pred) in HIGH_RISK_PAIRS)
    return {
        "HRMR": round(hr_errors / total, 4) if total else 0.0,
        "HRMR_among_errors": round(hr_errors / errors, 4) if errors else 0.0,
        "high_risk_count": int(hr_errors),
    }


def directional_error_rates(cm: np.ndarray):
    """分方向危险错误率：每个高风险方向错误数 / 该真实类样本数。"""
    out = {}
    for (true, pred), w in HIGH_RISK_PAIRS:
        class_total = cm[true].sum()
        name = f"{CLASSES[true]}->{CLASSES[pred]}"
        out[name] = round(cm[true, pred] / class_total, 4) if class_total else 0.0
    return out


def cost_weighted_error_rate(cm: np.ndarray):
    """CWER：按高风险方向权重加权的错误率。"""
    weighted_err, weighted_total = 0, 0
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            w = HIGH_RISK_PAIRS.get((i, j), 1 if i != j else 0)
            weighted_err += w * cm[i, j]
            weighted_total += cm[i].sum()
    return round(weighted_err / weighted_total, 4) if weighted_total else 0.0


def evaluate_risks(y_true, y_pred):
    """汇总全部风险指标，供实验报告直接引用。"""
    cm = build_confusion(y_true, y_pred)
    return {
        "confusion_matrix": cm.tolist(),
        **high_risk_misclassification_rate(cm),
        "directional_error_rates": directional_error_rates(cm),
        "CWER": cost_weighted_error_rate(cm),
    }
