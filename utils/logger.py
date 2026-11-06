# -*- coding: utf-8 -*-
"""日志工具：同时输出到控制台与文件。"""
import logging
from pathlib import Path


def get_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("parcel_detection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
