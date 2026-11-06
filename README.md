# 基于 YOLOv8s 的快递包装类型检测与应用

面向快递分拣流水线，对 **硬纸箱（carton）、软包（soft_bag）、信封（envelope）** 三类包装进行目标检测，
在堆叠遮挡、光照多变条件下保持高精度，并通过类型感知注意力（TAM）与高风险误判独立量化提升工程可用性。

- 框架：PyTorch + Ultralytics YOLOv8
- 任务类型：CV 目标检测
- 适配环境：本地 GPU / Kaggle 免费 GPU（路径自动切换）

## 目录结构

```
parcel-yolo-detection/
├── configs/
│   ├── config.py            # 环境识别 + 路径/超参数集中配置
│   └── __init__.py
├── core/
│   ├── modules/
│   │   └── type_attention.py  # 类型感知注意力 TAM（通道+空间+原型）
│   ├── trainer.py             # 训练器，封装 Ultralytics 接口
│   ├── yolov8s_tam.yaml       # 带 TAM 的自定义网络结构
│   └── __init__.py
├── data_pipeline/
│   ├── preprocess.py          # 清洗去重、标注校验、损坏检测
│   ├── augment.py             # 离线拼接 / HSV 增强
│   └── dataset.py             # data.yaml 读取、类别分布、数据划分
├── utils/
│   ├── metrics.py             # 混淆矩阵 + HRMR + CWER 风险指标
│   ├── logger.py              # 日志
│   └── seed.py                # 随机种子
├── scripts/
│   └── run_train.sh
├── outputs/                   # 权重与日志（不入库）
├── main.py                    # 训练总入口
├── inference.py               # 推理脚本
├── requirements.txt
└── .gitignore
```

## 本地运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                 # 训练（启用 TAM）
python main.py --no-attention  # 基线对照
python inference.py --source path/to/image.jpg
```

## Kaggle 运行

1. 新建 Notebook，右侧 Add Input 挂载数据集（名称与 `config.py` 中 `DATA_ROOT` 对应）；
2. Settings 打开 GPU（T4 x2 / P100）；
3. 执行：

```bash
!git clone https://github.com/<你的用户名>/parcel-yolo-detection.git /kaggle/working/parcel-yolo-detection
%cd /kaggle/working/parcel-yolo-detection
!pip install -r requirements.txt
!python main.py
```

`config.py` 通过 `KAGGLE_KERNEL_RUN_TYPE` 与 `/kaggle` 目录自动识别环境，
Kaggle 下数据读取 `/kaggle/input/...`，输出写入 `/kaggle/working/outputs`，无需手动改路径。

## 自定义模块接入（TAM 注册）

Ultralytics 解析 yaml 时需识别 `TypeAwareAttention`，在训练前执行一次注册：

```python
from ultralytics.nn.modules import ...
from ultralytics.nn.tasks import parse_model
from core.modules.type_attention import TypeAwareAttention
# 将 TypeAwareAttention 加入模块映射字典，使 yolov8s_tam.yaml 可解析
```

> 若暂不接入自定义层，可先 `--no-attention` 跑通基线，再逐步接入，保证实验可对照。

## 评价指标

- 常规：Precision、Recall、F1、mAP50、mAP50:95、FPS；
- 风险量化（独立）：混淆矩阵、高风险误判率 HRMR、分方向危险错误率、代价加权错误率 CWER，
  计算入口见 `utils/metrics.py`，高风险混淆对在 `config.py` 中定义。
