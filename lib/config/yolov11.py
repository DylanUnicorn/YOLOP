"""
YOLOv11 Backbone 配置文件
使用 YOLOv11 作为 backbone 的 YOLOP 模型配置，支持无锚点检测头
"""
import os
import platform
from .default import _C as cfg

# 覆盖默认配置以使用 YOLOv11
cfg.MODEL.USE_YOLOV11 = True
cfg.MODEL.NAME = 'YOLOP_YOLOv11'  # 模型名称
cfg.MODEL.YOLOV11_SCALE = 's'  # 'n', 's', 'm', 'l', 'x'
cfg.MODEL.YOLOV11_WEIGHTS = ''  # 'weights/yolo11s.pt'
cfg.MODEL.FREEZE_BACKBONE = True  # 是否冻结 backbone

# 训练配置
cfg.TRAIN.BATCH_SIZE_PER_GPU = 8  # YOLOv11s 可以用更大的 batch size
cfg.TRAIN.END_EPOCH = 200  # YOLOv11 建议 200+ epochs

# 学习率配置（冻结 backbone 时可以用更大的学习率）
cfg.TRAIN.LR0 = 0.01  # 冻结 backbone 时增大学习率
cfg.TRAIN.LRF = 0.01
cfg.TRAIN.WARMUP_EPOCHS = 3.0  # 减少 warmup

# YOLOv11 使用新的损失函数，这些参数在 loss_v11.py 中定义
# 这里的配置主要用于兼容性
cfg.LOSS.MULTI_HEAD_LAMBDA = [7.5, 0.5, 1.5, 0.02, 0.04, 1.0]  # box, cls, dfl, da_seg, ll_seg, ll_iou
