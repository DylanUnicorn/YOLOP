"""
YOLOv11 Backbone 配置文件
使用 YOLOv11 作为 backbone 的 YOLOP 模型配置
"""
import os
import platform
from .default import _C as cfg

# 覆盖默认配置以使用 YOLOv11
cfg.MODEL.USE_YOLOV11 = True
cfg.MODEL.YOLOV11_SCALE = 's'  # 'n', 's', 'm', 'l', 'x'
cfg.MODEL.YOLOV11_WEIGHTS = ''  # 'weights/yolo11s.pt'
cfg.MODEL.FREEZE_BACKBONE = False  # 是否冻结 backbone

# 训练配置
cfg.TRAIN.BATCH_SIZE_PER_GPU = 8  # YOLOv11s 可以用更大的 batch size
cfg.TRAIN.END_EPOCH = 200  # YOLOv11 建议 200+ epochs

# 学习率配置（冻结 backbone 时可以用更大的学习率）
cfg.TRAIN.LR0 = 0.01  # 冻结 backbone 时增大学习率
cfg.TRAIN.LRF = 0.01
cfg.TRAIN.WARMUP_EPOCHS = 3.0  # 减少 warmup

# 损失权重（YOLOv11 适配）
cfg.LOSS.MULTI_HEAD_LAMBDA = [1.0, 1.0, 1.0, 0.5, 0.5, 0.8]
cfg.LOSS.BOX_GAIN = 7.5
cfg.LOSS.CLS_GAIN = 0.5
cfg.LOSS.DA_SEG_GAIN = 0.5
cfg.LOSS.LL_SEG_GAIN = 0.5
cfg.LOSS.LL_IOU_GAIN = 1.0
