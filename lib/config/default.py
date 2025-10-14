import os
import platform
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR = 'runs/'
_C.GPUS = (0,)  # 只使用第一块 GPU（你只有一块）
# Auto-detect optimal worker count based on OS and CPU cores
# Windows: use 0 to avoid multiprocessing issues (most stable)
# Linux: use 8 workers for better performance (4090D推荐8-16)
_C.WORKERS = 0 if platform.system() == 'Windows' else 8
_C.PIN_MEMORY = False  # 4090D 启用 PIN_MEMORY 加速数据传输
_C.PRINT_FREQ = 20
_C.AUTO_RESUME =False       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'yolop_yolov11_small'
_C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']
_C.MODEL.PRETRAINED = ""
_C.MODEL.PRETRAINED_DET = ""  # 已弃用
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)


# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
# YOLOv11 Multi-head lambda weights: [cls, obj(unused), box, da_seg, ll_seg, ll_iou]
# 这是在各个GAIN基础上的额外权重系数，最终损失 = 原始损失 × GAIN × LAMBDA
_C.LOSS.MULTI_HEAD_LAMBDA = [1.0, 1.0, 1.0, 0.5, 0.5, 0.8]  # 分割任务降权
_C.LOSS.FL_GAMMA = 0.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights (YOLOv5 only)
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 7.5  # box loss gain (YOLOv11 default, 适合小量级box loss)
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain (YOLOv5 only, unused in YOLOv11)
_C.LOSS.DA_SEG_GAIN = 0.2  # driving area seg loss (降低: 2.0->0.5, seg量级大需要小权重)
_C.LOSS.LL_SEG_GAIN = 0.2  # lane line seg loss (降低: 2.0->0.5)
_C.LOSS.LL_IOU_GAIN = 0.2  # lane line iou loss (降低: 2.0->1.0, iou量级适中)


# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.DATAROOT = 'E:\Hust\outside_study\YOLOP\dataset\\bdd100k_images_100k\\100k'       # the path of images folder
_C.DATASET.LABELROOT = 'E:\Hust\outside_study\YOLOP\dataset\det_annotations'      # the path of det_annotations folder label
_C.DATASET.MASKROOT = 'E:\Hust\outside_study\YOLOP\dataset\da_seg_annotations'                # the path of da_seg_annotations folder mask
_C.DATASET.LANEROOT = 'E:\Hust\outside_study\YOLOP\dataset\ll_seg_annotations'               # the path of ll_seg_annotations folder lane
_C.DATASET.DATASET = 'BddDataset'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [720, 1280]

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.5 # 0.25
_C.DATASET.ROT_FACTOR = 15 # 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add


# train
_C.TRAIN = CN(new_allowed=True)
# 大 batch_size (64) 需要更大的学习率，按比例缩放：lr = base_lr * (batch_size / base_batch)
# base: lr=0.001, batch=4  →  batch=64: lr = 0.001 * (64/4) = 0.016
_C.TRAIN.LR0 = 0.001  # initial learning rate for batch_size=64 (scaled from 0.001)
_C.TRAIN.LRF = 0.01  # final OneCycleLR learning rate (lr0 * lrf) - YOLOv11 uses 0.01
_C.TRAIN.WARMUP_EPOCHS = 5.0  # 大batch增加warmup到5 epoch
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 200  # YOLOv11 recommends 200+ epochs

_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.BATCH_SIZE_PER_GPU = 2  # 4090D (24GB) 可以支持 batch_size=64
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task




_C.TRAIN.PLOT = True                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 8
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True
_C.TEST.NMS_CONF_THRESHOLD  = 0.001
_C.TEST.NMS_IOU_THRESHOLD  = 0.6


def update_config(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir
    
    # if args.conf_thres:
    #     cfg.TEST.NMS_CONF_THRESHOLD = args.conf_thres

    # if args.iou_thres:
    #     cfg.TEST.NMS_IOU_THRESHOLD = args.iou_thres
    


    # cfg.MODEL.PRETRAINED = os.path.join(
    #     cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    # )
    #
    # if cfg.TEST.MODEL_FILE:
    #     cfg.TEST.MODEL_FILE = os.path.join(
    #         cfg.DATA_DIR, cfg.TEST.MODEL_FILE
    #     )

    cfg.freeze()
