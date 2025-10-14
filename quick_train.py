"""
快速训练脚本 - 用于测试和调试
只使用数据集的前100个样本进行快速多 epoch 测试
"""
import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.cuda import amp
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Quick train for testing')
    
    parser.add_argument('--config', type=str, default='yolov11',
                        help='config to use: default or yolov11')
    parser.add_argument('--samples', type=int, default=100,
                        help='number of samples to use for quick test')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs for quick test')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size for quick test')
    parser.add_argument('--yolo-scale', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv11 scale (only used if config=yolov11)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='freeze YOLOv11 backbone')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    
    args = parser.parse_args()
    return args


class SubsetDataset(torch.utils.data.Dataset):
    """数据集子集包装器"""
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError
        return self.dataset[idx]


def main():
    args = parse_args()
    
    # 加载配置
    if args.config == 'yolov11':
        from lib.config.yolov11 import cfg
        # 覆盖配置
        cfg.MODEL.YOLOV11_SCALE = args.yolo_scale
        cfg.MODEL.YOLOV11_WEIGHTS = f'weights/yolo11{args.yolo_scale}.pt'
        cfg.MODEL.FREEZE_BACKBONE = args.freeze_backbone
    else:
        from lib.config.default import _C as cfg
    
    # 覆盖快速测试配置
    cfg.TRAIN.BEGIN_EPOCH = 0
    cfg.TRAIN.END_EPOCH = args.epochs
    cfg.TRAIN.BATCH_SIZE_PER_GPU = args.batch_size
    cfg.WORKERS = args.workers
    cfg.PRINT_FREQ = 5  # 更频繁的打印
    
    # 设置日志
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'quick_train'
    )
    
    logger.info("="*80)
    logger.info("QUICK TRAIN MODE - Testing Configuration")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.config == 'yolov11':
        logger.info(f"YOLOv11 scale: {args.yolo_scale}")
        logger.info(f"Freeze backbone: {args.freeze_backbone}")
    logger.info("="*80)
    
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    
    # CUDNN 配置
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    # 创建模型
    logger.info("Building model...")
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU)
    
    if hasattr(cfg.MODEL, 'USE_YOLOV11') and cfg.MODEL.USE_YOLOV11:
        model = get_net(
            cfg, 
            yolo_scale=cfg.MODEL.YOLOV11_SCALE,
            yolo_weights_path=cfg.MODEL.YOLOV11_WEIGHTS,
            freeze_backbone=cfg.MODEL.FREEZE_BACKBONE
        ).to(device)
    else:
        model = get_net(cfg).to(device)
    
    logger.info("Model created successfully")

    print("++++++++++++++++++++++")
    print(model.model[model.detector_index])
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # 损失函数和优化器
    criterion = get_loss(cfg, device=device)
    optimizer = get_optimizer(cfg, model)
    
    # 学习率调度器
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # 创建数据集
    logger.info("Loading dataset...")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    # 只使用部分数据
    train_dataset = SubsetDataset(train_dataset, args.samples)
    logger.info(f"Using {len(train_dataset)} training samples")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    # 验证集
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = SubsetDataset(valid_dataset, args.samples // 2)
    logger.info(f"Using {len(valid_dataset)} validation samples")
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    # 混合精度训练
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    # 训练循环
    logger.info("Starting training...")
    logger.info("="*80)
    
    best_fitness = 0.0
    num_batch = len(train_loader)
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{cfg.TRAIN.END_EPOCH-1}")
        logger.info(f"{'='*80}")
        
        # 训练一个 epoch
        train(
            cfg, train_loader, model, criterion, optimizer,
            scaler, epoch, num_batch, num_warmup,
            writer_dict, logger, device
        )
        
        # 更新学习率
        lr_scheduler.step()
        
        # 验证
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH - 1):
            logger.info("\nValidating...")
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict, logger, device
            )
            
            # 计算 fitness
            fi = fitness(np.array(detect_results).reshape(1, -1))
            logger.info(f"Fitness: {fi.item():.4f}")
            
            # 保存最佳模型
            if fi > best_fitness:
                best_fitness = fi
                # logger.info(f"Fitness: {float(fi):.4f}")
                # logger.info(f"Fitness: {fi.item():.4f}")
                logger.info(f"New best fitness: {best_fitness.item():.4f}")
                save_checkpoint(
                    epoch= epoch + 1,
                    name='111',
                    model=model,
                    optimizer=optimizer,
                    output_dir=final_output_dir,
                    filename='checkpoint_best.pth',
                    is_best=True
                )
            
            # 保存检查点
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}.pth'
            )
    
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info(f"Best fitness: {best_fitness.item():.4f}")
    logger.info(f"Results saved to: {final_output_dir}")
    logger.info("="*80)
    
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
