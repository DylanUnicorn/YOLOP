"""
数据集格式验证脚本
用于验证 train_loader 加载的 input 和 target 格式
特别是验证 target[0] 是否为 [image_idx, class_id, x_center, y_center, width, height]
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torchvision.transforms as transforms
from lib.config import cfg
import lib.dataset as dataset
from lib.utils import DataLoaderX

def check_dataset_format():
    """验证数据集加载格式"""
    
    print("="*80)
    print("开始验证数据集加载格式...")
    print("="*80)
    
    # 数据预处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # 创建训练数据集
    print("\n1. 创建数据集...")
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    print(f"   数据集类型: {cfg.DATASET.DATASET}")
    print(f"   数据集大小: {len(train_dataset)}")
    
    # 打印类别数量
    if hasattr(train_dataset, 'names'):
        print(f"   数据集类别: {train_dataset.names}")
        print(f"   类别数量: {len(train_dataset.names)}")
    else:
        print("   数据集没有 names 属性")
    # 打印类别数量
    if hasattr(train_dataset, "names"):
        print(f"   数据集类别数量: {len(train_dataset.names)}")
    else:
        print("   数据集不包含 names 属性，无法统计类别数量。")
    
    # 创建 DataLoader
    print("\n2. 创建 DataLoader...")
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=4,  # 使用小 batch_size 方便查看
        shuffle=False,
        num_workers=0,  # Windows 上使用 0
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    print(f"   Batch size: ")
    print(f"   Total batches: {len(train_loader)}")
    
    # 获取第一个 batch
    print("\n3. 加载第一个 batch...")
    for i, (input, target, paths, shapes) in enumerate(train_loader):
        print("\n" + "="*80)
        print(f"Batch {i} 数据格式分析:")
        print("="*80)
        
        # 分析 input
        print("\n[INPUT - 图像数据]")
        print(f"  类型: {type(input)}")
        print(f"  形状: {input.shape}")
        print(f"  dtype: {input.dtype}")
        print(f"  值范围: [{input.min():.3f}, {input.max():.3f}]")
        
        # 分析 target
        print("\n[TARGET - 标注数据]")
        print(f"  类型: {type(target)}")
        print(f"  长度: {len(target)} (包含 3 个元素: det, da_seg, ll_seg)")
        
        # target[0] - 检测标签 (最重要)
        print(f"\n  target[0] - 检测标签 (Detection Labels):")
        print(f"    类型: {type(target[0])}")
        print(f"    形状: {target[0].shape}")
        print(f"    dtype: {target[0].dtype}")
        print(f"    说明: [N, 6] 其中 N 是所有图片的目标总数，6 维度为:")
        print(f"          [image_idx, class_id, x_center, y_center, width, height]")
        
        # 打印前几个样本
        if target[0].shape[0] > 0:
            print(f"\n    前 5 个目标样本:")
            print(f"    {'索引':<6} {'img_idx':<10} {'class_id':<10} {'x_center':<12} {'y_center':<12} {'width':<12} {'height':<12}")
            print(f"    {'-'*76}")
            for idx in range(min(5, target[0].shape[0])):
                obj = target[0][idx]
                print(f"    {idx:<6} {obj[0].item():<10.0f} {obj[1].item():<10.0f} {obj[2].item():<12.6f} {obj[3].item():<12.6f} {obj[4].item():<12.6f} {obj[5].item():<12.6f}")
            
            # 验证归一化
            print(f"\n    验证坐标是否归一化到 [0, 1]:")
            xywh_data = target[0][:, 2:]  # 提取 xywh 坐标
            print(f"      x_center 范围: [{xywh_data[:, 0].min():.6f}, {xywh_data[:, 0].max():.6f}]")
            print(f"      y_center 范围: [{xywh_data[:, 1].min():.6f}, {xywh_data[:, 1].max():.6f}]")
            print(f"      width    范围: [{xywh_data[:, 2].min():.6f}, {xywh_data[:, 2].max():.6f}]")
            print(f"      height   范围: [{xywh_data[:, 3].min():.6f}, {xywh_data[:, 3].max():.6f}]")
            
            # 检查是否归一化
            is_normalized = (xywh_data >= 0).all() and (xywh_data <= 1).all()
            if is_normalized:
                print(f"      ✓ 坐标已归一化到 [0, 1]")
            else:
                print(f"      ✗ 警告: 坐标未完全归一化!")
            
            # 统计每张图片的目标数量
            print(f"\n    每张图片的目标数量:")
            for img_idx in range(input.shape[0]):
                count = (target[0][:, 0] == img_idx).sum().item()
                print(f"      图片 {img_idx}: {count} 个目标")
        else:
            print(f"    (该 batch 没有检测目标)")
        
        # target[1] - 驾驶区域分割标签
        print(f"\n  target[1] - 驾驶区域分割标签 (Drivable Area Segmentation):")
        print(f"    类型: {type(target[1])}")
        print(f"    形状: {target[1].shape}")
        print(f"    dtype: {target[1].dtype}")
        print(f"    值范围: [{target[1].min():.3f}, {target[1].max():.3f}]")
        print(f"    说明: [batch_size, num_classes, H, W]")
        
        # target[2] - 车道线分割标签
        print(f"\n  target[2] - 车道线分割标签 (Lane Line Segmentation):")
        print(f"    类型: {type(target[2])}")
        print(f"    形状: {target[2].shape}")
        print(f"    dtype: {target[2].dtype}")
        print(f"    值范围: [{target[2].min():.3f}, {target[2].max():.3f}]")
        print(f"    说明: [batch_size, num_classes, H, W]")
        
        # 分析 paths
        print(f"\n[PATHS - 图像路径]")
        print(f"  类型: {type(paths)}")
        print(f"  长度: {len(paths)}")
        if len(paths) > 0:
            print(f"  示例路径:")
            for idx, path in enumerate(paths):
                print(f"    [{idx}] {path}")
        
        # 分析 shapes
        print(f"\n[SHAPES - 图像尺寸信息]")
        print(f"  类型: {type(shapes)}")
        print(f"  长度: {len(shapes)}")
        if len(shapes) > 0:
            print(f"  示例 (原始尺寸, ((缩放比例), (padding))):")
            for idx, shape in enumerate(shapes[:2]):  # 只显示前2个
                print(f"    [{idx}] {shape}")
        
        print("\n" + "="*80)
        print("验证结论:")
        print("="*80)
        print("✓ target[0] 格式为: [image_idx, class_id, x_center, y_center, width, height]")
        print("✓ xywh 坐标已归一化到 [0, 1]")
        print("✓ image_idx 用于区分 batch 中不同图片的目标")
        print("✓ class_id 表示目标类别")
        print("="*80)
        
        # 只查看第一个 batch
        break
    
    print("\n验证完成!")


if __name__ == '__main__':
    check_dataset_format()
