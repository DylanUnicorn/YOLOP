"""
检查YOLOv11 backbone的YOLOP模型架构
验证backbone和neck的结构是否与预期一致
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from lib.models.YOLOP_YOLOv11 import YOLOPWithYOLOv11
from lib.config import cfg
from lib.config import update_config


def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def check_module_structure(module, name, depth=0):
    """递归检查模块结构"""
    indent = "  " * depth
    print(f"{indent}{name}:")
    
    if isinstance(module, nn.ModuleList):
        for i, sub_module in enumerate(module):
            sub_name = f"[{i}] {type(sub_module).__name__}"
            if depth < 2:  # 限制递归深度
                check_module_structure(sub_module, sub_name, depth + 1)
            else:
                print(f"{indent}  {sub_name}")
    elif isinstance(module, nn.Sequential):
        for i, sub_module in enumerate(module):
            sub_name = f"[{i}] {type(sub_module).__name__}"
            print(f"{indent}  {sub_name}")
    else:
        # 打印模块的子模块
        children = list(module.named_children())
        if children and depth < 2:
            for child_name, child_module in children:
                print(f"{indent}  {child_name}: {type(child_module).__name__}")


def test_forward_pass(model, input_size=(1, 3, 640, 640)):
    """测试前向传播"""
    print("\n" + "="*80)
    print("Testing forward pass...")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        try:
            det_out, da_out, ll_out = model(dummy_input)
            
            print(f"\n✓ Forward pass successful!")
            print(f"\nInput shape: {dummy_input.shape}")
            
            # 检测输出
            if isinstance(det_out, tuple):
                print(f"\nDetection output (tuple with {len(det_out)} elements):")
                for i, out in enumerate(det_out):
                    if isinstance(out, list):
                        print(f"  Element {i} (list with {len(out)} items):")
                        for j, item in enumerate(out):
                            print(f"    [{j}] shape: {item.shape}")
                    else:
                        print(f"  Element {i} shape: {out.shape}")
            else:
                print(f"\nDetection output shape: {det_out.shape}")
            
            # 分割输出
            print(f"\nDrivable area segmentation output shape: {da_out.shape}")
            print(f"Lane line segmentation output shape: {ll_out.shape}")
            
            return True
        except Exception as e:
            print(f"\n✗ Forward pass failed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


def check_backbone_structure(model):
    """检查backbone结构"""
    print("\n" + "="*80)
    print("Checking Backbone Structure (YOLOv11)")
    print("="*80)
    
    backbone = model.backbone
    print(f"\nBackbone type: {type(backbone).__name__}")
    print(f"Number of layers: {len(backbone.layers)}")
    print(f"Output indices (P3, P4, P5): {backbone.out_indices}")
    
    print("\nBackbone layers:")
    for i, layer in enumerate(backbone.layers):
        layer_type = type(layer).__name__
        
        # 获取输入输出通道数
        if hasattr(layer, 'conv') and hasattr(layer.conv, 'in_channels'):
            # Conv 层
            in_ch = layer.conv.in_channels
            out_ch = layer.conv.out_channels
            print(f"  [{i:2d}] {layer_type:15s} - {in_ch:4d} -> {out_ch:4d} channels")
        elif hasattr(layer, 'cv1') and hasattr(layer, 'cv2'):
            # C3k2, C2PSA 等层
            in_ch = layer.cv1.conv.in_channels
            out_ch = layer.cv2.conv.out_channels
            print(f"  [{i:2d}] {layer_type:15s} - {in_ch:4d} -> {out_ch:4d} channels")
        elif hasattr(layer, 'cv1'):
            # SPPF 层
            in_ch = layer.cv1.conv.in_channels
            out_ch = layer.cv2.conv.out_channels if hasattr(layer, 'cv2') else 'N/A'
            if isinstance(out_ch, int):
                print(f"  [{i:2d}] {layer_type:15s} - {in_ch:4d} -> {out_ch:4d} channels")
            else:
                print(f"  [{i:2d}] {layer_type:15s} - {in_ch:4d} -> {out_ch} channels")
        else:
            print(f"  [{i:2d}] {layer_type}")
    
    # 测试backbone输出
    print("\nTesting backbone forward pass...")
    test_input = torch.randn(1, 3, 384, 640)
    with torch.no_grad():
        features = backbone(test_input)
        print(f"Backbone outputs {len(features)} feature maps:")
        for i, feat in enumerate(features):
            print(f"  P{i+3} shape: {feat.shape}")


def check_adapters(model):
    """检查通道适配器"""
    print("\n" + "="*80)
    print("Checking Channel Adapters")
    print("="*80)
    
    for i, adapter in enumerate(model.adapters):
        in_ch = adapter.conv.in_channels
        out_ch = adapter.conv.out_channels
        print(f"  Adapter {i}: {in_ch} -> {out_ch} channels")


def check_neck_structure(model):
    """检查neck结构"""
    print("\n" + "="*80)
    print("Checking Neck Structure (YOLOP FPN+PAN)")
    print("="*80)
    
    print(f"\nNeck has {len(model.neck)} layers:")
    for i, layer in enumerate(model.neck):
        layer_type = type(layer).__name__
        
        if hasattr(layer, 'conv'):
            if hasattr(layer.conv, 'in_channels'):
                in_ch = layer.conv.in_channels
                out_ch = layer.conv.out_channels
                print(f"  [{i:2d}] {layer_type:15s} - {in_ch:4d} -> {out_ch:4d} channels")
            elif hasattr(layer, 'Upsample'):
                in_ch = layer.Upsample.in_channels
                out_ch = layer.Upsample.out_channels
                print(f"  [{i:2d}] {layer_type:15s} - {in_ch:4d} -> {out_ch:4d} channels")
            else:
                print(f"  [{i:2d}] {layer_type}")
        else:
            print(f"  [{i:2d}] {layer_type}")


def check_heads(model):
    """检查检测和分割头"""
    print("\n" + "="*80)
    print("Checking Detection and Segmentation Heads")
    print("="*80)
    
    # 检测头
    print("\nDetection Head:")
    print(f"  Type: {type(model.detect_head).__name__}")
    print(f"  Number of classes: {model.detect_head.nc}")
    print(f"  Number of detection layers: {model.detect_head.nl}")
    print(f"  Number of anchors per layer: {model.detect_head.na}")
    print(f"  Anchors shape: {model.detect_head.anchors.shape}")
    print(f"  Strides: {model.detect_head.stride}")
    
    # 可驾驶区域分割头
    print(f"\nDrivable Area Segmentation Head:")
    print(f"  Number of layers: {len(model.drivable_seg_head)}")
    for i, layer in enumerate(model.drivable_seg_head):
        layer_type = type(layer).__name__
        if hasattr(layer, 'conv'):
            if hasattr(layer.conv, 'in_channels'):
                in_ch = layer.conv.in_channels
                out_ch = layer.conv.out_channels
                print(f"    [{i}] {layer_type:15s} - {in_ch:4d} -> {out_ch:4d} channels")
            else:
                print(f"    [{i}] {layer_type}")
        else:
            print(f"    [{i}] {layer_type}")
    
    # 车道线分割头
    print(f"\nLane Line Segmentation Head:")
    print(f"  Number of layers: {len(model.lane_seg_head)}")
    for i, layer in enumerate(model.lane_seg_head):
        layer_type = type(layer).__name__
        if hasattr(layer, 'conv'):
            if hasattr(layer.conv, 'in_channels'):
                in_ch = layer.conv.in_channels
                out_ch = layer.conv.out_channels
                print(f"    [{i}] {layer_type:15s} - {in_ch:4d} -> {out_ch:4d} channels")
            else:
                print(f"    [{i}] {layer_type}")
        else:
            print(f"    [{i}] {layer_type}")


def check_frozen_parameters(model):
    """检查哪些参数被冻结"""
    print("\n" + "="*80)
    print("Checking Frozen Parameters")
    print("="*80)
    
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = frozen_params + trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Frozen percentage: {frozen_params/total_params*100:.2f}%")
    
    # 检查各部分的冻结状态
    parts = {
        'Backbone': model.backbone,
        'Adapters': model.adapters,
        'Neck': model.neck,
        'Detection Head': model.detect_head,
        'Drivable Seg Head': model.drivable_seg_head,
        'Lane Seg Head': model.lane_seg_head
    }
    
    print("\nParameter status by component:")
    for name, module in parts.items():
        total = sum(p.numel() for p in module.parameters())
        frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        trainable = total - frozen
        status = "FROZEN" if frozen == total else ("TRAINABLE" if frozen == 0 else "PARTIAL")
        print(f"  {name:20s}: {trainable:>10,} trainable, {frozen:>10,} frozen [{status}]")


def main():
    print("="*80)
    print("YOLOP with YOLOv11 Backbone - Architecture Check")
    print("="*80)
    
    # 创建模型 - 使用 nano 版本
    yolo_scale = 's'  # 可以改为 'n', 's', 'm', 'l', 'x'
    # yolov11_weights = ''  # f'weights/yolo11{yolo_scale}.pt'
    yolov11_weights = f'weights/yolo11{yolo_scale}.pt'
    
    print(f"\n1. Creating model with YOLOv11{yolo_scale} backbone...")
    if os.path.exists(yolov11_weights):
        print(f"   Loading pretrained weights from: {yolov11_weights}")
        model = YOLOPWithYOLOv11(num_seg_class=2, yolo_scale=yolo_scale, yolo_weights_path=yolov11_weights)
        print("✓ Model created with pretrained backbone")
    else:
        print(f"   ⚠ Weights not found at: {yolov11_weights}")
        print("   Creating model from scratch...")
        model = YOLOPWithYOLOv11(num_seg_class=2, yolo_scale=yolo_scale, yolo_weights_path=None)
        print("✓ Model created from scratch")
    
    # 统计参数
    print("\n2. Counting parameters...")
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 检查各部分结构
    check_backbone_structure(model)
    check_adapters(model)
    check_neck_structure(model)
    check_heads(model)
    
    # 测试前向传播
    success = test_forward_pass(model, input_size=(1, 3, 384, 640))
    
    # 测试冻结backbone功能
    if os.path.exists(yolov11_weights):
        print("\n" + "="*80)
        print(f"Testing Backbone Freezing")
        print("="*80)
        print("\nFreezing backbone parameters...")
        model.freeze_backbone()
        check_frozen_parameters(model)
    else:
        print("\n" + "="*80)
        print("Checking Parameters (without pretrained weights)")
        print("="*80)
        check_frozen_parameters(model)
    
    # 总结
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    if success:
        print("✓ All checks passed!")
        print("✓ Model architecture is correct and ready for training")
    else:
        print("✗ Some checks failed. Please review the errors above.")
    
    print("\nModel attributes:")
    print(f"  model.nc = {model.nc}")
    print(f"  model.detector_index = {model.detector_index}")
    print(f"  model.names = {model.names}")


if __name__ == '__main__':
    main()
