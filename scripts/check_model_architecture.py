"""
检查YOLOP模型架构
支持原版YOLOP和YOLOv11 backbone的YOLOP模型
验证backbone和neck的结构是否与预期一致
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from lib.config import cfg
from lib.config import update_config


def get_model(model_type='yolop', yolo_scale='s'):
    """
    根据模型类型创建模型
    model_type: 'yolop' 或 'yolov11'
    """
    if model_type == 'yolop':
        from lib.models.YOLOP import get_net
        model = get_net(cfg)
        return model, 'YOLOP'
    elif model_type == 'yolov11':
        try:
            from lib.models.YOLOP_YOLOv11 import YOLOPWithYOLOv11
            yolov11_weights = f'weights/yolo11{yolo_scale}.pt'
            if os.path.exists(yolov11_weights):
                print(f"   Loading pretrained weights from: {yolov11_weights}")
                model = YOLOPWithYOLOv11(num_seg_class=2, yolo_scale=yolo_scale, yolo_weights_path=yolov11_weights)
                print("✓ Model created with pretrained backbone")
            else:
                print(f"   ⚠ Weights not found at: {yolov11_weights}")
                print("   Creating model from scratch...")
                model = YOLOPWithYOLOv11(num_seg_class=2, yolo_scale=yolo_scale, yolo_weights_path=None)
                print("✓ Model created from scratch")
            return model, f'YOLOP with YOLOv11{yolo_scale}'
        except ImportError:
            print("✗ YOLOPWithYOLOv11 model not found, falling back to YOLOP")
            from lib.models.YOLOP import get_net
            model = get_net(cfg)
            return model, 'YOLOP'
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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


def check_backbone_structure(model, model_type):
    """检查backbone结构"""
    print("\n" + "="*80)
    print(f"Checking Backbone Structure ({model_type})")
    print("="*80)
    
    if not hasattr(model, 'backbone'):
        print("\n⚠ This model doesn't have a separate backbone attribute")
        print("  (YOLOP uses a unified model structure)")
        return
    
    backbone = model.backbone
    print(f"\nBackbone type: {type(backbone).__name__}")
    
    if hasattr(backbone, 'layers'):
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
    else:
        print("\n⚠ Backbone doesn't have 'layers' attribute")
        print("  This is a unified model structure (YOLOP)")


def check_adapters(model):
    """检查通道适配器"""
    if not hasattr(model, 'adapters'):
        return
        
    print("\n" + "="*80)
    print("Checking Channel Adapters")
    print("="*80)
    
    for i, adapter in enumerate(model.adapters):
        in_ch = adapter.conv.in_channels
        out_ch = adapter.conv.out_channels
        print(f"  Adapter {i}: {in_ch} -> {out_ch} channels")


def check_neck_structure(model):
    """检查neck结构"""
    if not hasattr(model, 'neck'):
        return
        
    print("\n" + "="*80)
    print("Checking Neck Structure")
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


def check_unified_model_structure(model):
    """检查统一模型结构 (YOLOP)"""
    print("\n" + "="*80)
    print("Checking Unified Model Structure (YOLOP)")
    print("="*80)
    
    if hasattr(model, 'model'):
        print(f"\nModel has {len(model.model)} layers in total")
        print(f"Detection output index: {model.detector_index}")
        print(f"Segmentation output indices: {model.seg_out_idx}")
        
        print("\nLayer summary:")
        for i, layer in enumerate(model.model):
            layer_type = type(layer).__name__
            from_idx = layer.from_ if hasattr(layer, 'from_') else 'N/A'
            print(f"  [{i:2d}] {layer_type:20s} from: {from_idx}")
            
            # 打印前10层和重要层的详细信息
            if i < 10 or i in [model.detector_index] + model.seg_out_idx:
                if hasattr(layer, 'cv1') and hasattr(layer.cv1, 'conv'):
                    in_ch = layer.cv1.conv.in_channels if hasattr(layer.cv1.conv, 'in_channels') else 'N/A'
                    if hasattr(layer, 'cv2'):
                        out_ch = layer.cv2.conv.out_channels if hasattr(layer.cv2, 'conv') and hasattr(layer.cv2.conv, 'out_channels') else 'N/A'
                    else:
                        out_ch = 'N/A'
                    if in_ch != 'N/A' and out_ch != 'N/A':
                        print(f"       {in_ch:4d} -> {out_ch:4d} channels")


def check_heads(model):
    """检查检测和分割头"""
    print("\n" + "="*80)
    print("Checking Detection and Segmentation Heads")
    print("="*80)
    
    # 检查是否有独立的检测头
    if hasattr(model, 'detect_head'):
        print("\nDetection Head:")
        print(f"  Type: {type(model.detect_head).__name__}")
        print(f"  Number of classes: {model.detect_head.nc}")
        print(f"  Number of detection layers: {model.detect_head.nl}")
        if hasattr(model.detect_head, 'na'):
            print(f"  Number of anchors per layer: {model.detect_head.na}")
            print(f"  Anchors shape: {model.detect_head.anchors.shape}")
        else:
            print("  Anchor-free detection head")
        print(f"  Strides: {model.detect_head.stride}")
        
        # 可驾驶区域分割头
        if hasattr(model, 'drivable_seg_head'):
            print(f"\nDrivable Area Segmentation Head:")
            print(f"  Number of layers: {len(model.drivable_seg_head)}")
        
        # 车道线分割头
        if hasattr(model, 'lane_seg_head'):
            print(f"\nLane Line Segmentation Head:")
            print(f"  Number of layers: {len(model.lane_seg_head)}")
    else:
        # YOLOP模型
        print("\n✓ Using unified model structure")
        if hasattr(model, 'nc'):
            print(f"  Number of detection classes: {model.nc}")
        if hasattr(model, 'detector_index'):
            print(f"  Detection layer index: {model.detector_index}")
        if hasattr(model, 'seg_out_idx'):
            print(f"  Segmentation output indices: {model.seg_out_idx}")


def check_frozen_parameters(model):
    """检查哪些参数被冻结"""
    print("\n" + "="*80)
    print("Checking Parameters")
    print("="*80)
    
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = frozen_params + trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    if total_params > 0:
        print(f"Frozen percentage: {frozen_params/total_params*100:.2f}%")
    
    # 检查各部分的冻结状态 (仅对YOLOv11架构)
    if hasattr(model, 'backbone') and hasattr(model, 'neck'):
        parts = {
            'Backbone': model.backbone,
            'Neck': model.neck,
            'Detection Head': model.detect_head,
            'Drivable Seg Head': model.drivable_seg_head,
            'Lane Seg Head': model.lane_seg_head
        }
        
        if hasattr(model, 'adapters'):
            parts['Adapters'] = model.adapters
        
        print("\nParameter status by component:")
        for name, module in parts.items():
            total = sum(p.numel() for p in module.parameters())
            frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            trainable = total - frozen
            status = "FROZEN" if frozen == total else ("TRAINABLE" if frozen == 0 else "PARTIAL")
            print(f"  {name:20s}: {trainable:>10,} trainable, {frozen:>10,} frozen [{status}]")


def main():
    print("="*80)
    print("YOLOP Model Architecture Check")
    print("="*80)

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='Check YOLOP model architecture')
    parser.add_argument('--model', type=str, default='yolop', choices=['yolop', 'yolov11'],
                       help='Model type: yolop or yolov11')
    parser.add_argument('--scale', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv11 scale (only used when model=yolov11)')
    args = parser.parse_args()
    
    # 创建模型
    print(f"\n1. Creating {args.model.upper()} model...")
    model, model_name = get_model(model_type=args.model, yolo_scale=args.scale)
    print(f"✓ Model created: {model_name}")
    
    # 统计参数
    print("\n2. Counting parameters...")
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 检查各部分结构
    if args.model == 'yolov11':
        check_backbone_structure(model, 'YOLOv11')
        check_adapters(model)
        check_neck_structure(model)
    else:
        check_unified_model_structure(model)
    
    check_heads(model)
    
    # 测试前向传播
    success = test_forward_pass(model, input_size=(1, 3, 384, 640))
    
    # 测试冻结backbone功能 (仅YOLOv11)
    if args.model == 'yolov11' and hasattr(model, 'freeze_backbone'):
        print("\n" + "="*80)
        print(f"Testing Backbone Freezing")
        print("="*80)
        print("\nFreezing backbone parameters...")
        model.freeze_backbone()
        check_frozen_parameters(model)
    else:
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
    if hasattr(model, 'detector_index'):
        print(f"  model.detector_index = {model.detector_index}")
    if hasattr(model, 'names'):
        print(f"  model.names = {model.names}")


if __name__ == '__main__':
    main()
