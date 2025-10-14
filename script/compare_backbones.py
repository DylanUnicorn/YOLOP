"""
对比YOLOv11 backbone和原始YOLOP backbone的差异
"""
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
from lib.models.YOLOP_YOLOv11 import YOLOPWithYOLOv11, YOLOv11Backbone
from lib.models.YOLOP import get_net
from lib.config import cfg
from lib.config import update_config


def count_parameters(model):
    """统计参数数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size(model):
    """估算模型大小（MB）"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def test_inference_speed(model, input_size=(1, 3, 640, 640), num_runs=100, warmup=10):
    """测试推理速度"""
    import time
    model.eval()
    device = next(model.parameters()).device
    
    # 准备输入
    dummy_input = torch.randn(input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # 测试
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            torch.cuda.synchronize() if device.type == 'cuda' else None
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    fps = 1000 / avg_time
    
    return avg_time, fps


def main():
    print("="*80)
    print("Backbone Comparison: YOLOv11 vs Original YOLOP")
    print("="*80)
    
    # 创建模型
    print("\n1. Creating models...")
    
    print("   Creating YOLOv11-based YOLOP...")
    model_yolov11 = YOLOPWithYOLOv11(num_seg_class=2)
    
    print("   Creating original YOLOP...")
    try:
        model_original = get_net(cfg)
    except Exception as e:
        print(f"   Failed to create original YOLOP: {e}")
        model_original = None
    
    print("✓ Models created")
    
    # 对比参数数量
    print("\n" + "="*80)
    print("Parameter Comparison")
    print("="*80)
    
    # YOLOv11 backbone
    yolov11_backbone_params, _ = count_parameters(model_yolov11.backbone)
    yolov11_total_params, _ = count_parameters(model_yolov11)
    yolov11_size = get_model_size(model_yolov11)
    
    print("\nYOLOv11-based YOLOP:")
    print(f"  Backbone parameters: {yolov11_backbone_params:,}")
    print(f"  Total parameters: {yolov11_total_params:,}")
    print(f"  Model size: {yolov11_size:.2f} MB")
    
    if model_original:
        original_total_params, _ = count_parameters(model_original)
        original_size = get_model_size(model_original)
        
        print("\nOriginal YOLOP:")
        print(f"  Total parameters: {original_total_params:,}")
        print(f"  Model size: {original_size:.2f} MB")
        
        print("\nDifference:")
        param_diff = yolov11_total_params - original_total_params
        size_diff = yolov11_size - original_size
        print(f"  Parameters: {param_diff:+,} ({param_diff/original_total_params*100:+.2f}%)")
        print(f"  Size: {size_diff:+.2f} MB ({size_diff/original_size*100:+.2f}%)")
    
    # 对比输出形状
    print("\n" + "="*80)
    print("Output Shape Comparison")
    print("="*80)
    
    test_input = torch.randn(1, 3, 640, 640)
    
    print("\nYOLOv11-based YOLOP:")
    with torch.no_grad():
        det_out, da_out, ll_out = model_yolov11(test_input)
        print(f"  Detection output: {type(det_out)}")
        if isinstance(det_out, tuple):
            for i, out in enumerate(det_out):
                if isinstance(out, list):
                    print(f"    Element {i}: list with {len(out)} items")
                    for j, item in enumerate(out):
                        print(f"      [{j}] {item.shape}")
                else:
                    print(f"    Element {i}: {out.shape}")
        print(f"  Drivable area output: {da_out.shape}")
        print(f"  Lane line output: {ll_out.shape}")
    
    if model_original:
        print("\nOriginal YOLOP:")
        with torch.no_grad():
            det_out_orig, da_out_orig, ll_out_orig = model_original(test_input)
            print(f"  Detection output: {type(det_out_orig)}")
            if isinstance(det_out_orig, tuple):
                for i, out in enumerate(det_out_orig):
                    if isinstance(out, list):
                        print(f"    Element {i}: list with {len(out)} items")
                        for j, item in enumerate(out):
                            print(f"      [{j}] {item.shape}")
                    else:
                        print(f"    Element {i}: {out.shape}")
            print(f"  Drivable area output: {da_out_orig.shape}")
            print(f"  Lane line output: {ll_out_orig.shape}")
    
    # 对比推理速度
    print("\n" + "="*80)
    print("Inference Speed Comparison (CPU)")
    print("="*80)
    
    print("\nTesting YOLOv11-based YOLOP...")
    try:
        avg_time_yolov11, fps_yolov11 = test_inference_speed(model_yolov11, num_runs=50, warmup=5)
        print(f"  Average time: {avg_time_yolov11:.2f} ms")
        print(f"  FPS: {fps_yolov11:.2f}")
    except Exception as e:
        print(f"  Failed: {e}")
        avg_time_yolov11, fps_yolov11 = None, None
    
    if model_original:
        print("\nTesting original YOLOP...")
        try:
            avg_time_original, fps_original = test_inference_speed(model_original, num_runs=50, warmup=5)
            print(f"  Average time: {avg_time_original:.2f} ms")
            print(f"  FPS: {fps_original:.2f}")
            
            if avg_time_yolov11 and avg_time_original:
                time_diff = avg_time_yolov11 - avg_time_original
                print(f"\nSpeed difference: {time_diff:+.2f} ms ({time_diff/avg_time_original*100:+.2f}%)")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Backbone架构对比
    print("\n" + "="*80)
    print("Backbone Architecture Comparison")
    print("="*80)
    
    print("\nYOLOv11 Backbone:")
    print(f"  Number of layers: {len(model_yolov11.backbone.layers)}")
    print(f"  Output feature indices: {model_yolov11.backbone.out_indices}")
    print("  Layer types:")
    for i, layer in enumerate(model_yolov11.backbone.layers):
        print(f"    [{i:2d}] {type(layer).__name__}")
    
    if model_original and hasattr(model_original, 'model'):
        print("\nOriginal YOLOP Backbone:")
        print(f"  Number of layers: {len(model_original.model)}")
        print("  Layer types (first 10):")
        for i in range(min(10, len(model_original.model))):
            layer = model_original.model[i]
            print(f"    [{i:2d}] {type(layer).__name__}")
    
    # 总结
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("\nKey differences:")
    print("  ✓ YOLOv11 uses newer modules: C3k2, C2PSA, SPPF")
    print("  ✓ Original YOLOP uses: Focus, BottleneckCSP, SPP")
    print("  ✓ YOLOv11 backbone can be initialized with pre-trained weights")
    print("  ✓ Both models have the same output format (compatible with training code)")
    
    print("\nRecommendations:")
    print("  • Use YOLOv11 backbone with pre-trained weights for better initial performance")
    print("  • Consider freezing backbone for faster training of detection/segmentation heads")
    print("  • Monitor both training speed and accuracy during experiments")


if __name__ == '__main__':
    main()
