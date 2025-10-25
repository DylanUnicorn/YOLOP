import torch
import torch.nn as nn
import math
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, Concat
from lib.models.common import Focus, BottleneckCSP
from lib.models.yolov11_head import DetectV11
from lib.utils import check_anchor_order
import logging

class YOLOv11Backbone(nn.Module):
    def __init__(self, width_multiple=0.25, depth_multiple=0.50, yolo_model_path=None):
        """
        YOLOv11 Backbone - 直接从 ultralytics YOLO 模型提取
        
        Args:
            width_multiple: 通道数缩放因子 (n=0.25, s=0.50, m=1.00, l=1.00, x=1.50)
            depth_multiple: 深度缩放因子 (n=0.50, s=0.50, m=0.50, l=1.00, x=1.00)
            yolo_model_path: YOLOv11 预训练模型路径（可选）

        Warning:
            不同的yolo model(n, s, m, l, x)模型结构都会不同，目前这个是以 small 为例，
            恰好可以输出(128, 256, 512)通道数 (虽然有adapter也无所谓)
        """
        super().__init__()

        self.out_indices = [3, 6, 10]  # P3, P4, P5
        
        # 如果提供了预训练模型路径，直接加载
        if yolo_model_path:
            yolo = YOLO(yolo_model_path)
            yolo_model = yolo.model
            
            # 提取 backbone 层 (0-10)
            self.layers = nn.ModuleList([yolo_model.model[i] for i in range(11)])
            
            # 获取输出通道数 (C3k2 和 C2PSA 都有 cv2 属性)
            self.out_channels = [
                yolo_model.model[self.out_indices[0]].conv.out_channels,   # P3 (C3k2)
                yolo_model.model[self.out_indices[1]].cv2.conv.out_channels,   # P4 (C3k2)
                yolo_model.model[self.out_indices[2]].cv2.conv.out_channels,  # P5 (C2PSA)
            ]
        else:
            # 如果没有预训练模型，使用 ultralytics 的模块构建
            from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA
            
            # 根据 width_multiple 计算通道数
            def make_divisible(x, divisor=8):
                """确保通道数是 divisor 的倍数"""
                return int(math.ceil(x / divisor) * divisor)
            
            c1 = make_divisible(64 * width_multiple)
            c2 = make_divisible(128 * width_multiple)
            c3 = make_divisible(256 * width_multiple)
            c4 = make_divisible(512 * width_multiple)
            c5 = make_divisible(1024 * width_multiple)
            
            # 根据 depth_multiple 计算重复次数
            n1 = max(round(2 * depth_multiple), 1)  # C3k2 repeats
            
            self.layers = nn.ModuleList([
                Conv(3, c1, k=3, s=2),  # 0
                Conv(c1, c2, k=3, s=2),  # 1
                C3k2(c2, c3, n=n1, shortcut=False, e=0.25),  # 2
                Conv(c3, c3, k=3, s=2),  # 3
                C3k2(c3, c4, n=n1, shortcut=False, e=0.25),  # 4
                Conv(c4, c4, k=3, s=2),  # 5
                C3k2(c4, c4, n=n1, shortcut=True),  # 6
                Conv(c4, c5, k=3, s=2),  # 7
                C3k2(c5, c5, n=n1, shortcut=True),  # 8
                SPPF(c5, c5, k=5),  # 9
                C2PSA(c5, c5, n=n1),  # 10
            ])
            self.out_channels = []
            for i in self.out_indices:
                layer = self.layers[i]
                #（Conv)
                if hasattr(layer, 'conv'):
                    self.out_channels.append(layer.conv.out_channels)
                elif hasattr(layer, 'cv2'):  # (C3k2)
                    self.out_channels.append(layer.cv2.conv.out_channels)
                else:
                    raise AttributeError(f"Layer {i} 没有 conv 或 cv2 属性，请检查模块结构")
    
    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs

class ChannelAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class YOLOPWithYOLOv11(nn.Module):

    def __init__(self, num_seg_class=2, yolo_scale='n', yolo_weights_path=None):
        """
        YOLOP with YOLOv11 Backbone
        
        Args:
            num_seg_class: 分割类别数
            yolo_scale: YOLOv11 规模 ('n', 's', 'm', 'l', 'x')
            yolo_weights_path: YOLOv11 预训练权重路径（可选）
        """
        super().__init__()
        
        # YOLOv11 缩放参数
        scale_configs = {
            'n': {'width': 0.25, 'depth': 0.50},  # nano
            's': {'width': 0.50, 'depth': 0.50},  # small
            'm': {'width': 1.00, 'depth': 0.50},  # medium
            'l': {'width': 1.00, 'depth': 1.00},  # large
            'x': {'width': 1.50, 'depth': 1.00},  # xlarge
        }
        
        if yolo_scale not in scale_configs:
            raise ValueError(f"Invalid yolo_scale: {yolo_scale}. Must be one of {list(scale_configs.keys())}")
        
        scale = scale_configs[yolo_scale]
        
        # 如果提供了权重路径，直接从预训练模型提取 backbone
        if yolo_weights_path:
            self.backbone = YOLOv11Backbone(yolo_model_path=yolo_weights_path)
        else:
            self.backbone = YOLOv11Backbone(width_multiple=scale['width'], depth_multiple=scale['depth'])
        
        # 适配 YOLOv11 输出到 YOLOP neck 输入 [128, 256, 512]
        backbone_channels = self.backbone.out_channels
        neck_channels = [128, 256, 512]
        
        self.adapters = nn.ModuleList([
            ChannelAdapter(backbone_channels[0], neck_channels[0]),  # P3
            ChannelAdapter(backbone_channels[1], neck_channels[1]),  # P4
            ChannelAdapter(backbone_channels[2], neck_channels[2]),  # P5
        ])
        self.seg_adapter = nn.Conv2d(384, 256, kernel_size=1)
        # YOLOP neck (简化版，去掉冗余FPN卷积)
        # 适配后通道: P3=128, P4=256, P5=512
        self.neck = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest'),  # 0: P5上采样
            Concat(dimension=1),  # 1: Concat [P5_up, P4] 
            BottleneckCSP(512 + 256, 256, n=1, shortcut=False),  # 2: 融合P5+P4 -> 256
            nn.Upsample(scale_factor=2, mode='nearest'),  # 3: P4上采样
            Concat(dimension=1),  # 4: Concat [P4_up, P3]
            BottleneckCSP(256 + 128, 128, n=1, shortcut=False),  # 5: 融合P4+P3 -> 128
            Conv(128, 128, k=3, s=2),  # 6: P3下采样
            Concat(dimension=1),  # 7: Concat [P3_down, P4_fpn]
            BottleneckCSP(128 + 256, 256, n=1, shortcut=False),  # 8: 融合 -> 256
            Conv(256, 256, k=3, s=2),  # 9: P4下采样
            Concat(dimension=1),  # 10: Concat [P4_down, P5]
            BottleneckCSP(256 + 512, 512, n=1, shortcut=False),  # 11: 融合 -> 512
        ])
        # YOLOP heads - 使用YOLOv11无锚点检测头
        self.detect_head = DetectV11(nc=1, ch=(128, 256, 512))

        # 分割头输入从p3_fpn (128通道)
        self.drivable_seg_head = nn.ModuleList([
            Conv(256, 128, k=3, s=1),  # 0: 128->64
            nn.Upsample(scale_factor=2, mode='nearest'),  # 1
            BottleneckCSP(128, 64, n=1, shortcut=False),  # 2
            Conv(64, 32, k=3, s=1),  # 3
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4
            Conv(32, 16, k=3, s=1),  # 5
            BottleneckCSP(16, 8, n=1, shortcut=False),  # 6
            nn.Upsample(scale_factor=2, mode='nearest'),  # 7
            Conv(8, num_seg_class, k=3, s=1),  # 8
        ])
        self.lane_seg_head = nn.ModuleList([
            Conv(256, 128, k=3, s=1),  # 0: 128->64
            nn.Upsample(scale_factor=2, mode='nearest'),  # 1
            BottleneckCSP(128, 64, n=1, shortcut=False),  # 2
            Conv(64, 32, k=3, s=1),  # 3
            nn.Upsample(scale_factor=2, mode='nearest'),  # 4
            Conv(32, 16, k=3, s=1),  # 5
            BottleneckCSP(16, 8, n=1, shortcut=False),  # 6
            nn.Upsample(scale_factor=2, mode='nearest'),  # 7
            Conv(8, num_seg_class, k=3, s=1),  # 8
        ])

        # 初始化 Detection Head 的 stride
        s = 128
        with torch.no_grad():
            dummy = torch.zeros(1, 3, s, s)
            detect_out = self.detect_head(self.forward_backbone_neck(dummy))
            self.detect_head.stride = torch.tensor([s / x.shape[-2] for x in detect_out])
        
        self.stride = self.detect_head.stride
        self.detect_head.initialize_biases()

        print(f"Initialized DetectV11 head with strides: {self.detect_head.stride.tolist()}")
        
        # 添加必要的属性以兼容训练代码
        self.nc = 1  # number of classes
        self.detector_index = -1  # detector在模型中的索引
        self.names = ['vehicle']  # class names
        self.model = nn.ModuleList([
            self.backbone,
            self.adapters,
            self.neck,
            self.detect_head,
            self.drivable_seg_head,
            self.lane_seg_head
        ])
        self.detector_index = 3  # detect_head 在第4个位置
        self.det_out_idx = 22  # 更新检测输出索引 (原来是25，现在简化后是22)

        self.gr = 1.0 # giou loss ratio (obj loss ratio is 1-giou)
        
        # 初始化 Detection Head 的偏置
        self._initialize_biases()
    
    def freeze_backbone(self):
        """冻结backbone和adapters的参数"""
        logging.info("Freezing backbone parameters...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.adapters.parameters():
            param.requires_grad = False
        
        # 验证冻结状态
        frozen_count = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        frozen_count += sum(1 for p in self.adapters.parameters() if not p.requires_grad)
        total_count = sum(1 for _ in self.backbone.parameters())
        total_count += sum(1 for _ in self.adapters.parameters())
        logging.info(f"Frozen {frozen_count}/{total_count} backbone+adapter parameters")
        
    def unfreeze_backbone(self):
        """解冻backbone和adapters的参数"""
        logging.info("Unfreezing backbone parameters...")
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.adapters.parameters():
            param.requires_grad = True
    
    def _initialize_biases(self, cf=None):
        """初始化检测头的偏置 - 已在DetectV11中处理"""
        pass
    
    def forward_backbone_neck(self, x):
        """仅前向backbone和neck，用于初始化stride"""
        features = self.backbone(x)
        features = [adapter(f) for adapter, f in zip(self.adapters, features)]
        
        p3, p4, p5 = features[0], features[1], features[2]
        
        x = self.neck[0](p5)
        x = self.neck[1]([x, p4])
        p4_fpn = self.neck[2](x)
        
        x = self.neck[3](p4_fpn)
        x = self.neck[4]([x, p3])
        p3_fpn = self.neck[5](x)
        
        p3_out = p3_fpn
        
        x = self.neck[6](p3_fpn)
        x = self.neck[7]([x, p4_fpn])
        p4_out = self.neck[8](x)
        
        x = self.neck[9](p4_out)
        x = self.neck[10]([x, p5])
        p5_out = self.neck[11](x)
        
        return [p3_out, p4_out, p5_out]
        
    def load_yolov11_backbone_weights(self, weights_path, freeze_backbone=False):
        """
        从YOLOv11预训练模型加载backbone权重
        
        Args:
            weights_path: YOLOv11权重路径（.pt文件）
            freeze_backbone: 是否冻结backbone参数
        """
        try:
            from ultralytics import YOLO
            logging.info(f"Loading YOLOv11 weights from {weights_path}")
            
            # 加载YOLOv11模型
            yolo_model = YOLO(weights_path)
            yolo_state_dict = yolo_model.model.state_dict()
            
            # 映射YOLOv11的backbone权重到我们的模型
            # YOLOv11的backbone层索引: 0-10
            backbone_mapping = {
                # YOLOv11 layer -> our layer
                'model.0': 'backbone.layers.0',   # Conv 3->64
                'model.1': 'backbone.layers.1',   # Conv 64->128
                'model.2': 'backbone.layers.2',   # C3k2 128->256
                'model.3': 'backbone.layers.3',   # Conv 256->256
                'model.4': 'backbone.layers.4',   # C3k2 256->512
                'model.5': 'backbone.layers.5',   # Conv 512->512
                'model.6': 'backbone.layers.6',   # C3k2 512->512
                'model.7': 'backbone.layers.7',   # Conv 512->1024
                'model.8': 'backbone.layers.8',   # C3k2 1024->1024
                'model.9': 'backbone.layers.9',   # SPPF
                'model.10': 'backbone.layers.10', # C2PSA
            }
            
            # 构建新的state dict
            new_state_dict = {}
            loaded_keys = []
            for yolo_key, our_key in backbone_mapping.items():
                for k, v in yolo_state_dict.items():
                    if k.startswith(yolo_key + '.'):
                        new_key = k.replace(yolo_key, our_key)
                        new_state_dict[new_key] = v
                        loaded_keys.append(new_key)
            
            # 加载权重
            model_dict = self.state_dict()
            # 只更新存在的键
            new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            model_dict.update(new_state_dict)
            self.load_state_dict(model_dict)
            
            logging.info(f"Successfully loaded {len(loaded_keys)} backbone parameters from YOLOv11")
            
            # 冻结backbone
            if freeze_backbone:
                self.freeze_backbone()
                logging.info("Backbone frozen successfully")
                
        except Exception as e:
            logging.warning(f"Failed to load YOLOv11 weights: {e}")
            logging.warning("Training will start from scratch")
    
    def forward(self, x):
        features = self.backbone(x)  # YOLOv11 输出 [P3, P4, P5]
        features = [adapter(f) for adapter, f in zip(self.adapters, features)]  # 适配到 [128, 256, 512]
        
        # 简化的 Neck 前向传播 (直接使用YOLOv11的FPN输出)
        p3, p4, p5 = features[0], features[1], features[2]  # [128, 256, 512]通道
        
        # P5 -> P4 融合
        x = self.neck[0](p5)  # 0: P5上采样
        x = self.neck[1]([x, p4])  # 1: concat [P5_up, P4]
        p4_fpn = self.neck[2](x)  # 2: 融合后的P4特征 (256通道)
        
        # P4 -> P3 融合  
        x = self.neck[3](p4_fpn)  # 3: P4上采样
        seg_fpn = self.neck[4]([x, p3])  # 4: concat [P4_up, P3]
        p3_fpn = self.neck[5](seg_fpn)  # 5: 融合后的P3特征 (128通道)
        
        # 构建检测用的多尺度特征
        p3_out = p3_fpn  # 检测用P3 (128通道)
        
        x = self.neck[6](p3_fpn)  # 6: P3下采样
        x = self.neck[7]([x, p4_fpn])  # 7: 与P4_fpn融合  
        p4_out = self.neck[8](x)  # 8: 检测用P4 (256通道)
        
        x = self.neck[9](p4_out)  # 9: P4下采样
        x = self.neck[10]([x, p5])  # 10: 与P5融合
        p5_out = self.neck[11](x)  # 11: 检测用P5 (512通道)
        
        # Heads
        detect_out = self.detect_head([p3_out, p4_out, p5_out])  
        
        # 分割头使用P3的FPN特征 (256通道，与原始设计一致) 后面还可以在这里动手脚，比如添加一个激活块什么的Conv都可以尝试
        seg_fpn = self.seg_adapter(seg_fpn)
        drivable_out = seg_fpn
        for layer in self.drivable_seg_head:
            drivable_out = layer(drivable_out)

        lane_out = seg_fpn
        for layer in self.lane_seg_head:
            lane_out = layer(lane_out)

        drivable_out = torch.sigmoid(drivable_out)
        lane_out = torch.sigmoid(lane_out)

        return [detect_out, drivable_out, lane_out]


def get_net_yolov11(cfg, **kwargs):
    """
    获取带有YOLOv11 backbone的YOLOP模型
    
    Args:
        cfg: 配置对象
        **kwargs: 其他参数，包括：
            - yolov11_weights: YOLOv11预训练权重路径
            - freeze_backbone: 是否冻结backbone
            - yolo_scale: YOLOv11规模 ('n', 's', 'm', 'l', 'x')
    """
    num_seg_class = cfg.num_seg_class if hasattr(cfg, 'num_seg_class') else 2
    yolo_scale = kwargs.get('yolo_scale', 'n')  # 默认使用 nano
    
    # 如果提供了权重路径，直接用权重初始化
    yolov11_weights = kwargs.get('yolov11_weights', f'weights/yolo11{yolo_scale}.pt')
    freeze_backbone = kwargs.get('freeze_backbone', False)
    
    # 在初始化时就加载预训练权重
    import os
    if os.path.exists(yolov11_weights):
        logging.info(f"Creating model with YOLOv11{yolo_scale} pretrained weights from {yolov11_weights}")
        model = YOLOPWithYOLOv11(num_seg_class=num_seg_class, yolo_scale=yolo_scale, yolo_weights_path=yolov11_weights)
        if freeze_backbone:
            model.freeze_backbone()
    else:
        logging.warning(f"YOLOv11 weights not found at {yolov11_weights}, creating model from scratch")
        model = YOLOPWithYOLOv11(num_seg_class=num_seg_class, yolo_scale=yolo_scale, yolo_weights_path=None)
    
    return model