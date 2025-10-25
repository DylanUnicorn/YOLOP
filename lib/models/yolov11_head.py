"""
YOLOv11风格的无锚点检测头
基于民间YOLOv11实现，适配YOLOP
"""
import torch
import torch.nn as nn
import math
from lib.models.common import Conv


class DFL(nn.Module):
    """Distribution Focal Loss (DFL)"""
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class DetectV11(nn.Module):
    """YOLOv11风格的无锚点检测头"""
    
    def __init__(self, nc=1, ch=(128, 256, 512)):
        """
        Args:
            nc: 类别数
            ch: 输入通道数 tuple (P3, P4, P5)
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        # 构建检测头
        c_box = max(64, ch[0] // 4)
        c_cls = max(80, ch[0], self.nc)
        
        self.box_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        
        for c in ch:
            # Box分支
            self.box_convs.append(nn.Sequential(
                Conv(c, c_box, 3, 1),
                Conv(c_box, c_box, 3, 1),
                nn.Conv2d(c_box, 4 * self.reg_max, 1)
            ))
            # Class分支
            self.cls_convs.append(nn.Sequential(
                Conv(c, c, 3, 1, g=c),  # depthwise
                Conv(c, c_cls, 1, 1),
                Conv(c_cls, c_cls, 3, 1, g=c_cls),  # depthwise
                Conv(c_cls, c_cls, 1, 1),
                nn.Conv2d(c_cls, self.nc, 1)
            ))
        
        self.dfl = DFL(self.reg_max)
    
    def forward(self, x):
        """
        Args:
            x: list of 3 feature maps [P3, P4, P5]
        Returns:
            if training: list of 3 tensors [box+cls per layer]
            else: (inference, train_out)
        """
        outputs = []
        for i in range(self.nl):
            box = self.box_convs[i](x[i])
            cls = self.cls_convs[i](x[i])
            outputs.append(torch.cat([box, cls], dim=1))
        
        if self.training:
            return outputs
        
        # 推理模式
        return self.inference(outputs), outputs
    
    def inference(self, x):
        """推理输出，转换为 [batch, anchors, 4+nc] 格式"""
        # 生成anchor points
        anchors, strides = self._make_anchors(x)
        
        # 拼接所有层的输出
        x_cat = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x_cat.split([4 * self.reg_max, self.nc], dim=1)
        
        # DFL解码box
        # box = box.permute(0, 2, 1).contiguous()
        box = self.dfl(box)  # [b, 4, anchors]
        
        # 转换为 xyxy 格式
        box = box.permute(0, 2, 1).contiguous()  # [b, anchors, 4]
        lt, rb = box.chunk(2, -1)
        x1y1 = anchors.unsqueeze(0) - lt
        x2y2 = anchors.unsqueeze(0) + rb
        box = torch.cat([x1y1, x2y2], dim=-1) * strides.unsqueeze(0)
        
        # 分类分数
        cls = cls.permute(0, 2, 1).sigmoid()
        
        # 拼接为 [batch, anchors, 4+nc]
        return torch.cat([box, cls], dim=-1)
    
    def _make_anchors(self, x, offset=0.5):
        """生成anchor points和strides"""
        anchors, strides = [], []
        for i, xi in enumerate(x):
            _, _, h, w = xi.shape
            dtype, device = xi.dtype, xi.device
            stride = self.stride[i]
            
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchors.append(torch.stack([sx, sy], -1).view(-1, 2))
            strides.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
        return torch.cat(anchors), torch.cat(strides)
    
    def initialize_biases(self):
        """初始化偏置"""
        for box_conv, cls_conv, stride in zip(self.box_convs, self.cls_convs, self.stride):
            # Box分支
            box_conv[-1].bias.data[:] = 1.0
            # Class分支
            cls_conv[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)
