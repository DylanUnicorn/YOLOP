from .YOLOP import get_net as get_net_yolop
from .YOLOP_YOLOv11 import get_net_yolov11

def get_net(cfg, **kwargs):
    """
    根据配置选择合适的模型
    
    Args:
        cfg: 配置对象
        **kwargs: 额外参数（用于 YOLOv11）
    """
    if hasattr(cfg.MODEL, 'USE_YOLOV11') and cfg.MODEL.USE_YOLOV11:
        return get_net_yolov11(cfg, **kwargs)
    else:
        return get_net_yolop(cfg, **kwargs)
