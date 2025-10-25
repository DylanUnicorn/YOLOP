"""
YOLOv11风格的损失函数
支持无锚点检测 + 分割损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def make_anchors(features, strides, offset=0.5):
    """生成anchor points"""
    anchors, stride_tensors = [], []
    for i, feat in enumerate(features):
        _, _, h, w = feat.shape
        dtype, device = feat.dtype, feat.device
        stride = strides[i]
        
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchors.append(torch.stack([sx, sy], -1).view(-1, 2))
        stride_tensors.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    
    return torch.cat(anchors), torch.cat(stride_tensors)


def compute_iou(box1, box2, eps=1e-7):
    """计算IoU (CIoU)"""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    
    # CIoU
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class TaskAlignedAssigner(nn.Module):
    """Task-Aligned Assigner for YOLO"""
    def __init__(self, nc=1, top_k=10, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.top_k = top_k
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        batch_size = pd_scores.size(0)
        num_max_boxes = gt_bboxes.size(1)

        if num_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        num_anchors = anc_points.shape[0]
        
        # 计算anchor是否在GT box内
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2)
        mask_in_gts = bbox_deltas.amin(dim=-1).gt(self.eps)
        mask_in_gts = mask_in_gts.view(batch_size, num_max_boxes, num_anchors)
        
        na = pd_bboxes.shape[-2]
        mask_gt = (mask_in_gts * mask_gt).bool()
        
        overlaps = torch.zeros([batch_size, num_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([batch_size, num_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, batch_size, num_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=batch_size).view(-1, 1).expand(-1, num_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        # top_k_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        # top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k, dim=-1, largest=True)
        # if top_k_mask is not None:
        #     top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)
        # top_k_indices.masked_fill_(~top_k_mask, 0)

        # mask_top_k = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        # ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        # for k in range(self.top_k):
        #     mask_top_k.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        # mask_top_k.masked_fill_(mask_top_k > 1, 0)
        # mask_pos = mask_top_k.to(align_metric.dtype) * mask_in_gts * mask_gt
        
        # ensure top_k not larger than candidate anchors
        top_k = min(self.top_k, align_metric.shape[-1])

        # top-k values and their anchor indices along last dim (anchors)
        top_k_metrics, top_k_indices = torch.topk(align_metric, top_k, dim=-1, largest=True)  # [B, G, top_k]

        # mask invalid top-k where metric is too small
        valid_topk = (top_k_metrics > self.eps)  # [B, G, top_k]
        # get mask_gt values at those top_k indices: mask_gt has shape [B, G, NA]
        topk_mask_gt = mask_gt.gather(dim=-1, index=top_k_indices)  # [B, G, top_k]

        # only keep topk indices that both valid metric and mask_gt==True
        keep_topk = (valid_topk & topk_mask_gt.bool())

        # zero out invalid indices (so scatter_add won't place them)
        top_k_indices = top_k_indices.masked_fill(~keep_topk, 0)

        # build mask_top_k of shape [B, G, NA] by scattering 1s into top_k_indices positions
        mask_top_k = torch.zeros_like(align_metric, dtype=torch.int8)  # [B, G, NA]
        ones = torch.ones((top_k_indices.shape[0], top_k_indices.shape[1], 1), dtype=torch.int8, device=top_k_indices.device)
        for k in range(top_k):
            mask_top_k.scatter_add_(-1, top_k_indices[:, :, k:k + 1], ones)
        mask_top_k = mask_top_k.clamp_max(1)  # prevent multi-count >1

        # final positive mask (dtype float for later computations)
        mask_pos = mask_top_k.to(align_metric.dtype) * mask_in_gts * mask_gt.float()


        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        
        target_gt_idx = mask_pos.argmax(-2)

        # Assigned target
        batch_index = torch.arange(end=batch_size, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_index = target_gt_idx + batch_index * num_max_boxes
        target_labels = gt_labels.long().flatten()[target_index]
        target_bboxes = gt_bboxes.view(-1, 4)[target_index]

        target_labels.clamp_(0)
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                    dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()


class BboxLoss(nn.Module):
    """Box + DFL loss"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        a, b = target_bboxes.chunk(2, -1)
        target = torch.cat((anchor_points - a, b - anchor_points), -1)
        target = target.clamp(0, self.reg_max - 0.01)
        loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target[fg_mask])
        loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        left_loss = F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)
        right_loss = F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)
        return (left_loss * wl + right_loss * wr).mean(-1, keepdim=True)


class YOLOv11Loss(nn.Module):
    """YOLOv11风格的损失函数"""
    def __init__(self, model, device):
        super().__init__()
        self.device = device
        self.nc = model.nc
        self.reg_max = model.detect_head.reg_max
        self.no = model.detect_head.no
        self.stride = model.detect_head.stride
        
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss(self.reg_max - 1)
        self.assigner = TaskAlignedAssigner(nc=self.nc, top_k=10, alpha=0.5, beta=6.0)
        
        self.project = torch.arange(self.reg_max, dtype=torch.float, device=device)
        
        # 分割损失
        self.bce_seg = nn.BCELoss(reduction='mean')
        
        # 损失权重
        self.hyp_box = 7.5
        self.hyp_cls = 0.5
        self.hyp_dfl = 1.5
        self.hyp_seg_da = 0.02
        self.hyp_seg_ll = 0.04

    def box_decode(self, anchor_points, pred_dist):
        """解码box"""
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        return torch.cat([anchor_points - lt, anchor_points + rb], dim=-1)

    def preprocess_targets(self, targets, batch_size, scale_tensor):
        """预处理目标"""
        if targets[0].shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)

        idx = targets[0][:, 0].view(-1, 1)
        cls = targets[0][:, 1].view(-1, 1)
        box = targets[0][:, 2:6]

        targets_cat = torch.cat((idx, cls, box), dim=1).to(self.device)
        
        i = targets_cat[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets_cat[matches, 1:]
        
        # 转换为 xyxy
        # xy = out[..., 1:3] * scale_tensor
        # scale_xy = scale_tensor[[1, 0]]  # 对应 (width, height)
        xy = out[..., 1:3] * scale_tensor[:2]
        wh = out[..., 3:5] * scale_tensor[2:]
        out[..., 1:5] = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)
        
        return out

    def forward(self, outputs, targets, shapes, model):
        """
        Args:
            outputs: [det_out, da_seg_out, ll_seg_out]
            targets: [det_target, da_target, ll_target]
        """
        det_out, da_seg_out, ll_seg_out = outputs
        det_target, da_target, ll_target = targets
        
        loss = torch.zeros(3, device=self.device)
        
        # === 检测损失 ===
        if self.training:
            # Training模式: det_out是list of 3 feature maps
            feats = det_out
        else:
            # Inference模式: det_out是tuple (inference_out, train_out)
            _, feats = det_out
        
        # 拼接特征
        batch_size = feats[0].shape[0]
        dtype = feats[0].dtype
        
        x_cat = torch.cat([i.view(batch_size, self.no, -1) for i in feats], dim=2)
        pred_distri, pred_scores = x_cat.split([self.reg_max * 4, self.nc], dim=1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        
        # 生成anchor points
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        # 预处理目标
        targets_det = self.preprocess_targets(targets, batch_size, imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets_det.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        # 解码预测框
        pred_bboxes = self.box_decode(anchor_points, pred_distri)
        
        # Task-aligned assignment
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # 分类损失
        loss[1] = self.bce_cls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        
        # Box损失
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss_iou, loss_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            loss[0] = loss_iou + loss_dfl
        
        # === 分割损失 ===
        loss_da = self.bce_seg(da_seg_out, da_target)
        loss_ll = self.bce_seg(ll_seg_out, ll_target)
        
        # 加权总损失
        loss[0] *= self.hyp_box  # box
        loss[1] *= self.hyp_cls  # cls
        loss[2] = loss_da * self.hyp_seg_da + loss_ll * self.hyp_seg_ll  # seg
        
        total_loss = loss.sum()
        
        # 返回损失细节
        head_losses = {
            'box_loss': loss[0].item(),
            'cls_loss': loss[1].item(),
            'seg_loss': loss[2].item(),
            'da_loss': loss_da.item(),
            'll_loss': loss_ll.item(),
        }
        
        return total_loss, head_losses
