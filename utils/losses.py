import torch
import torch.nn.functional as F

def l1_loss(refined_depth, gt_depth):
    """Computes L1 loss, ignoring invalid ground truth pixels."""
    mask = gt_depth > 0
    # 如果尺寸不匹配，对 refined_depth 进行上采样
    if refined_depth.shape[-2:] != gt_depth.shape[-2:]:
        refined_depth = F.interpolate(
            refined_depth.unsqueeze(1), # 增加一个通道维度 (B, 1, H, W)
            size=gt_depth.shape[-2:],    # 目标尺寸
            mode='bilinear',             # 使用双线性插值
            align_corners=False
        ).squeeze(1) # 移除通道维度 (B, H, W)

    return F.l1_loss(refined_depth[mask], gt_depth[mask])

def smoothness_loss(depth, image):
    """Computes edge-aware smoothness loss."""
    # 如果尺寸不匹配，对 image 进行下采样
    if depth.shape[-2:] != image.shape[-2:]:
        image = F.interpolate(
            image,
            size=depth.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    def gradient(x):
        # 已经在内部处理了 unsqueeze，所以这里不需要
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        l = x
        r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        t = x
        b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        return torch.abs(r - l), torch.abs(b - t)

    depth_dx, depth_dy = gradient(depth.unsqueeze(1))
    image_dx, image_dy = gradient(torch.mean(image, dim=1, keepdim=True))

    loss_dx = torch.mean(depth_dx * torch.exp(-torch.mean(image_dx, dim=1, keepdim=True)))
    loss_dy = torch.mean(depth_dy * torch.exp(-torch.mean(image_dy, dim=1, keepdim=True)))

    return loss_dx + loss_dy

class CombinedLoss(torch.nn.Module):
    def __init__(self, smoothness_weight=0.02):
        super().__init__()
        self.smoothness_weight = smoothness_weight

    def forward(self, outputs, batch):
        refined_depth = outputs['refined_depth']
        # gt_depth 的形状是 [B, 1, H, W]，需要 squeeze
        gt_depth = batch['gt_depth'].squeeze(1)
        ref_image = batch['images'][:, 0]

        # l1_loss 内部会处理尺寸对齐
        main_loss = l1_loss(refined_depth, gt_depth)
        # smoothness_loss 内部也会处理尺寸对齐
        smooth_loss = smoothness_loss(refined_depth, ref_image)

        total_loss = main_loss + self.smoothness_weight * smooth_loss

        return {
            "total_loss": total_loss,
            "l1_loss": main_loss,
            "smoothness_loss": smooth_loss
        }
