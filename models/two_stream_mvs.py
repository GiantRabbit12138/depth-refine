import torch
import torch.nn as nn
import torch.nn.functional as F
# --- 新增导入 ---
from torch.utils.checkpoint import checkpoint
# ------------------
from models.rgb_branch import RGBBranch
from models.depth_branch import DepthBranch
from models.fusion_net import UNet3D
from utils.depth_utils import soft_argmax

class TwoStreamMVS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # RGB Branch
        self.rgb_branch = RGBBranch(fpn_channels=config['model']['fpn_channels'])

        # Depth Prior Branch
        self.depth_branch = DepthBranch(feature_channels=config['model']['depth_net_channels'])

        # Fusion and Regularization Network
        fusion_in_channels = 1 + config['model']['depth_net_channels']
        self.fusion_net = UNet3D(
            in_channels=fusion_in_channels,
            base_channels=config['model']['unet_base_channels']
        )

    def forward(self, batch):
        images = batch['images']
        init_depths = batch['init_depths']
        cam_intrinsics = batch['cam_intrinsics']
        cam_poses = batch['cam_poses']
        B, _, _, H, W = images.shape

        depth_hypotheses = torch.linspace(
            self.config['model']['min_depth'],
            self.config['model']['max_depth'],
            self.config['model']['num_depth_hypotheses'],
            device=images.device
        ).unsqueeze(0).repeat(B, 1)

        # 1. RGB Branch -> Cost Volume
        cost_volume = self.rgb_branch(images,
                                      cam_intrinsics,
                                      cam_poses,
                                      depth_hypotheses)

        # 2. Depth Branch -> Prior Volume
        ref_init_depth = init_depths[:, 0]
        prior_volume = self.depth_branch(ref_init_depth, depth_hypotheses)

        # 对齐两个分支的分辨率
        target_h, target_w = cost_volume.shape[-2:]
        B, C, D, H_prior, W_prior = prior_volume.shape
        prior_volume_reshaped = prior_volume.view(B * D, C, H_prior, W_prior)
        downsampled_prior_volume = F.avg_pool2d(prior_volume_reshaped, kernel_size=4, stride=4)
        prior_volume_aligned = downsampled_prior_volume.view(B, C, D, target_h, target_w)

        # 3. Fusion
        fused_volume = torch.cat([cost_volume, prior_volume_aligned], dim=1)

        # 4. 3D U-Net Regularization with Gradient Checkpointing
        # --- 修改开始 ---
        # 使用梯度检查点来减少内存消耗。
        # checkpoint 会在前向传播时运行一次 fusion_net，但不保存中间激活值。
        # 在反向传播时，它会重新计算 fusion_net 的前向传播来获取梯度所需的激活值。
        # if self.training:
        #     # `use_reentrant=False` 是新版PyTorch推荐的模式，效率更高。
        #     prob_volume = checkpoint(self.fusion_net, fused_volume, use_reentrant=False)
        # else:
        #     # 在评估/推理时，不需要计算梯度，正常调用即可，以避免不必要的重计算开销。
        #     prob_volume = self.fusion_net(fused_volume)
        # --- 修改结束 ---
        prob_volume = checkpoint(self.fusion_net, fused_volume, use_reentrant=False)

        # 5. Softmax over depth dimension to get probabilities
        prob_volume = torch.softmax(prob_volume, dim=1)

        # 6. Regress final depth map
        refined_depth = soft_argmax(prob_volume, depth_hypotheses)

        return {"refined_depth": refined_depth, "prob_volume": prob_volume}
