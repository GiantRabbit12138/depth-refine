import torch
import torch.nn.functional as F

def warp_features(src_feat, ref_cam_pose, src_cam_pose, ref_cam_intr, src_cam_intr, depth_hypotheses):
    """
    使用单应性变换将源视图特征翘曲到参考视图。

    Args:
        src_feat (torch.Tensor): 源视图的特征图 (B, C, H, W)
        ref_cam_pose (torch.Tensor): 参考视图的相机位姿 (camera-to-world) (B, 4, 4)
        src_cam_pose (torch.Tensor): 源视图的相机位姿 (camera-to-world) (B, 4, 4)
        ref_cam_intr (torch.Tensor): 参考视图的相机内参 (B, 3, 3)
        src_cam_intr (torch.Tensor): 源视图的相机内参 (B, 3, 3)
        depth_hypotheses (torch.Tensor): 深度假设值 (B, D)

    Returns:
        torch.Tensor: 翘曲后的特征体积 (B, C, D, H, W)
    """
    B, C, H, W = src_feat.shape
    D = depth_hypotheses.shape[1]
    device = src_feat.device

    # 1. 创建参考视图的像素坐标网格
    # meshgrid创建一个(H, W)的网格, y_coords和x_coords的形状都是(H, W)
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    # 将坐标展平为(H*W)，并添加齐次坐标1
    # pixel_coords 形状: (3, H*W)
    pixel_coords = torch.stack((x_coords.flatten(), y_coords.flatten(), torch.ones_like(x_coords.flatten())), dim=0).float()

    # 扩展到整个批次
    # pixel_coords 形状: (B, 3, H*W)
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1)

    # 2. 从像素坐标到相机坐标
    # 使用内参矩阵的逆，将像素坐标反投影到参考相机的归一化平面上
    # K_inv @ pixels -> (B, 3, 3) @ (B, 3, H*W) -> (B, 3, H*W)
    # cam_coords_ref 形状: (B, 3, H*W)
    ref_cam_intr_inv = torch.inverse(ref_cam_intr)
    cam_coords_ref = torch.bmm(ref_cam_intr_inv, pixel_coords)

    # 3. 从参考相机坐标到世界坐标
    # 将每个深度假设应用到归一化平面上的点，得到在参考相机坐标系下的3D点
    # depth_hypotheses: (B, D) -> (B, 1, D)
    # cam_coords_ref: (B, 3, H*W) -> (B, 3, H*W, 1)
    # points_in_ref_cam: (B, 3, H*W, D)
    points_in_ref_cam = cam_coords_ref.unsqueeze(3) * depth_hypotheses.view(B, 1, 1, D)

    # 添加齐次坐标，使其成为4D点
    # points_in_ref_cam_homo: (B, 4, H*W, D)
    points_in_ref_cam_homo = torch.cat([points_in_ref_cam, torch.ones(B, 1, H*W, D, device=device)], dim=1)

    # 使用参考相机的位姿(camera-to-world)变换到世界坐标系
    # ref_cam_pose: (B, 4, 4) -> (B, 4, 4, 1, 1)
    # world_coords: (B, 4, H*W, D)
    world_coords = torch.einsum('bxy,byzd->bxzd', ref_cam_pose, points_in_ref_cam_homo)

    # 4. 从世界坐标到源相机坐标
    # 使用源相机位姿的逆(world-to-camera)变换到源相机坐标系
    # src_cam_pose_inv: (B, 4, 4)
    # points_in_src_cam_homo: (B, 4, H*W, D)
    src_cam_pose_inv = torch.inverse(src_cam_pose)
    points_in_src_cam_homo = torch.einsum('bxy,byzd->bxzd', src_cam_pose_inv, world_coords)

    # 5. 从源相机坐标到源像素坐标
    # points_in_src_cam: (B, 3, H*W, D)
    points_in_src_cam = points_in_src_cam_homo[:, :3, :, :]

    # 投影到源图像平面
    # projected_coords: (B, 3, H*W, D)
    projected_coords = torch.einsum('bxy,byzd->bxzd', src_cam_intr, points_in_src_cam)

    # 归一化，将齐次坐标转换为2D像素坐标 (u, v)
    # 加上一个很小的epsilon防止除以零
    u = projected_coords[:, 0:1, :, :] / (projected_coords[:, 2:3, :, :] + 1e-8)
    v = projected_coords[:, 1:2, :, :] / (projected_coords[:, 2:3, :, :] + 1e-8)

    # 6. 创建采样网格 (Grid) 并进行采样
    # F.grid_sample需要标准化的坐标，范围从-1到1
    # u: (0, W-1) -> (-1, 1)
    # v: (0, H-1) -> (-1, 1)
    normalized_u = (u * 2.0 / (W - 1)) - 1.0
    normalized_v = (v * 2.0 / (H - 1)) - 1.0

    # 将u,v合并成grid, 形状: (B, H*W, D, 2)
    grid = torch.cat([normalized_u, normalized_v], dim=1).permute(0, 2, 3, 1).view(B, H*W, D, 2)

    # `F.grid_sample` 需要的grid形状是 (B, D_out, H_out, 2)
    # 我们这里将深度D看作是输出的高度，H*W看作是输出的宽度
    grid_for_sample = grid.view(B, H, W, D, 2).permute(0, 3, 1, 2, 4).reshape(B, D, H*W, 2)

    # src_feat: (B, C, H, W)
    # F.grid_sample要求输入特征图是4D的
    warped_feat = F.grid_sample(src_feat, grid_for_sample, mode='bilinear', padding_mode='zeros', align_corners=True)

    # 恢复形状为 (B, C, D, H, W)
    return warped_feat.view(B, C, D, H, W)