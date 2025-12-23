import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from datasets.mvs_dataset import MVSDataset
from models.two_stream_mvs import TwoStreamMVS
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import gc
import cv2 # 用于保存16位PNG

# compute_depth_metrics 函数保持不变
def compute_depth_metrics(pred, gt):
    """计算单个深度图的评估指标"""
    if pred.shape[-2:] != gt.shape[-2:]:
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=gt.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
    mask = gt > 0
    if not mask.any():
        return None
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    pred_valid = torch.clamp(pred_valid, min=1e-3)
    gt_valid = torch.clamp(gt_valid, min=1e-3)
    thresh = torch.max((gt_valid / pred_valid), (pred_valid / gt_valid))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    rmse = torch.sqrt(((gt_valid - pred_valid) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(gt_valid) - torch.log(pred_valid)) ** 2).mean())
    abs_rel = torch.mean(torch.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = torch.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    return {'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 'rmse': rmse.item(),
            'rmse_log': rmse_log.item(), 'a1': a1.item(), 'a2': a2.item(), 'a3': a3.item()}

def save_depth_maps(output_dir, meta_info, pred_depth, depth_scale=1000.0):
    """
    将预测深度和真实深度保存为16位图和伪彩色图，并按新结构存放。
    - pred_depth: 预测的深度图 (Tensor, HxW)
    - meta_info: 包含场景、图像名和GT路径的字典
    """
    # 1. 解析元数据
    scene_name = meta_info['scene'][0]
    ref_image_name = meta_info['ref_image_name'][0]
    gt_depth_path = meta_info['gt_depth_path'][0]

    # 2. 创建目录结构
    gt_16bit_dir = os.path.join(output_dir, scene_name, 'gt_16bit')
    gt_color_dir = os.path.join(output_dir, scene_name, 'gt_color')
    pred_16bit_dir = os.path.join(output_dir, scene_name, 'pred_16bit_upsampled')
    pred_color_dir = os.path.join(output_dir, scene_name, 'pred_color_upsampled')

    os.makedirs(gt_16bit_dir, exist_ok=True)
    os.makedirs(gt_color_dir, exist_ok=True)
    os.makedirs(pred_16bit_dir, exist_ok=True)
    os.makedirs(pred_color_dir, exist_ok=True)

    # 3. 加载原始GT深度图
    gt_depth_raw = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
    if gt_depth_raw is None:
        print(f"Warning: Could not load GT depth from {gt_depth_path}")
        return
    gt_depth_np = gt_depth_raw.astype(np.float32) / depth_scale

    # 4. 上采样预测深度图到GT尺寸
    gt_h, gt_w = gt_depth_np.shape
    pred_depth_upsampled = F.interpolate(
        pred_depth.unsqueeze(0).unsqueeze(0),
        size=(gt_h, gt_w),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()

    # 5. 保存16位深度图 (uint16)
    pred_16bit = (pred_depth_upsampled * depth_scale).astype(np.uint16)
    # gt_16bit 就是原始文件，直接复制或重新写入
    gt_16bit = (gt_depth_np * depth_scale).astype(np.uint16)

    cv2.imwrite(os.path.join(pred_16bit_dir, ref_image_name), pred_16bit)
    cv2.imwrite(os.path.join(gt_16bit_dir, ref_image_name), gt_16bit)

    # 6. 保存伪彩色可视化图
    def _save_color_map(path, depth_map_np):
        valid_mask = depth_map_np > 0
        if not valid_mask.any():
            min_val, max_val = 0, 1
        else:
            min_val, max_val = depth_map_np[valid_mask].min(), depth_map_np[valid_mask].max()

        depth_normalized = (depth_map_np - min_val) / (max_val - min_val + 1e-8)
        depth_normalized[~valid_mask] = 0
        colormap = plt.get_cmap('viridis')
        depth_colored = (colormap(depth_normalized) * 255).astype(np.uint8)
        Image.fromarray(depth_colored[:, :, :3]).save(path)

    _save_color_map(os.path.join(pred_color_dir, ref_image_name), pred_depth_upsampled)
    _save_color_map(os.path.join(gt_color_dir, ref_image_name), gt_depth_np)


def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 修改：移除硬编码尺寸，保留分块大小 ---
    chunk_size = 10 # 每10个样本清理一次内存
    # ------------------------------------

    output_dir = os.path.join(config['train']['log_path'], "eval_outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Evaluation outputs will be saved to: {output_dir}")

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 修改：从config文件读取resize_scale，并传递给数据集 ---
    resize_scale = config['model'].get('resize_scale', 1.0)
    print(f"Using resize_scale: {resize_scale} for evaluation.")

    eval_dataset = MVSDataset(
        data_path=config['train']['data_path'],
        num_views=config['model']['num_views'],
        transform=eval_transform,
        resize_scale=resize_scale # 统一使用 resize_scale
    )
    # -----------------------------------------------------------

    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    model = TwoStreamMVS(config).to(device)
    model_path = os.path.join(config['train']['log_path'], "final_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(config['train']['log_path'], "checkpoint_epoch_20.pth")
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 后续的分块评估和内存清理逻辑保持不变
    all_metrics = []
    chunk_metrics = []
    print("Starting evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            # 将所有Tensor移动到设备
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch)
            refined_depth = outputs['refined_depth']

            # 使用缩放后的gt_depth计算指标
            gt_depth_scaled = batch['gt_depth'].squeeze(1)
            metrics = compute_depth_metrics(refined_depth, gt_depth_scaled)
            if metrics:
                all_metrics.append(metrics)
                chunk_metrics.append(metrics)

            # --- 修改：调用新的保存函数 ---
            save_depth_maps(
                output_dir=output_dir,
                meta_info=batch['meta_info'],
                pred_depth=refined_depth.squeeze(), # 移除batch和channel维度
                depth_scale=eval_dataset.depth_scale
            )

            if (i + 1) % chunk_size == 0 or (i + 1) == len(eval_loader):
                if chunk_metrics:
                    avg_chunk_metrics = {key: np.mean([m[key] for m in chunk_metrics]) for key in chunk_metrics[0]}
                    print(f"\n--- Chunk {i//chunk_size + 1} (samples {i-len(chunk_metrics)+2}-{i+1}) Avg Metrics ---")
                    print(f"  AbsRel: {avg_chunk_metrics['abs_rel']:.4f}, RMSE: {avg_chunk_metrics['rmse']:.4f}")
                    chunk_metrics = []
                print(f"--- Clearing CUDA cache at sample {i+1} ---\n")
                del outputs, refined_depth, gt_depth_scaled, batch
                gc.collect()
                torch.cuda.empty_cache()

    if all_metrics:
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
        print("\n" + "="*40 + "\nEvaluation Finished. Final Average Metrics:\n" + "="*40)
        print(f"Absolute Relative Error (AbsRel): {avg_metrics['abs_rel']:.4f}")
        print(f"Squared Relative Error (SqRel):  {avg_metrics['sq_rel']:.4f}")
        print(f"Root Mean Squared Error (RMSE):  {avg_metrics['rmse']:.4f}")
        print(f"Log RMSE (RMSElog):              {avg_metrics['rmse_log']:.4f}")
        print(f"Threshold δ < 1.25 (a1):         {avg_metrics['a1']:.4f}")
        print(f"Threshold δ < 1.25² (a2):        {avg_metrics['a2']:.4f}")
        print(f"Threshold δ < 1.25³ (a3):        {avg_metrics['a3']:.4f}")
        print("="*40)

        # Save metrics to file
        metrics_file = os.path.join(output_dir, "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write("="*40 + "\nEvaluation Finished. Final Average Metrics:\n" + "="*40 + "\n")
            f.write(f"Absolute Relative Error (AbsRel): {avg_metrics['abs_rel']:.4f}\n")
            f.write(f"Squared Relative Error (SqRel):  {avg_metrics['sq_rel']:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE):  {avg_metrics['rmse']:.4f}\n")
            f.write(f"Log RMSE (RMSElog):              {avg_metrics['rmse_log']:.4f}\n")
            f.write(f"Threshold δ < 1.25 (a1):         {avg_metrics['a1']:.4f}\n")
            f.write(f"Threshold δ < 1.25² (a2):        {avg_metrics['a2']:.4f}\n")
            f.write(f"Threshold δ < 1.25³ (a3):        {avg_metrics['a3']:.4f}\n")
            f.write("="*40 + "\n")
        print(f"Metrics saved to: {metrics_file}")
    else:
        print("Evaluation could not be completed as no valid data was found.")

if __name__ == '__main__':
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    evaluate(config)
