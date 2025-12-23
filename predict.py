import torch
import yaml
import numpy as np
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import cv2

# 假设模型定义文件在 models/two_stream_mvs.py
from models.two_stream_mvs import TwoStreamMVS

class DepthPredictor:
    """
    一个封装了模型加载和深度预测的类。
    """
    def __init__(self, config_path, model_path):
        """
        初始化预测器。
        Args:
            config_path (str): base_config.yaml 文件的路径。
            model_path (str): 训练好的 .pth 模型权重文件路径。
        """
        print("Initializing predictor...")
        # 1. 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 2. 初始化模型
        self.model = TwoStreamMVS(self.config).to(self.device)

        # 3. 加载权重
        print(f"Loading model weights from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # 4. 准备图像预处理
        self.resize_scale = self.config['model'].get('resize_scale', 1.0)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Predictor initialized successfully.")

    def _preprocess_image(self, image_path):
        """加载并预处理单张图像"""
        try:
            img = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None

        # 根据配置文件中的resize_scale进行缩放
        if self.resize_scale != 1.0:
            original_size = img.size
            new_width = int(original_size[0] * self.resize_scale)
            new_height = int(original_size[1] * self.resize_scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        return self.transform(img)

    def predict(self, ref_image_path, src_image_paths):
        """
        对一组多视角图像进行深度预测。
        Args:
            ref_image_path (str): 参考图像的路径。
            src_image_paths (list[str]): 源图像路径的列表。
        Returns:
            torch.Tensor: 预测的深度图 (H, W)。
        """
        # 1. 预处理所有图像
        ref_image = self._preprocess_image(ref_image_path)
        if ref_image is None: return None

        src_images = [self._preprocess_image(p) for p in src_image_paths]
        if any(img is None for img in src_images): return None

        all_images = [ref_image] + src_images
        images_tensor = torch.stack(all_images).to(self.device)

        # 2. 伪造一个batch数据
        num_views = len(all_images)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(num_views, 1, 1).float().to(self.device)
        poses = torch.eye(4).unsqueeze(0).repeat(num_views, 1, 1).float().to(self.device)

        _, h, w = ref_image.shape
        init_depths = torch.zeros(num_views, 1, h, w, dtype=torch.float32).to(self.device)

        batch = {
            "images": images_tensor.unsqueeze(0),
            "init_depths": init_depths.unsqueeze(0),
            "cam_intrinsics": intrinsics.unsqueeze(0),
            "cam_poses": poses.unsqueeze(0)
        }

        # 3. 执行预测
        print("Running model inference...")
        with torch.no_grad():
            outputs = self.model(batch)

        refined_depth = outputs['refined_depth'].squeeze()
        print("Inference complete.")
        return refined_depth

def save_depth_visualization(path, depth_map, output_size=None, save_16bit=False):
    """
    将深度图保存为伪彩色图，并可选保存16位图。
    Args:
        path (str): 输出文件路径 (不含扩展名)。
        depth_map (torch.Tensor): 深度图。
        output_size (tuple, optional): (width, height) 用于上采样。
        save_16bit (bool): 是否保存16位图。
    """
    # --- 修改开始：在保存前确保目录存在 ---
    output_dir = os.path.dirname(path)
    if output_dir: # 确保目录名不为空（当只提供文件名时）
        os.makedirs(output_dir, exist_ok=True)
    # --- 修改结束 ---

    # 1. 处理上采样
    if output_size:
        depth_map = F.interpolate(
            depth_map.unsqueeze(0).unsqueeze(0),
            size=(output_size[1], output_size[0]), # H, W
            mode='bilinear',
            align_corners=False
        ).squeeze()

    depth_np = depth_map.cpu().numpy()

    # 2. 保存伪彩色图
    valid_mask = depth_np > 0
    if not valid_mask.any():
        min_val, max_val = 0, 1
    else:
        min_val, max_val = depth_np[valid_mask].min(), depth_np[valid_mask].max()

    depth_normalized = (depth_np - min_val) / (max_val - min_val + 1e-8)
    depth_normalized[~valid_mask] = 0
    colormap = plt.get_cmap('magma')
    depth_colored = (colormap(depth_normalized) * 255).astype(np.uint8)

    color_path = f"{path}_color.png"
    Image.fromarray(depth_colored[:, :, :3]).save(color_path)
    print(f"Saved color depth map to: {color_path}")

    # 3. 可选：保存16位深度图
    if save_16bit:
        depth_16bit = (depth_np * 1000).astype(np.uint16)
        bit_path = f"{path}_16bit.png"
        cv2.imwrite(bit_path, depth_16bit)
        print(f"Saved 16-bit depth map to: {bit_path}")


def main():
    parser = argparse.ArgumentParser(description="Run depth prediction using a trained TwoStreamMVS model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument("--ref_image", type=str, required=True, help="Path to the reference image.")
    parser.add_argument("--src_images", type=str, nargs='+', required=True, help="Paths to one or more source images.")
    parser.add_argument("--config_path", type=str, default="configs/base_config.yaml", help="Path to the model config file.")
    parser.add_argument("--output_path", type=str, default="predicted_depth", help="Path and base name for the output files (without extension).")
    parser.add_argument("--output_size", type=str, default=None, help="Optional output size in format 'WIDTHxHEIGHT' (e.g., '1280x720').")
    parser.add_argument("--save_16bit", action='store_true', help="If set, saves a 16-bit depth map in addition to the color map.")

    args = parser.parse_args()

    # 检查源图像数量是否符合模型要求
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        num_src_views_needed = config['model']['num_views'] - 1
        if len(args.src_images) != num_src_views_needed:
            print(f"Warning: Model was trained with {num_src_views_needed} source views, but {len(args.src_images)} were provided.")
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_path}")
        return
    except KeyError:
        print("Error: Could not find 'num_views' in the model configuration.")
        return


    # 初始化预测器
    predictor = DepthPredictor(args.config_path, args.model_path)

    # 执行预测
    predicted_depth = predictor.predict(args.ref_image, args.src_images)

    if predicted_depth is not None:
        # 解析输出尺寸
        output_size = None
        if args.output_size:
            try:
                w, h = map(int, args.output_size.lower().split('x'))
                output_size = (w, h)
            except ValueError:
                print("Error: Invalid format for --output_size. Use 'WIDTHxHEIGHT'.")
                return

        # 保存结果
        save_depth_visualization(args.output_path, predicted_depth, output_size, args.save_16bit)

if __name__ == '__main__':
    main()
