import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import cv2 # 使用OpenCV读取16位PNG
from utils.colmap_utils import read_cameras_binary, read_images_binary, qvec2rotmat

class MVSDataset(Dataset):
    """
    适用于COLMAP格式的MVS数据集加载器。
    - data_path: 数据集根目录, 包含多个场景 (e.g., 'BOOK')
    """
    # 在构造函数中添加 resize_scale 参数
    def __init__(self, data_path, num_views=3, transform=None, depth_scale=1000.0, resize_scale=1.0):
        super().__init__()
        self.data_path = data_path
        self.num_views = num_views
        self.transform = transform
        self.depth_scale = depth_scale # 16位PNG深度图的缩放因子
        self.resize_scale = resize_scale # 新增: 图像缩放比例
        self.scenes = [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]
        self.metas = self._build_metas()

    def _build_metas(self):
        metas = []
        for scene in self.scenes:
            scene_path = os.path.join(self.data_path, scene)
            sparse_path = os.path.join(scene_path, 'sparse', '0')

            # 读取COLMAP的输出
            cameras = read_cameras_binary(os.path.join(sparse_path, 'cameras.bin'))
            images = read_images_binary(os.path.join(sparse_path, 'images.bin'))

            # 按图像名称排序，以确保一致的视图选择
            sorted_images = sorted(images.values(), key=lambda im: im.name)

            for i in range(len(sorted_images)):
                # 将第i个视图作为参考视图
                if i + self.num_views <= len(sorted_images):
                    ref_image_info = sorted_images[i]
                    src_images_info = [sorted_images[i + j] for j in range(1, self.num_views)]

                    # 提取相机参数
                    view_infos = [ref_image_info] + src_images_info
                    all_cam_params = []
                    for im_info in view_infos:
                        cam = cameras[im_info.camera_id]

                        # 内参矩阵 K
                        K = np.eye(3)
                        if cam.model == 'SIMPLE_PINHOLE':
                            fx, cx, cy = cam.params
                            K[0, 0] = fx
                            K[1, 1] = fx
                            K[0, 2] = cx
                            K[1, 2] = cy
                        elif cam.model == 'PINHOLE':
                            fx, fy, cx, cy = cam.params
                            K[0, 0] = fx
                            K[1, 1] = fy
                            K[0, 2] = cx
                            K[1, 2] = cy
                        else:
                            # 为其他相机模型添加支持
                            raise ValueError(f"Unsupported camera model: {cam.model}")

                        # 外参矩阵 E = [R|t]
                        R = qvec2rotmat(im_info.qvec)
                        t = im_info.tvec.reshape(3, 1)
                        E = np.hstack((R, t))

                        all_cam_params.append({'K': K, 'E': E, 'name': im_info.name})

                    metas.append((scene, all_cam_params))
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scene, cam_params = self.metas[idx]
        ref_image_name = cam_params[0]['name']

        images = []
        init_depths = []
        cam_intrinsics = []
        cam_poses = []

        for cam_info in cam_params:
            img_name = cam_info['name']

            # 1. 加载RGB图像并进行缩放
            img_path = os.path.join(self.data_path, scene, 'images', img_name)
            img = Image.open(img_path).convert('RGB')

            # --- 缩放图像 ---
            if self.resize_scale != 1.0:
                new_width = int(img.width * self.resize_scale)
                new_height = int(img.height * self.resize_scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            if self.transform:
                img = self.transform(img)
            images.append(img)

            # 2. 加载初始深度图并进行缩放
            depth_path = os.path.join(self.data_path, scene, 'depths', img_name)
            init_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if init_depth is not None:
                # --- 缩放深度图 ---
                if self.resize_scale != 1.0:
                    new_width = int(init_depth.shape[1] * self.resize_scale)
                    new_height = int(init_depth.shape[0] * self.resize_scale)
                    init_depth = cv2.resize(init_depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

                init_depth = init_depth.astype(np.float32) / self.depth_scale
                init_depth = torch.from_numpy(init_depth).unsqueeze(0)
            else:
                h, w = img.shape[1:]
                init_depth = torch.zeros(1, h, w, dtype=torch.float32)
            init_depths.append(init_depth)

            # 3. 整理相机参数并根据缩放比例进行调整
            K = torch.from_numpy(cam_info['K']).float()

            # --- 调整相机内参 ---
            if self.resize_scale != 1.0:
                K[0, :] *= self.resize_scale # 缩放 fx, cx
                K[1, :] *= self.resize_scale # 缩放 fy, cy

            R, t = cam_info['E'][:, :3], cam_info['E'][:, 3:]
            R_inv = R.T
            t_inv = -R.T @ t
            E_inv = np.hstack((R_inv, t_inv))
            E_inv = np.vstack((E_inv, [0,0,0,1]))

            cam_intrinsics.append(K)
            cam_poses.append(torch.from_numpy(E_inv).float())

        # 4. 加载用于模型计算的GT深度图 (缩放版本)
        gt_depth_path = os.path.join(self.data_path, scene, 'depths', ref_image_name)
        gt_depth_scaled_raw = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        if gt_depth_scaled_raw is None:
            raise FileNotFoundError(f"Ground truth depth not found at: {gt_depth_path}")

        if self.resize_scale != 1.0:
            new_width = int(gt_depth_scaled_raw.shape[1] * self.resize_scale)
            new_height = int(gt_depth_scaled_raw.shape[0] * self.resize_scale)
            gt_depth_scaled_raw = cv2.resize(gt_depth_scaled_raw, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        gt_depth_scaled = torch.from_numpy(gt_depth_scaled_raw.astype(np.float32) / self.depth_scale).unsqueeze(0)

        return {
            "images": torch.stack(images),
            "init_depths": torch.stack(init_depths),
            "cam_intrinsics": torch.stack(cam_intrinsics),
            "cam_poses": torch.stack(cam_poses),
            "gt_depth": gt_depth_scaled, # 用于计算loss的缩放后版本
            # --- 修改：只传递路径，不加载原始图像 ---
            "meta_info": {
                "scene": scene,
                "ref_image_name": ref_image_name,
                "gt_depth_path": gt_depth_path
            }
        }
