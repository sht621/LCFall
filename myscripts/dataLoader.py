import numpy as np
import torch
import json
import os
from pathlib import Path
from LiCamPoseUtils.datasets.panoptic import Panoptic
from myscripts.my_pose2d import YOLOPose, cv2

class dataLoader(Panoptic):
    """
    Panoptic を継承し、2-D キーポイントを “その場で推論” して Heat-Map を作る
    カメラ1台対応 (フォルダ/ファイル名も現状のまま)
    """

    def __init__(self, cfg, datadir):
        super().__init__(cfg, datadir)
        self.pose2d = YOLOPose(device='cuda:0')
        self.cfg = cfg  # Panoptic本家では未保存なので自前で保存

        # --- frame_idリストの生成（points_pedのファイル名から）
        points_ped_folder = os.path.join(datadir, 'sorted_data', 'points_ped')
        files = sorted([f for f in os.listdir(points_ped_folder) if f.endswith('.ply')])
        self.frame_ids = [os.path.splitext(f)[0].replace('_001','') for f in files]
        self.image_dir = os.path.join(datadir, 'sorted_data', 'hdImgs')
        self.lidar_dir = os.path.join(datadir, 'sorted_data', 'points_ped')

        # カメラパラメータも（本家形式）
        # 例: calibration_cam0.json から取得
        cali_path = [f for f in os.listdir(datadir) if f.startswith('calibration_') and f.endswith('.json')]
        if not cali_path:
            raise FileNotFoundError('calibration_*.json not found in data dir')
        with open(os.path.join(datadir, cali_path[0]), 'r') as f:
            calib_data = json.load(f)
        cam = calib_data["cameras"][0]
        K = np.array(cam['K']).reshape(3, 3)
        R = np.array(cam['R']).reshape(3, 3)
        T = np.array(cam['t']).reshape(3, 1)
        self.camera = {
            "K": torch.tensor(K, dtype=torch.float32),
            "R": torch.tensor(R, dtype=torch.float32),
            "T": torch.tensor(T, dtype=torch.float32).view(3, 1),
            "fx": K[0,0],
            "fy": K[1,1],
            "cx": K[0,2],
            "cy": K[1,2],
            "distCoef": torch.tensor(cam.get('distCoef', np.zeros(5)), dtype=torch.float32),
            "k": torch.tensor(cam.get('distCoef', np.zeros(5))[[0,1,4]].reshape(3,1), dtype=torch.float32),
            "p": torch.tensor(cam.get('distCoef', np.zeros(5))[[2,3]].reshape(2,1), dtype=torch.float32)
        }

    def _read_point_cloud(self, ply_path):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        return np.asarray(pcd.points)

    def _infer_pose2d(self, img_path):
        img = cv2.imread(str(img_path))
        kpts = self.pose2d(img)
        # === ここでCOCO 17点→15点抽出 ===
        # COCOからPanoptic形式: 0〜13番＋16番
        indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,16]
        if isinstance(kpts, np.ndarray) and kpts.shape[0] >= 17:
            kpts = kpts[indices, :]   # 指定indexのみ
        return kpts  # (15,3)

    def _generate_heatmap(self, kpts, heatmap_size, image_size):
        # kpts: (15,3) or (1,15,3)
        if isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[0] == 1:
            kpts = kpts[0]
        joints = [kpts]
        joints_vis = [kpts[:,2:3]]
        return np.array(self.generate_input_heatmap(joints, joints_vis)[0])

    def __getitem__(self, index: int):
        fid = self.frame_ids[index]           # '000000' など

        # ファイルパス組み立て
        img_path = Path(self.image_dir) / f"{fid}.jpg"
        ply_path = Path(self.lidar_dir) / f"{fid}_001.ply"

        # LiDAR点群ロード
        xyz = self._read_point_cloud(ply_path)

        # 2Dキーポイント推論（+ 必ず15点に揃える）
        kpts = self._infer_pose2d(img_path)   # (15,3)

        # Heatmap生成
        heatmaps = self._generate_heatmap(
            kpts,
            self.cfg.NETWORK.HEATMAP_SIZE,
            self.cfg.NETWORK.IMAGE_SIZE
        )  # (15, h, w)

        # Occupancy voxel & pelvis
        lidar_center = 0.5 * (np.max(xyz, axis=0) + np.min(xyz, axis=0))
        input3d = self.generate_3d_input(xyz, lidar_center)

        # 本家Panopticに合わせたprojectionM
        projectionM = [self.camera]

        # テンソル化
        input3d = torch.from_numpy(input3d)[None].float()       # (1, d, w, h)
        heatmaps = torch.from_numpy(heatmaps)[None].float()     # (1, 15, h, w)
        grid_centers = torch.from_numpy(lidar_center)[None].float()  # (1, 3)

        return input3d, [heatmaps], projectionM, grid_centers

    def __len__(self):
        return len(self.frame_ids)
