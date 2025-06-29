import numpy as np
import torch
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
        self.cfg = cfg  # ←Panoptic本家では未保存なので自前で保存

        # --- frame_idリストの生成（points_pedのファイル名から）
        points_ped_folder = os.path.join(datadir, 'sorted_data', 'points_ped')
        files = sorted([f for f in os.listdir(points_ped_folder) if f.endswith('.ply')])
        self.frame_ids = [os.path.splitext(f)[0].replace('_001','') for f in files]
        self.image_dir = os.path.join(datadir, 'sorted_data', 'hdImgs')
        self.lidar_dir = os.path.join(datadir, 'sorted_data', 'points_ped')

    def _infer_pose2d(self, img_path):
        img = cv2.imread(str(img_path))
        kpts = self.pose2d(img)
        if isinstance(kpts, np.ndarray) and kpts.ndim == 3 and kpts.shape[0] == 1:
            kpts = kpts[0]  # (17,3)
        return kpts  # (17,3)

    def _generate_heatmap(self, kpts, heatmap_size, image_size):
        # kpts: (17,3) or (1,17,3)
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

        # 2Dキーポイント推論
        kpts = self._infer_pose2d(img_path)   # (17,3)

        # Heatmap生成
        heatmaps = self._generate_heatmap(
            kpts,
            self.cfg.NETWORK.HEATMAP_SIZE,
            self.cfg.NETWORK.IMAGE_SIZE
        )  # (17, h, w)

        # Occupancy voxel & pelvis
        input3d, grid_centers = self.generate_3d_input(xyz, kpts)

        # 射影行列（本家流で単カメラ対応）
        projectionM = [torch.eye(4)]  # ダミー値（実際は本家のカメラ行列等に合わせて調整）

        # テンソル化
        input3d = torch.from_numpy(input3d)[None].float()       # (1, d, w, h)
        heatmaps = torch.from_numpy(heatmaps)[None].float()     # (1, 17, h, w)
        grid_centers = torch.from_numpy(grid_centers)[None].float()

        return input3d, [heatmaps], projectionM, grid_centers
