# myscripts/dataLoader.py
"""
Production-ready DataLoader:
    * 画像 + LiDAR 点群だけを入力
    * 毎フレーム GPU/CPU で 2-D Pose 推論（YOLOPose 例）
    * Heat-Map 生成は Panoptic オリジナル _generate_heatmap を呼ぶ
    * Occupancy voxel／pelvis 位置決定なども Panoptic 実装を再利用
"""

from pathlib import Path
import torch, cv2
import numpy as np

from LiCamPoseUtils.datasets.panoptic import Panoptic
from myscripts.my_pose2d import YOLOPose            # ← 任意の 2-D 推定器 (返り shape = (17,3))


class dataLoader(Panoptic):
    """Panoptic を継承し、2-D キーポイントを “その場で推論” して Heat-Map を作る"""

    def __init__(self, cfg, datadir):
        super().__init__(cfg, datadir)        # frame_ids, camera_dict など初期化
        self.pose2d = YOLOPose(device='cuda:0')

    # ───────────────────────────────────────────
    def _infer_pose2d(self, img_path: Path) -> np.ndarray:
        img = cv2.imread(str(img_path))
        return self.pose2d(img)               # (J,3) ndarray  (u,v,score)

    # ───────────────────────────────────────────
    def __getitem__(self, index: int):
        fid = self.frame_ids[index]           # '000123'

        # ---------- ファイルパス ----------
        img_path = self.image_dir / f"{fid}.jpg"
        ply_path = self.lidar_dir / f"lidar_{fid}.ply"

        # ---------- LiDAR 点群 ----------
        xyz = self._read_point_cloud(ply_path)        # Panoptic util

        # ---------- 2-D キーポイント ----------
        kpts = self._infer_pose2d(img_path)           # (J,3)

        # ---------- Heat-Map (Panoptic 本家関数) ----------
        heatmaps = self._generate_heatmap(
            kpts,
            self.cfg.NETWORK.HEATMAP_SIZE,
            self.cfg.NETWORK.IMAGE_SIZE
        )                                             # (J,h,w) float32

        # ---------- Occupancy voxel & pelvis ----------
        input3d, grid_centers = self.generate_3d_input(xyz, kpts)

        # ---------- 射影行列リスト ----------
        projectionM = [
            torch.from_numpy(P).float()
            for P in self.camera_dict.values()
        ]

        # ---------- テンソル化 (B=1) ----------
        input3d  = torch.from_numpy(input3d)[None].float()       # 1×1×D×H×W
        heatmaps = torch.from_numpy(heatmaps)[None].float()      # 1×J×h×w
        grid_centers = torch.from_numpy(grid_centers)[None].float()

        return input3d, [heatmaps], projectionM, grid_centers
