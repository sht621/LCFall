import os
import numpy as np
from glob import glob
from LiCamPoseUtils.datasets.panoptic import Panoptic
from myscripts.my_pose2d import YOLOPose
import cv2
import torch

class dataLoader(Panoptic):
    """
    Panoptic本家を変更せず、1カメ(cam0)専用で自前データにフィットさせる継承DataLoader
    """

    def __init__(self, cfg, datadir):
        self.cfg = cfg
        super().__init__(cfg, datadir)
        self.pose2d = YOLOPose(device='cuda:0')
        # 本家は5台想定だが、ここで上書き
        self.views_num = 1
        self.camera_names = ['cam0']

        # --- 画像・点群用ディレクトリ設定 ---
        # panoptic.py側は points_ped_folder などを使うが1カメ用にpaths生成
        self.image_dir = os.path.join(datadir, "sorted_data", "hdImgs")
        self.lidar_dir = os.path.join(datadir, "sorted_data", "points_ped")
        self.pred2d_dir = os.path.join(datadir, "sorted_data", "pred_2d")

        # --- frame_idsを自動生成（例: 000000_001.ply → 000000_001） ---
        ply_files = sorted(glob(os.path.join(self.lidar_dir, "*.ply")))
        self.frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in ply_files]

    def _infer_pose2d(self, img_path):
        img = cv2.imread(str(img_path))
        return self.pose2d(img)

    def _generate_heatmap(self, kpts, heatmap_size, image_size):
        # Panoptic本家ではviews_num個まとめて扱うが、ここではcam0のみの(17,3)
        # 必要ならPanoptic本家の関数呼び出しに合わせること
        joints = [kpts]  # 1カメでもリストで渡す
        joints_vis = [kpts[:, 2:3]]
        # 本家のgenerate_input_heatmap(self, joints, joints_vis)
        return np.array(self.generate_input_heatmap(joints, joints_vis)[0])

    def _read_point_cloud(self, ply_path):
        # Open3Dで点群ロード
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        xyz = np.asarray(pcd.points)
        return xyz

    def __getitem__(self, index):
        fid = self.frame_ids[index]  # 例: 000000_001

        # ファイルパス構築
        img_path = os.path.join(self.image_dir, f"{fid.split('_')[0]}.jpg")
        ply_path = os.path.join(self.lidar_dir, f"{fid}.ply")

        # LiDAR点群取得
        xyz = self._read_point_cloud(ply_path)
        # 2D keypoints（推論）
        kpts = self._infer_pose2d(img_path)  # (17,3)
        # Heatmap
        heatmaps = self._generate_heatmap(kpts, self.cfg.NETWORK.HEATMAP_SIZE, self.cfg.NETWORK.IMAGE_SIZE)  # (17, h, w)
        # 3D occupancy voxel/center
        input3d, grid_centers = self.generate_3d_input(xyz, kpts)
        # 射影行列リスト（カメラ1台のみ）
        projectionM = [torch.from_numpy(P).float() for P in self.cameras.values()]

        # テンソル化(B=1)
        input3d = torch.from_numpy(input3d)[None].float()          # 1x1xDHW
        heatmaps = torch.from_numpy(heatmaps)[None].float()        # 1xJxhxw
        grid_centers = torch.from_numpy(grid_centers)[None].float()

        return input3d, [heatmaps], projectionM, grid_centers

    def __len__(self):
        return len(self.frame_ids)
