import numpy as np
import torch
import os
from pathlib import Path
from LiCamPoseUtils.datasets.panoptic import Panoptic
from myscripts.my_pose2d import YOLOPose, cv2
import json

class dataLoader(Panoptic):
    """
    Panoptic を継承し、2-D キーポイントを “その場で推論” して Heat-Map を作る
    カメラ1台対応 (フォルダ/ファイル名も現状のまま)
    """

    def __init__(self, cfg, datadir):
        super().__init__(cfg, datadir)
        self.pose2d = YOLOPose(device='cuda:0')
        self.cfg = cfg  # Panoptic本家では未保存なので自前で保存

        # --- カメラパラメータを本家形式でロード ---
        calib_files = [f for f in os.listdir(datadir) if f.startswith('calibration_') and f.endswith('.json')]
        if not calib_files:
            raise FileNotFoundError("calibration_*.json が datadir にありません")
        cam_json_path = os.path.join(datadir, calib_files[0])
        with open(cam_json_path, 'r') as f:
            calib_data = json.load(f)
        cam = calib_data["cameras"][0]  # 1台だけ使う

        # --- 必要なパラメータを本家の形で辞書化 ---
        K = np.array(cam['K']).reshape(3, 3)
        R = np.array(cam['R']).reshape(3, 3)
        T = np.array(cam['t']).reshape(3, 1)
        distCoef = np.array(cam.get('distCoef', [0, 0, 0, 0, 0]))  # なければゼロ
        # k, p は歪みパラメータ。たいていは [0,1,4] と [2,3] を使うが存在しなければゼロ
        k = distCoef[[0, 1, 4]].reshape(3, 1)
        p = distCoef[[2, 3]].reshape(2, 1)
        # そのまま
        self.camera = {
            'K': K,
            'R': R,
            'T': T,
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, 2],
            'cy': K[1, 2],
            'distCoef': distCoef,
            'k': k,
            'p': p
        }

        # --- frame_idリストの生成（points_pedのファイル名から） ---
        points_ped_folder = os.path.join(datadir, 'sorted_data', 'points_ped')
        files = sorted([f for f in os.listdir(points_ped_folder) if f.endswith('.ply')])
        self.frame_ids = [os.path.splitext(f)[0].replace('_001','') for f in files]
        self.image_dir = os.path.join(datadir, 'sorted_data', 'hdImgs')
        self.lidar_dir = os.path.join(datadir, 'sorted_data', 'points_ped')

    def _read_point_cloud(self, ply_path):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        return np.asarray(pcd.points)

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
        lidar_center = 0.5 * (np.max(xyz, axis=0) + np.min(xyz, axis=0))
        input3d = self.generate_3d_input(xyz, lidar_center)

        # --- 本家Panopticに合わせてprojectionMは「リスト内dict」で返す ---
        projectionM = [self.camera]

        # テンソル化
        input3d = torch.from_numpy(input3d)[None].float()       # (1, d, w, h)
        heatmaps = torch.from_numpy(heatmaps)[None].float()     # (1, 17, h, w)
        grid_centers = torch.from_numpy(lidar_center)[None].float()  # (1, 3)

        return input3d, [heatmaps], projectionM, grid_centers

    def __len__(self):
        return len(self.frame_ids)
