# myscripts/test.py
"""
LiCamPose 推論チェーンの健全性チェック（ダミー入力版）
- cfg を YAML から読み込み
- cfg パラメータに合わせてゼロ埋めテンソルを生成
- VoxelFusionNet.forward() を 1 回呼び shape を確認
実データ・2-D 推定器・DataLoader を用意しなくても動く
"""

import torch
from LiCamPoseUtils.config import config, update_config
from LiCamPoseUtils.model.voxel_fusion_net import VoxelFusionNet

# ─────────────────────────────────────────────
# 1) YAML 読み込み（パスは適宜変更）
update_config("config/mydata.yaml")

# 2) cfg から各寸法を取得
cube_x, cube_y, cube_z = config.PICT_STRUCT.CUBE_SIZE   # 例 [64,64,64]
heat_w, heat_h         = config.NETWORK.HEATMAP_SIZE    # 例 [480,270]
num_joints             = config.NUM_JOINTS              # 例 17
B, V                   = 1, 1                           # バッチ=1, view=1

# 3) ダミー入力テンソル
dummy_3d  = torch.zeros((B, 1, cube_z, cube_y, cube_x), dtype=torch.float32)
dummy_hm  = [torch.zeros((B, num_joints, heat_h, heat_w), dtype=torch.float32)]

proj_single   = torch.eye(4, dtype=torch.float32)[:3]            # (3,4)
projectionM   = [proj_single.unsqueeze(0).repeat(B, 1, 1)]       # list[Tensor(B,3,4)]
grid_centers  = torch.zeros((B, 3), dtype=torch.float32)         # pelvis 原点

# 4) モデル生成（重みは未ロード＝ランダム）
model = VoxelFusionNet(config).eval()

# 5) 前向き実行
with torch.no_grad():
    pred, _ = model(dummy_3d, dummy_hm, projectionM, grid_centers)

print("forward OK, output shape:", pred.shape)  # → torch.Size([1, 17, 4])
