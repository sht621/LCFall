"""
test.py  ―― LiCamPose 推論チェーンの健全性チェック用
  1) YAML から cfg を読み込む
  2) cfg の寸法に合わせてダミー入力を組み立てる
  3) VoxelFusionNet.forward() を 1 回呼んで shape を確認
"""

import torch
from LiCamPoseUtils.config import config, update_config
from LiCamPoseUtils.model.voxel_fusion_net import VoxelFusionNet

# ───────────────────────────────────────────
# ① YAML 読み込み
update_config("config/mydata.yaml")

# ───────────────────────────────────────────
# ② ダミー入力を cfg から動的生成
#    ・Occupancy voxel  : 1 × 1 × D × H × W
#    ・Heat-Map         : [1 × J × h × w]  （view = 1）
#    ・projectionM      : [ torch.eye(3,4) ]  （view = 1）
cube_x, cube_y, cube_z = config.PICT_STRUCT.CUBE_SIZE   # 例 [64,64,64]
heat_w, heat_h         = config.NETWORK.HEATMAP_SIZE    # 例 [480,270]
num_joints             = config.NUM_JOINTS              # 例 17

dummy_3d  = torch.zeros((1, 1, cube_z, cube_y, cube_x), dtype=torch.float32)
dummy_hm  = [torch.zeros((1, num_joints, heat_h, heat_w), dtype=torch.float32)]
projectionM = [torch.eye(4, dtype=torch.float32)[:3]]   # shape (3,4)
grid_center = torch.zeros((1, 3), dtype=torch.float32)

# ───────────────────────────────────────────
# ③ モデルを作って forward
model = VoxelFusionNet(config).eval()     # 重みは未ロード（ランダム初期化）

with torch.no_grad():
    pred, _ = model(dummy_3d, dummy_hm, projectionM, grid_center)

print("forward OK, output shape:", pred.shape)   # → torch.Size([1, J, 4])
