import torch
import numpy as np
import yaml
from easydict import EasyDict
from myscripts.dataLoader import dataLoader
from model.voxel_fusion_net import VoxelFusionNet

# ----- 1. configロード -----
cfg_path = 'config/mydata.yaml'
datadir = 'data'
model_path = 'LiCamPoseUtils/model/model_best.pth'
output_dir = 'results'   # 出力ディレクトリ

with open(cfg_path) as f:
    cfg = EasyDict(yaml.safe_load(f))

# ----- 2. データローダ（推論用） -----
dl = dataLoader(cfg, datadir)
loader = torch.utils.data.DataLoader(dl, batch_size=1, shuffle=False)

# ----- 3. モデル準備 -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VoxelFusionNet(cfg).to(device)
state_dict = torch.load(model_path, map_location=device)
# key名が 'state_dict' の場合
if 'state_dict' in state_dict:
    model.load_state_dict(state_dict['state_dict'])
else:
    model.load_state_dict(state_dict)
model.eval()

import os
os.makedirs(output_dir, exist_ok=True)

# ----- 4. 推論ループ（本家ロジックそのまま） -----
with torch.no_grad():
    for i, batch in enumerate(loader):
        input3d, input_heatmap, projectionM, grid_centers = batch
        input3d = input3d.to(device)
        # input_heatmapはリスト型
        input_heatmap = [h.to(device) for h in input_heatmap]
        grid_centers = grid_centers.to(device)

        # ---- 推論: 本家ロジックと同じ ----
        pred_kp, voxel_prob = model(
            input3d, input_heatmap, 
            projection=projectionM, 
            centers=grid_centers
        )  # pred_kp: (B, J, 4)

        # ----- 出力保存 (例: npyで) -----
        np.save(f"{output_dir}/pred_kp_{i:06d}.npy", pred_kp[0].cpu().numpy())
        # 追加で可視化やcsv保存等も可能

        print(f"[{i}] 推論完了, pred_kp shape:", pred_kp.shape)
