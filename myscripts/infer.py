import torch
import numpy as np
import yaml
from easydict import EasyDict
from myscripts.dataLoader import dataLoader
from model.voxel_fusion_net import VoxelFusionNet
import os

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
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
# module. を除去
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('module.', '')
    new_state_dict[new_k] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()

os.makedirs(output_dir, exist_ok=True)

# ----- 4. 推論ループ（本家ロジックそのまま） -----
with torch.no_grad():
    for i, batch in enumerate(loader):
        input3d, input_heatmap, projectionM, grid_centers = batch
        input3d = input3d.to(device)
        input_heatmap = [h.to(device) for h in input_heatmap]
        grid_centers = grid_centers.to(device)

        pred_kp, voxel_prob = model(
        input3d, input_heatmap, 
        projectionM, grid_centers
        )

        np.save(f"{output_dir}/pred_kp_{i:06d}.npy", pred_kp[0].cpu().numpy())

        print(f"[{i}] 推論完了, pred_kp shape:", pred_kp.shape)
