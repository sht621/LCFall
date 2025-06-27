# scripts/dry_run.py
import torch, yaml, numpy as np
from easydict import EasyDict
from licampose.config import config, update_config
from licampose.model.voxel_fusion_net import VoxelFusionNet

update_config(config, "config/my_infer.yaml")
model = VoxelFusionNet(config).eval()
dummy_3d  = torch.zeros((1,1,80,80,40))       # LiDAR
dummy_hm  = [torch.zeros((1,17,64,64))]        # 1 view
P         = [torch.eye(3,4)]                   # ダミー射影
center    = torch.zeros((1,3))

with torch.no_grad():
    pred,_ = model(dummy_3d, dummy_hm, P, center)
print("forward OK, output shape:", pred.shape)   # → (1,17,4)
