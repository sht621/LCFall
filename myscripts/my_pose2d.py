import numpy as np
import cv2
from ultralytics import YOLO

class YOLOPose:
    def __init__(self, device='cuda:0'):
        self.model = YOLO('yolov8n-pose.pt')
        self.model.to(device)
        self.device = device

    def __call__(self, bgr_img: np.ndarray) -> np.ndarray:
        res = self.model(bgr_img, verbose=False)
        if len(res) == 0 or len(res[0].keypoints) == 0:
            return np.zeros((17, 3), dtype=np.float32)
        # ここが重要！
        kp = res[0].keypoints[0].data.cpu().numpy().astype(np.float32)
        return kp
