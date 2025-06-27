import numpy as np
import cv2                           # opencv-python-headless で OK
from ultralytics import YOLO         # pip install ultralytics

class YOLOPose:
    """
    ラッパークラス:
      __init__(device='cuda:0')   でモデルをロード
      __call__(bgr_img)           → (17,3) ndarray (u,v,score)
    """

    def __init__(self, device='cuda:0'):
        self.model = YOLO('yolov8n-pose.pt')   # 自動DL
        self.model.to(device)
        self.device = device

    def __call__(self, bgr_img: np.ndarray) -> np.ndarray:
        # Ultralytics は BGR ndarray をそのまま受け取れる
        res = self.model(bgr_img, verbose=False)
        if len(res) == 0 or len(res[0].keypoints) == 0:
            # 検出ゼロなら全0を返す (17,3)
            return np.zeros((17, 3), dtype=np.float32)
        # ここがポイント！res[0].keypoints[0] は torch.Tensor (17,3)
        kp = res[0].keypoints[0].cpu().numpy().astype(np.float32)  # ← .numpy() で十分
        return kp

# cv2 も必要なら外部へエクスポート
