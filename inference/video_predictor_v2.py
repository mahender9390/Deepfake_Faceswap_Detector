"""
inference/video_predictor_v2.py  — Final Version
──────────────────────────────────────────────────
V2-aug Pure Temporal Deepfake Detector.
Model: checkpoints_video_v2/best_video_model_v2.pth (AUC 0.998)
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2, numpy as np, torch, yaml
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training_video.video_model_v2 import get_video_model_v2

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class VideoPredictorV2:
    def __init__(
        self,
        checkpoint_path: str | Path = "C:/imagevideo/checkpoints_video_v2/best_video_model_v2.pth",
        config_path:     str | Path = "C:/imagevideo/training_video/video_config_v2.yaml",
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        config_path     = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_frames = self.config["model"]["n_frames"]
        self.size     = self.config["model"]["image_size"]

        self.model = get_video_model_v2(self.config, self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.mtcnn = MTCNN(
            image_size=self.size, margin=20, min_face_size=40,
            thresholds=[0.6, 0.7, 0.7], factor=0.709,
            post_process=False, keep_all=False, device=self.device,
        )

        self.transform = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])

        print(f"VideoPredictorV2 (AUC 0.998) ready on {self.device}")

    def _crop_face(self, rgb: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(rgb)
        try:
            boxes, probs = self.mtcnn.detect(pil)
            if boxes is not None and len(boxes) > 0:
                b = int(np.argmax(probs))
                x1, y1, x2, y2 = boxes[b]
                w, h = pil.size; m = 20
                x1 = max(0, int(x1)-m); y1 = max(0, int(y1)-m)
                x2 = min(w, int(x2)+m); y2 = min(h, int(y2)+m)
                if x2 > x1 and y2 > y1:
                    return np.array(
                        pil.crop((x1,y1,x2,y2)).resize((self.size,self.size), Image.LANCZOS),
                        dtype=np.uint8)
        except Exception:
            pass
        w, h = pil.size
        return np.array(
            pil.crop((w//4, 0, 3*w//4, 2*h//3)).resize((self.size,self.size), Image.LANCZOS),
            dtype=np.uint8)

    def _load_clip(self, video_path: Path) -> torch.Tensor | None:
        cap   = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1: cap.release(); return None

        tensors = []
        for idx in np.linspace(0, total-1, self.n_frames, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, bgr = cap.read()
            if not ret: continue
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            face = self._crop_face(rgb)
            tensors.append(self.transform(image=face)["image"])
        cap.release()

        if not tensors: return None
        while len(tensors) < self.n_frames:
            tensors.append(tensors[-1])
        return torch.stack(tensors[:self.n_frames], dim=0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, video_path: str | Path) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        clip = self._load_clip(video_path)
        if clip is None:
            return {"prediction":"UNKNOWN","confidence":0.0,
                    "is_fake":False,"raw_score":0.0,"frames_analyzed":0}

        raw_score  = torch.sigmoid(self.model(clip)).item()
        is_fake    = raw_score >= 0.5
        confidence = (raw_score if is_fake else 1.0 - raw_score) * 100.0

        return {
            "prediction":      "FAKE" if is_fake else "REAL",
            "confidence":      round(confidence, 2),
            "is_fake":         is_fake,
            "raw_score":       round(raw_score, 6),
            "frames_analyzed": self.n_frames,
        }