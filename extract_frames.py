"""
extract_frames.py  (v2 — improved)
────────────────────────────────────
Extracts face frames from:
  - FaceForensics++ (Face2Face, Deepfakes, FaceSwap, YouTube-real)
  - Celeb-DF v2 (Celeb-synthesis, Celeb-real, YouTube-real)
  - DFDC part 48 (uses metadata.json for labels)

Improvements over v1:
  - 30 frames per video (3x more clips for training)
  - DFDC support via metadata.json
  - Better face detection fallback

Output:
  C:/imagevideo/data_video/train/real/
  C:/imagevideo/data_video/train/fake/

Usage:
    python extract_frames.py
"""

from __future__ import annotations
import json, random, sys
from pathlib import Path

import cv2, numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_REAL  = Path("C:/imagevideo/data_video/train/real")
OUTPUT_FAKE  = Path("C:/imagevideo/data_video/train/fake")
FFPP         = Path("C:/Users/jashw/Downloads/FFPP")
CELEBDF      = Path("C:/Users/jashw/Downloads/Celeb-DF-v2")
DFDC         = Path("C:/Users/jashw/Downloads/dfdc_train_part_48/dfdc_train_part_48")

FRAMES       = 30        # ← 30 frames = 3 clips of 10 per video
SIZE         = 224
SEED         = 42
VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}

FAKE_SOURCES = [
    {"name": "Face2Face",       "path": FFPP / "manipulated_sequences/Face2Face/c23/videos",  "cap": None},
    {"name": "Deepfakes",       "path": FFPP / "manipulated_sequences/Deepfakes/c23/videos",  "cap": None},
    {"name": "FaceSwap",        "path": FFPP / "manipulated_sequences/FaceSwap/c23/videos",   "cap": None},
    {"name": "FaceShifter",     "path": FFPP / "manipulated_sequences/FaceShifter/c23/videos", "cap": 400},
    {"name": "Celeb-synthesis", "path": CELEBDF / "Celeb-synthesis",                           "cap": 800},
]

REAL_SOURCES = [
    {"name": "FF++-YouTube",    "path": FFPP / "original_sequences/youtube/c23/videos",       "cap": None},
    {"name": "Celeb-real",      "path": CELEBDF / "Celeb-real",                               "cap": None},
    {"name": "YouTube-real",    "path": CELEBDF / "YouTube-real",                             "cap": None},
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_videos(folder, cap):
    if not folder.exists():
        print(f"  [SKIP] {folder}")
        return []
    vids = []
    for ext in VIDEO_EXTS:
        vids.extend(folder.glob(f"*{ext}"))
        vids.extend(folder.glob(f"*{ext.upper()}"))
    vids = sorted(set(vids))
    if cap and len(vids) > cap:
        random.seed(SEED)
        vids = random.sample(vids, cap)
    return vids


def setup_mtcnn():
    from facenet_pytorch import MTCNN
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  MTCNN on: {dev}")
    return MTCNN(image_size=SIZE, margin=20, min_face_size=40,
                 thresholds=[0.6,0.7,0.7], factor=0.709,
                 post_process=False, keep_all=False, device=dev)


def crop_face(mtcnn, rgb):
    pil = Image.fromarray(rgb)
    try:
        boxes, probs = mtcnn.detect(pil)
        if boxes is not None and len(boxes) > 0:
            b = int(np.argmax(probs))
            x1,y1,x2,y2 = boxes[b]
            w,h = pil.size; m=20
            x1=max(0,int(x1)-m); y1=max(0,int(y1)-m)
            x2=min(w,int(x2)+m); y2=min(h,int(y2)+m)
            if x2>x1 and y2>y1:
                return pil.crop((x1,y1,x2,y2))
    except: pass
    w,h = pil.size
    return pil.crop((w//4, 0, 3*w//4, 2*h//3))


def process_video(mtcnn, path, out_dir, n, idx, label):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1: cap.release(); return 0
    saved = 0
    for fi in np.linspace(0, total-1, min(n,total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, bgr = cap.read()
        if not ret: continue
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face = crop_face(mtcnn, rgb)
        if face is None: continue
        face = face.resize((SIZE,SIZE), Image.LANCZOS)
        face.save(out_dir / f"{label}_{idx:05d}_{fi:06d}.jpg", quality=95)
        saved += 1
    cap.release()
    return saved


def process_sources(mtcnn, sources, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for src in sources:
        vids = get_videos(src["path"], src["cap"])
        if not vids: continue
        print(f"\n  [{label.upper()}] {src['name']} — {len(vids)} videos")
        for i,v in enumerate(tqdm(vids, desc=f"    {src['name']}", ncols=70)):
            total += process_video(mtcnn, v, out_dir, FRAMES, i, label[:4])
        print(f"    Saved so far: {total:,} frames")
    return total


def process_dfdc(mtcnn, dfdc_folder, fake_out, real_out):
    """Process DFDC using metadata.json for labels."""
    json_path = dfdc_folder / "metadata.json"
    if not json_path.exists():
        print(f"  [SKIP] DFDC metadata.json not found at {json_path}")
        return 0, 0

    with open(json_path) as f:
        metadata = json.load(f)

    fake_vids = [dfdc_folder / fn for fn, info in metadata.items()
                 if info["label"] == "FAKE" and (dfdc_folder / fn).exists()]
    real_vids = [dfdc_folder / fn for fn, info in metadata.items()
                 if info["label"] == "REAL" and (dfdc_folder / fn).exists()]

    # Cap to balance with other sources
    random.seed(SEED)
    fake_vids = random.sample(fake_vids, min(400, len(fake_vids)))
    real_vids = random.sample(real_vids, min(200, len(real_vids)))

    print(f"\n  [FAKE] DFDC-fake — {len(fake_vids)} videos")
    n_fake = 0
    for i,v in enumerate(tqdm(fake_vids, desc="    DFDC-fake", ncols=70)):
        n_fake += process_video(mtcnn, v, fake_out, FRAMES, 90000+i, "fake")
    print(f"    DFDC fake frames: {n_fake:,}")

    print(f"\n  [REAL] DFDC-real — {len(real_vids)} videos")
    n_real = 0
    for i,v in enumerate(tqdm(real_vids, desc="    DFDC-real", ncols=70)):
        n_real += process_video(mtcnn, v, real_out, FRAMES, 90000+i, "real")
    print(f"    DFDC real frames: {n_real:,}")

    return n_fake, n_real


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  Frame Extractor v2 — FF++ + Celeb-DF + DFDC")
    print("="*55)
    print(f"  Frames/video : {FRAMES} (= 3 clips of 10)")
    print(f"  Image size   : {SIZE}x{SIZE}")

    OUTPUT_REAL.mkdir(parents=True, exist_ok=True)
    OUTPUT_FAKE.mkdir(parents=True, exist_ok=True)

    print("\n  Loading MTCNN...")
    mtcnn = setup_mtcnn()

    # FF++ + Celeb-DF
    print("\n" + "="*55)
    print("  Extracting FAKE frames (FF++ + Celeb-DF)...")
    print("="*55)
    total_fake = process_sources(mtcnn, FAKE_SOURCES, OUTPUT_FAKE, "fake")

    print("\n" + "="*55)
    print("  Extracting REAL frames (FF++ + Celeb-DF)...")
    print("="*55)
    total_real = process_sources(mtcnn, REAL_SOURCES, OUTPUT_REAL, "real")

    # DFDC
    print("\n" + "="*55)
    print("  Extracting DFDC frames...")
    print("="*55)
    dfdc_fake, dfdc_real = process_dfdc(mtcnn, DFDC, OUTPUT_FAKE, OUTPUT_REAL)
    total_fake += dfdc_fake
    total_real += dfdc_real

    print("\n" + "="*55)
    print("  EXTRACTION COMPLETE")
    print("="*55)
    print(f"  FAKE frames : {total_fake:,}")
    print(f"  REAL frames : {total_real:,}")
    print(f"  Total       : {total_fake+total_real:,}")
    print(f"  Est. clips  : {(total_fake+total_real)//10:,} (at 10 frames/clip)")
    print()
    print("  Next: python training_video/video_train.py")
    print("="*55)


if __name__ == "__main__":
    main()