"""
test_all_datasets_v2.py  — Final Test Script (V2-aug model)
────────────────────────────────────────────────────────────
Tests the final V2-aug Pure Temporal model on all datasets.
200 videos per dataset.

Usage:
    python test_all_datasets_v2.py
"""
from __future__ import annotations
import random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference.video_predictor_v2 import VideoPredictorV2

_VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv"}
CHECKPOINT    = Path("C:/imagevideo/checkpoints_video_v2/best_video_model_v2.pth")
CONFIG        = Path("C:/imagevideo/training_video/video_config_v2.yaml")
VIDEOS_PER_DS = 200

DATASETS = [
    {"name": "FF++ FaceSwap",  "path": "C:/Users/jashw/Downloads/FFPP/manipulated_sequences/FaceSwap/c23/videos",    "label": "FAKE"},
    {"name": "FF++ Face2Face", "path": "C:/Users/jashw/Downloads/FFPP/manipulated_sequences/Face2Face/c23/videos",   "label": "FAKE"},
    {"name": "FF++ Deepfakes", "path": "C:/Users/jashw/Downloads/FFPP/manipulated_sequences/Deepfakes/c23/videos",   "label": "FAKE"},
    {"name": "FaceShifter",    "path": "C:/Users/jashw/Downloads/FFPP/manipulated_sequences/FaceShifter/c23/videos", "label": "FAKE"},
    {"name": "Celeb-synth",    "path": "C:/Users/jashw/Downloads/Celeb-DF-v2/Celeb-synthesis",                       "label": "FAKE"},
    {"name": "FF++ Real",      "path": "C:/Users/jashw/Downloads/FFPP/original_sequences/youtube/c23/videos",        "label": "REAL"},
    {"name": "Celeb-real",     "path": "C:/Users/jashw/Downloads/Celeb-DF-v2/Celeb-real",                            "label": "REAL"},
    {"name": "YouTube-real",   "path": "C:/Users/jashw/Downloads/Celeb-DF-v2/YouTube-real",                          "label": "REAL"},
]


def get_videos(folder, cap):
    p = Path(folder)
    if not p.exists(): return []
    vids = []
    for ext in _VIDEO_EXTS:
        vids.extend(p.glob(f"*{ext}"))
    vids = sorted(set(vids))
    if len(vids) > cap:
        random.seed(42); vids = random.sample(vids, cap)
    return vids


def test_dataset(predictor, videos, label):
    correct = 0; errors = 0; confs = []
    for v in videos:
        try:
            r = predictor.predict(v)
            if r["prediction"] == label: correct += 1
            confs.append(r["confidence"])
        except Exception: errors += 1
    total = len(videos) - errors
    return {
        "accuracy": correct / total * 100 if total > 0 else 0,
        "correct":  correct,
        "total":    total,
        "avg_conf": sum(confs) / len(confs) if confs else 0,
    }


def main():
    print("\n" + "="*60)
    print("  V2-aug Pure Temporal — Full Dataset Evaluation")
    print(f"  Model : best_video_model_v2.pth (AUC 0.998)")
    print(f"  Videos: {VIDEOS_PER_DS} per dataset")
    print("="*60)

    print("\n  Loading V2-aug model...")
    predictor = VideoPredictorV2(checkpoint_path=CHECKPOINT, config_path=CONFIG)

    results = []
    total_correct = 0; total_videos = 0

    for ds in DATASETS:
        videos = get_videos(ds["path"], VIDEOS_PER_DS)
        if not videos:
            print(f"\n  [SKIP] {ds['name']} — not found"); continue

        print(f"\n  Testing {ds['name']} ({len(videos)} videos, label={ds['label']})...")
        res = test_dataset(predictor, videos, ds["label"])
        results.append({**ds, **res})
        total_correct += res["correct"]
        total_videos  += res["total"]
        print(f"  → Accuracy: {res['accuracy']:.1f}%  ({res['correct']}/{res['total']})  Avg conf: {res['avg_conf']:.1f}%")

    print("\n" + "="*60)
    print("  FINAL RESULTS — V2-aug Pure Temporal Model")
    print("="*60)
    print(f"  {'Dataset':<18} {'Label':<6} {'Videos':<8} {'Accuracy':<10} {'Avg Conf'}")
    print("  " + "-"*55)

    fake_accs = []; real_accs = []
    for r in results:
        print(f"  {r['name']:<18} {r['label']:<6} {r['total']:<8} {r['accuracy']:>6.1f}%    {r['avg_conf']:>6.1f}%")
        if r["label"] == "FAKE": fake_accs.append(r["accuracy"])
        else: real_accs.append(r["accuracy"])

    print("  " + "-"*55)
    overall = total_correct / total_videos * 100 if total_videos > 0 else 0
    print(f"\n  Fake detection avg : {sum(fake_accs)/len(fake_accs):.1f}%" if fake_accs else "")
    print(f"  Real detection avg : {sum(real_accs)/len(real_accs):.1f}%" if real_accs else "")
    print(f"  Overall accuracy   : {overall:.1f}%")
    print("\n" + "="*60)
    print("  Testing complete! [V2-aug Pure Temporal Model]")
    print("="*60)


if __name__ == "__main__":
    main()
