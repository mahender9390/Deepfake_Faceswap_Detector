"""
training_video/video_train_v2.py
──────────────────────────────────
Training pipeline for V2 Pure Temporal model.

Anti-overfitting features:
  1. Strong augmentation (video_dataset_v2.py)
  2. MixUp augmentation — blends clips
  3. Dropout in model
  4. Weight decay
  5. Early stopping
  6. Consistent frame augmentation

Usage:
    python training_video/video_train_v2.py
"""

from __future__ import annotations
import csv, random, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch, torch.nn as nn, yaml
from sklearn.metrics import f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training_video.video_dataset_v2 import get_video_dataloaders_v2
from training_video.video_model_v2   import get_video_model_v2


def set_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def mixup_data(clips, labels, alpha=0.2):
    """MixUp: blend two clips to prevent memorization."""
    lam   = np.random.beta(alpha, alpha)
    index = torch.randperm(clips.size(0), device=clips.device)
    mixed = lam * clips + (1 - lam) * clips[index]
    return mixed, labels, labels[index], lam


def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


def build_opt1(model, config):
    t = config["training"]
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t["base_lr"], weight_decay=t["weight_decay"])


def build_opt2(model, config):
    t      = config["training"]
    bb     = list(model.backbone.parameters())
    bb_ids = {id(p) for p in bb}
    head   = [p for p in model.parameters() if id(p) not in bb_ids]
    return torch.optim.AdamW([
        {"params": bb,   "lr": t["backbone_lr"]},
        {"params": head, "lr": t["base_lr"]},
    ], weight_decay=t["weight_decay"])


def build_scheduler(opt, config):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=1, eta_min=1e-6)


def train_epoch(model, loader, opt, criterion, scaler, config, device, epoch):
    model.train()
    t         = config["training"]
    accum     = t["accumulation_steps"]
    use_amp   = t["mixed_precision"] and device.type == "cuda"
    use_mixup = epoch > 2  # start mixup after warmup
    loss_sum  = 0.0
    opt.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"  Train E{epoch:02d}", leave=False, ncols=90)

    for step, (clips, labels) in pbar:
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        if use_mixup and random.random() < 0.5:
            clips, ya, yb, lam = mixup_data(clips, labels)
            with autocast('cuda', enabled=use_amp):
                loss = mixup_criterion(criterion, model(clips), ya, yb, lam) / accum
        else:
            with autocast('cuda', enabled=use_amp):
                loss = criterion(model(clips), labels) / accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                scaler.step(opt); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                opt.step()
            opt.zero_grad()

        loss_sum += loss.item() * accum
        pbar.set_postfix(loss=f"{loss.item()*accum:.4f}")

    return loss_sum / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_probs, all_preds = 0.0, [], [], []

    for clips, labels in tqdm(loader, desc="  Val  ", leave=False, ncols=90):
        clips  = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        logits = model(clips)
        total_loss += criterion(logits, labels).item()
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).long()
        all_labels.extend(labels.squeeze(1).cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return (
        total_loss / len(loader),
        sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels),
        roc_auc_score(all_labels, all_probs),
        f1_score(all_labels, all_preds, zero_division=0),
    )


def plot_curves(log_path, out_path):
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("V2 Pure Temporal (Strong Aug) — Curves", fontweight="bold")
    axes[0].plot(df["epoch"], df["train_loss"], label="Train", lw=2)
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val",   lw=2, ls="--")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(df["epoch"], df["val_auc"], color="darkorange", lw=2)
    best = df["val_auc"].idxmax()
    axes[1].scatter(df.loc[best,"epoch"], df.loc[best,"val_auc"],
                    color="red", s=80, label=f'Best {df.loc[best,"val_auc"]:.4f}')
    axes[1].set(xlabel="Epoch", ylabel="AUC", title="Val AUC")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Curves → {out_path}")


def train(config):
    t = config["training"]; p = config["paths"]
    set_seeds(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading datasets with STRONG augmentation...")
    train_loader, val_loader, _ = get_video_dataloaders_v2(config)

    print("\nBuilding Pure Temporal Model V2...")
    model = get_video_model_v2(config, device)

    total_p, _ = model.get_num_params()
    eff_batch  = t["batch_size"] * t["accumulation_steps"]
    gpu_name   = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print()
    print("=" * 58)
    print("  V2 Pure Temporal — Strong Augmentation Training")
    print("=" * 58)
    print(f"  GPU        : {gpu_name}")
    print(f"  Train      : {len(train_loader.dataset)} clips")
    print(f"  Val        : {len(val_loader.dataset)} clips")
    print(f"  Frames     : {config['model']['n_frames']} per clip")
    print(f"  Params     : {total_p/1e6:.2f}M")
    print(f"  Epochs     : {t['epochs']}")
    print(f"  Batch      : {t['batch_size']} (effective: {eff_batch})")
    print(f"  MixUp      : ✅ (after epoch 2, 50% chance)")
    print(f"  Frame Drop : ✅ (30% chance per clip)")
    print(f"  Aug        : Heavy spatial + color + noise")
    print("=" * 58)

    criterion = nn.BCEWithLogitsLoss()
    use_amp   = t["mixed_precision"] and device.type == "cuda"
    scaler    = GradScaler('cuda', enabled=use_amp)

    Path(p["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(p["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    log_csv   = Path(p["log_dir"]) / "video_v2_aug_log.csv"
    best_path = Path(p["best_model"])
    last_path = Path(p["last_model"])

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss",
                                 "val_acc","val_auc","val_f1","lr","phase"])

    best_auc = 0.0; best_epoch = 0; patience_count = 0; phase = 1

    model.freeze_backbone()
    print(f"\nPhase 1: Backbone frozen (epochs 1-{t['freeze_epochs']})\n")
    opt = build_opt1(model, config)
    sch = build_scheduler(opt, config)

    try:
        for epoch in range(1, t["epochs"] + 1):

            if epoch == t["freeze_epochs"] + 1 and phase == 1:
                phase = 2
                model.unfreeze_backbone()
                print("\nPhase 2: Full model training")
                opt = build_opt2(model, config)
                sch = build_scheduler(opt, config)

            train_loss = train_epoch(
                model, train_loader, opt, criterion,
                scaler, config, device, epoch
            )
            val_loss, val_acc, val_auc, val_f1 = validate(
                model, val_loader, criterion, device
            )
            sch.step()
            lr = opt.param_groups[0]["lr"]

            print(f"Epoch {epoch:02d}/{t['epochs']:02d} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Acc: {val_acc*100:.1f}% | AUC: {val_auc:.3f} | "
                  f"F1: {val_f1:.3f} | LR: {lr:.2e}")

            # Check for overfitting
            gap = train_loss - val_loss
            if gap < -0.1:
                print(f"  ⚠️  Val loss lower than train (gap={gap:.3f}) — good generalization!")
            elif gap > 0.15:
                print(f"  ⚠️  Train-val gap large ({gap:.3f}) — watch for overfitting")

            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                    f"{val_acc:.6f}", f"{val_auc:.6f}",
                    f"{val_f1:.6f}", f"{lr:.8f}", f"phase{phase}"
                ])

            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_auc": val_auc, "config": config
            }, last_path)

            if val_auc > best_auc:
                best_auc = val_auc; best_epoch = epoch; patience_count = 0
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "val_auc": val_auc, "config": config
                }, best_path)
                print(f"  ✓ New best AUC: {best_auc:.3f} — model saved!")
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{t['patience']})")

            if patience_count >= t["patience"]:
                print(f"\nEarly stopping triggered.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted! Saving...")
        torch.save({"model_state": model.state_dict(), "config": config}, last_path)
        print("Saved.")
        return

    print(f"\nBest AUC: {best_auc:.4f} at epoch {best_epoch}")
    plot_curves(log_csv, Path(p["log_dir"]) / "video_v2_aug_curves.png")
    print("Training complete! ✅")


if __name__ == "__main__":
    config_path = Path(__file__).parent / "video_config_v2.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train(config)