"""
training_video/video_model_v2.py
─────────────────────────────────
Pure Temporal Deepfake Detector (V2)

Key improvement over V1:
  Frame Difference Module explicitly computes
  temporal changes between consecutive frames.
  This directly detects flickering and warping
  that deepfakes produce between frames.

Architecture:
  Input: (B, N, 3, 224, 224)  — N=10 frames
      ↓
  EfficientNet-B4 (shared backbone)
  → frame features: (B, N, 1792)
      ↓
  Frame Difference Module:
    diff[t] = feat[t+1] - feat[t]
    → (B, N-1, 1792) temporal changes
      ↓
  Concat [frame_feats, frame_diffs]
    → (B, N + N-1, 1792) = (B, 19, 1792)
      ↓
  Projection → (B, 19, 512)
      ↓
  [CLS] + positional encoding
    → (B, 20, 512)
      ↓
  Temporal Transformer (4 layers, 8 heads)
  Self-attention across ALL tokens
      ↓
  CLS output → (B, 512)
      ↓
  Classifier → (B, 1) logit
"""

from __future__ import annotations
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FrameDifferenceModule(nn.Module):
    """
    Computes differences between consecutive frame features.

    For N frames → produces N-1 difference vectors.
    Each diff = feat[t+1] - feat[t]

    This explicitly captures:
    - Temporal flickering (rapid changes)
    - Face warping between frames
    - Unnatural transitions unique to deepfakes
    """

    def __init__(self, feature_dim: int, d_model: int) -> None:
        super().__init__()
        # Project differences to same space as frame features
        self.diff_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            Shape (B, N, feature_dim) — frame features

        Returns
        -------
        torch.Tensor
            Shape (B, N-1, d_model) — projected frame differences
        """
        # Compute consecutive frame differences
        diffs = features[:, 1:, :] - features[:, :-1, :]  # (B, N-1, feature_dim)
        return self.diff_projection(diffs)                  # (B, N-1, d_model)


class PureTemporalDeepfakeDetector(nn.Module):
    """
    Pure temporal deepfake detector with explicit frame difference modeling.

    Unlike V1 which only used frame features, V2 also computes
    frame differences to explicitly detect temporal inconsistencies.
    """

    def __init__(
        self,
        backbone_name:  str   = "efficientnet_b4",
        pretrained:     bool  = True,
        feature_dim:    int   = 1792,
        d_model:        int   = 512,
        nhead:          int   = 8,
        num_layers:     int   = 4,
        dropout:        float = 0.3,
        n_frames:       int   = 10,
    ) -> None:
        super().__init__()
        self.n_frames    = n_frames
        self.feature_dim = feature_dim
        self.d_model     = d_model

        # ── 1. EfficientNet-B4 backbone (shared across frames) ────────────────
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )

        # ── 2. Frame feature projection ───────────────────────────────────────
        self.frame_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ── 3. Frame Difference Module (KEY IMPROVEMENT) ──────────────────────
        self.frame_diff = FrameDifferenceModule(feature_dim, d_model)

        # ── 4. CLS token ──────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 5. Positional encoding ────────────────────────────────────────────
        # n_frames frame tokens + (n_frames-1) diff tokens + 1 CLS
        n_tokens = n_frames + (n_frames - 1) + 1  # = 20
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ── 6. Temporal Transformer ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── 7. Classifier ─────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, N, 3, H, W)

        Returns
        -------
        torch.Tensor
            Shape (B, 1) raw logit
        """
        B, N, C, H, W = x.shape

        # ── Extract frame features ────────────────────────────────────────────
        feat_flat = self.backbone(x.view(B * N, C, H, W))   # (B*N, 1792)
        features  = feat_flat.view(B, N, -1)                 # (B, N, 1792)

        # ── Project frame features ────────────────────────────────────────────
        frame_tokens = self.frame_projection(features)        # (B, N, 512)

        # ── Compute frame differences ─────────────────────────────────────────
        diff_tokens = self.frame_diff(features)               # (B, N-1, 512)

        # ── Interleave frame tokens and diff tokens ───────────────────────────
        # Order: [frame0, diff0, frame1, diff1, ..., frame8, diff8, frame9]
        # This preserves temporal order: frame → change → next frame
        interleaved = []
        for i in range(N - 1):
            interleaved.append(frame_tokens[:, i:i+1, :])    # frame i
            interleaved.append(diff_tokens[:, i:i+1, :])     # diff i
        interleaved.append(frame_tokens[:, N-1:N, :])        # last frame
        temporal_tokens = torch.cat(interleaved, dim=1)       # (B, 19, 512)

        # ── Prepend CLS token ─────────────────────────────────────────────────
        cls_tokens = self.cls_token.expand(B, -1, -1)         # (B, 1, 512)
        sequence   = torch.cat([cls_tokens, temporal_tokens], dim=1)  # (B, 20, 512)

        # ── Add positional encoding ───────────────────────────────────────────
        sequence = sequence + self.pos_embedding

        # ── Temporal Transformer ──────────────────────────────────────────────
        encoded    = self.transformer(sequence)               # (B, 20, 512)
        cls_output = encoded[:, 0, :]                         # (B, 512)

        return self.classifier(cls_output)                    # (B, 1)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("Backbone unfrozen")

    def get_num_params(self) -> tuple[int, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def get_video_model_v2(config: dict, device: torch.device) -> PureTemporalDeepfakeDetector:
    """Build PureTemporalDeepfakeDetector from config."""
    m = config["model"]

    model = PureTemporalDeepfakeDetector(
        backbone_name = m["backbone"],
        pretrained    = False,
        feature_dim   = m["feature_dim"],
        d_model       = m["d_model"],
        nhead         = m["nhead"],
        num_layers    = m["transformer_layers"],
        dropout       = m["dropout"],
        n_frames      = m["n_frames"],
    )

    # Load backbone weights from image model
    ckpt_path = Path(m["pretrained_image_model"])
    if ckpt_path.exists():
        print(f"  Loading backbone from: {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone_weights = {
            k.replace("backbone.", ""): v
            for k, v in ckpt["model_state"].items()
            if k.startswith("backbone.")
        }
        model.backbone.load_state_dict(backbone_weights, strict=True)
        print("  Backbone weights loaded! ✅")
    else:
        print("  [WARNING] No pretrained model found, using ImageNet weights")
        model.backbone = timm.create_model(m["backbone"], pretrained=True, num_classes=0)

    model = model.to(device)

    total, trainable = model.get_num_params()
    print(f"  Total parameters     : {total/1e6:.2f}M")
    print(f"  Trainable parameters : {trainable/1e6:.2f}M")
    print(f"  Device               : {device}")

    return model


if __name__ == "__main__":
    config_path = Path(__file__).parent / "video_config_v2.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Building pure temporal video model v2...")
    model = get_video_model_v2(config, device)

    # Test forward pass
    dummy = torch.randn(2, 10, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"\nInput  shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print("\nVideo Model V2 OK! ✅")