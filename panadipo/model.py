"""PanAdipo = the public PanEcho video encoder (768-dimensional pooled embedding of a 16-frame clip) followed by a
dropout layer and a single linear projection to one logit. The whole network is fine-tuned end-to-end."""

from __future__ import annotations

import torch
from torch import nn

PANECHO_HUB = "CarDS-Yale/PanEcho"


class PanAdipo(nn.Module):
    def __init__(self, clip_len: int = 16, dropout: float = 0.3):
        super().__init__()
        self.backbone = torch.hub.load(PANECHO_HUB, "PanEcho", backbone_only=True, clip_len=clip_len, trust_repo=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, T, H, W) -> logits (B,)."""
        return self.head(self.backbone(x)).squeeze(1)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[PanAdipo, dict]:
    """Load a checkpoint written by panadipo.train (keys: model_state, args)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = PanAdipo(clip_len=args.get("clip_len", 16), dropout=args.get("dropout", 0.3))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), args
