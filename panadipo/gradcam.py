"""Grad-CAM++ saliency maps for individual clips (Figure 1A, Figure 2A, Supplemental Figure 4).

Maps are computed frame by frame from the final convolutional stage of the PanEcho ConvNeXt image encoder, with the
PanAdipo head applied to the frame embedding (the temporal transformer is bypassed). Each map is sharpened with an
unsharp mask (Gaussian sigma 3, gain 2), thresholded at a normalized activation of 0.25, and rescaled to 0-1.
Maps are illustrative: the model is weakly supervised by study-level report labels.

    python -m panadipo.gradcam --checkpoint <best_model.pt> --out_dir <folder> --fpaths <clip.avi> [<clip.avi> ...]

Requires the `grad-cam` package (pip install grad-cam).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.ndimage import gaussian_filter
from torch import nn

from .data import read_frames, resize, sample_frames, to_tensor
from .model import load_checkpoint


class FrameModel(nn.Module):
    """ConvNeXt frame encoder + PanAdipo head, so Grad-CAM++ can run on single (B, 3, H, W) frames."""

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, x):
        return self.head(self.encoder(x))           # (B, 1): the logit is the Grad-CAM target


def image_encoder(model) -> nn.Module:
    """The ConvNeXt image encoder inside the PanEcho backbone (module with `features` and `avgpool`)."""
    found = [m for _, m in model.backbone.named_modules() if hasattr(m, "features") and hasattr(m, "avgpool")]
    if not found:
        raise RuntimeError("ConvNeXt image encoder not found in the backbone")
    return found[-1]


def sharpen(cam: np.ndarray, sigma: float = 3.0, gain: float = 2.0, threshold: float = 0.25) -> np.ndarray:
    s = np.clip(cam + gain * (cam - gaussian_filter(cam, sigma)), 0, 1)
    s[s < threshold] = 0
    return s / s.max() if s.max() > 0 else s


def clip_maps(model, fpath: str, clip_len: int = 16, size: int = 224, layer: int = -1, **sharpen_kw):
    """Returns (frames uint8 (T, H, W, 3), maps float (T, H, W)) for one clip."""
    frames = resize(sample_frames(read_frames(fpath), clip_len), size, cv2.INTER_LINEAR)
    x = to_tensor(frames).unsqueeze(0).to(next(model.parameters()).device)     # (1, 3, T, H, W)
    enc = image_encoder(model)
    fm = FrameModel(enc, model.head).eval()
    maps = []
    with GradCAMPlusPlus(model=fm, target_layers=[enc.features[layer]]) as cam:
        for t in range(x.shape[2]):
            maps.append(sharpen(cam(input_tensor=x[:, :, t].contiguous(), targets=None)[0], **sharpen_kw))
    return frames, np.stack(maps)


def save_strip(frames: np.ndarray, maps: np.ndarray, out: Path, n: int = 8) -> None:
    """Two rows: n evenly spaced frames (top) and their Grad-CAM++ overlays (bottom)."""
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    top = np.concatenate([frames[i] for i in idx], axis=1)
    bottom = np.concatenate([show_cam_on_image(frames[i] / 255.0, maps[i], use_rgb=True) for i in idx], axis=1)
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate([top, bottom], axis=0), cv2.COLOR_RGB2BGR))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fpaths", nargs="+", required=True)
    p.add_argument("--layer", type=int, default=-1, help="ConvNeXt stage (-1 = last)")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--n_display", type=int, default=8)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args = load_checkpoint(a.checkpoint, device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(a.fpaths, 1):
        frames, maps = clip_maps(model, f, args.get("clip_len", 16), layer=a.layer, threshold=a.threshold)
        with torch.no_grad():
            score = torch.sigmoid(model(to_tensor(frames).unsqueeze(0).to(device))).item()
        save_strip(frames, maps, out / f"clip{k:02d}_score{score:.3f}.png", a.n_display)
        np.save(out / f"clip{k:02d}_maps.npy", maps)                 # for mean maps across clips


if __name__ == "__main__":
    main()
