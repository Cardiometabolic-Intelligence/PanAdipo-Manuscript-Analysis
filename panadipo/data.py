"""Echocardiography clip loading.

Clips are preprocessed upstream with the PanEcho pipeline (DICOM -> sector-masked, cropped, 224 x 224 within a
256 x 256 frame, MJPG .avi at the native frame rate). Each clip is sampled to `clip_len` evenly spaced frames and
ImageNet-normalized. Training uses PanEcho-style augmentation applied identically to every frame of a clip.

Clip tables are CSV files with one row per clip: `fpath` (video file) and, for training, `label` (0/1).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_frames(fpath: str, max_frames: int = 1000) -> np.ndarray:
    cap = cv2.VideoCapture(fpath)
    frames = []
    for _ in range(max_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError("no frames decoded")
    return np.stack(frames)


def sample_frames(video: np.ndarray, clip_len: int) -> np.ndarray:
    """Evenly spaced frames; short clips are padded by repeating the last frame."""
    t = len(video)
    if t >= clip_len:
        return video[np.linspace(0, t - 1, clip_len).astype(int)]
    idx = np.r_[np.arange(t), [t - 1] * (clip_len - t)]
    return video[idx]


def resize(video: np.ndarray, size: int, interp=cv2.INTER_AREA) -> np.ndarray:
    if video.shape[1:3] == (size, size):
        return video
    return np.stack([cv2.resize(f, (size, size), interpolation=interp) for f in video])


def augment(video: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Random zoom-out (p=0.5, up to 1.2x), random crop to size, horizontal flip (p=0.5), rotation (+/-15 degrees)."""
    t, h, w, c = video.shape
    if rng.random() < 0.5:
        s = rng.uniform(1.0, 1.2)
        nh, nw = int(h * s), int(w * s)
        top, left = rng.integers(0, nh - h + 1), rng.integers(0, nw - w + 1)
        pad = np.zeros((t, nh, nw, c), dtype=video.dtype)
        pad[:, top:top + h, left:left + w] = video
        video, h, w = pad, nh, nw
    top, left = rng.integers(0, max(h - size, 0) + 1), rng.integers(0, max(w - size, 0) + 1)
    video = resize(video[:, top:top + size, left:left + size], size, cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        video = video[:, :, ::-1].copy()
    angle = rng.uniform(-15, 15)
    if abs(angle) > 0.5:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        video = np.stack([cv2.warpAffine(f, m, (size, size), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) for f in video])
    return video


def center_crop(video: np.ndarray, size: int) -> np.ndarray:
    h, w = video.shape[1:3]
    top, left = max((h - size) // 2, 0), max((w - size) // 2, 0)
    return video[:, top:top + size, left:left + size]


def to_tensor(video: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (3, T, H, W) float, ImageNet-normalized."""
    v = (video.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(v).permute(3, 0, 1, 2).contiguous()


class ClipDataset(Dataset):
    """Clip table -> (video tensor, label, row index). Unlabeled clips return label NaN; unreadable clips return
    label -1 (skipped in training, left unscored at inference).

    mode: "train" (augmented), "val" (center crop of the 256 x 256 frame), or "predict" (resize to frame_size, as at
    inference).
    """

    def __init__(self, table: pd.DataFrame, clip_len: int = 16, frame_size: int = 224, mode: str = "val",
                 label_col: str = "label"):
        self.df = table.reset_index(drop=True)
        self.clip_len, self.size, self.mode, self.label_col = clip_len, frame_size, mode, label_col

    def __len__(self) -> int:
        return len(self.df)

    def load(self, fpath: str) -> torch.Tensor:
        video = sample_frames(read_frames(fpath), self.clip_len)
        if self.mode == "train":
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ())))     # torch seeds each loader worker
            video = augment(resize(video, 256), self.size, rng)
        elif self.mode == "val":
            video = center_crop(resize(video, 256), self.size)
        else:
            video = resize(video, self.size)
        return to_tensor(video)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = float(row[self.label_col]) if self.label_col in row and pd.notna(row[self.label_col]) else np.nan
        try:
            video = self.load(str(row["fpath"]))
        except Exception:  # noqa: BLE001 - corrupt or unreadable clip
            video, label = torch.zeros(3, self.clip_len, self.size, self.size), -1.0
        return video, torch.tensor(label, dtype=torch.float32), idx
