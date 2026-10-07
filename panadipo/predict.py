"""PanAdipo inference and study-level aggregation.

    python -m panadipo.predict --checkpoint <best_model.pt> --clips_csv <clip table> --out_csv <clip scores>

The clip table needs `fpath`; every other column (e.g., patient_id, study_id, view) is carried through to the output,
which adds `score` (sigmoid output, 0-1). Because of the positive-class weight, the score is not a calibrated
probability; it is analyzed as a continuous score without a threshold.

Study-level scores are computed from the clip scores with panadipo.scores.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import ClipDataset
from .model import load_checkpoint


def predict(checkpoint: str, clips: pd.DataFrame, batch_size: int = 64, num_workers: int = 8) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args = load_checkpoint(checkpoint, device)
    ds = ClipDataset(clips, args.get("clip_len", 16), args.get("frame_size", 224), mode="predict")
    scores = np.full(len(ds), np.nan)
    with torch.no_grad():
        for video, label, idx in DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True):
            p = torch.sigmoid(model(video.to(device))).cpu().numpy()
            readable = (label != -1).numpy()
            scores[idx.numpy()[readable]] = p[readable]
    return clips.assign(score=scores)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clips_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    a = p.parse_args()
    out = predict(a.checkpoint, pd.read_csv(a.clips_csv), a.batch_size, a.num_workers)
    out.to_csv(a.out_csv, index=False)
    print(f"scored {out['score'].notna().sum():,} of {len(out):,} clips")


if __name__ == "__main__":
    main()
