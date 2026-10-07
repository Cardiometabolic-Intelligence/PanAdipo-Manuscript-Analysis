"""Fine-tune PanAdipo (PanEcho backbone + linear head) for the report-derived label of a prominent epicardial fat depot.

Settings default to configs/train.yaml; any flag on the command line overrides the file. Early stopping, checkpoint
selection, and the learning-rate scheduler all follow the validation metric chosen with --monitor (auprc or auroc),
and training stops after --patience epochs without improvement (up to --epochs).

Distributed (6 GPUs x 24 clips = effective batch 144):
    torchrun --nproc_per_node 6 -m panadipo.train --config configs/train.yaml \
        --train_csv <train clips> --val_csv <validation clips> --out_dir <run folder>
Alternative stopping rule, e.g. validation AUROC with patience 7:
    torchrun --nproc_per_node 6 -m panadipo.train --config configs/train.yaml --monitor auroc --patience 7 ...
Single GPU or CPU (smoke test):
    python -m panadipo.train --config configs/train.yaml --per_gpu_batch 2 --epochs 1 ...

Clip tables: one row per clip with `fpath` and `label` (0/1); patients must not overlap between partitions.
Outputs in --out_dir: best_model.pt (best epoch by the monitored metric), training_log.csv, config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import ClipDataset
from .model import PanAdipo

METRICS = {"auroc": roc_auc_score, "auprc": average_precision_score}


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    defaults = yaml.safe_load(Path(known.config).read_text()) if known.config else {}

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file with default settings (configs/train.yaml)")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label_col", default="label")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--per_gpu_batch", type=int, default=24)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr_divisor", type=float, default=10)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pos_weight", type=float, default=15)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reduce_lr_patience", type=int, default=3)
    p.add_argument("--reduce_lr_factor", type=float, default=0.1)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--epochs", type=int, default=50, help="maximum number of epochs")
    p.add_argument("--monitor", choices=sorted(METRICS), default="auprc",
                   help="validation metric for early stopping, checkpoint selection, and the scheduler")
    p.add_argument("--patience", type=int, default=5, help="epochs without improvement before stopping (e.g., 5-7)")
    p.add_argument("--seed", type=int, default=20260401)
    unknown = set(defaults) - {a.dest for a in p._actions}
    if unknown:
        p.error(f"unknown keys in --config: {sorted(unknown)}")
    p.set_defaults(**defaults)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- distributed helpers
def setup() -> tuple[torch.device, int, int]:
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank), dist.get_rank(), dist.get_world_size()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1


def gather(obj, world: int) -> list:
    if world == 1:
        return [obj]
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out


# ----------------------------------------------------------------------------- one epoch
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False) -> tuple[float, dict]:
    """Returns (mean loss on this rank, {clip index: (label, probability)})."""
    train = optimizer is not None
    model.train(train)
    loss_sum, n, preds = 0.0, 0, {}
    with torch.set_grad_enabled(train):
        for video, label, idx in loader:
            keep = label >= 0                                        # skip unreadable clips
            if not keep.any():
                continue
            video, label, idx = video[keep].to(device, non_blocking=True), label[keep].to(device), idx[keep]
            with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
                logits = model(video)
                loss = criterion(logits.float(), label)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * len(label)
            n += len(label)
            for i, y, pr in zip(idx.tolist(), label.tolist(), torch.sigmoid(logits.float()).tolist()):
                preds[i] = (y, pr)
    return loss_sum / max(n, 1), preds


def score(preds: dict) -> dict:
    y, p = map(np.asarray, zip(*preds.values())) if preds else (np.array([]), np.array([]))
    ok = 0 < y.sum() < len(y)
    return {k: float(f(y, p)) if ok else float("nan") for k, f in METRICS.items()}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    args = parse_args(argv)
    device, rank, world = setup()
    main_proc = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    out = Path(args.out_dir)
    if main_proc:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({**vars(args), "world_size": world,
                                                     "effective_batch": args.per_gpu_batch * world}, indent=2))

    tables = {k: pd.read_csv(getattr(args, f"{k}_csv")).dropna(subset=[args.label_col]) for k in ("train", "val")}
    ds = {"train": ClipDataset(tables["train"], args.clip_len, args.frame_size,
                               "train" if args.augment else "val", args.label_col),
          "val": ClipDataset(tables["val"], args.clip_len, args.frame_size, "val", args.label_col)}
    samplers = {k: DistributedSampler(v, shuffle=(k == "train"), seed=args.seed) if world > 1 else None
                for k, v in ds.items()}
    loaders = {k: DataLoader(v, batch_size=args.per_gpu_batch, sampler=samplers[k],
                             shuffle=(k == "train" and world == 1), drop_last=(k == "train"),
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=args.num_workers > 0) for k, v in ds.items()}

    model = PanAdipo(args.clip_len, args.dropout).to(device)
    backbone_lr = args.head_lr / args.backbone_lr_divisor                     # backbone unfrozen, 10-fold lower LR
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": backbone_lr},
                                   {"params": model.head.parameters(), "lr": args.head_lr}],
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.reduce_lr_factor,
                                                           patience=args.reduce_lr_patience, min_lr=args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    net = DDP(model, device_ids=[device.index]) if world > 1 else model

    best, waited, log = -np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if samplers["train"] is not None:
            samplers["train"].set_epoch(epoch)
        train_loss, train_preds = run_epoch(net, loaders["train"], criterion, device, optimizer, scaler, args.amp)
        val_loss, val_preds = run_epoch(net, loaders["val"], criterion, device, amp=args.amp)
        # metrics on the full validation set (predictions gathered from every rank; duplicates from padding removed)
        val_all = {k: v for part in gather(val_preds, world) for k, v in part.items()}
        train_all = {k: v for part in gather(train_preds, world) for k, v in part.items()}
        vm, tm = score(val_all), score(train_all)
        current = vm[args.monitor]
        scheduler.step(current)
        improved = np.isfinite(current) and current > best
        log.append({"epoch": epoch, "lr_head": optimizer.param_groups[1]["lr"], "train_loss": train_loss,
                    "train_auroc": tm["auroc"], "train_auprc": tm["auprc"], "val_loss": val_loss,
                    "val_auroc": vm["auroc"], "val_auprc": vm["auprc"], "best": improved, "seconds": time.time() - t0})
        if improved:
            best, waited = current, 0
            if main_proc:
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "args": vars(args),
                            f"val_{args.monitor}": current, "val_metrics": vm}, out / "best_model.pt")
        else:
            waited += 1
        if main_proc:
            pd.DataFrame(log).to_csv(out / "training_log.csv", index=False)
            print(f"epoch {epoch:03d}  val AUROC {vm['auroc']:.4f}  AUPRC {vm['auprc']:.4f}  "
                  f"{'* best' if improved else f'no improvement ({waited}/{args.patience})'}", flush=True)
        if waited >= args.patience:
            if main_proc:
                print(f"early stopping: validation {args.monitor.upper()} did not improve for {args.patience} epochs")
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
