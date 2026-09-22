"""Training entry point.

Differences from the original loop, all of which were blocking a defensible
result rather than merely being untidy:

* **There is a validation set.**  The original trained for a fixed 100 epochs,
  printed the training loss and saved the last weights.  It could not detect
  overfitting, could not select a checkpoint, and produced no number that
  could be quoted.
* **Metrics are computed** every epoch (see ``metrics.py``), and the best
  checkpoint is chosen by a held-out score rather than by being last.
* **Runs are seeded and self-describing** - the resolved config is written
  next to the weights, with per-epoch history in a CSV.
* Cosine schedule with warmup, AMP, gradient clipping, optional early stopping.

Example
-------
    python src/train_unet.py --config configs/default.yaml
    python src/train_unet.py --config configs/default.yaml --epochs 200 --lr 5e-4
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labels as L
from coastline_dataset import AugmentConfig, CoastlineDataset, DatasetConfig
from losses import CombinedLoss
from metrics import EdgeMetrics, SegMetrics
from UNet import UNet


# --------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(cfg):
    d = cfg["data"]
    ds_cfg = DatasetConfig(
        image_dir=d["image_dir"],
        mask_dir=d["mask_dir"],
        cloud_mask_dir=d["cloud_mask_dir"],
        img_size=tuple(d["img_size"]),
        night_luma=d["night_luma"],
        mean=tuple(d["mean"]),
        std=tuple(d["std"]),
        val_fraction=d["val_fraction"],
        split_seed=d["split_seed"],
        augment=AugmentConfig(**cfg["augment"]),
    )
    return CoastlineDataset(ds_cfg, "train"), CoastlineDataset(ds_cfg, "val")


def build_scheduler(cfg, optimizer, steps_per_epoch):
    t = cfg["train"]
    kind = (t.get("scheduler") or "none").lower()
    if kind == "cosine":
        warmup = t.get("warmup_epochs", 0) * steps_per_epoch
        total = max(t["epochs"] * steps_per_epoch, 1)

        def lr_lambda(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            p = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + np.cos(np.pi * min(p, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=t.get("monitor_mode", "max"),
            factor=0.5,
            patience=max(t["epochs"] // 20, 3),
        ), "epoch"
    return None, "none"


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    seg_m = SegMetrics(num_classes, L.SEG_CLASS_NAMES)
    edge_m = {c: EdgeMetrics() for c in range(L.NUM_EDGE_CHANNELS)}
    loss_sum, n = 0.0, 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        seg = batch["seg"].to(device, non_blocking=True)
        edges = batch["edges"].to(device, non_blocking=True)
        valid = batch["edge_valid"].to(device, non_blocking=True)

        seg_out, edge_out, _ = model(image)
        loss, _ = criterion(seg_out, edge_out, seg, edges, valid, ())
        loss_sum += float(loss) * image.size(0)
        n += image.size(0)

        seg_m.update(seg_out.argmax(1).cpu().numpy(), seg.cpu().numpy())
        prob = torch.sigmoid(edge_out).cpu().numpy()
        e_np, v_np = edges.cpu().numpy(), valid.cpu().numpy()
        for b in range(image.size(0)):
            for c in range(L.NUM_EDGE_CHANNELS):
                edge_m[c].update(prob[b, c], e_np[b, c] > 0.5, v_np[b, c] > 0.5)

    out = {"loss": loss_sum / max(n, 1)}
    out.update({f"seg/{k}": v for k, v in seg_m.compute().items()})
    for c, m in edge_m.items():
        for k, v in m.compute().items():
            out[f"edge/{L.EDGE_CHANNEL_NAMES[c]}/{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--out-dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="stop each epoch early; for smoke tests only")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key, val in (("epochs", args.epochs), ("batch_size", args.batch_size),
                     ("lr", args.lr), ("out_dir", args.out_dir), ("seed", args.seed)):
        if val is not None:
            cfg["train"][key] = val
    if args.no_amp:
        cfg["train"]["amp"] = False

    t = cfg["train"]
    set_seed(t["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = t["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_ds, val_ds = build_datasets(cfg)
    print(f"train={len(train_ds)} images   val={len(val_ds)} images   device={device}")
    if len(val_ds) == 0:
        raise RuntimeError(
            "validation split is empty - lower data.val_fraction or add images. "
            "Training without a held-out set produces no reportable result."
        )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                              num_workers=t["num_workers"], pin_memory=pin,
                              drop_last=len(train_ds) > t["batch_size"],
                              persistent_workers=t["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            num_workers=t["num_workers"], pin_memory=pin,
                            persistent_workers=t["num_workers"] > 0)

    model = UNet(in_channels=3, num_classes=L.NUM_SEG_CLASSES,
                 num_edge_channels=L.NUM_EDGE_CHANNELS, **cfg["model"]).to(device)
    criterion = CombinedLoss(num_classes=L.NUM_SEG_CLASSES, **cfg["loss"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=t["lr"], weight_decay=t["weight_decay"],
    )
    scheduler, sched_when = build_scheduler(cfg, optimizer, max(len(train_loader), 1))
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    resolved = copy.deepcopy(cfg)
    resolved["_meta"] = {"n_params": n_params, "device": str(device),
                         "train_images": len(train_ds), "val_images": len(val_ds),
                         "torch": str(torch.__version__)}
    with open(os.path.join(out_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    monitor, mode = t["monitor"], t["monitor_mode"]
    best = -np.inf if mode == "max" else np.inf
    best_epoch, since_improved = -1, 0
    hist_path = os.path.join(out_dir, "history.csv")
    history = []
    total_start = time.time()

    for epoch in range(t["epochs"]):
        model.train()
        ep_start = time.time()
        sums, nb = {}, 0

        for i, batch in enumerate(train_loader):
            if args.limit_batches and i >= args.limit_batches:
                break
            image = batch["image"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            edges = batch["edges"].to(device, non_blocking=True)
            valid = batch["edge_valid"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                seg_out, edge_out, aux = model(image)
                loss, parts = criterion(seg_out, edge_out, seg, edges, valid, aux)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if t.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()),
                    t["grad_clip"],
                )
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and sched_when == "step":
                scheduler.step()

            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            nb += 1

        train_stats = {f"train/{k}": v / max(nb, 1) for k, v in sums.items()}
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
               "epoch_sec": time.time() - ep_start, **train_stats}

        if (epoch + 1) % t.get("eval_every", 1) == 0 or epoch == t["epochs"] - 1:
            val_stats = evaluate(model, val_loader, criterion, device, L.NUM_SEG_CLASSES)
            row.update({f"val/{k}": v for k, v in val_stats.items()})

            score = val_stats.get(monitor)
            if score is None:
                raise KeyError(
                    f"monitor {monitor!r} is not a validation metric. Available: "
                    + ", ".join(sorted(val_stats))
                )
            improved = (score > best) if mode == "max" else (score < best)
            if improved and np.isfinite(score):
                best, best_epoch, since_improved = score, epoch + 1, 0
                torch.save(
                    {"model": model.state_dict(), "criterion": criterion.state_dict(),
                     "epoch": epoch + 1, "config": resolved,
                     "metric": {monitor: score}},
                    os.path.join(out_dir, "best.pth"),
                )
            else:
                since_improved += 1

            print(
                f"epoch {epoch+1:4d}/{t['epochs']}  "
                f"loss {train_stats['train/total']:8.4f}  "
                f"seg {train_stats['train/seg']:7.4f}  edge {train_stats['train/edge']:7.4f}  "
                f"| val mIoU {val_stats['seg/mIoU']:.4f}  "
                f"coast ODS {val_stats.get('edge/coastline/ODS_F1', float('nan')):.4f}  "
                f"limb ODS {val_stats.get('edge/limb/ODS_F1', float('nan')):.4f}  "
                f"| {monitor.split('/')[-1]} {score:.4f}"
                f"{'  *best*' if improved else ''}  "
                f"({row['epoch_sec']:.1f}s)"
            )

            if scheduler is not None and sched_when == "epoch":
                scheduler.step(score)
            if t.get("early_stop_patience") and since_improved >= t["early_stop_patience"]:
                print(f"early stop: no improvement in {since_improved} evaluations")
                break
        else:
            print(f"epoch {epoch+1:4d}/{t['epochs']}  "
                  f"loss {train_stats['train/total']:8.4f}  ({row['epoch_sec']:.1f}s)")

        # Rewrite the whole CSV each epoch: epochs that skip validation have
        # fewer keys, so the column set is only known incrementally, and a
        # crashed run still leaves a complete file behind.
        history.append(row)
        fieldnames = list(dict.fromkeys(k for r in history for k in r))
        with open(hist_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in history:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    torch.save({"model": model.state_dict(), "criterion": criterion.state_dict(),
                "epoch": t["epochs"], "config": resolved},
               os.path.join(out_dir, "last.pth"))

    mins = (time.time() - total_start) / 60
    print(f"\ndone in {mins:.1f} min. best {monitor} = {best:.4f} at epoch {best_epoch}")
    print(f"weights: {os.path.join(out_dir,'best.pth')} / last.pth")
    print(f"history: {hist_path}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"best_metric": monitor, "best_value": float(best),
                   "best_epoch": best_epoch, "minutes": mins}, f, indent=2)


if __name__ == "__main__":
    main()
