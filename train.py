"""Train one U-Net experiment. Never touches the test set.

    U1 baseline            python train.py --data_root D --run_id U1  --loss ce
    U2 weights + Dice      python train.py --data_root D --run_id U2  --loss wce_dice
    ablations (optional)   --loss wce   (weights only)    --loss ce_dice  (Dice only)

Keep --seed/--height/--width/--bs/--epochs/--lr/--aug identical inside a controlled pair so only
the thing you are studying changes. Outputs (shared result contract) go to results/<run_id>/.
last.pth is saved every epoch; add --resume to continue a stopped run.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from oilspill.data import (NORMS, OilSpillDataset, class_pixel_counts, discover, find_pairs, load_colors,
                           load_or_make_split, select)
from oilspill.losses import NEEDS_WEIGHTS, build_loss, class_weights_from_counts
from oilspill.metrics import confusion, summarize
from oilspill.unet import UNet, count_params


def train_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--loss", choices=["ce", "wce", "ce_dice", "wce_dice"], default="ce")
    ap.add_argument("--weight_method", choices=["inv_sqrt", "median_freq"], default="inv_sqrt")
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--aug", choices=["none", "flip", "flip_int"], default="flip")
    ap.add_argument("--norm", choices=list(NORMS), default="half")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=512, help="images are ~2:1 wide; avoid squashing to a square")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--deterministic", action="store_true", help="slower; makes runs repeatable on the same GPU")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--class_map", default="audit/class_map.json")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--out", default="results")
    return ap


def seed_all(s, deterministic=False):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.use_deterministic_algorithms(True, warn_only=True)


def run_epoch(model, loader, crit, dev, opt=None, scaler=None, amp=False):
    train = opt is not None
    model.train(train)
    cm = torch.zeros(5, 5, dtype=torch.long, device=dev)
    loss_sum, ce_sum, n = 0.0, 0.0, 0
    ce_plain = torch.nn.CrossEntropyLoss()
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
            logits = logits.float()  # loss in fp32: fp16 logits overflow plain CE under AMP
            loss = crit(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
            b = x.size(0)
            loss_sum += loss.item() * b
            ce_sum += ce_plain(logits, y).item() * b  # unweighted CE: comparable across runs
            n += b
            cm += confusion(logits.argmax(1), y)
    return loss_sum / n, ce_sum / n, summarize(cm.cpu().numpy())


COLS = ["run_id", "epoch", "train_loss", "val_loss", "train_ce", "val_ce", "train_mIoU", "val_mIoU",
        "val_oil_IoU", "val_ship_IoU", "val_lookalike_IoU", "epoch_time_s"]


def run_train(a):
    seed_all(a.seed, a.deterministic)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not a.no_amp
    size_hw = (a.height, a.width)
    colors = load_colors(a.class_map)
    found = discover(a.data_root)
    train_pairs = find_pairs(*found["train"])[0]
    test_ids = [p[0] for p in find_pairs(*found["test"], prefix="test_")[0]] if "test" in found else []
    split = load_or_make_split(a.splits, [p[0] for p in train_pairs], test_ids, a.val_frac, a.seed)
    tr_items, va_items = select(train_pairs, split["train"]), select(train_pairs, split["val"])
    print(f"[{a.run_id}] train={len(tr_items)} val={len(va_items)} (test={len(test_ids)} untouched) "
          f"device={dev} amp={amp} input={a.height}x{a.width} norm={a.norm} aug={a.aug} loss={a.loss}")

    weights = None
    if a.loss in NEEDS_WEIGHTS:  # weights from the TRAIN split only
        counts = class_pixel_counts(tr_items, size_hw, colors)
        weights = class_weights_from_counts(counts, a.weight_method)
        print("train pixel counts:", counts.tolist(), "-> class weights:", [round(float(w), 3) for w in weights])
        weights = weights.to(dev)
    crit = build_loss(a.loss, weights, a.dice_w)

    def loader(items, aug, shuffle):
        ds = OilSpillDataset(items, size_hw, aug, colors, a.norm)
        return DataLoader(ds, batch_size=a.bs, shuffle=shuffle, num_workers=a.workers, pin_memory=dev.type == "cuda")
    dl_tr, dl_va = loader(tr_items, a.aug, True), loader(va_items, "none", False)

    model = UNet(3, 5, a.base, a.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda") if amp else None

    out = Path(a.out) / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(a), "n_params": count_params(model), "class_weights": None if weights is None else weights.tolist(),
              "norm_mean_std": NORMS[a.norm], "colors": [list(c) for c in colors], "pretrained": False,
              "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "torch": torch.__version__,
              "augmentation": {"none": "none", "flip": "random h+v flip", "flip_int": "flips + brightness/contrast jitter (image only)"}[a.aug]}
    json.dump(config, open(out / "config.json", "w"), indent=1)

    start, best, best_ep, best_val, elapsed, hist = 1, -1.0, 0, None, 0.0, []
    ckpt = out / "last.pth"
    if a.resume and ckpt.exists():
        s = torch.load(ckpt, map_location=dev, weights_only=True)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        if scaler and s.get("scaler"):
            scaler.load_state_dict(s["scaler"])
        start, best, best_ep, best_val, elapsed = s["epoch"] + 1, s["best"], s["best_ep"], s["best_val"], s["elapsed"]
        hist = [[r[0]] + [float(v) for v in r[1:]] for r in list(csv.reader(open(out / "history.csv")))[1:]]
        print(f"resumed from epoch {s['epoch']} (best so far epoch {best_ep}, val mIoU {best:.4f})")

    t0 = time.time() - elapsed
    nan = float("nan")
    with open(out / "history.csv", "a" if start > 1 else "w", newline="") as fh:
        w = csv.writer(fh)
        if start == 1:
            w.writerow(COLS)
        for ep in range(start, a.epochs + 1):
            te = time.time()
            trl, trce, trm = run_epoch(model, dl_tr, crit, dev, opt, scaler, amp)
            val, vace, vam = run_epoch(model, dl_va, crit, dev, amp=amp)
            sched.step()
            g = lambda d, k: d[k] if d[k] is not None else nan
            row = [a.run_id, ep, trl, val, trce, vace, g(trm, "mIoU"), g(vam, "mIoU"), g(vam, "IoU_oil_spill"),
                   g(vam, "IoU_ship"), g(vam, "IoU_look_alike"), time.time() - te]
            w.writerow(row); fh.flush(); hist.append(row)
            score = g(vam, "mIoU")
            if score > best:  # checkpoint rule: highest VALIDATION mIoU
                best, best_ep, best_val = score, ep, vam
                torch.save(model.state_dict(), out / "best_model.pth")
            torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict() if scaler else None, "best": best, "best_ep": best_ep,
                        "best_val": best_val, "elapsed": time.time() - t0}, ckpt)  # survives a crash / timeout
            print(f"ep {ep:3d} | train {trl:.4f} val {val:.4f} | mIoU tr {g(trm,'mIoU'):.3f} va {score:.3f} | "
                  f"oil {row[8]:.3f} ship {row[9]:.3f} | {row[-1]:.0f}s" + ("  *best" if best_ep == ep else ""))

    total = time.time() - t0
    json.dump({"run_id": a.run_id, "selection_rule": "best validation mIoU", "best_epoch": best_ep,
               "train_time_s": total, "val": best_val}, open(out / "metrics.json", "w"), indent=1)

    arr = np.array([r[1:] for r in hist], dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(arr[:, 0], arr[:, 1], label="train"); ax[0].plot(arr[:, 0], arr[:, 2], label="val"); ax[0].set_title("training objective")
    ax[1].plot(arr[:, 0], arr[:, 3], label="train"); ax[1].plot(arr[:, 0], arr[:, 4], label="val"); ax[1].set_title("plain CE (comparable across runs)")
    ax[2].plot(arr[:, 0], arr[:, 5], label="train mIoU"); ax[2].plot(arr[:, 0], arr[:, 6], label="val mIoU")
    ax[2].plot(arr[:, 0], arr[:, 7], "--", label="val oil IoU"); ax[2].plot(arr[:, 0], arr[:, 8], "--", label="val ship IoU"); ax[2].set_title("IoU")
    for x in ax:
        x.set_xlabel("epoch"); x.legend(); x.axvline(best_ep, color="gray", ls=":")
    fig.suptitle(a.run_id); fig.tight_layout(); fig.savefig(out / "curves.png", dpi=140); plt.close(fig)
    print(f"[{a.run_id}] done in {total/60:.1f} min | best epoch {best_ep} val mIoU {best:.4f} | outputs in {out}")


# ---- CLI ----
if __name__ == "__main__":
    run_train(train_parser().parse_args())
