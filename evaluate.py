"""Final evaluation of a finished run (confusion matrix, prediction panels, failure cases).

Defaults to the VALIDATION split. The held-out test set is only used with --final, which you
run ONCE per run after all experiments and checkpoint choices are frozen.

    python evaluate.py --data_root D --run_id U1            # validation
    python evaluate.py --data_root D --run_id U1 --final     # test, final numbers
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from oilspill.data import OilSpillDataset, discover, find_pairs, select
from oilspill.metrics import NAMES, confusion, summarize
from oilspill.unet import UNet
from oilspill.viz import colorize, denorm


def eval_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--final", action="store_true", help="evaluate on the held-out TEST set")
    ap.add_argument("--out", default="results")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--n_panels", type=int, default=6)
    return ap


def run_eval(a):
    run = Path(a.out) / a.run_id
    cfg = json.load(open(run / "config.json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colors = [tuple(c) for c in cfg["colors"]]
    size_hw, norm = (cfg["height"], cfg["width"]), cfg["norm"]
    found = discover(a.data_root)
    split = json.load(open(a.splits))
    if a.final:
        items, tag = find_pairs(*found["test"], prefix="test_")[0], "test"
        assert {i[0] for i in items} == set(split["test"]), "test IDs changed since the split was made"
    else:
        items, tag = select(find_pairs(*found["train"])[0], split["val"]), "val"
    print(f"evaluating {a.run_id} on {tag}: {len(items)} images")

    model = UNet(3, 5, cfg["base"], cfg["depth"]).to(dev)
    model.load_state_dict(torch.load(run / "best_model.pth", map_location=dev, weights_only=True))
    model.eval()
    ds = OilSpillDataset(items, size_hw, "none", colors, norm, return_id=True)
    cm = torch.zeros(5, 5, dtype=torch.long, device=dev)
    per_img = []  # (oil IoU or nan, missed-ship flag, index)
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, _ = ds[i]
            pred = model(x[None].to(dev)).argmax(1)[0]
            y = y.to(dev)
            cm += confusion(pred, y)
            tp = ((pred == 1) & (y == 1)).sum().item(); un = ((pred == 1) | (y == 1)).sum().item()
            per_img.append((tp / un if (y == 1).any() else np.nan, bool((y == 3).any() and not (pred == 3).any()), i))
    m = summarize(cm.cpu().numpy())
    print(json.dumps(m, indent=1))

    mp = run / "metrics.json"
    allm = json.load(open(mp))
    allm[tag] = m
    json.dump(allm, open(mp, "w"), indent=1)

    c = cm.cpu().numpy().astype(float)
    norm_cm = c / np.maximum(c.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{norm_cm[i, j]:.2f}", ha="center", va="center", color="w" if norm_cm[i, j] > .5 else "k", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(NAMES, rotation=30); ax.set_yticks(range(5)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth"); ax.set_title(f"{a.run_id} {tag}: row-normalised confusion")
    fig.tight_layout(); fig.savefig(run / "confusion_matrix.png", dpi=150); plt.close(fig)

    def panel(idxs, fname, title):
        if not idxs:
            return
        fig, axs = plt.subplots(len(idxs), 3, figsize=(12, 2.6 * len(idxs)), squeeze=False)
        for r, i in enumerate(idxs):
            x, y, sid = ds[i]
            with torch.no_grad():
                p = model(x[None].to(dev)).argmax(1)[0].cpu().numpy()
            for k, (im, t) in enumerate([(denorm(x, norm), f"input {sid}"), (colorize(y.numpy(), colors), "ground truth"),
                                         (colorize(p, colors), "prediction")]):
                axs[r, k].imshow(im); axs[r, k].set_title(t, fontsize=8); axs[r, k].axis("off")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(run / fname, dpi=130); plt.close(fig)

    rnd = list(range(len(ds)))
    random.Random(0).shuffle(rnd)
    panel(rnd[: a.n_panels], "predictions.png", f"{a.run_id} {tag}: random samples")
    worst_oil = [i for s, _, i in sorted([t for t in per_img if not np.isnan(t[0])])[: a.n_panels // 2 + 1]]
    missed = [i for _, f, i in per_img if f][: a.n_panels // 2]
    panel(list(dict.fromkeys(worst_oil + missed)), "failure_cases.png",
          f"{a.run_id} {tag}: lowest oil IoU (oil vs look-alike) + missed ships")
    print(f"wrote confusion_matrix.png, predictions.png, failure_cases.png and metrics.json[{tag}] in {run}")


# ---- CLI ----
if __name__ == "__main__":
    run_eval(eval_parser().parse_args())
